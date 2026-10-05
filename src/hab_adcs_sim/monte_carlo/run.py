"""Simulate and analyse the runs of a campaign.

Runs with output already on disk are skipped, so an interrupted campaign
resumes when run again. A failed run writes its traceback to error.txt.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from hab_adcs_sim.simulation_runner import run_simulation, save_results

from .generate import PARAMS_FILE, RUN_FILE
from .per_run_analysis import ANALYSIS_VERSION, analyze

SIM_OUTPUT = "sim_output.pkl"
ANALYSIS_OUTPUT = "analysis.pkl"
ERROR_FILE = "error.txt"

STAGES = ("sim", "analysis", "all")

# Retry the rename, OneDrive sometimes locks files while syncing
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_S = 0.4


def _replace_with_retry(tmp: Path, final: Path) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, final)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))


def _write_pickle_atomically(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f)
    _replace_with_retry(tmp, path)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_run_dirs(campaign_dir: Path, only_case: str | None = None) -> list[Path]:
    campaign_dir = Path(campaign_dir)
    run_dirs = sorted(
        path.parent
        for path in campaign_dir.glob("*/run_*/" + PARAMS_FILE)
        if only_case is None or path.parent.parent.name == only_case
    )
    return run_dirs


def _analysis_is_current(run_dir: Path) -> bool:
    analysis_path = run_dir / ANALYSIS_OUTPUT
    if not analysis_path.is_file():
        return False
    try:
        with analysis_path.open("rb") as f:
            return pickle.load(f).get("analysis_version") == ANALYSIS_VERSION
    except (pickle.UnpicklingError, EOFError, AttributeError):
        return False


def simulate_run(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    params = _read_json(run_dir / PARAMS_FILE)
    meta = _read_json(run_dir / RUN_FILE)

    df = run_simulation(params, seed=meta["noise_seed"], wind_seed=meta.get("wind_seed"))

    # Same format as a single run's results
    tmp = run_dir / (SIM_OUTPUT + ".tmp")
    save_results(tmp, df, params)
    _replace_with_retry(tmp, run_dir / SIM_OUTPUT)


def analyze_run(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / RUN_FILE)
    with (run_dir / SIM_OUTPUT).open("rb") as f:
        cached = pickle.load(f)

    payload = analyze(cached["data"], cached["parameters"])
    payload.update(
        {
            "run_id": meta["run_id"],
            "case": meta["case"],
            "sample": meta["sample"],
            "is_nominal": meta["is_nominal"],
            "dispersed": meta["dispersed"],
        }
    )
    _write_pickle_atomically(run_dir / ANALYSIS_OUTPUT, payload)


def _execute(run_dir_str: str, stage: str) -> tuple[str, str, float, str]:
    """Run one stage on one run directory, failures are recorded instead of raised."""
    run_dir = Path(run_dir_str)
    started = time.perf_counter()
    try:
        if stage == "sim":
            simulate_run(run_dir)
        else:
            analyze_run(run_dir)
    except Exception:
        (run_dir / ERROR_FILE).write_text(
            f"stage: {stage}\n\n{traceback.format_exc()}", encoding="utf-8"
        )
        return run_dir_str, "failed", time.perf_counter() - started, traceback.format_exc().splitlines()[-1]

    error_file = run_dir / ERROR_FILE
    if error_file.is_file():
        error_file.unlink()
    return run_dir_str, "done", time.perf_counter() - started, ""


def _pending(campaign_dir: Path, stage: str, only_case: str | None, rerun: bool) -> list[Path]:
    pending = []
    for run_dir in iter_run_dirs(campaign_dir, only_case):
        if stage == "sim":
            if rerun or not (run_dir / SIM_OUTPUT).is_file():
                pending.append(run_dir)
        elif (run_dir / SIM_OUTPUT).is_file() and (rerun or not _analysis_is_current(run_dir)):
            pending.append(run_dir)
    return pending


def run_campaign(
    campaign_dir: str | Path,
    stage: str = "all",
    workers: int = 3,
    only_case: str | None = None,
    rerun: bool = False,
    echo=print,
) -> dict[str, int]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}, expected one of {STAGES}")
    campaign_dir = Path(campaign_dir)
    stages = ("sim", "analysis") if stage == "all" else (stage,)

    totals = {"done": 0, "failed": 0, "skipped": 0}
    for current in stages:
        pending = _pending(campaign_dir, current, only_case, rerun)
        all_runs = iter_run_dirs(campaign_dir, only_case)
        totals["skipped"] += len(all_runs) - len(pending)
        if not pending:
            echo(f"[{current}] nothing to do ({len(all_runs)} run(s) already complete)")
            continue

        echo(f"[{current}] {len(pending)} run(s) to process on {workers or 1} worker(s)")
        elapsed_total = 0.0
        for index, (run_dir_str, status, elapsed, message) in enumerate(
            _dispatch(pending, current, workers), start=1
        ):
            totals[status] += 1
            elapsed_total += elapsed
            eta = (elapsed_total / index) * (len(pending) - index) / max(workers, 1)
            suffix = f" - {message}" if message else ""
            echo(
                f"[{current}] {index}/{len(pending)} {status} "
                f"{Path(run_dir_str).parent.name}/{Path(run_dir_str).name} "
                f"({elapsed:.1f}s, eta {eta / 60:.1f} min){suffix}"
            )

    return totals


def _dispatch(pending: list[Path], stage: str, workers: int):
    if workers <= 0:
        for run_dir in pending:
            yield _execute(str(run_dir), stage)
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_execute, str(run_dir), stage) for run_dir in pending]
        for future in as_completed(futures):
            yield future.result()


def campaign_status(campaign_dir: str | Path) -> pd.DataFrame:
    rows = []
    for run_dir in iter_run_dirs(campaign_dir):
        rows.append(
            {
                "case": run_dir.parent.name,
                "generated": 1,
                "simulated": int((run_dir / SIM_OUTPUT).is_file()),
                "analysed": int(_analysis_is_current(run_dir)),
                "failed": int((run_dir / ERROR_FILE).is_file()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["case", "generated", "simulated", "analysed", "failed"])
    return pd.DataFrame(rows).groupby("case", as_index=False).sum()
