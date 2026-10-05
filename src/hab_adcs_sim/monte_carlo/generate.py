"""Write one directory per run with its full parameter file and seeds."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

from .params import get_path, set_path
from .sampling import disperse_value, linked_values, unit_samples
from .spec import REPO_ROOT, CampaignSpec, Case, SpecError, load_spec

DEFAULT_ROOT = REPO_ROOT / "monte_carlo"

CAMPAIGN_FILE = "campaign.toml"
MANIFEST_FILE = "manifest.csv"
PARAMS_FILE = "params.json"
RUN_FILE = "run.json"


def run_dir_name(index: int) -> str:
    return f"run_{index:04d}"


def wind_seed(spec: CampaignSpec, run_index: int) -> int | None:
    """Wind seed of run `run_index`, the same in every case. None keeps the wind fixed."""
    if not spec.vary_wind:
        return None
    return int(np.random.SeedSequence([spec.noise_seed, run_index]).generate_state(1)[0])


def build_run_params(
    spec: CampaignSpec, case: Case, unit_row: np.ndarray | None
) -> tuple[dict, dict[str, float]]:
    """Parameters for one run and the dispersed values.

    Applied in order: campaign overrides, case overrides, dispersions.
    """
    params = copy.deepcopy(spec.base_params)
    for path, value in spec.overrides.items():
        set_path(params, path, value)
    for path, value in case.overrides.items():
        set_path(params, path, value)

    dispersed: dict[str, float] = {}
    if unit_row is None:
        return params, dispersed

    for column, dispersion in enumerate(spec.dispersions):
        nominal = get_path(params, dispersion.path)
        value = disperse_value(nominal, float(unit_row[column]), dispersion)
        set_path(params, dispersion.path, value)
        dispersed[dispersion.path] = value

        linked_nominals = {path: get_path(params, path) for path in dispersion.link}
        for path, linked_value in linked_values(nominal, value, linked_nominals).items():
            set_path(params, path, linked_value)
            dispersed[path] = linked_value

    return params, dispersed


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


def _write_manifest(campaign_dir: Path, rows: list[dict], dispersed_columns: list[str]) -> None:
    columns = [
        "run_id", "case", "run_index", "sample", "is_nominal", "noise_seed", "wind_seed",
        *dispersed_columns,
    ]
    with (campaign_dir / MANIFEST_FILE).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_campaign(
    spec: CampaignSpec, root: Path | None = None, force: bool = False
) -> tuple[Path, int]:
    """Write the run directories that don't exist yet. Returns (campaign_dir, created)."""
    root = DEFAULT_ROOT if root is None else Path(root)
    campaign_dir = root / spec.name
    existing_spec = campaign_dir / CAMPAIGN_FILE

    if existing_spec.is_file():
        if existing_spec.read_text(encoding="utf-8") != spec.source_text and not force:
            raise SpecError(f"{campaign_dir} was generated from a different campaign file, use --force")

    campaign_dir.mkdir(parents=True, exist_ok=True)
    existing_spec.write_text(spec.source_text, encoding="utf-8")

    unit = unit_samples(
        spec.samples_per_case, len(spec.dispersions), spec.sampling_seed, spec.sampling_method
    )

    rows: list[dict] = []
    dispersed_columns: list[str] = []
    created = 0

    for case in spec.cases:
        for run_index in range(spec.runs_per_case):
            if spec.include_nominal and run_index == 0:
                sample, unit_row = -1, None
            else:
                sample = run_index - 1 if spec.include_nominal else run_index
                unit_row = unit[sample]

            params, dispersed = build_run_params(spec, case, unit_row)
            for path in dispersed:
                if path not in dispersed_columns:
                    dispersed_columns.append(path)

            run_dir = campaign_dir / case.name / run_dir_name(run_index)
            run_id = f"{case.name}/{run_dir_name(run_index)}"
            if not (run_dir / PARAMS_FILE).is_file():
                run_dir.mkdir(parents=True, exist_ok=True)
                _write_json(run_dir / PARAMS_FILE, params)
                _write_json(
                    run_dir / RUN_FILE,
                    {
                        "campaign": spec.name,
                        "run_id": run_id,
                        "case": case.name,
                        "run_index": run_index,
                        "sample": sample,
                        "is_nominal": unit_row is None,
                        "noise_seed": spec.noise_seed,
                        "wind_seed": wind_seed(spec, run_index),
                        "case_overrides": case.overrides,
                        "dispersed": dispersed,
                    },
                )
                created += 1

            rows.append(
                {
                    "run_id": run_id,
                    "case": case.name,
                    "run_index": run_index,
                    "sample": sample,
                    "is_nominal": unit_row is None,
                    "noise_seed": spec.noise_seed,
                    "wind_seed": wind_seed(spec, run_index),
                    **dispersed,
                }
            )

    _write_manifest(campaign_dir, rows, dispersed_columns)
    return campaign_dir, created


def generate_from_file(spec_path: str | Path, root: Path | None = None, force: bool = False):
    return generate_campaign(load_spec(spec_path), root=root, force=force)
