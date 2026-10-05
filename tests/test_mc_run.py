import json
import pickle
import textwrap

import pandas as pd
import pytest

from hab_adcs_sim.monte_carlo import run as mc_run
from hab_adcs_sim.monte_carlo.generate import generate_campaign
from hab_adcs_sim.monte_carlo.run import campaign_status, run_campaign
from hab_adcs_sim.monte_carlo.spec import load_spec

SPEC = """
    name = "unit"
    base_parameters = "config/sim_params/parameters_ec60.json"
    samples_per_case = 1
    include_nominal = true
    noise_seed = 0
    sampling = { method = "random", seed = 2 }

    [overrides]
    "simulation.duration" = 0.05

    [cases.dump_on]
    "lt_motor.activate" = true

    [cases.dump_off]
    "lt_motor.activate" = false

    [dispersions."Payload_params.Ip"]
    dist = "normal"
    sigma_rel = 0.1
"""


@pytest.fixture
def campaign_dir(tmp_path):
    spec_path = tmp_path / "unit.toml"
    spec_path.write_text(textwrap.dedent(SPEC), encoding="utf-8")
    campaign_dir, _ = generate_campaign(load_spec(spec_path), root=tmp_path / "mc")
    return campaign_dir


def silent(*args, **kwargs) -> None:
    pass


def test_run_campaign_simulates_and_analyses_every_run(campaign_dir):
    totals = run_campaign(campaign_dir, workers=0, echo=silent)

    assert totals["failed"] == 0
    assert totals["done"] == 8  # 4 runs x 2 stages
    for run_dir in mc_run.iter_run_dirs(campaign_dir):
        assert (run_dir / "sim_output.pkl").is_file()
        assert (run_dir / "analysis.pkl").is_file()


def test_analysis_holds_only_plain_types(campaign_dir):
    run_campaign(campaign_dir, workers=0, echo=silent)
    payload = pickle.load((campaign_dir / "dump_on" / "run_0000" / "analysis.pkl").open("rb"))

    assert isinstance(payload["metrics"], dict)
    assert all(isinstance(value, float) for value in payload["metrics"].values())
    assert isinstance(payload["timeseries"], pd.DataFrame)
    assert payload["case"] == "dump_on"
    assert "time" in payload["timeseries"].columns


def test_sim_output_keeps_the_single_run_envelope(campaign_dir):
    run_campaign(campaign_dir, stage="sim", workers=0, echo=silent)
    cached = pickle.load((campaign_dir / "dump_on" / "run_0000" / "sim_output.pkl").open("rb"))
    assert set(cached) == {"parameters", "data"}
    assert isinstance(cached["data"], pd.DataFrame)


def test_a_second_run_skips_completed_work(campaign_dir):
    run_campaign(campaign_dir, workers=0, echo=silent)
    totals = run_campaign(campaign_dir, workers=0, echo=silent)

    assert totals["done"] == 0
    assert totals["failed"] == 0
    assert totals["skipped"] == 8


def test_stale_analysis_is_recomputed_without_resimulating(campaign_dir, monkeypatch):
    run_campaign(campaign_dir, workers=0, echo=silent)
    sim_output = campaign_dir / "dump_on" / "run_0000" / "sim_output.pkl"
    simulated_at = sim_output.stat().st_mtime_ns

    monkeypatch.setattr(mc_run, "ANALYSIS_VERSION", 999)
    totals = run_campaign(campaign_dir, workers=0, echo=silent)

    assert totals["done"] == 4  # the analysis stage only
    assert sim_output.stat().st_mtime_ns == simulated_at


def test_one_bad_run_does_not_stop_the_campaign(campaign_dir):
    broken = campaign_dir / "dump_on" / "run_0001"
    params = json.loads((broken / "params.json").read_text())
    del params["simulation"]["duration"]
    (broken / "params.json").write_text(json.dumps(params))

    totals = run_campaign(campaign_dir, workers=0, echo=silent)

    assert totals["failed"] == 1
    assert (broken / "error.txt").is_file()
    assert "duration" in (broken / "error.txt").read_text()
    assert (campaign_dir / "dump_off" / "run_0001" / "analysis.pkl").is_file()


def test_status_counts_each_stage(campaign_dir):
    before = campaign_status(campaign_dir)
    assert before["simulated"].sum() == 0

    run_campaign(campaign_dir, workers=0, echo=silent)
    after = campaign_status(campaign_dir)

    assert list(after["case"]) == ["dump_off", "dump_on"]
    assert after["generated"].sum() == 4
    assert after["analysed"].sum() == 4
    assert after["failed"].sum() == 0


def test_only_case_restricts_the_work(campaign_dir):
    run_campaign(campaign_dir, only_case="dump_on", workers=0, echo=silent)

    assert (campaign_dir / "dump_on" / "run_0000" / "analysis.pkl").is_file()
    assert not (campaign_dir / "dump_off" / "run_0000" / "sim_output.pkl").exists()
