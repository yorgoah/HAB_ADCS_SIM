import csv
import json
import textwrap

import pytest

from hab_adcs_sim.monte_carlo.generate import generate_campaign
from hab_adcs_sim.monte_carlo.spec import SpecError, load_spec

SPEC = """
    name = "unit"
    base_parameters = "config/sim_params/parameters_ec60.json"
    samples_per_case = 3
    include_nominal = true
    noise_seed = 5
    sampling = { method = "random", seed = 11 }

    [overrides]
    "simulation.duration" = 0.05

    [cases.dump_on]
    "lt_motor.activate" = true

    [cases.dump_off]
    "lt_motor.activate" = false

    [dispersions."Payload_params.Ip"]
    dist = "normal"
    sigma_rel = 0.1
    min = 0.005

    [dispersions."rw_motor.torque_constant"]
    dist = "normal"
    sigma_rel = 0.05
    link = ["rw_motor.back_emf_constant"]
"""


@pytest.fixture
def spec_path(tmp_path):
    path = tmp_path / "unit.toml"
    path.write_text(textwrap.dedent(SPEC), encoding="utf-8")
    return path


@pytest.fixture
def campaign(tmp_path, spec_path):
    campaign_dir, created = generate_campaign(load_spec(spec_path), root=tmp_path / "mc")
    return campaign_dir, created


def read_params(campaign_dir, run_id):
    with (campaign_dir / run_id / "params.json").open() as f:
        return json.load(f)


def test_generate_writes_every_run_directory(campaign):
    campaign_dir, created = campaign
    assert created == 8  # 2 cases x (1 nominal + 3 samples)
    for case in ("dump_on", "dump_off"):
        for index in range(4):
            run_dir = campaign_dir / case / f"run_{index:04d}"
            assert (run_dir / "params.json").is_file()
            assert (run_dir / "run.json").is_file()


def test_nominal_run_is_the_base_plus_overrides_only(campaign, base_params):
    campaign_dir, _ = campaign
    params = read_params(campaign_dir, "dump_on/run_0000")
    assert params["simulation"]["duration"] == 0.05
    assert params["Payload_params"]["Ip"] == base_params["Payload_params"]["Ip"]
    assert params["rw_motor"]["torque_constant"] == base_params["rw_motor"]["torque_constant"]


def test_case_override_reaches_the_written_parameters(campaign):
    campaign_dir, _ = campaign
    assert read_params(campaign_dir, "dump_on/run_0001")["lt_motor"]["activate"] is True
    assert read_params(campaign_dir, "dump_off/run_0001")["lt_motor"]["activate"] is False


def test_dispersed_runs_differ_from_the_nominal(campaign, base_params):
    campaign_dir, _ = campaign
    nominal_ip = base_params["Payload_params"]["Ip"]
    drawn = [
        read_params(campaign_dir, f"dump_on/run_{i:04d}")["Payload_params"]["Ip"] for i in (1, 2, 3)
    ]
    assert all(value != nominal_ip for value in drawn)
    assert len(set(drawn)) == 3
    assert all(value >= 0.005 for value in drawn)


def test_linked_parameter_tracks_its_partner(campaign):
    campaign_dir, _ = campaign
    params = read_params(campaign_dir, "dump_on/run_0002")["rw_motor"]
    assert params["back_emf_constant"] == pytest.approx(params["torque_constant"])


def test_cases_are_paired_run_for_run(campaign):
    campaign_dir, _ = campaign
    on = read_params(campaign_dir, "dump_on/run_0002")
    off = read_params(campaign_dir, "dump_off/run_0002")
    assert on["Payload_params"]["Ip"] == off["Payload_params"]["Ip"]
    assert on["rw_motor"]["torque_constant"] == off["rw_motor"]["torque_constant"]


def test_run_metadata_records_the_seed_and_draws(campaign):
    campaign_dir, _ = campaign
    with (campaign_dir / "dump_off" / "run_0001" / "run.json").open() as f:
        meta = json.load(f)
    assert meta["case"] == "dump_off"
    assert meta["noise_seed"] == 5
    assert meta["is_nominal"] is False
    assert set(meta["dispersed"]) == {
        "Payload_params.Ip",
        "rw_motor.torque_constant",
        "rw_motor.back_emf_constant",
    }


def test_manifest_lists_every_run_with_its_draws(campaign):
    campaign_dir, _ = campaign
    with (campaign_dir / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert {row["case"] for row in rows} == {"dump_on", "dump_off"}
    assert "Payload_params.Ip" in rows[0]


def test_regenerating_the_same_spec_creates_nothing(tmp_path, spec_path, campaign):
    campaign_dir, _ = campaign
    _, created_again = generate_campaign(load_spec(spec_path), root=tmp_path / "mc")
    assert created_again == 0


def test_a_changed_spec_is_refused_unless_forced(tmp_path, spec_path, campaign):
    spec_path.write_text(
        textwrap.dedent(SPEC).replace("samples_per_case = 3", "samples_per_case = 4"),
        encoding="utf-8",
    )
    changed = load_spec(spec_path)

    with pytest.raises(SpecError, match="different campaign file"):
        generate_campaign(changed, root=tmp_path / "mc")

    _, created = generate_campaign(changed, root=tmp_path / "mc", force=True)
    assert created == 2  # one more sample per case, existing runs untouched


def read_meta(campaign_dir, run_id):
    with (campaign_dir / run_id / "run.json").open() as f:
        return json.load(f)


@pytest.fixture
def windy_campaign(tmp_path):
    path = tmp_path / "windy.toml"
    path.write_text(
        textwrap.dedent(SPEC).replace("noise_seed = 5", "noise_seed = 5\nvary_wind = true"),
        encoding="utf-8",
    )
    campaign_dir, _ = generate_campaign(load_spec(path), root=tmp_path / "mc")
    return campaign_dir


def test_wind_is_held_fixed_unless_asked_to_vary(campaign):
    campaign_dir, _ = campaign
    assert read_meta(campaign_dir, "dump_on/run_0001")["wind_seed"] is None


def test_varied_wind_differs_between_runs_but_pairs_across_cases(windy_campaign):
    seeds = [read_meta(windy_campaign, f"dump_on/run_{i:04d}")["wind_seed"] for i in range(4)]
    assert all(isinstance(seed, int) for seed in seeds)
    assert len(set(seeds)) == 4
    assert read_meta(windy_campaign, "dump_off/run_0002")["wind_seed"] == seeds[2]


def test_manifest_records_the_wind_seed(windy_campaign):
    with (windy_campaign / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_id = {row["run_id"]: row for row in rows}
    assert int(by_id["dump_on/run_0003"]["wind_seed"]) == read_meta(windy_campaign, "dump_on/run_0003")["wind_seed"]
