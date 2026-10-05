import textwrap

import pytest

from hab_adcs_sim.monte_carlo.params import get_path, is_numeric_leaf, set_path
from hab_adcs_sim.monte_carlo.spec import SpecError, load_spec

BASE = 'base_parameters = "config/sim_params/parameters_ec60.json"'


def write_spec(tmp_path, body: str):
    path = tmp_path / "campaign.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_get_path_reads_a_nested_leaf(base_params):
    assert get_path(base_params, "rw_motor.max_current") == base_params["rw_motor"]["max_current"]


def test_get_path_raises_naming_the_whole_path(base_params):
    with pytest.raises(KeyError, match="rw_motor.nope"):
        get_path(base_params, "rw_motor.nope")


def test_set_path_writes_an_existing_leaf(base_params):
    set_path(base_params, "Payload_params.Ip", 0.02)
    assert base_params["Payload_params"]["Ip"] == 0.02


def test_set_path_refuses_to_create_a_new_key(base_params):
    with pytest.raises(KeyError):
        set_path(base_params, "Payload_params.Iq", 1.0)
    assert "Iq" not in base_params["Payload_params"]


def test_bool_is_not_a_numeric_leaf():
    assert is_numeric_leaf(1.5) and is_numeric_leaf(3)
    assert not is_numeric_leaf(True)
    assert not is_numeric_leaf("0.5")


def test_load_spec_reads_a_minimal_campaign(tmp_path):
    spec = load_spec(write_spec(tmp_path, f"""
        name = "unit"
        {BASE}
        samples_per_case = 4

        [dispersions."Payload_params.Ip"]
        dist = "normal"
        sigma_rel = 0.1
    """))
    assert spec.name == "unit"
    assert spec.samples_per_case == 4
    assert spec.runs_per_case == 5  # include_nominal defaults on
    assert [case.name for case in spec.cases] == ["nominal"]
    assert spec.total_runs == 5


def test_unknown_dispersion_path_is_rejected(tmp_path):
    with pytest.raises(SpecError, match="not in the parameter file"):
        load_spec(write_spec(tmp_path, f"""
            name = "unit"
            {BASE}
            samples_per_case = 2

            [dispersions."rw_motor.made_up_gain"]
            dist = "normal"
            sigma_rel = 0.1
        """))


def test_dispersing_a_switch_is_rejected(tmp_path):
    with pytest.raises(SpecError, match="not a number"):
        load_spec(write_spec(tmp_path, f"""
            name = "unit"
            {BASE}
            samples_per_case = 2

            [dispersions."lt_motor.activate"]
            dist = "normal"
            sigma_rel = 0.1
        """))


def test_normal_dispersion_needs_exactly_one_width(tmp_path):
    with pytest.raises(SpecError, match="sigma or sigma_rel"):
        load_spec(write_spec(tmp_path, f"""
            name = "unit"
            {BASE}
            samples_per_case = 2

            [dispersions."Payload_params.Ip"]
            dist = "normal"
            sigma = 0.001
            sigma_rel = 0.1
        """))


def test_a_parameter_cannot_be_dispersed_twice(tmp_path):
    with pytest.raises(SpecError, match="more than once"):
        load_spec(write_spec(tmp_path, f"""
            name = "unit"
            {BASE}
            samples_per_case = 2

            [dispersions."rw_motor.torque_constant"]
            dist = "normal"
            sigma_rel = 0.05
            link = ["rw_motor.back_emf_constant"]

            [dispersions."rw_motor.back_emf_constant"]
            dist = "normal"
            sigma_rel = 0.05
        """))


def test_samples_without_dispersions_is_rejected(tmp_path):
    with pytest.raises(SpecError, match="no \\[dispersions\\]"):
        load_spec(write_spec(tmp_path, f"""
            name = "unit"
            {BASE}
            samples_per_case = 10
        """))


def test_unknown_case_override_is_rejected(tmp_path):
    with pytest.raises(SpecError, match="not in the parameter file"):
        load_spec(write_spec(tmp_path, f"""
            name = "unit"
            {BASE}
            samples_per_case = 0

            [cases.bad]
            "lt_motor.enabled" = false
        """))


def test_cases_cross_with_the_grid(tmp_path):
    spec = load_spec(write_spec(tmp_path, f"""
        name = "unit"
        {BASE}
        samples_per_case = 0

        [cases.dump_on]

        [cases.dump_off]
        "lt_motor.activate" = false

        [grid]
        "rw_motor.proportional_gain" = [6000.0, 8000.0]
    """))

    names = [case.name for case in spec.cases]
    assert names == [
        "dump_on__proportional_gain=6000",
        "dump_on__proportional_gain=8000",
        "dump_off__proportional_gain=6000",
        "dump_off__proportional_gain=8000",
    ]
    dump_off_6000 = spec.cases[2]
    assert dump_off_6000.overrides["lt_motor.activate"] is False
    assert dump_off_6000.overrides["rw_motor.proportional_gain"] == 6000.0


def test_case_names_stay_filesystem_safe(tmp_path):
    spec = load_spec(write_spec(tmp_path, f"""
        name = "unit"
        {BASE}
        samples_per_case = 0

        [cases."dump: on/off"]
    """))
    assert spec.cases[0].name == "dump__on_off"


def test_vary_wind_defaults_off(tmp_path):
    spec = load_spec(write_spec(tmp_path, f"""
        name = "unit"
        {BASE}
        samples_per_case = 0
    """))
    assert spec.vary_wind is False


def test_vary_wind_with_a_replayed_log_is_rejected(tmp_path):
    with pytest.raises(SpecError, match="replay"):
        load_spec(write_spec(tmp_path, f"""
            name = "unit"
            {BASE}
            samples_per_case = 0
            vary_wind = true

            [cases.surrogate]

            [cases.replayed]
            "wind_params.model" = "replay"
        """))


def test_vary_wind_must_be_a_boolean(tmp_path):
    with pytest.raises(SpecError, match="vary_wind"):
        load_spec(write_spec(tmp_path, f"""
            name = "unit"
            {BASE}
            samples_per_case = 0
            vary_wind = "yes"
        """))
