import copy

import numpy as np
import pytest

from hab_adcs_sim.simulation_runner import run_simulation
from sim_tools.disturbance import DisturbanceGenerator
from sim_tools.wind_surrogate import gust_rms, load_yaw_acceleration, log_path, steady_torque

GUST_LAW_KEYS = ("gust_rms_sea_level", "gust_density_exponent", "gust_rms_log_sigma")


@pytest.fixture
def flight_data_params(short_params):
    params = copy.deepcopy(short_params)
    params["wind_params"]["model"] = "replay"
    return params


@pytest.fixture
def synthetic_wind_params(short_params):
    params = copy.deepcopy(short_params)
    params["wind_params"]["model"] = "synthetic"
    return params


@pytest.fixture
def surrogate_params(short_params):
    params = copy.deepcopy(short_params)
    params["wind_params"]["model"] = "surrogate"
    params["simulation"]["duration"] = 5.0
    return params


def torque_series(gen, duration, step=0.01):
    return np.array([gen.generate_torque_disturbance(t) for t in np.arange(0.0, duration, step)])


def test_flight_data_disturbance_is_finite_across_run(flight_data_params):
    gen = DisturbanceGenerator(flight_data_params)
    dt = flight_data_params["simulation"]["time_step"]
    duration = flight_data_params["simulation"]["duration"]
    t = 0.0
    while t <= duration:
        value = gen.generate_torque_disturbance(t)
        assert np.isfinite(value)
        t += dt


@pytest.mark.parametrize("model", ["replay", "surrogate"])
def test_flight_disturbance_ignores_the_simulated_payload_inertia(short_params, model):
    params = copy.deepcopy(short_params)
    params["wind_params"]["model"] = model
    heavier = copy.deepcopy(params)
    heavier["Payload_params"]["Ip"] *= 2.0

    a = torque_series(DisturbanceGenerator(params, rng=np.random.default_rng(1)), 0.2)
    b = torque_series(DisturbanceGenerator(heavier, rng=np.random.default_rng(1)), 0.2)
    np.testing.assert_array_equal(a, b)


def test_flight_disturbance_scales_with_the_flight_inertia(flight_data_params):
    gen_a = DisturbanceGenerator(flight_data_params)

    params_b = copy.deepcopy(flight_data_params)
    params_b["wind_params"]["flight_inertia"] *= 2.0
    gen_b = DisturbanceGenerator(params_b)

    assert gen_b.generate_torque_disturbance(0.0) == pytest.approx(
        2.0 * gen_a.generate_torque_disturbance(0.0)
    )


def test_parameter_files_without_a_wind_model_keep_their_old_meaning(short_params):
    # Older files have a `simulated` switch and scale the log by Ip
    legacy = copy.deepcopy(short_params)
    wind = legacy["wind_params"]
    for key in ("model", "file", "flight_inertia", "source_range"):
        del wind[key]
    wind["simulated"] = False
    legacy["Payload_params"]["Ip"] = 0.03

    gen = DisturbanceGenerator(legacy)
    assert gen.model == "replay"
    expected = 0.03 * np.interp(gen.start_time, *load_yaw_acceleration(log_path("log137")))
    assert gen.generate_torque_disturbance(0.0) == pytest.approx(expected)

    wind["simulated"] = True
    assert DisturbanceGenerator(legacy, rng=np.random.default_rng(0)).model == "synthetic"


def test_unknown_wind_model_is_rejected(short_params):
    short_params["wind_params"]["model"] = "gusty"
    with pytest.raises(ValueError, match="gusty"):
        DisturbanceGenerator(short_params)


def test_surrogate_disturbance_is_finite_past_the_end_of_the_run(surrogate_params):
    gen = DisturbanceGenerator(surrogate_params, rng=np.random.default_rng(0))
    duration = surrogate_params["simulation"]["duration"]
    dt = surrogate_params["simulation"]["time_step"]
    tau = torque_series(gen, duration + 2 * dt, step=dt)
    assert np.all(np.isfinite(tau))
    assert np.std(tau) > 0.0


def test_surrogate_reproduces_from_its_seed(surrogate_params):
    duration = surrogate_params["simulation"]["duration"]
    a = torque_series(DisturbanceGenerator(surrogate_params, rng=np.random.default_rng(7)), duration)
    b = torque_series(DisturbanceGenerator(surrogate_params, rng=np.random.default_rng(7)), duration)
    c = torque_series(DisturbanceGenerator(surrogate_params, rng=np.random.default_rng(8)), duration)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def without_steady_torque(params):
    params = copy.deepcopy(params)
    params["wind_params"]["steady_torque_per_density"] = 0.0
    return params


def test_surrogate_adds_the_steady_torque_of_the_ascent(surrogate_params):
    # Same seed so the gusts match
    duration = surrogate_params["simulation"]["duration"]
    with_it = torque_series(DisturbanceGenerator(surrogate_params, rng=np.random.default_rng(3)), duration)
    gusts = torque_series(
        DisturbanceGenerator(without_steady_torque(surrogate_params), rng=np.random.default_rng(3)), duration
    )
    t = np.arange(0.0, duration, 0.01)
    assert surrogate_params["wind_params"]["steady_torque_per_density"] > 0.0
    np.testing.assert_allclose(with_it - gusts, steady_torque(t, surrogate_params["wind_params"]), rtol=1e-9)


@pytest.mark.parametrize("model", ["replay", "synthetic"])
def test_steady_torque_belongs_to_the_surrogate_only(short_params, model):
    params = copy.deepcopy(short_params)
    params["wind_params"]["model"] = model
    a = torque_series(DisturbanceGenerator(params, rng=np.random.default_rng(1)), 0.2)
    b = torque_series(DisturbanceGenerator(without_steady_torque(params), rng=np.random.default_rng(1)), 0.2)
    np.testing.assert_array_equal(a, b)
    assert np.isnan(DisturbanceGenerator(params, rng=np.random.default_rng(1)).altitude(0.0))


def without_gust_law(params):
    params = copy.deepcopy(params)
    for key in GUST_LAW_KEYS:
        params["wind_params"].pop(key, None)
    return params


def test_parameter_files_without_a_steady_torque_run_without_one(surrogate_params):
    legacy = without_gust_law(surrogate_params)
    for key in ("steady_torque_per_density", "start_altitude_m", "ascent_rate_m_s"):
        del legacy["wind_params"][key]
    duration = legacy["simulation"]["duration"]
    old = DisturbanceGenerator(legacy, rng=np.random.default_rng(5))
    gusts = DisturbanceGenerator(
        without_steady_torque(without_gust_law(surrogate_params)), rng=np.random.default_rng(5)
    )
    np.testing.assert_array_equal(torque_series(old, duration), torque_series(gusts, duration))
    assert np.isnan(old.altitude(0.0))


def gust_law_params(params, duration=600.0, sigma=0.0):
    """A fast ascent, 0 to 18 km in ten minutes, with gusts only."""
    params = without_steady_torque(params)
    params["simulation"]["duration"] = duration
    params["wind_params"].update({
        "start_altitude_m": 0.0,
        "ascent_rate_m_s": 18000.0 / duration,
        "gust_rms_sea_level": 0.05,
        "gust_density_exponent": 1.0,
        "gust_rms_log_sigma": sigma,
    })
    return params


def test_gusts_follow_the_air_density_along_the_ascent(surrogate_params):
    params = gust_law_params(surrogate_params)
    wind = params["wind_params"]
    t = np.arange(0.0, 600.0, 0.02)
    tau = torque_series(DisturbanceGenerator(params, rng=np.random.default_rng(2)), 600.0, step=0.02)
    low, high = t < 60.0, t > 540.0
    unit = tau / gust_rms(t, wind)
    assert np.std(unit[low]) == pytest.approx(1.0, rel=0.25)
    assert np.std(unit[high]) == pytest.approx(1.0, rel=0.25)
    ratio = np.std(tau[high]) / np.std(tau[low])
    want = np.mean(gust_rms(t[high], wind)) / np.mean(gust_rms(t[low], wind))
    assert want < 0.2
    assert ratio == pytest.approx(want, rel=0.3)


def test_gust_intensity_varies_between_runs_by_the_log_sigma(surrogate_params):
    t = np.arange(0.0, 60.0, 0.02)
    def level(sigma, seed):
        params = gust_law_params(surrogate_params, duration=60.0, sigma=sigma)
        tau = torque_series(DisturbanceGenerator(params, rng=np.random.default_rng(seed)), 60.0, step=0.02)
        return np.std(tau / gust_rms(t, params["wind_params"]))
    fixed = [level(0.0, seed) for seed in range(6)]
    spread = [level(0.5, seed) for seed in range(6)]
    np.testing.assert_allclose(fixed, 1.0, rtol=0.05)
    assert np.std(np.log(spread)) > 0.2


def test_gust_law_without_an_ascent_is_rejected(surrogate_params):
    params = gust_law_params(surrogate_params)
    del params["wind_params"]["ascent_rate_m_s"]
    with pytest.raises(ValueError, match="gust_density_exponent"):
        DisturbanceGenerator(params, rng=np.random.default_rng(0))


def test_altitude_follows_the_modelled_ascent(surrogate_params):
    gen = DisturbanceGenerator(surrogate_params, rng=np.random.default_rng(0))
    wind = surrogate_params["wind_params"]
    assert gen.altitude(100.0) == pytest.approx(wind["start_altitude_m"] + 100.0 * wind["ascent_rate_m_s"])


def test_run_logs_the_modelled_altitude(short_params):
    df = run_simulation(short_params, seed=0)
    wind = short_params["wind_params"]
    np.testing.assert_allclose(df["altitude"], wind["start_altitude_m"] + wind["ascent_rate_m_s"] * df["time"])


def test_surrogate_source_range_off_the_log_is_rejected(surrogate_params):
    surrogate_params["wind_params"]["source_range"] = [1e7, 2e7]
    with pytest.raises(ValueError, match="source_range"):
        DisturbanceGenerator(surrogate_params, rng=np.random.default_rng(0))


def test_flight_data_index_out_of_range_start_is_clamped_not_crashed(flight_data_params):
    params = copy.deepcopy(flight_data_params)
    params["wind_params"]["start"] = 1e9
    gen = DisturbanceGenerator(params)
    value = gen.generate_torque_disturbance(params["simulation"]["duration"])
    assert np.isfinite(value)


def test_synthetic_wind_disturbance_is_finite_across_run(synthetic_wind_params):
    np.random.seed(0)
    gen = DisturbanceGenerator(synthetic_wind_params)
    dt = synthetic_wind_params["simulation"]["time_step"]
    duration = synthetic_wind_params["simulation"]["duration"]
    t = 0.0
    while t <= duration + dt:  # RK4's h4 stage probes one dt past `duration`
        value = gen.generate_torque_disturbance(t)
        assert np.isfinite(value)
        t += dt


def test_synthetic_wind_disturbance_does_not_index_out_of_bounds_at_boundary(synthetic_wind_params):
    np.random.seed(0)
    gen = DisturbanceGenerator(synthetic_wind_params)
    duration = synthetic_wind_params["simulation"]["duration"]
    gen.generate_torque_disturbance(duration)
    gen.generate_torque_disturbance(duration + 10.0)
