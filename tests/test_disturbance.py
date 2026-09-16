import copy

import numpy as np
import pytest

from sim_tools.disturbance import DisturbanceGenerator


@pytest.fixture
def flight_data_params(short_params):
    params = copy.deepcopy(short_params)
    params["wind_params"]["simulated"] = False
    return params


@pytest.fixture
def synthetic_wind_params(short_params):
    params = copy.deepcopy(short_params)
    params["wind_params"]["simulated"] = True
    return params


def test_flight_data_disturbance_is_finite_across_run(flight_data_params):
    gen = DisturbanceGenerator(flight_data_params)
    dt = flight_data_params["simulation"]["time_step"]
    duration = flight_data_params["simulation"]["duration"]
    t = 0.0
    while t <= duration:
        value = gen.generate_torque_disturbance(t)
        assert np.isfinite(value)
        t += dt


def test_flight_data_disturbance_scales_with_payload_inertia(flight_data_params):
    gen_a = DisturbanceGenerator(flight_data_params)

    params_b = copy.deepcopy(flight_data_params)
    params_b["Payload_params"]["Ip"] *= 2.0
    gen_b = DisturbanceGenerator(params_b)

    assert gen_b.generate_torque_disturbance(0.0) == pytest.approx(
        2.0 * gen_a.generate_torque_disturbance(0.0)
    )


def test_flight_data_index_out_of_range_start_is_clamped_not_crashed(flight_data_params):
    # A "start" time far beyond the loaded CSV's span used to be able to walk
    # generate_torque_disturbance's index past the end of the array.
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
    # Exactly at and beyond the nominal end of the run.
    gen.generate_torque_disturbance(duration)
    gen.generate_torque_disturbance(duration + 10.0)
