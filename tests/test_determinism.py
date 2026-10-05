import copy

import numpy as np
import pandas as pd

from hab_adcs_sim.simulation_runner import run_simulation
from sim_tools.integrator import ModelIntegrator
from sim_tools.sensor import Sensor

NOISY = {"sampling_rate": 20, "noise_density": 0.01, "random_walk": 0.001}


def test_same_seed_reproduces_the_whole_run(short_params):
    a = run_simulation(short_params, seed=3)
    b = run_simulation(short_params, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_noise(short_params):
    params = copy.deepcopy(short_params)
    params["gyroscope"]["noise_density"] = 0.05

    a = run_simulation(params, seed=1)
    b = run_simulation(params, seed=2)
    assert not a["yaw"].equals(b["yaw"])


def test_seeded_sensors_are_independent_of_each_other(short_params):
    """Test changing one sensor does not change another's noise."""
    params = copy.deepcopy(short_params)
    params["gyroscope"]["noise_density"] = 0.05
    baseline = run_simulation(params, seed=4)

    params["gps"]["sampling_rate"] = 2
    changed = run_simulation(params, seed=4)

    assert baseline["ang_vel"].equals(changed["ang_vel"])


def test_unseeded_sensor_still_follows_the_global_random_state():
    np.random.seed(0)
    first = Sensor(NOISY, duration=1.0, initial_value=0.0).white_noise.copy()
    np.random.seed(0)
    second = Sensor(NOISY, duration=1.0, initial_value=0.0).white_noise

    np.testing.assert_array_equal(first, second)


def test_seeded_sensor_ignores_the_global_random_state():
    rng = np.random.default_rng(7)
    np.random.seed(0)
    seeded = Sensor(NOISY, duration=1.0, initial_value=0.0, rng=rng).white_noise.copy()

    rng = np.random.default_rng(7)
    np.random.seed(999)
    again = Sensor(NOISY, duration=1.0, initial_value=0.0, rng=rng).white_noise

    np.testing.assert_array_equal(seeded, again)


def test_wind_seed_redraws_the_wind_but_not_the_sensor_noise(short_params):
    params = copy.deepcopy(short_params)
    params["wind_params"]["model"] = "surrogate"
    params["gyroscope"]["noise_density"] = 0.05
    init = params["simulation"]["initial_state"]
    dt = params["simulation"]["time_step"]

    a = ModelIntegrator(dt, init, params, seed=4, wind_seed=1)
    b = ModelIntegrator(dt, init, params, seed=4, wind_seed=2)

    np.testing.assert_array_equal(a.gyro.white_noise, b.gyro.white_noise)
    assert not np.array_equal(a.disturbance.torque, b.disturbance.torque)
