import numpy as np
import pytest

from sim_tools.sensor import Sensor

NOISE_FREE = {"sampling_rate": 10, "noise_density": 0.0, "random_walk": 0.0}


def test_noise_free_sensor_returns_true_value():
    sensor = Sensor(NOISE_FREE, duration=1.0, initial_value=0.0)
    assert sensor.get_measurement(true_value=3.14, time=0.0) == pytest.approx(3.14)
    assert sensor.get_measurement(true_value=-2.0, time=0.5) == pytest.approx(-2.0)


def test_measurement_holds_between_samples():
    # sampling_rate=10 Hz -> a new sample every 0.1s. Calls that land in the
    # same 0.1s bucket must return the identical cached value even if the
    # "true" underlying value has moved on.
    sensor = Sensor(NOISE_FREE, duration=1.0, initial_value=0.0)
    first = sensor.get_measurement(true_value=1.0, time=0.05)
    held = sensor.get_measurement(true_value=999.0, time=0.09)
    assert held == first


def test_measurement_updates_on_new_sample_interval():
    sensor = Sensor(NOISE_FREE, duration=1.0, initial_value=0.0)
    sensor.get_measurement(true_value=1.0, time=0.0)
    updated = sensor.get_measurement(true_value=2.0, time=0.1)
    assert updated == pytest.approx(2.0)


def test_repeated_calls_at_same_time_do_not_resample():
    # Regression test: get_measurement is called multiple times per RK4 step
    # at the same simulated time (h1..h4 stage re-evaluations at a shared t).
    # It must not draw a fresh noise sample on every call, only on a genuine
    # time-bucket change.
    params = {"sampling_rate": 10, "noise_density": 1.0, "random_walk": 0.0}
    sensor = Sensor(params, duration=1.0, initial_value=0.0)
    values = [sensor.get_measurement(true_value=0.0, time=0.05) for _ in range(4)]
    assert len(set(values)) == 1


def test_noisy_sensor_measurement_is_finite_and_bounded_by_noise_scale():
    params = {"sampling_rate": 20, "noise_density": 0.01, "random_walk": 0.0}
    sensor = Sensor(params, duration=1.0, initial_value=0.0)
    m = sensor.get_measurement(true_value=0.0, time=0.0)
    assert np.isfinite(m)
    # White noise std is noise_density * sqrt(sampling_rate/2); a single draw
    # landing within 10 sigma is an astronomically safe bound for a flaky-free test.
    sigma = params["noise_density"] * np.sqrt(params["sampling_rate"] / 2)
    assert abs(m) < 10 * sigma
