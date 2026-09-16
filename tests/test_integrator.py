import copy

import numpy as np
import pytest

from sim_tools.integrator import ModelIntegrator


def run(params, n_steps=None):
    dt = params["simulation"]["time_step"]
    duration = params["simulation"]["duration"]
    init_state = params["simulation"]["initial_state"]
    model = ModelIntegrator(init_state, dt, params)

    state = np.array(init_state, dtype=float)
    states = [state.copy()]
    t = 0.0
    steps = n_steps if n_steps is not None else int(duration / dt)
    for _ in range(steps):
        state = model.rk4_step(state, t)
        t += dt
        states.append(state.copy())
    return np.array(states)


def test_momentum_management_flag_follows_lt_motor_activate(short_params):
    params_on = copy.deepcopy(short_params)
    params_on["lt_motor"]["activate"] = True
    model_on = ModelIntegrator(
        params_on["simulation"]["initial_state"],
        params_on["simulation"]["time_step"],
        params_on,
    )
    assert model_on.momentum_management is True

    params_off = copy.deepcopy(short_params)
    params_off["lt_motor"]["activate"] = False
    model_off = ModelIntegrator(
        params_off["simulation"]["initial_state"],
        params_off["simulation"]["time_step"],
        params_off,
    )
    assert model_off.momentum_management is False


@pytest.mark.parametrize("momentum_management", [True, False])
@pytest.mark.parametrize("simulated_wind", [True, False])
def test_short_run_produces_finite_states(short_params, momentum_management, simulated_wind):
    params = copy.deepcopy(short_params)
    params["lt_motor"]["activate"] = momentum_management
    params["wind_params"]["simulated"] = simulated_wind
    if simulated_wind:
        np.random.seed(0)

    states = run(params)
    assert np.all(np.isfinite(states))


def test_reaction_wheel_current_never_exceeds_max_current(short_params):
    params = copy.deepcopy(short_params)
    # A large yaw offset drives a large control effort, which is exactly the
    # scenario the current-limit clip in rk4_step exists to guard against.
    params["simulation"]["initial_state"][0] = 100.0

    states = run(params)
    max_current = params["rw_motor"]["max_current"]
    rw_current = states[:, 2]
    assert np.all(np.abs(rw_current) <= max_current + 1e-9)


def test_logged_disturbance_and_torque_channels_stay_bounded(short_params):
    # Regression test: indices 3, 7, 8 (lt_torque, tau_d, rw_torque) are
    # logging channels, not real ODE states. A prior bug let RK4 integrate
    # them like states, so they grew roughly linearly in time instead of
    # tracking the instantaneous disturbance/torque. Over this run they
    # should stay within a small physically-reasonable band, not drift.
    states = run(short_params)
    disturbance = states[:, 7]
    rw_torque = states[:, 8]
    assert np.max(np.abs(disturbance)) < 1.0
    assert np.max(np.abs(rw_torque)) < 1.0


def test_yaw_error_shrinks_from_a_large_initial_offset(short_params):
    # Displace yaw well away from the ground-station bearing and check the
    # closed loop actually regulates the error down, not just "stays finite".
    params = copy.deepcopy(short_params)
    params["simulation"]["duration"] = 5.0
    params["simulation"]["initial_state"][0] += 1.0  # ~57 deg yaw offset

    states = run(params)
    yaw = states[:, 0]
    x, y = states[:, 5], states[:, 6]
    error = yaw - np.arctan2(y, x)
    assert np.abs(error[-1]) < 0.5 * np.abs(error[0])
