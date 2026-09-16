"""Regression tests for the physics and discrete-time bugs fixed in the model.

Each test here corresponds to a specific defect that the simulation used to
have. They are deliberately written as physical invariants (momentum is
conserved, a rating is not exceeded, a discrete integral advances once per
step) rather than as golden numbers, so they keep their meaning if the
parameters are retuned.
"""

import copy

import numpy as np
import pytest

from sim_tools.disturbance import DisturbanceGenerator
from sim_tools.integrator import ModelIntegrator, wrap_angle


def run(params, initial_state=None):
    dt = params["simulation"]["time_step"]
    duration = params["simulation"]["duration"]
    init = params["simulation"]["initial_state"] if initial_state is None else initial_state
    model = ModelIntegrator(init, dt, params)

    state = np.array(init, dtype=float)
    states = [state.copy()]
    t = 0.0
    for _ in range(int(duration / dt)):
        state = model.rk4_step(state, t)
        t += dt
        states.append(state.copy())
    return model, np.array(states)


# --- angular momentum -----------------------------------------------------
def test_angular_momentum_is_conserved_while_coasting(base_params):
    """Test bearing drag stays internal to the payload/wheel pair."""
    params = copy.deepcopy(base_params)
    params["simulation"]["duration"] = 20.0
    params["Payload_params"]["Cp"] = 0.0  # no external damping
    params["Payload_params"]["Kp"] = 0.0  # no tether spring
    params["lt_motor"]["activate"] = False  # no external dump torque
    params["wind_params"]["simulated"] = True
    params["wind_params"]["sigma_noise"] = 0.0  # no wind
    # Open loop: the only thing acting is the spinning wheel's own friction.
    params["rw_motor"]["proportional_gain"] = 0.0
    params["rw_motor"]["derivative_gain"] = 0.0
    params["rw_motor"]["integral_gain"] = 0.0
    params["rw_motor"]["rpm_bias"] = 0.0

    np.random.seed(0)
    init = [0.0, 0.0, 0.0, 0.0, 150.0, 200.0, 200.0, 0.0, 0.0]
    model, states = run(params, initial_state=init)

    momentum = np.array([model.angular_momentum(s) for s in states])
    drift = abs(momentum[-1] - momentum[0]) / abs(momentum[0])
    assert drift < 1e-6, f"angular momentum drifted {100 * drift:.3f}%"


def test_wheel_friction_transfers_momentum_to_the_payload(base_params):
    """Test momentum transfer from frictional effects."""
    params = copy.deepcopy(base_params)
    params["simulation"]["duration"] = 20.0
    params["Payload_params"]["Cp"] = 0.0
    params["lt_motor"]["activate"] = False
    params["wind_params"]["simulated"] = True
    params["wind_params"]["sigma_noise"] = 0.0
    params["rw_motor"]["proportional_gain"] = 0.0
    params["rw_motor"]["derivative_gain"] = 0.0
    params["rw_motor"]["integral_gain"] = 0.0
    params["rw_motor"]["rpm_bias"] = 0.0

    np.random.seed(0)
    init = [0.0, 0.0, 0.0, 0.0, 150.0, 200.0, 200.0, 0.0, 0.0]
    model, states = run(params, initial_state=init)

    Ip = params["Payload_params"]["Ip"]
    J = params["rw_motor"]["rotor_inertia"] + params["rw_motor"]["load_inertia"]
    # Wheel slowed down, so the payload must have sped up by the matching amount.
    lost_by_wheel = J * (states[0, 4] - states[-1, 4])
    gained_by_payload = Ip * (states[-1, 1] - states[0, 1])
    assert lost_by_wheel > 0
    assert gained_by_payload == pytest.approx(lost_by_wheel, rel=1e-6)


# --- discrete-time / RK4 separation ---------------------------------------
def test_pid_integral_advances_once_per_step(short_params):
    """Test controller is computed once per step."""
    params = copy.deepcopy(short_params)
    params["simulation"]["duration"] = 2.0
    params["rw_motor"]["integral_gain"] = 1.0  # off in the shipped config
    dt = params["simulation"]["time_step"]

    model = ModelIntegrator(params["simulation"]["initial_state"], dt, params)
    state = np.array(params["simulation"]["initial_state"], dtype=float)
    t = 0.0
    errors = []
    for _ in range(int(2.0 / dt)):
        state = model.rk4_step(state, t)
        errors.append(model.yaw_error)
        t += dt

    true_integral = np.trapezoid(errors, dx=dt)
    assert model.rw_controller.e_int == pytest.approx(true_integral, rel=1e-2)


def test_commands_are_held_constant_across_the_rk4_stages(short_params):
    params = copy.deepcopy(short_params)
    dt = params["simulation"]["time_step"]
    model = ModelIntegrator(params["simulation"]["initial_state"], dt, params)
    state = np.array(params["simulation"]["initial_state"], dtype=float)

    seen = []
    original = model._dynamics

    def spy(s, t, rw_voltage, lt_current):
        seen.append((rw_voltage, lt_current))
        return original(s, t, rw_voltage, lt_current)

    model._dynamics = spy
    model.rk4_step(state, 0.0)

    assert len(seen) == 4  # h1..h4
    assert len(set(seen)) == 1  # all four stages saw the same command


def test_sensor_index_stays_in_range_at_the_end_of_a_run(short_params):
    """Testing time stepping does not step past duration."""
    params = copy.deepcopy(short_params)
    dt = params["simulation"]["time_step"]
    model = ModelIntegrator(params["simulation"]["initial_state"], dt, params)
    state = np.array(params["simulation"]["initial_state"], dtype=float)

    t = 0.0
    duration = params["simulation"]["duration"]
    while t <= duration + 5 * dt:  # deliberately overrun
        state = model.rk4_step(state, t)
        t += dt
    assert np.all(np.isfinite(state))


# --- angle wrapping -------------------------------------------------------
def test_wrap_angle_takes_the_short_way_round():
    assert wrap_angle(3.1 - -3.1) == pytest.approx(-0.0831853, abs=1e-6)
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert wrap_angle(2 * np.pi + 0.25) == pytest.approx(0.25, abs=1e-9)


def test_yaw_error_stays_small_when_the_bearing_crosses_the_branch_cut(base_params):
    params = copy.deepcopy(base_params)
    params["simulation"]["duration"] = 40.0
    # The payload drifts at (x_dot, y_dot) = (-2.5, +2.5) m/s. Starting at
    # (30, -50) it reaches (-20, 0) at t = 20 s, so y changes sign while x is
    # negative and the bearing atan2(y, x) crosses -pi mid-run. Closest approach
    # is ~14 m, which keeps the bearing rate well within what the wheel can track.
    init = list(params["simulation"]["initial_state"])
    init[0] = np.arctan2(-50.0, 30.0)
    init[5], init[6] = 30.0, -50.0

    _, states = run(params, initial_state=init)
    error = wrap_angle(states[:, 0] - np.arctan2(states[:, 6], states[:, 5]))

    bearing = np.arctan2(states[:, 6], states[:, 5])
    assert np.abs(np.diff(bearing)).max() > 1.0, "test did not actually cross the cut"
    assert np.rad2deg(np.abs(error)).max() < 30.0


# --- actuator ratings -----------------------------------------------------
def test_wheel_speed_stays_inside_the_commanded_rpm_clamp(base_params):
    params = copy.deepcopy(base_params)
    params["simulation"]["duration"] = 30.0
    params["lt_motor"]["activate"] = False  # let the wheel wind up freely
    # Start the wheel already at the top of the clamp and drive hard against it.
    top_speed = (params["rw_motor"]["rpm_bias"] + params["rw_motor"]["max_rpm"]) * np.pi / 30
    init = list(params["simulation"]["initial_state"])
    init[0] += 1.0
    init[4] = top_speed

    _, states = run(params, initial_state=init)
    assert np.abs(states[:, 4]).max() <= top_speed + 1e-9


def test_dump_motor_torque_respects_its_current_rating(base_params):
    params = copy.deepcopy(base_params)
    params["simulation"]["duration"] = 20.0
    params["lt_motor"]["activate"] = True
    # A big yaw offset winds the wheel up hard, which rails the dump loop.
    init = list(params["simulation"]["initial_state"])
    init[0] += 1.5

    _, states = run(params, initial_state=init)
    max_torque = params["lt_motor"]["torque_constant"] * params["lt_motor"]["max_current"]
    assert np.abs(states[:, 3]).max() <= max_torque + 1e-9


def test_wheel_torque_respects_its_current_rating(base_params):
    params = copy.deepcopy(base_params)
    params["simulation"]["duration"] = 0.5
    params["lt_motor"]["activate"] = False
    # Without bearing drag the change in wheel speed is exactly the RK4 average
    # of the drive torque, with nothing to subtract.
    params["rw_motor"]["viscous_friction_coeff"] = 0.0
    init = list(params["simulation"]["initial_state"])
    init[0] += 1.5  # a big yaw offset rails the wheel drive

    model, states = run(params, initial_state=init)
    dt = params["simulation"]["time_step"]
    delivered = model.rw_motor.J * np.diff(states[:, 4]) / dt
    rated = model.rw_motor.Kt * model.rw_motor.max_current

    assert np.abs(states[:, 2]).max() == pytest.approx(model.rw_motor.max_current), "drive never hit its limit"
    assert np.abs(delivered).max() <= rated * (1 + 1e-9)


# --- disturbance playback -------------------------------------------------
def test_flight_disturbance_is_continuous_in_time(base_params):
    params = copy.deepcopy(base_params)
    params["wind_params"]["simulated"] = False
    gen = DisturbanceGenerator(params)

    t = np.linspace(10.0, 10.1, 2001)
    tau = np.array([gen.generate_torque_disturbance(x) for x in t])
    jumps = np.abs(np.diff(tau))
    # No single sub-millisecond step should carry a large fraction of the
    # signal's whole range.
    assert jumps.max() < 0.05 * (tau.max() - tau.min())


def test_flight_disturbance_tracks_the_logs_own_timestamps(base_params):
    params = copy.deepcopy(base_params)
    params["wind_params"]["simulated"] = False
    gen = DisturbanceGenerator(params)

    start = params["wind_params"]["start"]
    for offset in (0.0, 1.0, 37.5, 500.0):
        expected = np.interp(start + offset, gen.time, gen.torque)
        assert gen.generate_torque_disturbance(offset) == pytest.approx(expected)