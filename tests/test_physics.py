import copy

import numpy as np
import pytest

from sim_tools.disturbance import DisturbanceGenerator
from sim_tools.integrator import ModelIntegrator, wrap_angle
from sim_tools.pointing import PointingState


def run(params, initial_state=None):
    dt = params["simulation"]["time_step"]
    duration = params["simulation"]["duration"]
    init = params["simulation"]["initial_state"] if initial_state is None else initial_state
    model = ModelIntegrator(dt, init, params)

    state = np.array(init, dtype=float)
    states = [state.copy()]
    t = 0.0
    for _ in range(int(duration / dt)):
        state = model.rk4_step(state, t)
        t += dt
        states.append(state.copy())
    return model, np.array(states)


def test_angular_momentum_is_conserved_while_coasting(base_params):
    """Test bearing drag stays internal to the payload/wheel pair."""
    params = copy.deepcopy(base_params)
    params["simulation"]["duration"] = 20.0
    params["Payload_params"]["Cp"] = 0.0  # no external damping
    params["Payload_params"]["Kp"] = 0.0  # no tether spring
    params["lt_motor"]["activate"] = False  # no external dump torque
    params["lt_motor"]["viscous_friction_coeff"] = 0.0  # no bearing drag to the flight train
    params["lt_motor"]["coulomb_friction"] = 0.0
    params["wind_params"]["model"] = "synthetic"
    params["wind_params"]["sigma_noise"] = 0.0  # no wind
    # Open loop, only the wheel friction acts
    params["rw_motor"]["proportional_gain"] = 0.0
    params["rw_motor"]["derivative_gain"] = 0.0
    params["rw_motor"]["integral_gain"] = 0.0
    params["rw_motor"]["rpm_bias"] = 0.0
    params["pointing"]["state_machine"] = False  # no desaturation ramp either

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
    params["lt_motor"]["viscous_friction_coeff"] = 0.0  # no bearing drag to the flight train
    params["lt_motor"]["coulomb_friction"] = 0.0
    params["wind_params"]["model"] = "synthetic"
    params["wind_params"]["sigma_noise"] = 0.0
    params["rw_motor"]["proportional_gain"] = 0.0
    params["rw_motor"]["derivative_gain"] = 0.0
    params["rw_motor"]["integral_gain"] = 0.0
    params["rw_motor"]["rpm_bias"] = 0.0
    params["pointing"]["state_machine"] = False  # no desaturation ramp either

    np.random.seed(0)
    init = [0.0, 0.0, 0.0, 0.0, 150.0, 200.0, 200.0, 0.0, 0.0]
    model, states = run(params, initial_state=init)

    Ip = params["Payload_params"]["Ip"]
    J = params["rw_motor"]["rotor_inertia"] + params["rw_motor"]["load_inertia"]
    # Wheel slowed down, payload sped up by the same momentum
    lost_by_wheel = J * (states[0, 4] - states[-1, 4])
    gained_by_payload = Ip * (states[-1, 1] - states[0, 1])
    assert lost_by_wheel > 0
    assert gained_by_payload == pytest.approx(lost_by_wheel, rel=1e-6)


@pytest.mark.parametrize("dump_active", [True, False])
def test_dump_motor_bearing_drag_always_damps_the_payload(base_params, dump_active):
    """Test the dump motor's bearing drag acts on the payload whether or not it is driven."""
    params = copy.deepcopy(base_params)
    params["simulation"]["duration"] = 3.0
    params["lt_motor"]["activate"] = dump_active
    params["lt_motor"]["coulomb_friction"] = 0.0  # isolate the viscous term
    params["wind_params"]["model"] = "synthetic"
    for key in ("proportional_gain", "derivative_gain", "integral_gain", "rpm_bias"):
        params["rw_motor"][key] = 0.0  # idle wheel: only damping acts on the payload
    params["pointing"]["state_machine"] = False
    dt = params["simulation"]["time_step"]
    init = [0.0, 0.5, 0.0, 0.0, 0.0, 200.0, 200.0, 0.0, 0.0]
    model = ModelIntegrator(dt, init, params)
    model.disturbance.generate_torque_disturbance = lambda t: 0.0

    state, t = np.array(init), 0.0
    for _ in range(int(3.0 / dt)):
        state = model.rk4_step(state, t)
        t += dt

    Ip = params["Payload_params"]["Ip"]
    damping = params["Payload_params"]["Cp"] + params["lt_motor"]["viscous_friction_coeff"]
    assert state[1] == pytest.approx(0.5 * np.exp(-damping / Ip * 3.0), rel=1e-6)


def test_dump_motor_coulomb_friction_brakes_the_payload_at_a_constant_rate(base_params):
    """Test Coulomb drag slows the payload linearly, then holds it near rest."""
    params = copy.deepcopy(base_params)
    params["simulation"]["duration"] = 2.0
    params["lt_motor"]["activate"] = True
    params["lt_motor"]["viscous_friction_coeff"] = 0.0  # isolate the Coulomb term
    params["Payload_params"]["Cp"] = 0.0
    params["wind_params"]["model"] = "synthetic"
    for key in ("proportional_gain", "derivative_gain", "integral_gain", "rpm_bias"):
        params["rw_motor"][key] = 0.0  # idle wheel: only friction acts on the payload
    params["pointing"]["state_machine"] = False
    dt = params["simulation"]["time_step"]
    init = [0.0, 0.5, 0.0, 0.0, 0.0, 200.0, 200.0, 0.0, 0.0]
    model = ModelIntegrator(dt, init, params)
    model.disturbance.generate_torque_disturbance = lambda t: 0.0

    state, t = np.array(init), 0.0
    rates = [state[1]]
    for _ in range(round(2.0 / dt)):
        state = model.rk4_step(state, t)
        t += dt
        rates.append(state[1])
    rates = np.array(rates)

    decel = params["lt_motor"]["coulomb_friction"] / params["Payload_params"]["Ip"]
    stop_time = 0.5 / decel
    assert stop_time < 1.0, "payload should come to rest well inside the run"
    # Constant torque while turning, so RK4 is exact
    assert rates[round(0.5 * stop_time / dt)] == pytest.approx(0.5 - decel * 0.5 * stop_time, rel=1e-9)
    # Once stopped sgn() chatters around zero, by at most one step
    assert np.abs(rates[round((stop_time + 0.1) / dt):]).max() <= decel * dt


def test_pid_integral_advances_once_per_controller_update(short_params):
    """Test the integral charges once per controller period, not per RK4 stage or step."""
    params = copy.deepcopy(short_params)
    params["simulation"]["duration"] = 2.0
    params["rw_motor"]["integral_gain"] = 1.0  # off in the shipped config
    # No output clamp so anti-windup never kicks in
    params["rw_motor"]["max_rpm"] = 1e9
    dt = params["simulation"]["time_step"]

    model = ModelIntegrator(dt, params["simulation"]["initial_state"], params)
    state = np.array(params["simulation"]["initial_state"], dtype=float)
    t = 0.0
    errors = []
    for _ in range(int(2.0 / dt)):
        state = model.rk4_step(state, t)
        errors.append(model.yaw_error)
        t += dt

    # The pointing task samples the error every hold_steps steps
    assert model.pointing.state == PointingState.POINTING, "left POINTING, so the integral stopped charging"
    controller = model.pointing.controllers["pointing"]
    sampled = np.sum(errors[::model.pointing.hold_steps]) * controller.period
    assert controller.e_int == pytest.approx(sampled, rel=1e-9)


def test_commands_are_held_constant_across_the_rk4_stages(short_params):
    params = copy.deepcopy(short_params)
    dt = params["simulation"]["time_step"]
    model = ModelIntegrator(dt, params["simulation"]["initial_state"], params)
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
    model = ModelIntegrator(dt, params["simulation"]["initial_state"], params)
    state = np.array(params["simulation"]["initial_state"], dtype=float)

    t = 0.0
    duration = params["simulation"]["duration"]
    while t <= duration + 5 * dt:  # deliberately overrun
        state = model.rk4_step(state, t)
        t += dt
    assert np.all(np.isfinite(state))


def test_wrap_angle_takes_the_short_way_round():
    assert wrap_angle(3.1 - -3.1) == pytest.approx(-0.0831853, abs=1e-6)
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert wrap_angle(2 * np.pi + 0.25) == pytest.approx(0.25, abs=1e-9)


def test_yaw_error_stays_small_when_the_bearing_crosses_the_branch_cut(base_params):
    params = copy.deepcopy(base_params)
    params["simulation"]["duration"] = 40.0
    # Drifting at (-2.5, 2.5) m/s from (30, -50), the bearing crosses -pi at
    # t = 20 s. Drag and steady wind are off so the wheel does not saturate.
    params["lt_motor"]["viscous_friction_coeff"] = 0.0
    params["lt_motor"]["coulomb_friction"] = 0.0
    params["wind_params"]["steady_torque_per_density"] = 0.0
    init = list(params["simulation"]["initial_state"])
    init[0] = np.arctan2(-50.0, 30.0)
    init[5], init[6] = 30.0, -50.0

    _, states = run(params, initial_state=init)
    error = wrap_angle(states[:, 0] - np.arctan2(states[:, 6], states[:, 5]))

    bearing = np.arctan2(states[:, 6], states[:, 5])
    assert np.abs(np.diff(bearing)).max() > 1.0, "test did not actually cross the cut"
    assert np.rad2deg(np.abs(error)).max() < 30.0


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
    # No bearing drag, so the wheel acceleration is the drive torque only
    params["rw_motor"]["viscous_friction_coeff"] = 0.0
    init = list(params["simulation"]["initial_state"])
    init[0] += 1.5  # a big yaw offset rails the wheel drive

    model, states = run(params, initial_state=init)
    dt = params["simulation"]["time_step"]
    delivered = model.rw_motor.J * np.diff(states[:, 4]) / dt
    rated = model.rw_motor.Kt * model.rw_motor.max_current

    assert np.abs(states[:, 2]).max() == pytest.approx(model.rw_motor.max_current), "drive never hit its limit"
    assert np.abs(delivered).max() <= rated * (1 + 1e-9)


def test_flight_disturbance_is_continuous_in_time(base_params):
    params = copy.deepcopy(base_params)
    params["wind_params"]["model"] = "replay"
    gen = DisturbanceGenerator(params)

    t = np.linspace(10.0, 10.1, 2001)
    tau = np.array([gen.generate_torque_disturbance(x) for x in t])
    jumps = np.abs(np.diff(tau))
    assert jumps.max() < 0.05 * (tau.max() - tau.min())


def test_flight_disturbance_tracks_the_logs_own_timestamps(base_params):
    params = copy.deepcopy(base_params)
    params["wind_params"]["model"] = "replay"
    gen = DisturbanceGenerator(params)

    start = params["wind_params"]["start"]
    for offset in (0.0, 1.0, 37.5, 500.0):
        expected = np.interp(start + offset, gen.time, gen.torque)
        assert gen.generate_torque_disturbance(offset) == pytest.approx(expected)