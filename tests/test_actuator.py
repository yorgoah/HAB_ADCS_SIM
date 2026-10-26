import numpy as np
import pytest

from sim_tools.actuator import Motor, VescSpeedControl

PARAMS = {
    "rotor_inertia": 1e-4,
    "load_inertia": 2e-4,
    "torque_constant": 0.05,
    "back_emf_constant": 0.05,
    "resistance": 0.3,
    "inductance": 3e-4,
    "viscous_friction_coeff": 1e-4,
    "max_current": 5.0,
}


def make_motor(**overrides):
    return Motor({**PARAMS, **overrides})


def test_torque_proportional_to_current():
    motor = make_motor()
    torque, _ = motor.torque(voltage=0.0, current=2.0, angular_velocity=0.0)
    assert torque == pytest.approx(PARAMS["torque_constant"] * 2.0)


def test_current_derivative_matches_electrical_equation():
    motor = make_motor()
    voltage, current, ang_vel = 1.5, 1.5, 10.0
    _, di_dt = motor.torque(voltage=voltage, current=current, angular_velocity=ang_vel)
    expected = (voltage - PARAMS["resistance"] * current - PARAMS["back_emf_constant"] * ang_vel) / PARAMS["inductance"]
    assert di_dt == pytest.approx(expected)


def test_torque_includes_viscous_drag():
    motor = make_motor()
    current, ang_vel = 3.0, 20.0
    torque, _ = motor.torque(voltage=0.0, current=current, angular_velocity=ang_vel)
    # The returned torque is the net torque between rotor and housing, bearing drag included.
    assert torque == pytest.approx(PARAMS["torque_constant"] * current - PARAMS["viscous_friction_coeff"] * ang_vel)


@pytest.mark.parametrize("ang_vel", [-5.0, -1e-3, 1e-3, 5.0])
def test_coulomb_friction_opposes_motion_with_constant_magnitude(ang_vel):
    motor = make_motor(viscous_friction_coeff=0.0, coulomb_friction=0.01)
    torque, _ = motor.torque(voltage=0.0, current=0.0, angular_velocity=ang_vel)
    assert torque == pytest.approx(-0.01 if ang_vel > 0 else 0.01)


def test_coulomb_friction_vanishes_at_rest():
    motor = make_motor(coulomb_friction=0.01)
    torque, _ = motor.torque(voltage=0.0, current=0.0, angular_velocity=0.0)
    assert torque == pytest.approx(0.0)


def test_torque_saturates_at_the_current_rating():
    motor = make_motor()
    rated = PARAMS["torque_constant"] * PARAMS["max_current"]
    torque, _ = motor.torque(voltage=0.0, current=3 * PARAMS["max_current"], angular_velocity=0.0)
    assert torque == pytest.approx(rated)
    torque, _ = motor.torque(voltage=0.0, current=-3 * PARAMS["max_current"], angular_velocity=0.0)
    assert torque == pytest.approx(-rated)


def test_zero_current_gives_zero_torque():
    motor = make_motor()
    torque, _ = motor.torque(voltage=0.0, current=0.0, angular_velocity=0.0)
    assert torque == pytest.approx(0.0)


def test_back_emf_opposes_current_growth():
    # At matched voltage/back-emf (voltage == Kb * angular_velocity), di_dt should
    # only depend on the resistive drop.
    motor = make_motor()
    ang_vel = 40.0
    voltage = PARAMS["back_emf_constant"] * ang_vel
    _, di_dt = motor.torque(voltage=voltage, current=1.0, angular_velocity=ang_vel)
    expected = (-PARAMS["resistance"] * 1.0) / PARAMS["inductance"]
    assert di_dt == pytest.approx(expected)

VESC = {"kp": 0.004, "ki": 0.004, "pole_pairs": 7}


def make_vesc(dt=1e-3, **overrides):
    motor = make_motor()
    return VescSpeedControl({**VESC, **overrides}, motor, dt), motor


def test_vesc_at_the_commanded_speed_only_cancels_the_back_emf():
    vesc, motor = make_vesc()
    speed = 100.0
    assert vesc.voltage(speed * 30 / np.pi, speed) == pytest.approx(motor.Kb * speed)


def test_vesc_proportional_current_is_a_fraction_of_the_current_limit():
    vesc, motor = make_vesc(ki=0.0)
    speed, rpm = 100.0, 100.0 * 30 / np.pi + 10.0  # 10 rpm low, 70 ERPM
    current = (vesc.voltage(rpm, speed) - motor.Kb * speed) / motor.R
    assert current == pytest.approx(VESC["kp"] * 10.0 * VESC["pole_pairs"] * PARAMS["max_current"])


def test_vesc_current_request_is_capped_at_the_current_limit():
    vesc, motor = make_vesc()
    for rpm in (1e5, -1e5):
        current = (vesc.voltage(rpm, 0.0) - 0.0) / motor.R
        assert current == pytest.approx(np.sign(rpm) * PARAMS["max_current"])


def test_vesc_integral_is_clamped_like_the_firmware():
    vesc, _ = make_vesc()
    for _ in range(10_000):
        vesc.voltage(1e5, 0.0)
    assert vesc.i_term == pytest.approx(1.0)


def test_vesc_holds_the_commanded_speed_against_a_load():
    # Wheel held by a constant load torque: the integral removes the steady error
    dt = 1e-4
    vesc, motor = make_vesc(dt=dt)
    rpm, load = 2000.0, 0.02
    speed, current = rpm * np.pi / 30, 0.0
    for _ in range(round(5.0 / dt)):
        voltage = vesc.voltage(rpm, speed)
        torque, di_dt = motor.torque(voltage, current, speed)
        current += di_dt * dt
        speed += (torque - load) / motor.J * dt
    assert speed * 30 / np.pi == pytest.approx(rpm, abs=1.0)
