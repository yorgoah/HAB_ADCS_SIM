import pytest

from sim_tools.actuator import Motor

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