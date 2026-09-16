import pytest

from sim_tools.controller import Controller

PARAMS = {
    "proportional_gain": 2.0,
    "derivative_gain": 0.5,
    "integral_gain": 1.0,
}

# The limit is whatever the output drives: a supply voltage, a current rating,
# or a wheel speed command. It is passed in rather than read from params.
LIMIT = 10.0


def make_controller(output_limit=LIMIT, **overrides):
    params = {**PARAMS, **overrides}
    return Controller(params=params, dt=0.1, output_limit=output_limit)


def test_proportional_term_only_on_first_call():
    # e_prev and e_int start at 0, so the first call is pure P + I(dt) with no
    # history to differentiate against.
    c = make_controller(derivative_gain=0.0, integral_gain=0.0)
    assert c.output(3.0) == pytest.approx(PARAMS["proportional_gain"] * 3.0)


def test_numerical_derivative_uses_previous_error():
    c = make_controller(output_limit=1000.0, proportional_gain=0.0, integral_gain=0.0)
    c.output(1.0)  # e_prev becomes 1.0
    out = c.output(4.0)  # derivative = (4.0 - 1.0) / dt
    expected = PARAMS["derivative_gain"] * (4.0 - 1.0) / c.dt
    assert out == pytest.approx(expected)


def test_explicit_derivative_overrides_numerical_one():
    c = make_controller(proportional_gain=0.0, integral_gain=0.0)
    out = c.output(error=1.0, error_derivative=7.0)
    assert out == pytest.approx(PARAMS["derivative_gain"] * 7.0)


def test_integral_accumulates_across_calls():
    c = make_controller(proportional_gain=0.0, derivative_gain=0.0)
    c.output(2.0)
    out = c.output(2.0)
    # e_int = 2.0*dt + 2.0*dt = 2*2.0*dt
    expected = PARAMS["integral_gain"] * (2 * 2.0 * c.dt)
    assert out == pytest.approx(expected)


def test_output_is_clipped_to_the_output_limit():
    c = make_controller(proportional_gain=1000.0, integral_gain=0.0, derivative_gain=0.0)
    assert c.output(1.0) == pytest.approx(LIMIT)
    assert c.output(-1.0) == pytest.approx(-LIMIT)


def test_a_different_limit_moves_the_saturation_point():
    # A current-command loop saturates at the current rating, a speed command at
    # the rpm clamp. The controller saturates at whatever limit it is given.
    c = make_controller(output_limit=3.0, proportional_gain=1000.0)
    assert c.output(1.0) == pytest.approx(3.0)
    assert c.output(-1.0) == pytest.approx(-3.0)


def test_zero_error_gives_zero_output():
    c = make_controller()
    assert c.output(0.0) == pytest.approx(0.0)


def test_integral_does_not_wind_up_while_output_is_saturated():
    c = make_controller(proportional_gain=1000.0, derivative_gain=0.0)
    for _ in range(50):
        c.output(1.0)  # pinned at +LIMIT throughout
    assert c.e_int == pytest.approx(0.0)


def test_integral_still_unwinds_while_output_is_saturated():
    # Pinned high by the derivative term while the error is negative: integrating
    # pulls the output back out of the limit, so it must not be frozen.
    c = make_controller(proportional_gain=1.0, derivative_gain=1.0)
    assert c.output(error=-1.0, error_derivative=100.0) == pytest.approx(LIMIT)
    assert c.e_int == pytest.approx(-1.0 * c.dt)


def test_output_is_held_between_control_updates():
    # A 0.1 s loop called every 0.025 s updates on every fourth call only.
    c = Controller(params={**PARAMS, "period_s": 0.1}, dt=0.025, output_limit=LIMIT)
    first = c.output(1.0)
    assert [c.output(5.0) for _ in range(3)] == [first] * 3
    assert c.output(5.0) != first


def test_integral_accumulates_over_the_control_period():
    params = {**PARAMS, "proportional_gain": 0.0, "derivative_gain": 0.0, "period_s": 0.1}
    c = Controller(params=params, dt=0.025, output_limit=LIMIT)
    for _ in range(8):  # two updates
        c.output(2.0)
    assert c.e_int == pytest.approx(2 * 2.0 * 0.1)
