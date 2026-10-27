import pytest

from sim_tools.pwmodulator import Modulator

DT = 0.01
BIAS = 1500.0
PARAMS = {
    "rw_motor": {"rpm_bias": BIAS},
    "lt_motor": {
        "pulse_s": 0.05,
        "pulse_current": 5.8,
        "settle_time": 0.1,
        "integration_threshold": 2.0,
    },
}


def test_no_pulse_while_the_accumulator_is_under_the_threshold():
    m = Modulator(PARAMS, DT)
    # 100 rpm over the bias adds 1 rpm s per step, crosses on the second step
    assert m.modulate(BIAS + 100.0) == 0.0
    assert m.accumulator == pytest.approx(1.0)


def test_one_pulse_then_settle_with_the_accumulator_frozen():
    m = Modulator(PARAMS, DT)
    m.modulate(BIAS + 100.0)
    out = [m.modulate(BIAS + 100.0) for _ in range(5)]
    assert out == [5.8] * 5  # 0.05 s pulse
    frozen = m.accumulator
    # Settling ignores the input, even far over the threshold
    assert [m.modulate(BIAS + 1000.0) for _ in range(10)] == [0.0] * 10
    assert m.accumulator == 0.0, "accumulator is cleared at the end of the settle"
    assert frozen == pytest.approx(2.0)


def test_integration_restarts_from_zero_after_settling():
    m = Modulator(PARAMS, DT)
    for _ in range(1 + 5 + 10):  # integrate, pulse, settle
        m.modulate(BIAS + 100.0)
    assert m.modulate(BIAS + 100.0) == 0.0
    assert m.accumulator == pytest.approx(1.0)
    assert m.modulate(BIAS + 100.0) == 5.8


def test_wheel_under_the_bias_pulses_negative():
    m = Modulator(PARAMS, DT)
    out = [m.modulate(BIAS - 150.0) for _ in range(2)]
    assert out == [0.0, -5.8]


def test_zero_settle_time_resumes_integrating_right_after_the_pulse():
    params = {**PARAMS, "lt_motor": {**PARAMS["lt_motor"], "settle_time": 0.0}}
    m = Modulator(params, DT)
    out = [m.modulate(BIAS + 100.0) for _ in range(8)]
    assert out == [0.0] + [5.8] * 5 + [0.0, 5.8]
