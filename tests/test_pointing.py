"""Pointing state machine tests, one call per task cycle (dt = period_s).

Transition counts allow one cycle of slack for rounding of the task time.
"""

import copy

import pytest

from hab_adcs_sim.simulation_runner import run_simulation
from sim_tools.pointing import (
    DESATURATE_MIN_S,
    DESATURATION_RATE_RPM,
    STABILIZE_TIMEOUT_S,
    PointingState,
    PointingStateMachine,
)

MID_RPM = 1500.0  # wheel speed well away from both saturation edges


def make(base_params, rw_motor=None, **pointing):
    params = copy.deepcopy(base_params)
    params["rw_motor"].update(rw_motor or {})
    params["pointing"].update(pointing)
    return PointingStateMachine(params, dt=params["rw_motor"]["period_s"]), params


def cycles(sm, seconds):
    return round(seconds / sm.period)


def cycles_until(sm, state, error=0.0, yaw_rate=0.0, rw_rpm=MID_RPM, limit=100_000):
    """Task cycles run, with fixed measurements, until the machine enters `state`."""
    for k in range(1, limit + 1):
        sm.rpm(error, yaw_rate, rw_rpm)
        if sm.state == state:
            return k
    return None


def within_a_cycle(k, nominal):
    return k is not None and abs(k - nominal) <= 1


def test_starts_in_pointing_with_the_flight_pd_law(base_params):
    sm, params = make(base_params)
    rw = params["rw_motor"]
    error, rate = 0.01, 0.02

    assert sm.state == PointingState.POINTING
    expected = int(rw["proportional_gain"] * error + rw["derivative_gain"] * rate + rw["rpm_bias"])
    assert sm.rpm(error, rate, MID_RPM) == expected


def test_command_is_clipped_to_the_driver_range(base_params):
    # Off-centre bias so the 0..3000 rpm clip is hit
    high, _ = make(base_params, rw_motor={"rpm_bias": 2000.0})
    assert high.rpm(1.0, 0.0, MID_RPM) == 3000

    low, _ = make(base_params, rw_motor={"rpm_bias": 1000.0})
    assert low.rpm(-1.0, 0.0, MID_RPM) == 0


def test_command_is_held_between_task_cycles(base_params):
    params = copy.deepcopy(base_params)
    period = params["rw_motor"]["period_s"]
    sm = PointingStateMachine(params, dt=period / 4)

    first = sm.rpm(0.01, 0.0, MID_RPM)
    assert [sm.rpm(0.05, 0.0, MID_RPM) for _ in range(3)] == [first] * 3
    assert sm.rpm(0.05, 0.0, MID_RPM) != first


@pytest.mark.parametrize("rw_rpm", [2860.0, 40.0], ids=["upper", "lower"])
def test_saturation_is_declared_once_the_flag_has_held_for_saturation_s(base_params, rw_rpm):
    sm, params = make(base_params)
    # Flagged from the first cycle, declared saturation_s later.
    k = cycles_until(sm, PointingState.SATURATED, rw_rpm=rw_rpm)
    assert within_a_cycle(k, cycles(sm, params["pointing"]["saturation_s"]) + 1)


def test_a_break_in_the_saturation_flag_restarts_the_timer(base_params):
    sm, params = make(base_params)
    n = cycles(sm, params["pointing"]["saturation_s"])
    for _ in range(n - 5):
        sm.rpm(0.0, 0.0, 2860.0)
    sm.rpm(0.0, 0.0, MID_RPM)  # one clear cycle

    assert sm.state == PointingState.POINTING
    assert within_a_cycle(cycles_until(sm, PointingState.SATURATED, rw_rpm=2860.0), n + 1)


def test_desaturation_walks_the_command_to_the_bias(base_params):
    sm, params = make(base_params)
    bias = params["rw_motor"]["rpm_bias"]
    cycles_until(sm, PointingState.SATURATED, error=1.0, rw_rpm=2900.0)
    assert sm.rpm_command == 3000  # the transition cycle sent nothing new

    ramp = [sm.rpm(0.0, 0.0, 2900.0) for _ in range(16)]
    steps = int((3000 - bias) / DESATURATION_RATE_RPM)
    assert ramp == [3000 - DESATURATION_RATE_RPM * (i + 1) for i in range(steps)] + [bias] * (16 - steps)


def test_desaturation_hands_over_to_stabilize_only_in_band_and_after_5_s(base_params):
    sm, _ = make(base_params)
    cycles_until(sm, PointingState.SATURATED, error=1.0, rw_rpm=2900.0)

    for _ in range(4 * cycles(sm, DESATURATE_MIN_S)):
        sm.rpm(0.0, 0.0, 1650.0)
    assert sm.state == PointingState.SATURATED

    sm, _ = make(base_params)
    cycles_until(sm, PointingState.SATURATED, error=1.0, rw_rpm=2900.0)
    k = cycles_until(sm, PointingState.STABILIZE, rw_rpm=1550.0)
    assert within_a_cycle(k, cycles(sm, DESATURATE_MIN_S))


def test_sustained_spin_drops_pointing_to_stabilize(base_params):
    sm, params = make(base_params)
    rate = params["pointing"]["unstable_yaw_rate"] + 0.2
    window = sm.yaw_rate_window.maxlen
    nominal = window + cycles(sm, params["pointing"]["stability_threshold"])

    held = None
    for k in range(1, nominal + 3):
        before = sm.rpm_command
        sm.rpm(0.0, rate, MID_RPM)
        if sm.state == PointingState.STABILIZE:
            held = before
            break
    assert within_a_cycle(k, nominal)
    assert sm.rpm_command == held  # the transition cycle sent nothing new


def test_isolated_spikes_do_not_leave_pointing(base_params):
    sm, _ = make(base_params)
    # At most 4 spikes in any 10 samples, so the median never sees them.
    for k in range(400):
        sm.rpm(0.0, 5.0 if k % 3 == 0 else 0.0, MID_RPM)
    assert sm.state == PointingState.POINTING


def test_stabilize_damps_the_yaw_rate(base_params):
    sm, params = make(base_params)
    cycles_until(sm, PointingState.STABILIZE, yaw_rate=2.0)

    stab = params["rw_motor"]["stabilize"]
    command = sm.rpm(0.0, 2.0, MID_RPM)
    assert command == int(stab["proportional_gain"] * 2.0 + params["rw_motor"]["rpm_bias"])


def test_stabilize_returns_to_pointing_once_the_rate_settles(base_params):
    sm, params = make(base_params)
    # Entering SATURATED empties the median filter
    cycles_until(sm, PointingState.SATURATED, error=1.0, rw_rpm=2900.0)
    cycles_until(sm, PointingState.STABILIZE, rw_rpm=MID_RPM)
    assert len(sm.yaw_rate_window) == 0

    k = cycles_until(sm, PointingState.POINTING, yaw_rate=0.2)
    nominal = sm.yaw_rate_window.maxlen + cycles(sm, params["pointing"]["stability_threshold"])
    assert within_a_cycle(k, nominal)


def test_stabilize_gives_up_after_its_timeout(base_params):
    sm, _ = make(base_params)
    cycles_until(sm, PointingState.STABILIZE, yaw_rate=2.0)

    k = cycles_until(sm, PointingState.POINTING, yaw_rate=2.0)
    assert within_a_cycle(k, cycles(sm, STABILIZE_TIMEOUT_S) + 1)


def test_returning_to_pointing_resets_its_controller(base_params):
    sm, _ = make(base_params, rw_motor={"integral_gain": 1.0})
    for _ in range(50):
        sm.rpm(0.01, 0.0, MID_RPM)
    assert sm.controllers["pointing"].e_int > 0.0

    cycles_until(sm, PointingState.STABILIZE, error=0.01, yaw_rate=2.0)
    cycles_until(sm, PointingState.POINTING, yaw_rate=0.2)
    sm.rpm(0.01, 0.0, MID_RPM)  # first POINTING cycle: reset, then one charge
    assert sm.controllers["pointing"].e_int == pytest.approx(0.01 * sm.period)


def test_disabled_state_machine_stays_in_pointing(base_params):
    sm, params = make(base_params, state_machine=False)
    rw = params["rw_motor"]
    commands = [sm.rpm(-0.05, 3.0, 2900.0) for _ in range(1000)]

    assert sm.state == PointingState.POINTING
    law = rw["proportional_gain"] * -0.05 + rw["derivative_gain"] * 3.0
    assert commands[-1] == int(min(law, rw["max_rpm"]) + rw["rpm_bias"])


def test_run_logs_the_pointing_state(short_params):
    df = run_simulation(short_params, seed=0)
    assert (df["pointing_state"] == PointingState.POINTING.value).all()
