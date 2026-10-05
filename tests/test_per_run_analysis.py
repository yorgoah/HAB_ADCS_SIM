import numpy as np
import pandas as pd
import pytest

from hab_adcs_sim.monte_carlo import per_run_analysis as pra


def frame(error_deg, rpm=None) -> pd.DataFrame:
    error_deg = np.asarray(error_deg, dtype=float)
    rpm = np.full(error_deg.size, 1500.0) if rpm is None else np.asarray(rpm, dtype=float)
    return pd.DataFrame(
        {
            "time": np.arange(error_deg.size) * 0.1,
            "error": np.deg2rad(error_deg),
            "rw_vel": rpm * np.pi / 30.0,
        }
    )


def test_each_separate_excursion_counts_once():
    # One long positive excursion, one negative, one at the very end.
    error = [0, 9, 10, 9, 0, 0, -9, -9, 0, 12]
    assert pra.error_bound_violations(frame(error), {}) == 3.0


def test_a_run_starting_outside_the_bound_counts_that_excursion():
    assert pra.error_bound_violations(frame([9, 9, 0, 0]), {}) == 1.0


def test_a_run_that_never_leaves_the_bound_has_no_violations():
    df = frame([0, 3, -7.9, 7.9])
    assert pra.error_bound_violations(df, {}) == 0.0
    assert pra.pct_time_within_bound(df, {}) == 100.0


def test_sitting_on_the_bound_counts_as_within_it():
    df = frame([8.0, -8.0, 0.0, 8.5])
    assert pra.pct_time_within_bound(df, {}) == pytest.approx(75.0)
    assert pra.error_bound_violations(df, {}) == 1.0


def blocks(*specs) -> list[float]:
    """Error pattern from (within_bound, n_samples) pairs, n samples last (n - 1) / 10 s."""
    values: list[float] = []
    for inside, count in specs:
        values.extend([0.0 if inside else 20.0] * count)
    return values


def test_the_longest_stretch_wins_not_the_last_one():
    df = frame(blocks((True, 51), (False, 5), (True, 121)))  # 5.0 s then 12.0 s
    assert pra.longest_pointing_window_s(df, {}) == pytest.approx(12.0)


def test_windows_are_packed_into_each_stretch_and_never_span_an_excursion():
    # 95 s, 40 s and 12 s of unbroken pointing: 3 + 1 + 0 whole 30 s windows.
    df = frame(blocks(
        (False, 1), (True, 951), (False, 1), (True, 401), (False, 1), (True, 121)
    ))
    assert pra.pointing_windows(df, {}) == 4.0
    assert pra.longest_pointing_window_s(df, {}) == pytest.approx(95.0)


def test_a_run_that_never_holds_the_bound_has_no_window():
    df = frame([20.0] * 100)
    assert pra.longest_pointing_window_s(df, {}) == 0.0
    assert pra.pointing_windows(df, {}) == 0.0


def test_a_run_that_never_leaves_the_bound_is_one_long_window():
    df = frame(np.zeros(6001))  # 600.0 s
    assert pra.longest_pointing_window_s(df, {}) == pytest.approx(600.0)
    assert pra.pointing_windows(df, {}) == 20.0


def test_pointing_held_shorter_than_one_window_counts_for_nothing():
    df = frame(blocks((False, 1), (True, 290), (False, 1)))  # 28.9 s
    assert pra.pointing_windows(df, {}) == 0.0
    assert pra.longest_pointing_window_s(df, {}) == pytest.approx(28.9)


def test_saturation_follows_the_flight_thresholds():
    # Flagged at >= 2850 rpm or <= 50 rpm
    rpm = [1500, 2860, 2840, 40, 60, -10, 3000, 1500]
    df = frame(np.zeros(len(rpm)), rpm)
    assert pra.rw_saturated_pct(df, {}) == pytest.approx(50.0)


def test_a_wheel_near_its_bias_is_never_saturated():
    df = frame(np.zeros(5), [1400, 1500, 1600, 2000, 1000])
    assert pra.rw_saturated_pct(df, {}) == 0.0


def test_analyze_records_the_limits_it_used():
    payload = pra.analyze(frame([0, 9, 0]), {})

    assert payload["analysis_version"] == pra.ANALYSIS_VERSION
    assert payload["limits"] == {
        "error_bound_deg": 8.0,
        "rw_saturation_low_rpm": 50.0,
        "rw_saturation_high_rpm": 2850.0,
        "pointing_window_s": 30.0,
    }
    assert set(payload["metrics"]) == {
        "error_bound_violations",
        "pct_time_within_bound",
        "longest_pointing_window_s",
        "pointing_windows",
        "rw_saturated_pct",
    }
    assert list(payload["timeseries"].columns) == ["time", "error"]
