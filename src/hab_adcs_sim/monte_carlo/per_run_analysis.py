"""Metrics computed for each Monte Carlo run.

Bump ANALYSIS_VERSION after changing a metric, limit or channel so that
`run --stage analysis` recomputes them without simulating again.
"""

import numpy as np
import pandas as pd

ANALYSIS_VERSION = 3

# Channels kept for plotting, "time" is required
CHANNELS = ("time", "error")

ERROR_BOUND_DEG = 8.0  # azimuth pointing requirement
POINTING_WINDOW_S = 30.0  # unbroken time within the bound for one observation

# Same saturation rule as the flight computer (saturation_rpm and 50 rpm margin)
RW_SATURATION_RPM = 2900.0
RW_SATURATION_MARGIN_RPM = 50.0

RAD_S_TO_RPM = 30.0 / np.pi

LIMITS = {
    "error_bound_deg": ERROR_BOUND_DEG,
    "rw_saturation_low_rpm": RW_SATURATION_MARGIN_RPM,
    "rw_saturation_high_rpm": RW_SATURATION_RPM - RW_SATURATION_MARGIN_RPM,
    "pointing_window_s": POINTING_WINDOW_S,
}


def _outside_bound(df: pd.DataFrame) -> np.ndarray:
    return np.abs(np.rad2deg(df["error"].to_numpy())) > ERROR_BOUND_DEG


def error_bound_violations(df: pd.DataFrame, params: dict) -> float:
    """Number of separate excursions beyond the pointing bound."""
    outside = _outside_bound(df)
    if outside.size == 0:
        return 0.0
    entries = np.count_nonzero(outside[1:] & ~outside[:-1])
    return float(entries + int(outside[0]))


def pct_time_within_bound(df: pd.DataFrame, params: dict) -> float:
    return float(100.0 * np.mean(~_outside_bound(df)))


def _within_bound_durations(df: pd.DataFrame) -> np.ndarray:
    """Duration (s) of each unbroken stretch inside the bound, first to last sample."""
    inside = ~_outside_bound(df)
    if inside.size == 0:
        return np.empty(0)

    time = df["time"].to_numpy()
    edges = np.diff(inside.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1)
    if inside[0]:
        starts = np.concatenate(([0], starts))
    if inside[-1]:
        ends = np.concatenate((ends, [inside.size - 1]))
    return time[ends] - time[starts]


def longest_pointing_window_s(df: pd.DataFrame, params: dict) -> float:
    durations = _within_bound_durations(df)
    return float(durations.max()) if durations.size else 0.0


def pointing_windows(df: pd.DataFrame, params: dict) -> float:
    """Number of whole POINTING_WINDOW_S windows, e.g. a 95 s stretch gives three."""
    durations = _within_bound_durations(df)
    return float(np.sum(np.floor(durations / POINTING_WINDOW_S)))


def rw_saturated_pct(df: pd.DataFrame, params: dict) -> float:
    """Percent of the run with the saturation flag set (not the SATURATED state)."""
    rpm = df["rw_vel"].to_numpy() * RAD_S_TO_RPM
    upper = rpm >= RW_SATURATION_RPM - RW_SATURATION_MARGIN_RPM
    lower = rpm <= RW_SATURATION_MARGIN_RPM
    return float(100.0 * np.mean(upper | lower))


METRICS = {
    "error_bound_violations": error_bound_violations,
    "pct_time_within_bound": pct_time_within_bound,
    "longest_pointing_window_s": longest_pointing_window_s,
    "pointing_windows": pointing_windows,
    "rw_saturated_pct": rw_saturated_pct,
}


def analyze(df: pd.DataFrame, params: dict) -> dict:
    """Metrics, limits and plotting channels of one run, as plain types."""
    metrics = {name: float(fn(df, params)) for name, fn in METRICS.items()}
    channels = [c for c in CHANNELS if c in df.columns]
    return {
        "analysis_version": ANALYSIS_VERSION,
        "metrics": metrics,
        "limits": dict(LIMITS),
        "timeseries": df[channels].copy(),
    }
