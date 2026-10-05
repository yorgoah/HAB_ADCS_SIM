"""Reaction wheel pointing state machine of the flight computer.

Same logic as PointingTask in Altairfc_V2 (tasks/pointing_task.py and
drivers/rw_driver.py). Pointing is enabled on the ground in flight, so a run
starts in POINTING and never sees IDLE or SPINUP.
"""

from collections import deque
from enum import Enum

import numpy as np

from sim_tools.controller import Controller

RAD_S_TO_RPM = 30.0 / np.pi

# Hard-coded in the flight code
SATURATION_MARGIN_RPM = 50.0
STABILIZE_TIMEOUT_S = 300.0
DESATURATE_MIN_S = 5.0
DESATURATE_BAND_RPM = 100.0  # desaturation ends within the bias +/- 100 rpm
DESATURATION_RATE_RPM = 100  # per task cycle
MIN_RPM, MAX_RPM = 0, 3000
MEDIAN_WINDOW_S = 0.5  # yaw rate median filter, at least 5 samples


class PointingState(Enum):
    IDLE = 0
    SPINUP = 1
    STABILIZE = 2
    POINTING = 3
    SATURATED = 4


class PointingStateMachine:
    """Runs every rw_motor.period_s, the VESC holds the last command in between."""

    def __init__(self, params: dict, dt: float):
        rw = params["rw_motor"]
        cfg = params["pointing"]
        self.period = rw["period_s"]
        self.hold_steps = max(1, round(self.period / dt))
        self.rpm_bias = rw["rpm_bias"]
        # dt = period so every call is an update, the task cycle sets the timing
        self.controllers = {
            "pointing": Controller(rw, dt=self.period, output_limit=rw["max_rpm"]),
            "stabilize": Controller({**rw, **rw["stabilize"]}, dt=self.period, output_limit=rw["max_rpm"]),
        }
        self.mode = "stabilize"
        # With the state machine off the wheel stays in POINTING
        self.transitions = cfg["state_machine"]
        self.stabilize_yaw_rate = cfg["stabilize_yaw_rate"]
        self.unstable_yaw_rate = cfg["unstable_yaw_rate"]
        self.stability_threshold = cfg["stability_threshold"]
        self.saturation_rpm = cfg["saturation_rpm"]
        self.saturation_s = cfg["saturation_s"]

        self.state = PointingState.POINTING
        self.state_started = 0.0
        self.saturated_since = None
        self.stable_since = None
        self.unstable_since = None
        self.yaw_rate_window = deque(maxlen=max(5, int(MEDIAN_WINDOW_S / self.period)))
        self.rpm_command = 0
        self.step = 0
        self.cycle = 0

    def rpm(self, yaw_error: float, yaw_rate: float, rw_rpm: float) -> int:
        """Wheel speed command (rpm) from the measured error, yaw rate and wheel speed."""
        self.step += 1
        if (self.step - 1) % self.hold_steps:
            return self.rpm_command
        # Task time from the cycle count, avoids drift of the accumulated t
        now = self.cycle * self.period
        self.cycle += 1

        if self.state == PointingState.STABILIZE:
            self._stabilize(now, yaw_rate, rw_rpm)
        elif self.state == PointingState.POINTING:
            self._point(now, yaw_error, yaw_rate, rw_rpm)
        elif self.state == PointingState.SATURATED:
            self._desaturate(now, rw_rpm)
        return self.rpm_command

    def _point(self, now, error, yaw_rate, rw_rpm):
        self._set_mode("pointing")
        if self.transitions:
            if self._is_saturated(now, rw_rpm):
                self._set_state(PointingState.SATURATED, now)
                return
            if self._is_unstable(now, yaw_rate):
                self._set_state(PointingState.STABILIZE, now)
                return
        self._set_rpm(self.controllers["pointing"].output(error, yaw_rate) + self.rpm_bias)

    def _stabilize(self, now, yaw_rate, rw_rpm):
        self._set_mode("stabilize")
        if self._is_saturated(now, rw_rpm):
            self._set_state(PointingState.SATURATED, now)
            return
        stable = self._is_stable(now, yaw_rate)
        if stable or now - self.state_started > STABILIZE_TIMEOUT_S:
            self._set_state(PointingState.POINTING, now)
            return
        self._set_rpm(self.controllers["stabilize"].output(yaw_rate) + self.rpm_bias)

    def _desaturate(self, now, rw_rpm):
        # Ramp the command back to the bias
        step = np.clip(self.rpm_bias - self.rpm_command, -DESATURATION_RATE_RPM, DESATURATION_RATE_RPM)
        self.rpm_command = int(self.rpm_command + step)
        in_band = abs(abs(rw_rpm) - self.rpm_bias) < DESATURATE_BAND_RPM
        if now - self.state_started >= DESATURATE_MIN_S and in_band:
            self._set_state(PointingState.STABILIZE, now)

    def _set_rpm(self, rpm):
        self.rpm_command = int(np.clip(int(rpm), MIN_RPM, MAX_RPM))

    def _is_saturated(self, now, rw_rpm):
        upper = rw_rpm >= self.saturation_rpm - SATURATION_MARGIN_RPM
        lower = rw_rpm <= SATURATION_MARGIN_RPM
        if not (upper or lower):
            self.saturated_since = None
            return False
        if self.saturated_since is None:
            self.saturated_since = now
        return now - self.saturated_since >= self.saturation_s

    def _filtered_yaw_rate(self, yaw_rate):
        self.yaw_rate_window.append(yaw_rate)
        if len(self.yaw_rate_window) < self.yaw_rate_window.maxlen:
            return None
        return float(np.median(self.yaw_rate_window))

    def _is_stable(self, now, yaw_rate):
        filtered = self._filtered_yaw_rate(yaw_rate)
        if filtered is None or abs(filtered) > self.stabilize_yaw_rate:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = now
            return False
        return now - self.stable_since >= self.stability_threshold

    def _is_unstable(self, now, yaw_rate):
        filtered = self._filtered_yaw_rate(yaw_rate)
        if filtered is None:
            return False
        if abs(filtered) <= self.unstable_yaw_rate:
            self.unstable_since = None
            return False
        if self.unstable_since is None:
            self.unstable_since = now
            return False
        return now - self.unstable_since >= self.stability_threshold

    def _set_mode(self, mode):
        if mode != self.mode:
            self.mode = mode
            self.controllers[mode].reset_integrator()

    def _set_state(self, state, now):
        if state != self.state:
            self.state = state
            self.state_started = now
            self.stable_since = None
            self.unstable_since = None
            self.saturated_since = None
            if state in (PointingState.IDLE, PointingState.SPINUP, PointingState.SATURATED):
                self.yaw_rate_window.clear()
