import numpy as np

RAD_S_TO_RPM = 30.0 / np.pi

class Modulator:
    def __init__(self, parameters: dict):
        self.pulse_current = parameters["lt_motor"]["pulse_current"]
        self.settle_time = parameters["lt_motor"]["settle_time"]
        self.integration_threshold = parameters["lt_motor"]["integration_threshold"]
        self.deadband = parameters["lt_motor"]["deadband"]  # normalized offset
        self.loop_gain = parameters["lt_motor"]["loop_gain"]
        self.reference_speed = parameters["rw_motor"]["rpm_bias"]
        self.max_offset = parameters["rw_motor"]["max_rpm"]
        self.dt = parameters["simulation"]["time_step"]
        self.lookahead = parameters["lt_motor"]["lookahead_s"]
        self.wheel_inertia = parameters["rw_motor"]["load_inertia"] + parameters["rw_motor"]["rotor_inertia"]
        self.wheel_torque_constant = parameters["rw_motor"]["torque_constant"]
        self.wheel_friction = parameters["rw_motor"]["viscous_friction_coeff"]
        self.time_constant = (self.wheel_inertia * (self.max_offset / RAD_S_TO_RPM)) / (parameters["lt_motor"]["torque_constant"] * self.pulse_current)
        self.rw_time_constant = self.wheel_torque_constant / (self.wheel_inertia)
        self.accumulator = 0.0
        self.pulse_duration = 0.0
        self.pulse_start = None
        self.direction = 0.0

    def modulate(self, signal, current, t) -> float:
        """Returns a current command based on the accumulator value.

        signal is the measured wheel speed (rpm) and current the wheel motor current (A).
        Pulsing is handled internally.
        """
        if self.pulse_start is not None:
            elapsed = t - self.pulse_start
            if elapsed < self.pulse_duration:
                return self.direction * self.pulse_current
            if elapsed < self.pulse_duration + self.settle_time:
                return 0.0  # settling, accumulator frozen
            self.pulse_start = None
            self.accumulator = 0.0

        
        offset = (signal - self.reference_speed) / self.max_offset
        accel = (self.wheel_torque_constant * current - self.wheel_friction * signal / RAD_S_TO_RPM) / (self.wheel_inertia * self.max_offset / RAD_S_TO_RPM)
        predicted = offset + self.lookahead * accel
        self.accumulator += np.sign(predicted) * max(abs(predicted) - self.deadband / 2.0, 0.0) * self.dt
        if (abs(self.accumulator) >= self.integration_threshold and np.sign(predicted) == np.sign(self.accumulator)
                and abs(predicted) > self.deadband / 2.0):
            impulse_bit = min(max(self.loop_gain * abs(predicted), self.deadband), self.loop_gain)
            self.pulse_duration = impulse_bit * self.time_constant
            self.direction = np.sign(predicted)
            self.pulse_start = t
            return self.direction * self.pulse_current
        return 0.0