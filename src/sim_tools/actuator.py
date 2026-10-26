import numpy as np

class Motor:
    def __init__(self, params: dict):
        self.J = params["rotor_inertia"] + params["load_inertia"]
        self.Kt = params["torque_constant"]
        self.Kb = params["back_emf_constant"]
        self.R = params["resistance"]
        self.L = params["inductance"]
        self.b = params["viscous_friction_coeff"]
        # Bearing drag (N-m), only set for the pivot motor
        self.coulomb = params.get("coulomb_friction", 0.0)
        self.max_current = params["max_current"]

    def torque(self, voltage: float | None=0.0, current: float | None=0.0, angular_velocity: float | None=0.0) -> tuple[float, float]:
        back_emf = self.Kb * angular_velocity
        voltage = np.clip(voltage, back_emf - self.R * self.max_current, back_emf + self.R * self.max_current)
        di_dt = (voltage - self.R * current - back_emf) / self.L
        friction = self.b * angular_velocity + self.coulomb * np.sign(angular_velocity)
        torque = self.Kt * np.clip(current, -self.max_current, self.max_current) - friction
        return torque, di_dt

    def voltage(self, rpm: float) -> float:
        """Computes voltage given an rpm input.
        
        Converts rpm to rad/s and multiplies by back EMF constant."""
        return self.Kb * rpm * np.pi / 30

class VescSpeedControl:
    """RPM mode of the VESC: a PI speed loop sets the current, the FOC current loop tracks it.

    Gains are in VESC Tool units (per ERPM). The output is a fraction of the
    current limit and the integral is clamped to +/-1, as in the firmware. The
    current loop is taken as ideal: R * i_cmd plus back-EMF feedforward.
    """

    def __init__(self, params: dict, motor: Motor, dt: float):
        self.kp = params["kp"]
        self.ki = params["ki"]
        self.erpm_per_rad_s = params["pole_pairs"] * 30 / np.pi
        self.motor = motor
        self.dt = dt
        self.i_term = 0.0

    def voltage(self, rpm: float, angular_velocity: float) -> float:
        error = (rpm * np.pi / 30 - angular_velocity) * self.erpm_per_rad_s
        self.i_term = np.clip(self.i_term + error * self.ki * self.dt, -1.0, 1.0)
        output = np.clip(error * self.kp + self.i_term, -1.0, 1.0)
        current = output * self.motor.max_current
        return self.motor.R * current + self.motor.Kb * angular_velocity
