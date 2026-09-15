import numpy as np

class Motor:
    def __init__(self, params: dict):
        self.J = params["rotor_inertia"] + params["load_inertia"]
        self.Kt = params["torque_constant"]
        self.Kb = params["back_emf_constant"]
        self.R = params["resistance"]
        self.L = params["inductance"]
        self.b = params["viscous_friction_coeff"]
        self.max_current = params["max_current"]

    def torque(self, voltage: float | None=0.0, current: float | None=0.0, angular_velocity: float | None=0.0) -> tuple[float, float, float]:
        back_emf = self.Kb * angular_velocity
        voltage = np.clip(voltage, back_emf - self.R * self.max_current, back_emf + self.R * self.max_current)
        di_dt = (voltage - self.R * current - back_emf) / self.L
        torque = self.Kt * np.clip(current, -self.max_current, self.max_current) - self.b * angular_velocity
        acc = torque / self.J
        return torque, di_dt, acc

    def voltage(self, rpm: float) -> float:
        """Computes voltage given an rpm input.
        
        Converts rpm to rad/s and multiplies by back EMF constant."""
        return self.Kb * rpm * np.pi / 30