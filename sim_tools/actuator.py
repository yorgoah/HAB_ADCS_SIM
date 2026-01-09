import numpy as np

class Motor:
    def __init__(self, params: dict):
        self.J = params["rotor_inertia"] + params["load_inertia"]
        self.Kt = params["torque_constant"]
        self.Kb = params["back_emf_constant"]
        self.R = params["resistance"]
        self.L = params["inductance"]
        self.b = params["viscous_friction_coeff"]

    def torque(self, voltage: float, current: float, angular_velocity: float) -> float:
        di_dt = (voltage - self.R * current - self.Kb * angular_velocity) / self.L
        torque = self.Kt * current - self.b * angular_velocity
        rw_acc = torque/self.J
        return torque, di_dt, rw_acc