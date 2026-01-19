import numpy as np

class Controller:
    def __init__(self, params: dict, dt:float):
        self.Kp = params["proportional_gain"]
        self.Kd = params["derivative_gain"]
        self.Ki = params["integral_gain"]
        self.max_value = params["max_voltage"]
        self.dt = dt

        self.e_prev = 0.0
        self.e_int = 0.0

    def output(self, error: float, error_derivative: float | None = None):
        P = self.Kp * error
        D = self.Kd * (error - self.e_prev)/self.dt if error_derivative is None else self.Kd * error_derivative

        self.e_int += error * self.dt
        I = self.Ki * self.e_int
        self.e_prev = error

        output = P + D + I
        output = np.clip(output, -self.max_value, self.max_value)

        return output