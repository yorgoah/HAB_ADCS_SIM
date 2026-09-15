import numpy as np

class Controller:
    def __init__(self, params: dict, dt:float, output_limit: float):
        self.Kp = params["proportional_gain"]
        self.Kd = params["derivative_gain"]
        self.Ki = params["integral_gain"]
        self.max_value = output_limit 
        self.dt = params.get("period_s", dt)
        self.hold_steps = max(1, round(self.dt / dt))
        self.step = 0
        self.last_output = 0.0

        self.e_prev = 0.0
        self.e_int = 0.0

    def output(self, error: float, error_derivative: float | None = None):
        self.step += 1
        if (self.step - 1) % self.hold_steps:
            return self.last_output

        P = self.Kp * error
        D = self.Kd * (error - self.e_prev)/self.dt if error_derivative is None else self.Kd * error_derivative

        self.e_prev = error

        # Conditional integration (anti-windup).
        e_int = self.e_int + error * self.dt
        unclipped = P + D + self.Ki * e_int
        if abs(unclipped) <= self.max_value or np.sign(error) != np.sign(unclipped):
            self.e_int = e_int

        output = P + D + self.Ki * self.e_int
        self.last_output = np.clip(output, -self.max_value, self.max_value)

        return self.last_output