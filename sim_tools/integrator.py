import numpy as np
from controller import Controller
from actuator import Motor
from disturbance import DisturbanceGenerator

class ModelIntegrator:
    def __init__(self, init_state: numpy.ndarray = None, dt: float = 0.1, constants: dict = {}):
        self.state = numpy.zeros(2) if init_state is None else init_state
        self.dt = dt
        self.constants = constants
        self.Ip = constants['Payload_params']['Ip']
        self.Kp = constants['Payload_params']['Kp']
        self.Cp = constants['Payload_params']['Cp']
        self.disturbance = DisturbanceGenerator(constants)

        self.rw_controller = Controller(
            params=constants['rw_motor'],
            dt=dt
        )
        self.rw_motor = Motor(constants["rw_motor"])
        self.max_current = constants['rw_motor']['max_current']
        self.lt_controller = Controller(
            params=constants['lt_motor'],
            dt=dt
        )
        self.lt_motor = Motor(constants["lt_motor"])

    def _dynamics(self, state, t):
        yaw = state[0]
        ang_vel = state[1]
        rw_i = state[2]
        lt_i = state[3]
        rw_vel = state[4]

        tau_d = self.disturbance.generate_torque_disturbance(t)

        error = yaw

        lt_voltage = self.lt_controller.output(rw_vel)
        lt_torque, di_dt_lt, _ = self.lt_motor.torque(lt_voltage, lt_i, ang_vel)

        rw_voltage = self.rw_controller.output(error)
        rw_torque, di_dt_rw, rw_acc = self.rw_motor.torque(rw_voltage, rw_i, rw_vel)
        
        
        ang_acc = (tau_d - rw_torque - lt_torque - self.Cp*ang_vel) / self.Ip
        return np.array([ang_vel, ang_acc, di_dt_rw, di_dt_lt, rw_acc])
    
    def rk4_step(self, state: np.ndarray, t: float):
        dt = self.dt
        h1 = self._dynamics(state, t)
        h2 = self._dynamics(state + 0.5 * dt * h1, t + 0.5 * dt)
        h3 = self._dynamics(state + 0.5 * dt * h2, t + 0.5 * dt)
        h4 = self._dynamics(state + dt * h3, t + dt)

        new_state = (h1 + 2*h2 + 2*h3 + h4)*dt + state
        new_state[2] = float(np.clip(new_state[2], -0.45, 0.45))
        new_state[3] = float(np.clip(new_state[3], -0.45, 0.45))
        return new_state