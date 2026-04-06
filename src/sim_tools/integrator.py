import numpy as np
from sim_tools.controller import Controller
from sim_tools.actuator import Motor
from sim_tools.disturbance import DisturbanceGenerator
from sim_tools.sensor import Sensor

class ModelIntegrator:
    def __init__(self, init_state: numpy.ndarray = None, dt: float = 0.1, constants: dict = {}):
        self.state = numpy.zeros(2) if init_state is None else init_state
        self.dt = dt
        self.constants = constants
        self.duration = constants['simulation']['duration']
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
        self.imu = Sensor(constants["inertial_measurement_unit"], duration=self.duration, initial_value=init_state[0])
        self.gps = Sensor(constants["gps"], duration=self.duration, initial_value=init_state[5])
        self.tachometer = Sensor(constants["tachometer"], duration=self.duration, initial_value=init_state[4])
        self.gyro = Sensor(constants["gyroscope"], duration=self.duration, initial_value=init_state[1])
    
    def _dynamics(self, state, t):
        yaw = state[0]
        ang_vel = state[1]
        rw_i = state[2]
        lt_i = state[3]
        rw_vel = state[4]
        x = state[5]
        y = state[6]

        x_dot = -2.5
        y_dot = 2.5
        payload_pos = self.gps.get_measurement(np.array([x, y]), t)

        tau_d = self.disturbance.generate_torque_disturbance(t)
        yaw_measured = self.imu.get_measurement(yaw, t)
        yaw_rate_measured = self.gyro.get_measurement(ang_vel, t)
        rw_vel_measured = self.tachometer.get_measurement(rw_vel, t)

        yaw_error = yaw_measured - np.arctan2(payload_pos[1], payload_pos[0])

        rw_voltage = self.rw_controller.output(yaw_error, yaw_rate_measured)
        rw_torque, di_dt_rw, rw_acc = self.rw_motor.torque(rw_voltage, rw_i, rw_vel)
        
        if self.momentum_management:
            lt_i = self.lt_controller.output(rw_vel_measured)
            lt_torque, _, _ = self.lt_motor.torque(current=lt_i)
            ang_acc = (tau_d - rw_torque - lt_torque - self.Cp*ang_vel) / self.Ip
            return np.array([ang_vel, ang_acc, di_dt_rw, lt_torque, rw_acc, x_dot, y_dot, tau_d, rw_torque])
        else:
            ang_acc = (tau_d - rw_torque - self.Cp*ang_vel) / self.Ip
            return np.array([ang_vel, ang_acc, di_dt_rw, 0.0, rw_acc, x_dot, y_dot, tau_d, rw_torque])
    
    def rk4_step(self, state: np.ndarray, t: float):
        dt = self.dt
        h1 = self._dynamics(state, t)
        h2 = self._dynamics(state + 0.5 * dt * h1, t + 0.5 * dt)
        h3 = self._dynamics(state + 0.5 * dt * h2, t + 0.5 * dt)
        h4 = self._dynamics(state + dt * h3, t + dt)

        new_state = (h1 + 2*h2 + 2*h3 + h4)*dt + state
        new_state[2] = float(np.clip(new_state[2], -self.max_current, self.max_current))
        new_state[3] = float(np.clip(new_state[3], -self.max_current, self.max_current))
        return new_state