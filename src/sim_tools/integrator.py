import numpy as np
from sim_tools.controller import Controller
from sim_tools.actuator import Motor
from sim_tools.disturbance import DisturbanceGenerator
from sim_tools.pointing import PointingStateMachine, RAD_S_TO_RPM
from sim_tools.sensor import Sensor


def wrap_angle(angle):
    """Wrap an angle to (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class ModelIntegrator:
    def __init__(
        self,
        dt: float,
        init_state: np.ndarray = None,
        constants: dict = None,
        seed: int | None = None,
        wind_seed: int | None = None,
    ):
        constants = {} if constants is None else constants
        self.state = np.zeros(9) if init_state is None else init_state
        self.dt = dt
        self.constants = constants
        self.duration = constants['simulation']['duration']
        self.Ip = constants['Payload_params']['Ip']
        self.Kp = constants['Payload_params']['Kp']
        self.Cp = constants['Payload_params']['Cp']
        if seed is None:
            dist_rng = imu_rng = gps_rng = tach_rng = gyro_rng = None
        else:
            dist_rng, imu_rng, gps_rng, tach_rng, gyro_rng = (
                np.random.default_rng(child)
                for child in np.random.SeedSequence(seed).spawn(5)
            )
        # Separate wind seed so runs can share sensor noise but not wind
        if wind_seed is not None:
            dist_rng = np.random.default_rng(wind_seed)

        self.disturbance = DisturbanceGenerator(constants, rng=dist_rng)

        self.pointing = PointingStateMachine(constants, dt)
        self.rpm_bias = constants['rw_motor']['rpm_bias']
        self.rw_motor = Motor(constants["rw_motor"])
        self.momentum_management = constants['lt_motor']['activate']
        self.lt_max_current = constants['lt_motor']['max_current']
        self.lt_controller = Controller(
            params=constants['lt_motor'],
            dt=dt,
            output_limit=self.lt_max_current,
        )
        self.lt_motor = Motor(constants["lt_motor"])
        self.imu = Sensor(constants["inertial_measurement_unit"], duration=self.duration, initial_value=init_state[0], rng=imu_rng)
        self.gps = Sensor(constants["gps"], duration=self.duration, initial_value=init_state[5], rng=gps_rng)
        self.tachometer = Sensor(constants["tachometer"], duration=self.duration, initial_value=init_state[4], rng=tach_rng)
        self.gyro = Sensor(constants["gyroscope"], duration=self.duration, initial_value=init_state[1], rng=gyro_rng)
        self.rw_voltage = 0.0
        self.lt_current = 0.0
        self.yaw_error = 0.0

    def _update_commands(self, state, t):
        # Discrete sensors and controllers, updated once per step outside the RK4 stages
        yaw = state[0]
        ang_vel = state[1]
        rw_vel = state[4]
        x = state[5]
        y = state[6]

        payload_pos = self.gps.get_measurement(np.array([x, y]), t)
        yaw_measured = self.imu.get_measurement(yaw, t)
        yaw_rate_measured = self.gyro.get_measurement(ang_vel, t)
        rw_velocity_measurement = self.tachometer.get_measurement(rw_vel, t)

        self.yaw_error = wrap_angle(
            yaw_measured - np.arctan2(payload_pos[1], payload_pos[0])
        )

        rpm_command = self.pointing.rpm(
            self.yaw_error, yaw_rate_measured, rw_velocity_measurement * RAD_S_TO_RPM
        )
        self.rw_voltage = self.rw_motor.voltage(rpm_command)

        if self.momentum_management:
            self.lt_current = float(self.lt_controller.output(rw_velocity_measurement - (self.rpm_bias * np.pi / 30)))
        else:
            self.lt_current = 0.0

    def _dynamics(self, state, t, rw_voltage, lt_current):
        ang_vel = state[1]
        rw_i = state[2]
        rw_vel = state[4]

        x_dot = -2.5
        y_dot = 2.5

        tau_d = self.disturbance.generate_torque_disturbance(t)

        rw_torque, di_dt_rw = self.rw_motor.torque(rw_voltage, rw_i, rw_vel)
        rw_acc = rw_torque / self.rw_motor.J
        # Pivot motor bearing drag acts even when the motor is off
        lt_torque, _ = self.lt_motor.torque(current=lt_current, angular_velocity=-ang_vel)
        ang_acc = (tau_d - rw_torque - lt_torque - self.Cp*ang_vel) / self.Ip
        return np.array([ang_vel, ang_acc, di_dt_rw, lt_torque, rw_acc, x_dot, y_dot, tau_d, rw_torque])

    def rk4_step(self, state: np.ndarray, t: float):
        dt = self.dt
        self._update_commands(state, t)
        u_rw, i_lt = self.rw_voltage, self.lt_current

        h1 = self._dynamics(state, t, u_rw, i_lt)
        h2 = self._dynamics(state + 0.5 * dt * h1, t + 0.5 * dt, u_rw, i_lt)
        h3 = self._dynamics(state + 0.5 * dt * h2, t + 0.5 * dt, u_rw, i_lt)
        h4 = self._dynamics(state + dt * h3, t + dt, u_rw, i_lt)

        new_state = (h1 + 2*h2 + 2*h3 + h4)*dt / 6.0 + state
        # TODO: Add telemetry handling separately.
        new_state[3] = h4[3]
        new_state[7] = h4[7]
        new_state[8] = h4[8]
        return new_state

    def angular_momentum(self, state):
        """Total yaw angular momentum of the payload plus wheel."""
        return self.Ip * state[1] + self.rw_motor.J * state[4]
