import numpy as np

class Sensor:

    def __init__(self, params: dict, duration: float, initial_value: float):
        self.t = duration
        self.sampling_rate = params["sampling_rate"] #Hz
        self.N = (params["noise_density"]*np.sqrt(self.sampling_rate/2))
        self.K = (params["random_walk"]*np.sqrt(self.sampling_rate/2))
        self.measurement = initial_value
        def _generate_noise(self):
            self.white_noise = np.random.normal(0, self.N, int(self.sampling_rate*self.t + 1))
            self.random_walk = np.cumsum(np.random.normal(0, self.K, int(self.sampling_rate*self.t + 1)))

        _generate_noise(self)
        self.step = 0

    def get_measurement(self, true_value: float, time: float):
            dt = 1/self.sampling_rate
            index = int(time // dt)
            if self.step % self.sampling_rate == 0.0:
                self.measurement = true_value + self.white_noise[index] + self.random_walk[index]
                self.step += 1
                return self.measurement
            else:
                self.step += 1
                return self.measurement
            