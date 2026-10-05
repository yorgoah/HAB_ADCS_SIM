import numpy as np

class Sensor:

    def __init__(self, params: dict, duration: float, initial_value: float, rng=None):
        self.t = duration
        self.sampling_rate = params["sampling_rate"] #Hz
        self.N = (params["noise_density"]*np.sqrt(self.sampling_rate/2))
        self.K = (params["random_walk"]*np.sqrt(self.sampling_rate/2))
        self.measurement = initial_value
        self.rng = np.random if rng is None else rng
        def _generate_noise(self):
            self.white_noise = self.rng.normal(0, self.N, int(self.sampling_rate*self.t + 1))
            self.random_walk = np.cumsum(self.rng.normal(0, self.K, int(self.sampling_rate*self.t + 1)))

        _generate_noise(self)
        self.last_index = -1

    def get_measurement(self, true_value: float, time: float):
            dt = 1/self.sampling_rate
            index = min(int(time // dt), len(self.white_noise) - 1)
            if index != self.last_index:
                self.measurement = true_value + self.white_noise[index] + self.random_walk[index]
                self.last_index = index
            return self.measurement
