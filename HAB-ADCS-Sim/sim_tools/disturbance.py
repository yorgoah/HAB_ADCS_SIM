import numpy as np
from scipy.signal import lfilter

class DisturbanceGenerator:
    def __init__(self, params: dict):
        self.dt = params["simulation"]["time_step"]
        self.sampling_rate = 1/self.dt
        self.duration = params["simulation"]["duration"]
        self.p = params["wind_params"]["order"]
        self.ar_coeffs = params["wind_params"]["ar_coeffs"]
        self.sigma_noise = params["wind_params"]["sigma_noise"]
        self.f_min = params["wind_params"]["turb_f_min"]
        self.cutoff_freq = params["wind_params"]["cutoff_freq"]
    
    def generate_wind_disturbance(self, t):
        N = int(self.duration / self.dt) 
        white_noise = np.random.normal(0, self.sigma_noise, N)
        ar_output = lfilter([1], np.concatenate(([1], -np.array(self.ar_coeffs))), white_noise)

        def _von_karman_spectrum(n, f_min, f_max, sampling_rate):
            freqs = np.fft.fftfreq(n, d=1/sampling_rate)
            freqs = np.fft.fftshift(freqs)

            spectrum = np.zeros(n)
            for i, f in enumerate(freqs):
                if np.abs(f) > f_min:
                    spectrum[i] = 1 / (np.abs(f)**(5/3))

            random_phases = np.exp(2j * np.pi * np.random.random(n))
            noise_freq = np.sqrt(spectrum) * random_phases
            noise_time = np.fft.ifft(noise_freq)
            return np.real(noise_time)

        high_freq_turbulence = _von_karman_spectrum(N, self.f_min, self.cutoff_freq, self.sampling_rate)
        total_wind_speed = ar_output + high_freq_turbulence

        time_index = int(t // self.dt) - 2

        return total_wind_speed[time_index]
