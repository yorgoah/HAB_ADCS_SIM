import pandas as pd
import numpy as np
from scipy.signal import lfilter
from pathlib import Path
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
        self.simulated = params["wind_params"]["simulated"]
        repo_root = Path(__file__).resolve().parents[2]
        self.data_path = repo_root / "config" / "ressources" / "log100_vehicle_angular_velocity_0.csv"
        self.start = params["wind_params"]["start"]
        self.wind_torque = None

        if self.simulated:
            self.wind_torque = self._generate_wind_disturbance()
        else:
            df = pd.read_csv(self.data_path)
            self.time = df["timestamp"].to_numpy() / 1e6
            self.torque = params["Payload_params"]["Ip"] * df["xyz_derivative[2]"].to_numpy()
            self.start_idx = int(np.argmin(np.abs(self.time - self.start)))

    
    def _generate_wind_disturbance(self):
        N = int(self.duration / self.dt) + 2
        white_noise = np.random.normal(0, self.sigma_noise, N)
        ar_output = lfilter([1], np.concatenate(([1], -np.array(self.ar_coeffs))), white_noise)

        def _von_karman_spectrum(n, f_min, sampling_rate):
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

        high_freq_turbulence = _von_karman_spectrum(N, self.f_min, self.sampling_rate)
        total_wind_speed = ar_output + high_freq_turbulence
        return 0.05*0.191*total_wind_speed*np.abs(total_wind_speed)*0.8
    
    def generate_torque_disturbance(self, t):
        if self.simulated:
            idx = int(np.clip(np.floor(t / self.dt), 0, len(self.wind_torque) - 1))
            return self.wind_torque[idx]
        else:
            idx = self.start_idx + int(t / 0.02)
            idx = int(np.clip(idx, 0, len(self.torque) - 1))
            return self.torque[idx]
