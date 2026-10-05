import numpy as np
from scipy.signal import lfilter
from sim_tools.wind_surrogate import (
    ascent_altitude,
    gust_rms,
    load_yaw_acceleration,
    log_path,
    steady_torque,
    surrogate_torque,
)

# replay the flight log, a surrogate of it, or the AR + von Karman model
WIND_MODELS = ("replay", "surrogate", "synthetic")


def wind_params_with_defaults(params: dict) -> dict:
    """wind_params with defaults for keys missing from older parameter files."""
    wind = dict(params["wind_params"])
    if "model" not in wind:
        wind["model"] = "synthetic" if wind["simulated"] else "replay"
        wind["file"] = "log137"
        wind["flight_inertia"] = params["Payload_params"]["Ip"]
    wind.setdefault("steady_torque_per_density", 0.0)
    return wind


class DisturbanceGenerator:
    def __init__(self, params: dict, rng=None):
        self.rng = np.random if rng is None else rng # Seed
        self.dt = params["simulation"]["time_step"]
        self.sampling_rate = 1/self.dt
        self.duration = params["simulation"]["duration"]
        self.ar_coeffs = params["wind_params"]["ar_coeffs"]
        self.sigma_noise = params["wind_params"]["sigma_noise"]
        self.f_min = params["wind_params"]["turb_f_min"]
        wind = wind_params_with_defaults(params)
        self.model = wind["model"]
        if self.model not in WIND_MODELS:
            raise ValueError(f"wind_params.model is {self.model!r}, expected one of {WIND_MODELS}")
        self.data_path = log_path(wind["file"])
        # A start is provided to pick which disturbance profile from
        # the flight data we want to use. Essentially allowing us to
        # study simulation behaviour at different altitudes.
        self.start = wind["start"]
        self.wind_torque = None
        # Only the surrogate models an ascent
        self.ascent = wind if self.model == "surrogate" and "ascent_rate_m_s" in wind else None

        if self.model == "synthetic":
            self.wind_torque = self._generate_wind_disturbance()
        else:
            # Torque from the flight vehicle's inertia, not the simulated payload's
            log_time, acceleration = load_yaw_acceleration(self.data_path)
            torque = wind["flight_inertia"] * acceleration
            if self.model == "replay":
                # Writable copy, np.interp copies read-only arrays on every call
                self.time, self.torque = np.array(log_time), torque
                self.start_time = float(np.clip(self.start, log_time[0], log_time[-1]))
            else:
                scaled = "gust_density_exponent" in wind
                if scaled and self.ascent is None:
                    raise ValueError(
                        "wind_params.gust_density_exponent sets the gusts by altitude, "
                        "which needs start_altitude_m and ascent_rate_m_s"
                    )
                self.time, self.torque = surrogate_torque(
                    log_time, torque, wind["source_range"], self.duration, self.rng,
                    normalize=scaled,
                )
                if scaled:
                    # Gust level at altitude times a lognormal factor per run
                    intensity = np.exp(wind.get("gust_rms_log_sigma", 0.0) * self.rng.normal())
                    self.torque = self.torque * gust_rms(self.time, wind) * intensity
                # The surrogate is zero mean, add the steady part of the wind
                if wind["steady_torque_per_density"]:
                    self.torque = self.torque + steady_torque(self.time, wind)
                self.start_time = 0.0

    
    def _generate_wind_disturbance(self):
        N = int(self.duration / self.dt) + 2
        white_noise = self.rng.normal(0, self.sigma_noise, N)
        ar_output = lfilter([1], np.concatenate(([1], -np.array(self.ar_coeffs))), white_noise)

        def _von_karman_spectrum(n, f_min, sampling_rate):
            freqs = np.fft.fftfreq(n, d=1/sampling_rate)
            freqs = np.fft.fftshift(freqs)

            spectrum = np.zeros(n)
            for i, f in enumerate(freqs):
                if np.abs(f) > f_min:
                    spectrum[i] = 1 / (np.abs(f)**(5/3))

            random_phases = np.exp(2j * np.pi * self.rng.random(n))
            noise_freq = np.sqrt(spectrum) * random_phases
            noise_time = np.fft.ifft(noise_freq)
            return np.real(noise_time)

        high_freq_turbulence = _von_karman_spectrum(N, self.f_min, self.sampling_rate)
        total_wind_speed = ar_output + high_freq_turbulence
        return 0.05*0.191*total_wind_speed*np.abs(total_wind_speed)*0.8
    
    def generate_torque_disturbance(self, t):
        if self.model == "synthetic":
            idx = int(np.clip(np.floor(t / self.dt), 0, len(self.wind_torque) - 1))
            return self.wind_torque[idx]
        else:
            return float(np.interp(self.start_time + t, self.time, self.torque))

    def altitude(self, t):
        """Altitude (m) at time t, NaN if the run has no ascent."""
        if self.ascent is None:
            return np.nan
        return float(ascent_altitude(t, self.ascent))
