"""Wind torque histories generated from flight data.

IAAFT surrogates (Schreiber & Schmitz, Phys. Rev. Lett. 77, 1996) of the yaw
torque logged on a passive flight. They keep the power spectrum and the heavy
tailed amplitude distribution of the flight torque with new random phases.
The steady torque and the gust level both scale with air density along the
ascent, fitted to the May 27 2026 flight.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import next_fast_len
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

RESOURCES = Path(__file__).resolve().parents[2] / "config" / "ressources"

IAAFT_ITERATIONS = 100
MIN_SOURCE_S = 300.0  # shortest flight window used as a source
WELCH_SEGMENT_S = 160.0  # resolves the 0.2 Hz peak of the flight spectrum
ENVELOPE_S = 120.0  # running RMS window used to normalize the source

# US Standard Atmosphere 1976, the three layers below 32 km.
_R_AIR = 287.053  # J/(kg K)
_G0 = 9.80665  # m/s^2
_T0, _P0 = 288.15, 101325.0
_LAPSE_TROPOSPHERE = 0.0065  # K/m, 0 to 11 km
_T11 = _T0 - _LAPSE_TROPOSPHERE * 11000.0  # isothermal from 11 to 20 km
_LAPSE_STRATOSPHERE = -0.001  # K/m, 20 to 32 km
_P11 = _P0 * (_T11 / _T0) ** (_G0 / (_LAPSE_TROPOSPHERE * _R_AIR))
_P20 = _P11 * np.exp(-_G0 * 9000.0 / (_R_AIR * _T11))


def log_path(name: str) -> Path:
    return RESOURCES / f"{name}_yaw_acceleration.csv.gz"


@lru_cache(maxsize=4)
def load_yaw_acceleration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Log time (s) and yaw acceleration (rad/s^2)."""
    df = pd.read_csv(path, usecols=["timestamp", "xyz_derivative[2]"])
    time = df["timestamp"].to_numpy() / 1e6
    acceleration = df["xyz_derivative[2]"].to_numpy()
    time.flags.writeable = False
    acceleration.flags.writeable = False
    return time, acceleration


def isa_density(altitude_m):
    """Air density (kg/m^3) of the standard atmosphere, 0 to 32 km."""
    h = np.asarray(altitude_m, dtype=float)
    t_trop = _T0 - _LAPSE_TROPOSPHERE * h
    t_strat = _T11 - _LAPSE_STRATOSPHERE * (h - 20000.0)
    temperature = np.where(h < 11000.0, t_trop, np.where(h < 20000.0, _T11, t_strat))
    pressure = np.where(
        h < 11000.0,
        _P0 * (t_trop / _T0) ** (_G0 / (_LAPSE_TROPOSPHERE * _R_AIR)),
        np.where(
            h < 20000.0,
            _P11 * np.exp(-_G0 * (h - 11000.0) / (_R_AIR * _T11)),
            _P20 * (t_strat / _T11) ** (_G0 / (_LAPSE_STRATOSPHERE * _R_AIR)),
        ),
    )
    return pressure / (_R_AIR * temperature)


ISA_RHO0 = float(isa_density(0.0))


def ascent_altitude(t, wind: dict):
    """Altitude (m) at time t for a constant ascent rate."""
    return wind["start_altitude_m"] + wind["ascent_rate_m_s"] * np.asarray(t, dtype=float)


def steady_torque(t, wind: dict):
    """Steady wind torque (N m) at time t, proportional to air density."""
    return wind["steady_torque_per_density"] * isa_density(ascent_altitude(t, wind))


def gust_rms(t, wind: dict):
    """RMS gust torque (N m) at time t, gust_rms_sea_level * (rho / rho_0)**k."""
    ratio = isa_density(ascent_altitude(t, wind)) / ISA_RHO0
    return wind["gust_rms_sea_level"] * ratio ** wind["gust_density_exponent"]


def local_rms(x: np.ndarray, fs: float, window_s: float = ENVELOPE_S) -> np.ndarray:
    """Centred running RMS of x over window_s seconds."""
    size = max(1, min(x.size, int(round(window_s * fs))))
    return np.sqrt(uniform_filter1d(x * x, size=size, mode="nearest"))


def uniform_segment(
    time: np.ndarray, values: np.ndarray, t0: float, t1: float
) -> tuple[np.ndarray, float]:
    # The log is nominally 50 Hz but has jitter and gaps up to 0.66 s
    dt = float(np.median(np.diff(time)))
    t0 = max(t0, float(time[0]))
    t1 = min(t1, float(time[-1]))
    grid = np.arange(t0, t1, dt)
    return np.interp(grid, time, values), 1.0 / dt


def target_magnitude(source: np.ndarray, fs: float, n: int) -> np.ndarray:
    """Fourier magnitude of an n sample series with the source's Welch spectrum."""
    nperseg = min(source.size, max(8, int(WELCH_SEGMENT_S * fs)))
    f_welch, psd = welch(source, fs=fs, nperseg=nperseg)
    f_welch, psd = f_welch[1:], psd[1:]

    f_out = np.fft.rfftfreq(n, d=1.0 / fs)
    shape = np.interp(f_out, f_welch, psd)
    # f^2 roll-off at low frequency, the yaw rate is bounded so no power at DC
    below = f_out < f_welch[0]
    shape[below] = psd[0] * (f_out[below] / f_welch[0]) ** 2
    return np.sqrt(shape)


def iaaft(
    amplitudes: np.ndarray, magnitude: np.ndarray, rng, n_iter: int = IAAFT_ITERATIONS
) -> np.ndarray:
    """Series with exactly these amplitudes and close to this Fourier magnitude."""
    sorted_amplitudes = np.sort(amplitudes)
    n = sorted_amplitudes.size
    series = rng.permutation(sorted_amplitudes)
    order = np.argsort(series)
    for _ in range(n_iter):
        phases = np.angle(np.fft.rfft(series))
        smooth = np.fft.irfft(magnitude * np.exp(1j * phases), n)
        new_order = np.argsort(smooth)
        series = np.empty(n)
        series[new_order] = sorted_amplitudes
        if np.array_equal(new_order, order):
            break
        order = new_order
    return series


def surrogate_torque(
    time: np.ndarray,
    torque: np.ndarray,
    source_range: tuple[float, float],
    duration: float,
    rng,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """New torque history of `duration` seconds from a random window of the log.

    With normalize the source is divided by its running RMS and the output has
    unit standard deviation.
    """
    segment, fs = uniform_segment(time, torque, *source_range)
    if segment.size < 2:
        raise ValueError(
            f"wind source_range {list(source_range)} does not overlap the flight log "
            f"({time[0]:.0f} to {time[-1]:.0f} s)"
        )
    window = min(segment.size, max(int(round(max(duration, MIN_SOURCE_S) * fs)), 2))
    # rng may be the np.random module, which has no integers()
    first = int(rng.random() * (segment.size - window + 1))
    source = segment[first : first + window]
    source = source - source.mean()
    if normalize:
        source = source / local_rms(source, fs)
        source = source - source.mean()

    # Two extra samples for the last RK4 stage, padded to a fast FFT length
    n = next_fast_len(int(np.ceil(round(duration * fs, 6))) + 2, real=True)
    # Quantiles at n evenly spaced probabilities, much faster than np.quantile
    ranks = (np.arange(n) + 0.5) / n * (source.size - 1)
    amplitudes = np.interp(ranks, np.arange(source.size), np.sort(source))
    series = iaaft(amplitudes, target_magnitude(source, fs, n), rng)
    if normalize:
        series = series / series.std()
    return np.arange(n) / fs, series
