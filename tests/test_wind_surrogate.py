import numpy as np
import pytest
from scipy.signal import lfilter, welch
from scipy.stats import kurtosis

from sim_tools.wind_surrogate import (
    ISA_RHO0,
    ascent_altitude,
    gust_rms,
    iaaft,
    isa_density,
    steady_torque,
    surrogate_torque,
    target_magnitude,
)

FS = 50.0


def gusty_signal(seconds: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Student-t noise through a lightly damped resonance near 2.5 Hz."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    shocks = rng.standard_t(df=3, size=n)
    r, theta = 0.97, 2 * np.pi * 2.5 / FS
    signal = lfilter([1.0], [1.0, -2 * r * np.cos(theta), r * r], shocks)
    return np.arange(n) / FS, signal


def log_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, p = welch(x, fs=FS, nperseg=1024)
    return f[1:], np.log10(p[1:])


def test_iaaft_keeps_the_target_amplitudes_exactly():
    _, source = gusty_signal(200)
    magnitude = target_magnitude(source, FS, source.size)
    out = iaaft(source, magnitude, np.random.default_rng(0))
    np.testing.assert_array_equal(np.sort(out), np.sort(source))


def test_iaaft_reproduces_the_power_spectrum():
    _, source = gusty_signal(600)
    out = iaaft(source, target_magnitude(source, FS, source.size), np.random.default_rng(0))

    f, want = log_psd(source)
    _, got = log_psd(out)
    band = (f > 0.1) & (f < 20.0)
    assert np.median(np.abs(got[band] - want[band])) < 0.1
    assert f[np.argmax(got)] == pytest.approx(f[np.argmax(want)], abs=0.2)


def test_iaaft_output_is_a_new_history():
    _, source = gusty_signal(200)
    out = iaaft(source, target_magnitude(source, FS, source.size), np.random.default_rng(0))
    assert abs(np.corrcoef(source, out)[0, 1]) < 0.2


def test_surrogate_keeps_the_heavy_tails():
    # Run as long as the source so the whole source is used
    time, source = gusty_signal(600)
    _, out = surrogate_torque(time, source, (0.0, 600.0), 600.0, np.random.default_rng(3))
    assert kurtosis(source) > 1.0
    assert kurtosis(out) == pytest.approx(kurtosis(source), rel=0.05)


@pytest.mark.parametrize("duration", [0.2, 10.0, 2000.0])
def test_surrogate_length_follows_the_run_not_the_source(duration):
    time, source = gusty_signal(1000)
    t, out = surrogate_torque(time, source, (0.0, 1000.0), duration, np.random.default_rng(0))
    needed = int(np.ceil(duration * FS)) + 2  # covers RK4's step past the end
    assert out.size == t.size
    assert needed <= out.size <= max(needed + 8, 1.05 * needed)
    assert t[0] == 0.0
    np.testing.assert_allclose(np.diff(t), 1.0 / FS)


def test_surrogate_draws_only_from_the_source_range():
    # Huge values outside the range would show up in the output
    time, source = gusty_signal(1000)
    marked = source.copy()
    outside = (time < 300.0) | (time > 700.0)
    marked[outside] = 1e6
    for seed in range(5):
        _, out = surrogate_torque(time, marked, (300.0, 700.0), 60.0, np.random.default_rng(seed))
        assert np.abs(out).max() < 1e3


def test_surrogate_has_no_steady_torque():
    time, source = gusty_signal(1000)
    _, out = surrogate_torque(time, source + 5.0, (0.0, 1000.0), 600.0, np.random.default_rng(0))
    assert abs(out.mean()) < 0.01 * out.std()


def test_isa_density_matches_the_standard_atmosphere():
    # US Standard Atmosphere 1976
    assert isa_density(0.0) == pytest.approx(1.2250, rel=1e-4)
    assert isa_density(11000.0) == pytest.approx(0.36392, rel=1e-4)
    assert isa_density(20000.0) == pytest.approx(0.08803, rel=1e-3)
    assert isa_density(30000.0) == pytest.approx(0.01801, rel=1e-3)


@pytest.mark.parametrize("boundary", [11000.0, 20000.0])
def test_isa_density_is_continuous_across_the_layers(boundary):
    assert isa_density(boundary - 1e-6) == pytest.approx(isa_density(boundary + 1e-6), rel=1e-9)


def test_steady_torque_follows_the_density_along_the_ascent():
    wind = {"steady_torque_per_density": 2e-3, "start_altitude_m": 100.0, "ascent_rate_m_s": 4.0}
    t = np.array([0.0, 1000.0, 5000.0])
    np.testing.assert_allclose(ascent_altitude(t, wind), [100.0, 4100.0, 20100.0])
    np.testing.assert_allclose(steady_torque(t, wind), 2e-3 * isa_density([100.0, 4100.0, 20100.0]))


def test_normalized_surrogate_has_unit_intensity_and_keeps_the_tails():
    # Gust level ramps tenfold across the source
    time, source = gusty_signal(1200, seed=1)
    ramped = source * np.linspace(1.0, 10.0, source.size)
    _, out = surrogate_torque(time, ramped, (0.0, 1200.0), 1200.0, np.random.default_rng(0), normalize=True)
    assert np.std(out) == pytest.approx(1.0, rel=1e-9)
    half = out.size // 2
    assert np.std(out[:half]) == pytest.approx(np.std(out[half:]), rel=0.2)
    assert kurtosis(out) == pytest.approx(kurtosis(source), rel=0.25)


def test_gust_rms_follows_a_power_of_the_density():
    wind = {"start_altitude_m": 0.0, "ascent_rate_m_s": 5.0, "gust_rms_sea_level": 0.05,
            "gust_density_exponent": 1.0}
    t = np.array([0.0, 2200.0, 4000.0])
    rho = isa_density(ascent_altitude(t, wind))
    np.testing.assert_allclose(gust_rms(t, wind), 0.05 * rho / ISA_RHO0)
    np.testing.assert_allclose(gust_rms(t, {**wind, "gust_density_exponent": 0.5}), 0.05 * np.sqrt(rho / ISA_RHO0))
    np.testing.assert_allclose(gust_rms(t, {**wind, "gust_density_exponent": 0.0}), 0.05)


def test_target_spectrum_vanishes_at_dc():
    _, source = gusty_signal(400)
    magnitude = target_magnitude(source, FS, 50_000)  # longer than one Welch segment
    assert magnitude[0] == 0.0
    assert np.all(np.diff(magnitude[:5]) > 0)  # rising as f below the Welch range
