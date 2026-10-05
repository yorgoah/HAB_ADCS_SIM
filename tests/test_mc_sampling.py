import numpy as np
import pytest

from hab_adcs_sim.monte_carlo.sampling import disperse_value, linked_values, unit_samples
from hab_adcs_sim.monte_carlo.spec import Dispersion


def normal(**kwargs) -> Dispersion:
    return Dispersion(path="p", dist="normal", **kwargs)


def uniform(**kwargs) -> Dispersion:
    return Dispersion(path="p", dist="uniform", **kwargs)


@pytest.mark.parametrize("method", ["random", "lhs"])
def test_same_seed_reproduces_the_same_draws(method):
    a = unit_samples(8, 3, seed=42, method=method)
    b = unit_samples(8, 3, seed=42, method=method)
    assert a.shape == (8, 3)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("method", ["random", "lhs"])
def test_draws_stay_inside_the_unit_interval(method):
    u = unit_samples(16, 4, seed=1, method=method)
    assert np.all((u > 0) & (u < 1))


def test_lhs_puts_one_point_in_every_stratum():
    # 10 samples cover each tenth of every column exactly once
    n = 10
    u = unit_samples(n, 2, seed=3, method="lhs")
    for column in u.T:
        assert sorted((column * n).astype(int)) == list(range(n))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown sampling method"):
        unit_samples(4, 1, seed=0, method="sobol")


def test_normal_median_draw_returns_the_nominal():
    assert disperse_value(0.015, 0.5, normal(sigma_rel=0.1)) == pytest.approx(0.015)


def test_normal_sigma_rel_scales_with_the_nominal():
    # u = 0.8413 is +1 sigma.
    value = disperse_value(0.015, 0.8413447460685429, normal(sigma_rel=0.1))
    assert value == pytest.approx(0.015 * 1.1, rel=1e-6)


def test_normal_respects_a_lower_bound_without_piling_up_on_it():
    d = normal(sigma_rel=0.5, minimum=0.005)
    far_tail = disperse_value(0.015, 1e-9, d)
    near_tail = disperse_value(0.015, 1e-3, d)
    assert far_tail >= 0.005
    assert near_tail >= 0.005
    assert far_tail != near_tail


def test_uniform_rel_spans_the_nominal_band():
    d = uniform(rel=0.5)
    assert disperse_value(0.002, 0.0, d) == pytest.approx(0.001)
    assert disperse_value(0.002, 1.0, d) == pytest.approx(0.003)


def test_uniform_low_high_is_absolute():
    d = uniform(low=2.0, high=4.0)
    assert disperse_value(99.0, 0.5, d) == pytest.approx(3.0)


def test_linked_parameters_follow_the_same_factor():
    linked = linked_values(nominal=0.0525, value=0.0525 * 1.05, linked_nominals={"kb": 0.0525})
    assert linked["kb"] == pytest.approx(0.0525 * 1.05)


def test_linked_parameters_with_a_zero_nominal_use_the_offset():
    linked = linked_values(nominal=0.0, value=0.2, linked_nominals={"other": 1.0})
    assert linked["other"] == pytest.approx(1.2)
