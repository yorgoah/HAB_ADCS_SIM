"""Dispersed parameter values from uniform draws in (0, 1).

Every case uses the same draws, so run N of each case has the same parameters
apart from the case overrides.
"""

import numpy as np
from scipy.stats import norm, qmc, truncnorm

from .spec import Dispersion

METHODS = ("random", "lhs")


def unit_samples(n_samples: int, n_params: int, seed: int, method: str = "random") -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"unknown sampling method {method!r}, expected one of {METHODS}")
    if n_params == 0 or n_samples == 0:
        return np.empty((n_samples, n_params))
    if method == "lhs":
        return qmc.LatinHypercube(d=n_params, seed=seed).random(n_samples)
    return np.random.default_rng(seed).random((n_samples, n_params))


def _normal_value(nominal: float, u: float, d: Dispersion) -> float:
    sigma = d.sigma if d.sigma is not None else d.sigma_rel * abs(nominal)
    if sigma == 0:
        return nominal
    if d.minimum is None and d.maximum is None:
        return float(nominal + sigma * norm.ppf(u))
    # Truncated normal, clipping would put a whole tail of runs on the bound
    a = -np.inf if d.minimum is None else (d.minimum - nominal) / sigma
    b = np.inf if d.maximum is None else (d.maximum - nominal) / sigma
    return float(truncnorm.ppf(u, a, b, loc=nominal, scale=sigma))


def _uniform_value(nominal: float, u: float, d: Dispersion) -> float:
    if d.rel is not None:
        low, high = nominal * (1.0 - d.rel), nominal * (1.0 + d.rel)
    else:
        low, high = d.low, d.high
    if low > high:
        low, high = high, low
    value = low + u * (high - low)
    if d.minimum is not None:
        value = max(value, d.minimum)
    if d.maximum is not None:
        value = min(value, d.maximum)
    return float(value)


def disperse_value(nominal: float, u: float, d: Dispersion) -> float:
    if d.dist == "normal":
        return _normal_value(nominal, u, d)
    return _uniform_value(nominal, u, d)


def linked_values(nominal: float, value: float, linked_nominals: dict[str, float]) -> dict[str, float]:
    """Linked parameters scale by the same factor (e.g. Kt and Kb)."""
    if nominal == 0:
        delta = value - nominal
        return {path: nom + delta for path, nom in linked_nominals.items()}
    factor = value / nominal
    return {path: nom * factor for path, nom in linked_nominals.items()}
