from __future__ import annotations

import math

import numpy as np

from hj.config import (
    A_BR,
    A_MAX,
    A_MIN,
    E_INIT_MAX,
    E_INIT_RMS,
    M_BR,
    M_MIN,
    s1,
    s2,
)

__all__ = ["sample_e_init", "sample_a_init", "sample_m1", "sample_m2"]

_E_TWO_SIGMA_SQ: float = 2.0 * E_INIT_RMS**2
_E_F_MAX: float = 1.0 - math.exp(-(E_INIT_MAX**2) / (2.0 * E_INIT_RMS**2))

_A_S1P1: float = s1 + 1.0
_A_S2P1: float = s2 + 1.0
_A_I1: float = (
    math.log(A_BR / A_MIN) if s1 == -1 else (A_BR**_A_S1P1 - A_MIN**_A_S1P1) / _A_S1P1
)
_A_I2: float = (
    A_BR ** (s1 - s2) * math.log(A_MAX / A_BR)
    if s2 == -1
    else A_BR ** (s1 - s2) * (A_MAX**_A_S2P1 - A_BR**_A_S2P1) / _A_S2P1
)
_A_PROB1: float = _A_I1 / (_A_I1 + _A_I2)

_M1_MIN_POW: float = M_MIN**0.6
_M1_RANGE: float = M_BR**0.6 - M_MIN**0.6


def sample_e_init(n: int, rng: np.random.Generator) -> np.ndarray:
    """Initial eccentricity — truncated Rayleigh on [0.05, E_INIT_MAX]."""
    out = np.empty(n, dtype=np.float64)
    filled = 0
    while filled < n:
        need = n - filled
        u = rng.random(int(need * 1.1) + 8)
        cand = np.sqrt(_E_TWO_SIGMA_SQ * np.log(1.0 / (1.0 - u * _E_F_MAX)))
        accepted = cand[cand >= 0.05]
        take = min(accepted.size, need)
        out[filled : filled + take] = accepted[:take]
        filled += take
    return out


def sample_a_init(n: int, rng: np.random.Generator) -> np.ndarray:
    """Initial semi-major axis — broken power law on [A_MIN, A_MAX] (Fernandes 2019)."""
    u = rng.random(n)
    v = rng.random(n)
    in_low = u < _A_PROB1

    if s1 == -1:
        low = A_MIN * (A_BR / A_MIN) ** v
    else:
        low = (A_MIN**_A_S1P1 + v * (A_BR**_A_S1P1 - A_MIN**_A_S1P1)) ** (1.0 / _A_S1P1)

    if s2 == -1:
        high = A_BR * (A_MAX / A_BR) ** v
    else:
        high = (A_BR**_A_S2P1 + v * (A_MAX**_A_S2P1 - A_BR**_A_S2P1)) ** (1.0 / _A_S2P1)

    return np.where(in_low, low, high)


def sample_m1(n: int, rng: np.random.Generator) -> np.ndarray:
    """Stellar mass — 47 Tuc IMF (Giersz & Heggie 2011)."""
    y = rng.random(n)
    return (_M1_MIN_POW + y * _M1_RANGE) ** (1.0 / 0.6)


def sample_m2(n: int, rng: np.random.Generator) -> np.ndarray:
    """Planet mass — fixed Jupiter mass (1e-3 M_sun)."""
    return np.full(n, 1.0e-3, dtype=np.float64)
