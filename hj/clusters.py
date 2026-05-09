from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

from hj.config import A_OVER_RH, AU_PER_PSC, G

__all__ = ["LocalEnvironment", "Cluster", "Plummer"]


class LocalEnvironment(NamedTuple):
    n_tot: float | np.ndarray
    sigma_v: float | np.ndarray


class Cluster(ABC):
    def __init__(self, r_max: float = 100.0):
        self.r_max = r_max

    @abstractmethod
    def radius(self, lagrange: float | np.ndarray, t: float) -> float | np.ndarray:
        """Radius r(L, t) enclosing Lagrangian mass fraction `lagrange` at time t."""

    @abstractmethod
    def local_environment(self, r: float | np.ndarray, t: float) -> LocalEnvironment:
        """Number density and isotropic velocity dispersion, (n_tot, sigma_v), at radius r and time t."""

    @abstractmethod
    def enclosed_mass_fraction(self, r: float, t: float) -> float:
        """Cumulative mass fraction M(<r, t) / M_tot within radius r."""

    def sample_lagrange(
        self, n_samples: int, t: float, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Stratified-uniform Lagrange in [0, M(<r_max, t) / M_tot]."""
        if rng is None:
            rng = np.random.default_rng()
        cdf_max = self.enclosed_mass_fraction(self.r_max, t)
        if not (0 < cdf_max <= 1):
            raise ValueError(
                f"Invalid CDF value {cdf_max:.3f} returned for r_max={self.r_max} at time t={t}"
            )
        bins = np.linspace(0.0, cdf_max, n_samples + 1)
        return rng.uniform(bins[:-1], bins[1:])


_PLUMMER_N0: float = 2.0e6
_PLUMMER_R0: float = 1.91
_PLUMMER_A_EXPANSION: float = 6.991e-4
_PLUMMER_M_AVG: float = 0.8
_PLUMMER_M0: float = 1.64e6
_PLUMMER_M1: float = 0.9e6
_PLUMMER_T_AGE: float = 12000.0


class Plummer(Cluster):
    """Time-dependent Plummer profile for 47 Tucanae."""

    def __init__(
        self,
        N0: float = _PLUMMER_N0,
        R0: float = _PLUMMER_R0,
        A: float = _PLUMMER_A_EXPANSION,
        M_avg: float = _PLUMMER_M_AVG,
        M0: float = _PLUMMER_M0,
        M1: float = _PLUMMER_M1,
        T_AGE: float = _PLUMMER_T_AGE,
        r_max: float = 100.0,
    ):
        super().__init__(r_max=r_max)
        self.N0 = N0
        self.R0 = R0
        self.A = A
        self.M_avg = M_avg
        self.M0 = M0
        self.M1 = M1
        self.T_AGE = T_AGE
        self.M_fixed = N0 * M_avg

    def _a_t(self, t: float) -> float:
        """Plummer scale a(t) = A_OVER_RH * (R0^1.5 + A t)^(2/3)."""
        rh = (self.R0**1.5 + self.A * t) ** (2.0 / 3.0)
        return A_OVER_RH * rh

    def _M_t(self, t: float) -> float:
        """Total cluster mass linearly interpolated from M0 to M1 over T_AGE."""
        return self.M0 + (self.M1 - self.M0) * (t / self.T_AGE)

    def radius(self, lagrange: float | np.ndarray, t: float) -> float | np.ndarray:
        a_t = self._a_t(t)
        u = lagrange ** (2.0 / 3.0)
        return np.sqrt(u / (1.0 - u)) * a_t

    def local_environment(self, r: float | np.ndarray, t: float) -> LocalEnvironment:
        a_t = self._a_t(t)
        m_avg_inv_e6 = 1.0 / (self.M_avg * 1.0e6)
        rho = (
            (3.0 * self.M_fixed)
            / (4.0 * np.pi * a_t**3)
            * (1.0 + (r / a_t) ** 2) ** -2.5
        )
        n_tot = rho * m_avg_inv_e6
        r_au = r * AU_PER_PSC
        a_au = a_t * AU_PER_PSC
        sigma_v = np.sqrt(G * self._M_t(t) / (6.0 * np.sqrt(r_au**2 + a_au**2)))
        return LocalEnvironment(n_tot=n_tot, sigma_v=sigma_v)

    def enclosed_mass_fraction(self, r: float, t: float) -> float:
        a_t = self._a_t(t)
        x = r / a_t
        return float((x**3) / ((1.0 + x**2) ** 1.5))

    def number_density(self, r: float | np.ndarray, t: float) -> float | np.ndarray:
        return self.local_environment(r, t).n_tot

    def velocity_dispersion(
        self, r: float | np.ndarray, t: float
    ) -> float | np.ndarray:
        return self.local_environment(r, t).sigma_v
