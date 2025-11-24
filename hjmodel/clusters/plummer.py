"""
An example density profile: a time-dependent Plummer profile for 47 Tuc
"""

import numpy as np

from hjmodel.clusters import DensityProfile
from hjmodel.config import AU_PER_PSC, G

__all__ = ["Plummer"]

_N0: float = 2.0e6
_R0: float = 1.91
_A: float = 6.991e-4
_M_AVG: float = 0.8
_M0: float = 1.64e6
_M1: float = 0.9e6
_T_AGE: float = 12000.0


_A_OVER_RH = 0.766


class Plummer(DensityProfile):
    def __init__(
        self,
        N0: float = _N0,
        R0: float = _R0,
        A: float = _A,
        M_avg: float = _M_AVG,
        M0: float = _M0,
        M1: float = _M1,
        T_AGE: float = _T_AGE,
    ):
        self.N0 = N0
        self.R0 = R0
        self.A = A
        self.M_avg = M_avg
        self.M0 = M0
        self.M1 = M1
        self.T_AGE = T_AGE
        self.M_fixed = N0 * M_avg

    def rh(self, t: float) -> float:
        return (self.R0**1.5 + self.A * t) ** (2 / 3)

    def a(self, t: float) -> float:
        return _A_OVER_RH * self.rh(t)

    def M_t(self, t: float) -> float:
        return self.M0 + (self.M1 - self.M0) * (t / self.T_AGE)

    def get_number_density(self, r: float, t: float) -> float:
        a_t = self.a(t)
        rho = (3 * self.M_fixed) / (4 * np.pi * a_t**3) * (1 + (r / a_t) ** 2) ** -2.5
        return rho / self.M_avg / 1e6

    def get_isotropic_velocity_dispersion(self, r: float, t: float) -> float:
        r_scaled = r * AU_PER_PSC
        a_scaled = self.a(t) * AU_PER_PSC
        return np.sqrt(G * self.M_t(t) / (6 * np.sqrt(r_scaled**2 + a_scaled**2)))

    def get_radius(self, lagrange: float, t: float) -> float:
        a_t = self.a(t)
        r_scaled = (lagrange ** (2 / 3)) / (1 - lagrange ** (2 / 3))
        r_scaled = r_scaled**0.5
        return r_scaled * a_t

    def get_mass_fraction_within_radius(self, r: float, t: float) -> float:
        a_t = self.a(t)
        r_scaled = r / a_t
        return (r_scaled**3) / ((1 + r_scaled**2) ** 1.5)
