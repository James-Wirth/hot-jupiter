"""
An example density profile: a time-dependent Plummer profile for 47 Tuc
"""

import numpy as np

from clusters import DensityProfile
from hjmodel.config import G, AU_PER_PSC


class Plummer(DensityProfile):
    def __init__(
        self,
        N0: float,
        R0: float,
        A: float,
        M_avg: float = 0.8,
        M0: float = 1.64e6,
        M1: float = 0.9e6,
        T_AGE: float = 12000.0,
    ):
        self.N0 = N0
        self.R0 = R0
        self.A = A
        self.M_avg = M_avg
        self.M0 = M0
        self.M1 = M1
        self.T_AGE = T_AGE
        self.M_fixed = N0 * M_avg

    def rh(self, t: float):
        return (self.R0**1.5 + self.A * t) ** (2 / 3)

    def a(self, t: float):
        return 0.766 * self.rh(t)

    def M_t(self, t: float):
        return self.M0 + (self.M1 - self.M0) * (t / self.T_AGE)

    def get_number_density(self, r: float, t: float):
        a_t = self.a(t)
        rho = (3 * self.M_fixed) / (4 * np.pi * a_t**3) * (1 + (r / a_t) ** 2) ** -2.5
        return rho / self.M_avg / 1e6

    def get_isotropic_velocity_dispersion(self, r: float, t: float):
        r_scaled = r * AU_PER_PSC
        a_scaled = self.a(t) * AU_PER_PSC
        return np.sqrt(G * self.M_t(t) / (6 * np.sqrt(r_scaled**2 + a_scaled**2)))

    def get_radius(self, lagrange: float, t: float):
        a_t = self.a(t)
        r_scaled = (lagrange ** (2 / 3)) / (1 - lagrange ** (2 / 3))
        r_scaled = r_scaled**0.5
        return r_scaled * a_t

    def get_mass_fraction_within_radius(self, r: float, t: float):
        a_t = self.a(t)
        r_scaled = r / a_t
        return (r_scaled**3) / ((1 + r_scaled**2) ** 1.5)
