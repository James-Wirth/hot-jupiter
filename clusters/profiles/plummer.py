"""
An example density profile
"""

import numpy as np

from clusters import DensityProfile
from hjmodel.config import G, AU_PER_PSC

class Plummer(DensityProfile):
    def __init__(self, N0, R0, A, M_avg=0.8, M0=1.64e6, M1=0.9e6):
        self.N0 = N0
        self.R0 = R0
        self.A = A
        self.M_avg = M_avg
        self.M0 = M0
        self.M1 = M1
        self.M_fixed = N0 * M_avg

    def rh(self, t): return (self.R0**1.5 + self.A * t)**(2/3)
    def a(self, t): return 0.766 * self.rh(t)
    def M_t(self, t): return self.M0 + (self.M1 - self.M0) * (t / 12000)

    def get_number_density(self, r, t):
        a_t = self.a(t)
        rho = (3 * self.M_fixed) / (4 * np.pi * a_t**3) * (1 + (r / a_t)**2)**-2.5
        return rho / self.M_avg / 1e6

    def get_isotropic_velocity_dispersion(self, r, t):
        r_scaled = r * AU_PER_PSC
        a_scaled = self.a(t) * AU_PER_PSC
        return np.sqrt(G * self.M_t(t) / (6 * np.sqrt(r_scaled**2 + a_scaled**2)))

    def get_radius(self, lagrange, t):
        a_t = self.a(t)
        r_scaled = (lagrange ** (2 / 3)) / (1 - lagrange ** (2 / 3))
        r_scaled = r_scaled ** 0.5
        return r_scaled * a_t

    def get_mass_fraction_within_radius(self, r, t):
        a_t = self.a(t)
        r_scaled = r / a_t
        return (r_scaled ** 3) / ((1 + r_scaled ** 2) ** 1.5)


