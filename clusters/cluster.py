import numpy as np
from abc import ABC, abstractmethod

class DensityProfile(ABC):
    @abstractmethod
    def get_number_density(self, r: float, t: float) -> float: pass

    @abstractmethod
    def get_isotropic_velocity_dispersion(self, r: float, t: float) -> float: pass

    @abstractmethod
    def get_mass_fraction_within_radius(self, r: float, t: float) -> float: pass



class Cluster:
    def __init__(self, profile: DensityProfile, r_max=100):
        self.profile = profile
        self.r_max = r_max

    def get_number_density(self, r, t): return self.profile.get_number_density(r, t)
    def get_isotropic_velocity_dispersion(self, r, t): return self.profile.get_isotropic_velocity_dispersion(r, t)

    def get_local_environment(self, r, t):
        return {
            'n_tot': self.get_number_density(r, t),
            'sigma_v': self.get_isotropic_velocity_dispersion(r, t)
        }

    def get_mass_fraction_within_radius(self, r, t):
        return self.profile.get_mass_fraction_within_radius(r, t)

    def get_radius(self, lagrange, t):
        from scipy.optimize import newton
        return newton(lambda r: self.get_mass_fraction_within_radius(r, t) - lagrange, x0=self.r_max / 2)

    def get_lagrange_distribution(self, n_samples, t, seed=None):
        if seed is not None:
            np.random.seed(seed)
        cdf_max = self.profile.get_mass_fraction_within_radius(self.r_max, t)
        bins = np.linspace(0, cdf_max, n_samples + 1)
        return np.array([np.random.uniform(bins[i], bins[i + 1]) for i in range(n_samples)])

