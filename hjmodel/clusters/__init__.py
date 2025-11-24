from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np

__all__ = ["LocalEnvironment", "DensityProfile", "Cluster"]


LocalEnvironment = namedtuple("LocalEnvironment", ["n_tot", "sigma_v"])


class DensityProfile(ABC):
    @abstractmethod
    def get_number_density(self, r: float, t: float) -> float:
        pass

    @abstractmethod
    def get_isotropic_velocity_dispersion(self, r: float, t: float) -> float:
        pass

    @abstractmethod
    def get_radius(self, lagrange: float, t: float) -> float:
        pass

    @abstractmethod
    def get_mass_fraction_within_radius(self, r: float, t: float) -> float:
        pass


class Cluster:
    def __init__(self, profile: DensityProfile, r_max: float = 100):
        self.profile = profile
        self.r_max = r_max

    def get_number_density(self, r: float, t: float) -> float:
        return self.profile.get_number_density(r, t)

    def get_isotropic_velocity_dispersion(self, r: float, t: float) -> float:
        return self.profile.get_isotropic_velocity_dispersion(r, t)

    def get_mass_fraction_within_radius(self, r: float, t: float) -> float:
        return self.profile.get_mass_fraction_within_radius(r, t)

    def get_radius(self, lagrange: float, t: float) -> float:
        if not (0 < lagrange < 1):
            raise ValueError(
                "Lagrange mass fraction must be between 0 and 1 (exclusive)"
            )
        return self.profile.get_radius(lagrange, t)

    def get_local_environment(self, r: float, t: float) -> LocalEnvironment:
        return LocalEnvironment(
            n_tot=self.get_number_density(r, t),
            sigma_v=self.get_isotropic_velocity_dispersion(r, t),
        )

    def get_lagrange_distribution(
        self, n_samples: int, t: float, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        cdf_max = self.profile.get_mass_fraction_within_radius(self.r_max, t)
        if not (0 < cdf_max <= 1):
            raise ValueError(
                f"Invalid CDF value {cdf_max:.3f} returned for r_max={self.r_max} at time t={t}"
            )

        bins = np.linspace(0, cdf_max, n_samples + 1)
        return rng.uniform(bins[:-1], bins[1:])
