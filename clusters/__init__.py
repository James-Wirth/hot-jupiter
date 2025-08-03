import numpy as np
from clusters.profiles import DensityProfile


class Cluster:
    """
    Wraps cluster methods for a given DensityProfile instance
    """

    def __init__(self, profile: DensityProfile, r_max: float = 100):
        self.profile = profile
        self.r_max = r_max

    def get_number_density(self, r: float, t: float):
        return self.profile.get_number_density(r, t)

    def get_isotropic_velocity_dispersion(self, r: float, t: float):
        return self.profile.get_isotropic_velocity_dispersion(r, t)

    def get_mass_fraction_within_radius(self, r: float, t: float):
        return self.profile.get_mass_fraction_within_radius(r, t)

    def get_radius(self, lagrange: float, t: float):
        if not (0 < lagrange < 1):
            raise ValueError(
                "Lagrange mass fraction must be between 0 and 1 (exclusive)"
            )
        return self.profile.get_radius(lagrange, t)

    def get_local_environment(self, r: float, t: float):
        return {
            "n_tot": self.get_number_density(r, t),
            "sigma_v": self.get_isotropic_velocity_dispersion(r, t),
        }

    def get_lagrange_distribution(self, n_samples: int, t: float, seed=None):
        if seed is not None:
            np.random.seed(seed)

        cdf_max = self.profile.get_mass_fraction_within_radius(self.r_max, t)
        if not (0 < cdf_max <= 1):
            raise ValueError(
                f"Invalid CDF value {cdf_max:.3f} returned for r_max={self.r_max} at time t={t}"
            )

        bins = np.linspace(0, cdf_max, n_samples + 1)
        return np.array(
            [np.random.uniform(bins[i], bins[i + 1]) for i in range(n_samples)]
        )
