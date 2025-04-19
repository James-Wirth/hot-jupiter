import numpy as np
from scipy.optimize import newton

from cluster import Cluster
from hjmodel.config import G, AU_PER_PSC, NUM_CPUS

class Plummer(Cluster):
    """
    Plummer class
    Calculates density and velocity dispersion profiles of globular clusters
    using a physically motivated time-dependent half-mass radius:
        r_h(t) = (R0^(3/2) + A * t)^(2/3)
    """

    def __init__(self, N0: float = 2E6, R0: float = 1.91, A: float = 6.991e-4, r_max: float = 100):
        super().__init__()
        self.N0 = N0
        self.R0 = R0
        self.A = A
        self.r_max = r_max

        self.M_avg = 0.8
        self.M = self.M_avg * self.N0

    def rh(self, t: float) -> float:
        return (self.R0 ** 1.5 + self.A * t) ** (2 / 3)

    def a(self, t: float) -> float:
        return 0.766 * self.rh(t)

    def M_time_dependent(self, t: float, M0: float = 1.64E6, M1: float = 0.9E6) -> float:
        return M0 + (M1-M0)*(t/12000)

    def particle_mass_density(self, r: float, t: float) -> float:
        """
        Returns the mass density profile corresponding to the fixed stellar population
        (not the time-dependent cluster mass)
        """
        a_t = self.a(t)
        return (3 * self.N0 * self.M_avg) / (4 * np.pi * a_t ** 3) * (1 + (r / a_t) ** 2) ** (-2.5)

    def number_density(self, r: float, t: float) -> float:
        """
        This returns the number density of the fixed particle population,
        based on a fixed average stellar mass (not inferred from M(t))
        """
        return (self.particle_mass_density(r, t) / self.M_avg) / 1e6  # per 10^6 pc³

    def isotropic_velocity_dispersion(self, r: float, t: float) -> float:
        a_t = self.a(t)
        r_scaled = r * AU_PER_PSC
        a_scaled = a_t * AU_PER_PSC
        return np.sqrt(G * self.M_time_dependent(t) / (6 * np.sqrt(r_scaled ** 2 + a_scaled ** 2)))

    def env_vars(self, r: float, t: float) -> dict[str, float]:
        return {
            'n_tot': self.number_density(r, t),
            'sigma_v': self.isotropic_velocity_dispersion(r, t)
        }

    def cdf(self, r: float, t: float) -> float:
        r_scaled = r / self.a(t)
        return (r_scaled ** 3) / (1 + r_scaled ** 2) ** 1.5


    def get_lagrange_distribution(self, n_samples: int, t: float, seed=None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        cdf_rt = self.cdf(self.r_max, t)
        bin_edges = np.linspace(0, cdf_rt, n_samples + 1)
        return np.array([
            np.random.uniform(bin_edges[i], bin_edges[i + 1])
            for i in range(n_samples)
        ])

    def get_radius(self, lagrange: float, t: float) -> float:
        return newton(lambda r: self.cdf(r, t) - lagrange, self.rh(t))
