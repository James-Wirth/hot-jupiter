import numpy as np
from scipy import integrate
from scipy.optimize import fsolve, newton
from joblib import delayed, Parallel

from hjmodel.config import G, AU_PER_PSC, NUM_CPUS

class Plummer:
    """
    Plummer class
    Calculates density and velocity dispersion profiles of globular clusters
    using a physically motivated time-dependent half-mass radius:
        r_h(t) = (R0^(3/2) + A * t)^(2/3)
    """

    def __init__(self, N0: float, R0: float, A: float, rt: float):
        self.N0 = N0
        self.R0 = R0
        self.A = A
        self.rt = rt

        self.M_avg = 0.8
        self.M = self.M_avg * self.N0

    def rh(self, t: float) -> float:
        return (self.R0 ** 1.5 + self.A * t) ** (2 / 3)

    def a(self, t: float) -> float:
        return 0.766 * self.rh(t)

    def density(self, r: float, t: float) -> float:
        a_t = self.a(t)
        return (3 * self.M) / (4 * np.pi * a_t ** 3) * (1 + (r / a_t) ** 2) ** (-2.5)

    def number_density(self, r: float, t: float) -> float:
        return (self.density(r, t) / self.M_avg) / 1e6  # per 10^6 pc³

    def isotropic_velocity_dispersion(self, r: float, t: float) -> float:
        a_t = self.a(t)
        r_scaled = r * AU_PER_PSC
        a_scaled = a_t * AU_PER_PSC
        return np.sqrt(G * self.M / (6 * np.sqrt(r_scaled ** 2 + a_scaled ** 2)))

    def env_vars(self, r: float, t: float) -> dict[str, float]:
        return {
            'n_tot': self.number_density(r, t),
            'sigma_v': self.isotropic_velocity_dispersion(r, t)
        }

    def mass_enclosed(self, r: float, t: float) -> float:
        mass_integrand = lambda _r: 4 * np.pi * _r ** 2 * self.density(_r, t)
        return integrate.quad(mass_integrand, 0, r)[0]

    def cdf(self, r: float, t: float) -> float:
        r_scaled = r / self.a(t)
        return (r_scaled ** 3) / (1 + r_scaled ** 2) ** 1.5

    def get_radial_distribution(self, n_samples: int, t: float) -> list:
        cdf_rt = self.cdf(self.rt, t)

        def inverse_cdf(y):
            return fsolve(lambda r: self.cdf(r, t) - y, self.rh(t))[0]

        return Parallel(n_jobs=NUM_CPUS)(delayed(inverse_cdf)(y)
                                         for y in np.linspace(0, cdf_rt, n_samples + 1)[1:])

    def get_lagrange_distribution(self, n_samples: int, t: float, seed=None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        cdf_rt = self.cdf(self.rt, t)
        bin_edges = np.linspace(0, cdf_rt, n_samples + 1)
        return np.array([
            np.random.uniform(bin_edges[i], bin_edges[i + 1])
            for i in range(n_samples)
        ])

    def map_lagrange_to_radius(self, lagrange: float, t: float) -> float:
        return newton(lambda r: self.cdf(r, t) - lagrange, self.rh(t))
