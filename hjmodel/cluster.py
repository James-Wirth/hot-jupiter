from scipy.interpolate import interp1d

from hjmodel.config import *
from joblib import delayed, Parallel
from scipy import integrate
from scipy.optimize import fsolve
from functools import lru_cache
from scipy.optimize import newton

from scipy.optimize import brentq
from scipy.interpolate import RegularGridInterpolator, interpolate

def interp(left: float, right: float, f: float) -> float:
    return left + (right - left) * f

class DynamicPlummer:
    """
    DynamicPlummer class
    Calculates density and velocity dispersion profiles of globular clusters with given parameters
    """

    def __init__(self, M0: (float, float), rt: (float, float), rh: (float, float), N: (float, float),
                 total_time: int, t_grid_size=500, lagrange_grid_size=500):
        self.M0 = M0
        self.rt = rt
        self.rh = rh
        self.N = N
        self.total_time = total_time
        self.a = lambda _rh: tuple(x * (1/(0.5**(2/3)) - 1)**(1/2) for x in _rh)

        """
        # Precompute grid
        self.t_grid = np.linspace(0, total_time, t_grid_size)
        self.lagrange_grid = np.linspace(0, 1, lagrange_grid_size)
        self.radius_grid = self._precompute_radius_grid()
        self.radius_interpolator = RegularGridInterpolator(
            (self.t_grid, self.lagrange_grid), self.radius_grid, bounds_error=False, fill_value=None
        )
        """

    @lru_cache(maxsize=None)
    def itrp(self, var, t):
        func = lambda left, right, f : left + (right - left) * f
        return func(var[0], var[1], t/self.total_time)

    # cluster methods
    def density(self, r: float, t: float) -> float:
        a_rh = self.itrp(self.a(self.rh), t)
        return ((3 * self.itrp(self.M0, t)) / (4 * np.pi * a_rh ** 3)) * (1 + r ** 2 / a_rh ** 2) ** (-5 / 2)

    # per 10^6 pc^3
    def number_density(self, r:float, t: float) -> float:
        M_avg = self.itrp(self.M0, t) / self.itrp(self.N, t)
        return (self.density(r, t)/M_avg) / 1E6

    def isotropic_velocity_dispersion(self, r: float, t: float) -> float:
        return np.sqrt(G * self.itrp(self.M0, t) / (6 * np.sqrt((r * AU_PER_PSC) ** 2 + (self.itrp(self.a(self.rh), t) * AU_PER_PSC) ** 2)))

    def env_vars(self, *args) -> dict[str, float]:
        return {'n_tot': self.number_density(*args), 'sigma_v': self.isotropic_velocity_dispersion(*args)}

    def mass_enclosed(self, r: float, t: float) -> float:
        mass_integrand = lambda _r: 4 * np.pi * _r ** 2 * self.density(_r, t)
        return integrate.quad(mass_integrand, 0, r)[0]

    def get_radial_distribution(self, n_samples:int, t: float) -> list:
        cdf = lambda r: (r / self.itrp(self.a(self.rh), t)) ** 3 / (1 + (r / self.itrp(self.a(self.rh), t)) ** 2) ** (3 / 2)
        inverse_cdf = lambda y: fsolve(lambda r: cdf(r) - y, self.itrp(self.rh, t))[0]
        y_cutoff = cdf(r=self.itrp(self.rt, t))
        return Parallel(n_jobs=NUM_CPUS)(delayed(inverse_cdf)(y)
                                         for y in np.linspace(0, y_cutoff, n_samples+1)[1:])

    def get_lagrange_distribution(self, n_samples: int, t: float) -> np.ndarray:
        cdf = lambda r: (r / self.itrp(self.a(self.rh), t)) ** 3 / (1 + (r / self.itrp(self.a(self.rh), t)) ** 2) ** (3 / 2)
        y_cutoff = cdf(r=self.itrp(self.rt, t))
        return np.linspace(0, y_cutoff, n_samples + 1)[1:]

    """
    def map_lagrange_to_radius_old(self, lagrange: float, t: float) -> float:
        cdf = lambda r: (r / self.itrp(self.a(self.rh), t)) ** 3 / (1 + (r / self.itrp(self.a(self.rh), t)) ** 2) ** (
                    3 / 2)
        inverse_cdf = lambda y: fsolve(lambda r: cdf(r) - y, self.itrp(self.rh, t))[0]
        return inverse_cdf(lagrange)
    """

    def map_lagrange_to_radius(self, lagrange: float, t: float) -> float:
        itrp_a_rh_t = self.itrp(self.a(self.rh), t)
        itrp_rh_t = self.itrp(self.rh, t)
        def cdf(r):
            r_scaled = r / itrp_a_rh_t
            return (r_scaled ** 3) / (1 + r_scaled ** 2) ** (3 / 2)
        def inverse_cdf(y):
            initial_guess = itrp_rh_t
            return newton(lambda r: cdf(r) - y, initial_guess)
        return inverse_cdf(lagrange)

    """
    def _precompute_radius_grid(self):
        # Vectorized precomputation
        radius_grid = np.zeros((len(self.t_grid), len(self.lagrange_grid)), dtype=np.float32)

        for i, t in enumerate(self.t_grid):
            itrp_a_rh_t = self.itrp(self.a(self.rh), t)
            itrp_rt_t = self.itrp(self.rt, t)

            radii = np.linspace(0, itrp_rt_t, 500000)
            cdf_values = (radii / itrp_a_rh_t) ** 3 / (1 + (radii / itrp_a_rh_t) ** 2) ** (3 / 2)

            # Precompute an interpolator for the current time step
            radius_interpolator = interpolate.interp1d(
                cdf_values, radii, kind='linear', bounds_error=False, fill_value=itrp_rt_t
            )

            # Apply the interpolator to the entire lagrange_grid
            radius_grid[i, :] = radius_interpolator(self.lagrange_grid)

        return radius_grid

    def map_lagrange_to_radius_precompute(self, lagrange, t):
        return self.radius_interpolator((t, lagrange))
    """
