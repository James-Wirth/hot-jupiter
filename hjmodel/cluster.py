from hjmodel.config import *
from joblib import delayed, Parallel
from scipy import integrate
from scipy.optimize import fsolve

def interp(left: float, right: float, f: float) -> float:
    return left + (right - left) * f

class DynamicPlummer:
    """
    DynamicPlummer class
    Calculates density and velocity dispersion profiles of globular clusters with given parameters
    """

    def __init__(self, M0: (float, float), rt: (float, float), rh: (float, float), N: (float, float),
                 total_time: int):
        self.M0 = M0
        self.rt = rt
        self.rh = rh
        self.N = N
        self.total_time = total_time
        self.a = lambda _rh: tuple(x * (1/(0.5**(2/3)) - 1)**(1/2) for x in _rh)

    def itrp(self, var, t):
        func = lambda left, right, f : left + (right - left) * f
        return func(var[0], var[1], t/self.total_time)

    # cluster methods
    def density(self, r: float, t: float) -> float:
        return ((3 * self.itrp(self.M0, t)) / (4 * np.pi * self.itrp(self.a(self.rh), t) ** 3)) * (1 + r ** 2 / self.itrp(self.a(self.rh), t) ** 2) ** (-5 / 2)

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
    
    # at time zero
    def get_radial_distribution(self, n_samples:int) -> list:
        cdf = lambda r: (r / self.itrp(self.a(self.rh), 0)) ** 3 / (1 + (r / self.itrp(self.a(self.rh), 0)) ** 2) ** (3 / 2)
        inverse_cdf = lambda y: fsolve(lambda r: cdf(r) - y, self.itrp(self.rh, 0))[0]
        y_cutoff = cdf(r=self.itrp(self.rt, 0))
        return Parallel(n_jobs=NUM_CPUS)(delayed(inverse_cdf)(y)
                                         for y in np.linspace(0, y_cutoff, n_samples+1)[1:])
