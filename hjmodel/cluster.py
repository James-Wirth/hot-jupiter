from hjmodel.config import *
from joblib import delayed, Parallel
from scipy import integrate
from scipy.optimize import fsolve

class Plummer:
    """
    Plummer class
    Calculates density and velocity dispersion profiles of globular clusters with given parameters

    Inputs
    ----------
    M0: float               Total mass of cluster                   M_solar
    rt:                     Tidal radius                            parsec
    rh:                     Half-mass radius                        parsec
    N:                      Total number of stars in cluster
    """

    def __init__(self, M0: float, rt: float, rh: float, N: float):
        self.M0 = M0
        self.rt = rt
        self.rh = rh
        self.N = N
        self.M_avg = M0/N
        self.a = self.rh * (1/(0.5**(2/3)) - 1)**(1/2)

    # in M_solar per pc^3
    def density(self, r: float) -> float:
        return ((3 * self.M0) / (4 * np.pi * self.a ** 3)) * (1 + r ** 2 / self.a ** 2) ** (-5 / 2)

    def number_density(self, r:float) -> float:
        return self.density(r)/self.M_avg

    # in au / yr
    def isotropic_velocity_dispersion(self, r: float) -> float:
        return np.sqrt(3) * np.sqrt(G * self.M0 / (6 * np.sqrt((r * AU_PER_PSC) ** 2 + (self.a * AU_PER_PSC) ** 2)))

    def mass_enclosed(self, r: float) -> float:
        mass_integrand = lambda _r: 4 * np.pi * _r ** 2 * self.density(_r)
        return integrate.quad(mass_integrand, 0, r)[0]

    def get_radial_distribution(self, n_samples:int) -> list:
        cdf = lambda r: (r / self.a) ** 3 / (1 + (r / self.a) ** 2) ** (3 / 2)
        inverse_cdf = lambda y: fsolve(lambda r: cdf(r) - y, self.rh)[0]
        return Parallel(n_jobs=NUM_CPUS)(delayed(inverse_cdf)(y)
                                         for y in np.linspace(0, 1, n_samples+1)[1:])
