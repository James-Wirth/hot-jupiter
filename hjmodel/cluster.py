import matplotlib
from matplotlib import pyplot as plt

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
        return np.sqrt(G * self.M0 / (6 * np.sqrt((r * AU_PER_PSC) ** 2 + (self.a * AU_PER_PSC) ** 2)))

    def mass_enclosed(self, r: float) -> float:
        mass_integrand = lambda _r: 4 * np.pi * _r ** 2 * self.density(_r)
        return integrate.quad(mass_integrand, 0, r)[0]

    def get_radial_distribution(self, n_samples:int) -> list:
        cdf = lambda r: (r / self.a) ** 3 / (1 + (r / self.a) ** 2) ** (3 / 2)
        inverse_cdf = lambda y: fsolve(lambda r: cdf(r) - y, self.rh)[0]
        return Parallel(n_jobs=NUM_CPUS)(delayed(inverse_cdf)(y)
                                         for y in np.linspace(0, 1, n_samples+1)[1:])


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

    # interpolated values
    def interp_M(self, t):
        return interp(self.M0[0], self.M0[1], t/self.total_time)

    def interp_rt(self, t):
        return interp(self.rt[0], self.rt[1], t/self.total_time)

    def interp_rh(self, t):
        return interp(self.rh[0], self.rh[1], t/self.total_time)

    def interp_N(self, t):
        return interp(self.N[0], self.N[1], t/self.total_time)

    def interp_a(self, t):
        return self.interp_rh(t) * (1/(0.5**(2/3)) - 1)**(1/2)

    # cluster methods
    def density(self, r: float, t: float) -> float:
        return ((3 * self.interp_M(t)) / (4 * np.pi * self.interp_a(t) ** 3)) * (1 + r ** 2 / self.interp_a(t) ** 2) ** (-5 / 2)

    def number_density(self, r:float, t: float) -> float:
        M_avg = self.interp_M(t) / self.interp_N(t)
        return self.density(r, t)/M_avg

    def isotropic_velocity_dispersion(self, r: float, t: float) -> float:
        return np.sqrt(G * self.interp_M(t) / (6 * np.sqrt((r * AU_PER_PSC) ** 2 + (self.interp_a(t) * AU_PER_PSC) ** 2)))

    def mass_enclosed(self, r: float, t: float) -> float:
        mass_integrand = lambda _r: 4 * np.pi * _r ** 2 * self.density(_r, t)
        return integrate.quad(mass_integrand, 0, r)[0]

    # at time zero
    def get_radial_distribution(self, n_samples:int) -> list:
        cdf = lambda r: (r / self.interp_a(0)) ** 3 / (1 + (r / self.interp_a(0)) ** 2) ** (3 / 2)
        inverse_cdf = lambda y: fsolve(lambda r: cdf(r) - y, self.interp_rh(0))[0]
        return Parallel(n_jobs=NUM_CPUS)(delayed(inverse_cdf)(y)
                                         for y in np.linspace(0, 1, n_samples+1)[1:])

if __name__ == '__main__':
    plummer = DynamicPlummer(M0=(1.64E6, 0.9E6),
                             rt=(86, 70),
                             rh=(1.91, 4.96),
                             N=(2E6, 1.85E6),
                             total_time=12000)
    r_values = np.geomspace(0.001, 86, 100)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [10, 1]})
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=12)

    for t_value in range(12):
        y = [plummer.isotropic_velocity_dispersion(r, t_value*1000)/0.211 for r in r_values]
        ax1.plot(r_values, y, label=t_value, color=cmap(norm(t_value)))
    ax1.set_xscale('symlog')
    # ax1.set_yscale('symlog')
    ax1.set_xlabel('$r / \\mathrm{pc}$')
    ax1.set_ylabel('$\\sigma / \\mathrm{km} \\ \\mathrm{s}^{-1}$')
    ax1.set_title('Plummer velocity dispersion')

    cb1 = matplotlib.colorbar.ColorbarBase(ax=ax2, cmap=cmap, norm=norm, orientation='vertical')
    ax2.set_ylabel('Time / Gyr')

    plt.show()