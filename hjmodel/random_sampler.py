import math
import random
from typing import Dict, List
from joblib import Parallel, delayed
from scipy.optimize import fsolve

from hjmodel.config import *
from hjmodel.model_utils import get_perts_per_Myr
from clusters.cluster import Cluster

class PlanetarySystem:
    sys: Dict[str, float]
    lagrange: float

    def __init__(self, sys, lagrange):
        self.sys = sys
        self.lagrange = lagrange
        return


class PlanetarySystemList(list):
    def __init__(self, n_samples: int, cluster: Cluster):
        # call the parent list.__init__ with your generated list
        super().__init__( sample_planetary_systems(n_samples=n_samples, cluster=cluster) )


class EncounterSampler:
    def __init__(self, sigma_v: float, override_b_max: float = B_MAX,
                 rng: random.Random = None):
        self.sigma_v = sigma_v
        self.override_b_max = override_b_max
        self.rng = rng or random.Random()

    def sample_b(self) -> float:
        return math.sqrt(self.rng.random() * self.override_b_max**2)

    def sample_v_infty(self) -> float:
        y = self.rng.random()
        sigma_rel = self.sigma_v * math.sqrt(2)

        def cdf(x: float) -> float:
            return (math.erf(x / (math.sqrt(2) * sigma_rel)) -
                    (math.sqrt(2) * x) / (math.sqrt(math.pi)
                                          * sigma_rel) *
                    math.exp(-x**2 / (2 * sigma_rel**2)))

        ans, *_ = fsolve(lambda x: cdf(x) - y, math.sqrt(2) * self.sigma_v)
        return ans

    def sample_orientation(self) -> Dict[str, float]:
        return {
            "Omega": self.rng.random() * 2 * math.pi,
            "omega": self.rng.random() * 2 * math.pi,
            "inc": math.acos(1 - 2 * self.rng.random())
        }

    def sample_m3(self) -> float:
        y = self.rng.random()
        a = (1.8 / (4 * M_BR**0.6 - 3 * M_MIN**0.6 - M_BR**2.4 * M_MAX**-1.8))
        b = a * M_BR**2.4
        y_crit = (a / 0.6) * (M_BR**0.6 - M_MIN**0.6)

        if y <= y_crit:
            return ((0.6 * y) / a + M_MIN**0.6)**(1 / 0.6)

        return ((M_BR**-1.8) + (1.8 / b) * (y_crit - y))**(-1 / 1.8)

    def sample_encounter(self) -> Dict[str, float]:
        params = self.sample_orientation()
        params.update({
            "v_infty": self.sample_v_infty(),
            "b": self.sample_b(),
            "m3": self.sample_m3()
        })
        return params

    @staticmethod
    def get_waiting_time(env_vars: Dict[str, float]) -> float:
        perts_per_Myr = get_perts_per_Myr(*env_vars.values())
        return np.random.exponential(1.0 / perts_per_Myr)


def sample_e_init() -> float:
    e_val = -1.0
    F_max = 1 - math.exp(-E_INIT_MAX**2 /
                          (2 * E_INIT_RMS**2))
    while e_val < 0.05:
        e_val = math.sqrt(
            2 * E_INIT_RMS**2 *
            math.log(1 / (1 - random.random() * F_max))
        )
    return e_val

def sample_a_init() -> float:
    return 10**(random.random() * math.log10(30))

def sample_m1() -> float:
    y = random.random()
    return (M_MIN**0.6 * (1 - y) + y * M_BR**0.6)**(1 / 0.6)

def sample_m2() -> float:
    return 1e-3

def sample_planetary_systems(
    n_samples: int,
    cluster: Cluster,
    num_cpus: int = NUM_CPUS
) -> List[PlanetarySystem]:

    lagrange_radii = cluster.get_lagrange_distribution(n_samples=n_samples, t=0)
    e_list = Parallel(n_jobs=num_cpus)(
        delayed(sample_e_init)() for _ in range(n_samples)
    )
    a_list = Parallel(n_jobs=num_cpus)(
        delayed(sample_a_init)() for _ in range(n_samples)
    )
    m1_list = Parallel(n_jobs=num_cpus)(
        delayed(sample_m1)() for _ in range(n_samples)
    )
    m2_list = Parallel(n_jobs=num_cpus)(
        delayed(sample_m2)() for _ in range(n_samples)
    )

    return [
        PlanetarySystem(
            sys={"e_init": e, "a_init": a, "m1": m1, "m2": m2},
            lagrange=lagrange
        )
        for e, a, m1, m2, lagrange in
        zip(e_list, a_list, m1_list, m2_list, lagrange_radii)
    ]
