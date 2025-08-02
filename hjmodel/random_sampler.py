import math
import logging
from typing import Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import fsolve

from dataclasses import dataclass

from clusters.cluster import Cluster
from hjmodel.config import (
    E_INIT_MAX,
    E_INIT_RMS,
    A_MIN,
    A_BR,
    A_MAX,
    s1,
    s2,
    M_MIN,
    M_BR,
    M_MAX,
    B_MAX,
    NUM_CPUS,
)
from hjmodel.model_utils import get_perts_per_Myr

logger = logging.getLogger(__name__)


@dataclass
class PlanetarySystem:
    sys: Dict[str, float]
    lagrange: float
    seed: int


class EncounterSampler:

    def __init__(
        self,
        sigma_v: float,
        override_b_max: float = B_MAX,
        rng: Optional[np.random.Generator] = None
    ):
        self.sigma_v = sigma_v
        self.override_b_max = override_b_max
        self.rng = rng or np.random.default_rng()

    def sample_b(self) -> float:
        return math.sqrt(self.rng.random() * self.override_b_max**2)

    def sample_v_infty(self) -> float:
        y = self.rng.random()
        sigma_rel = self.sigma_v * math.sqrt(2)

        def cdf(x: float) -> float:
            return (
                    math.erf(x / (math.sqrt(2) * sigma_rel))
                    - (math.sqrt(2) * x) / (math.sqrt(math.pi) * sigma_rel)
                    * math.exp(-x ** 2 / (2 * sigma_rel ** 2))
            )

        initial = math.sqrt(2) * self.sigma_v
        ans, infodict, ier, mesg = fsolve(lambda x: cdf(x) - y, initial, full_output=True)
        result = ans[0] if isinstance(ans, (list, np.ndarray)) else ans
        if ier != 1 or not np.isfinite(result) or result < 0:
            logger.warning(
                "v_infty root finding did not converge (ier=%s)! Fallback to sigma_v=%s; message=%s",
                ier,
                self.sigma_v,
                mesg,
            )
            return self.sigma_v
        return result

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
        params.update(
            {
                "v_infty": self.sample_v_infty(),
                "b": self.sample_b(),
                "m3": self.sample_m3(),
            }
        )
        return params

    def get_waiting_time(self, env_vars: Dict[str, float]) -> float:
        perts_per_Myr = get_perts_per_Myr(*env_vars.values())
        return self.rng.exponential(1.0 / perts_per_Myr)


def sample_e_init(rng: np.random.Generator) -> float:
    e_val = -1.0
    F_max = 1 - math.exp(-E_INIT_MAX**2 / (2 * E_INIT_RMS**2))
    while e_val < 0.05:
        u = rng.random() * F_max
        e_val = math.sqrt(2 * E_INIT_RMS**2 * math.log(1 / (1 - u)))
    return e_val


def sample_a_init(rng: np.random.Generator) -> float:
    def sample_segment(u: float, a1: float, a2: float, alpha: float) -> float:
        if alpha == -1:
            return a1 * (a2 / a1) ** u
        p = alpha + 1
        return (a1 ** p + u * (a2 ** p - a1 ** p)) ** (1 / p)

    def integral(a1: float, a2: float, alpha: float) -> float:
        if alpha == -1:
            return np.log(a2 / a1)
        return (a2 ** (alpha + 1) - a1 ** (alpha + 1)) / (alpha + 1)

    c2_over_c1 = A_BR ** (s1 - s2)
    I1 = integral(A_MIN, A_BR, s1)
    I2 = c2_over_c1 * integral(A_BR, A_MAX, s2)
    prob1 = I1 / (I1 + I2)

    u = rng.random()
    if u < prob1:
        return sample_segment(rng.random(), A_MIN, A_BR, s1)
    else:
        return sample_segment(rng.random(), A_BR, A_MAX, s2)


def sample_m1(rng: np.random.Generator) -> float:
    y = rng.random()
    return (M_MIN**0.6 * (1 - y) + y * M_BR**0.6) ** (1 / 0.6)


def sample_m2(_: Optional[np.random.Generator] = None) -> float:
    return 1e-3


def sample_planetary_systems(
    n_samples: int,
    cluster: Cluster,
    rng: np.random.Generator,
    num_cpus: int = NUM_CPUS
) -> List[PlanetarySystem]:

    lagrange_radii = cluster.get_lagrange_distribution(n_samples=n_samples, t=0)
    system_seeds = rng.integers(0, 2**32 - 1, size=n_samples)

    def _make_system(system_seed: int, lagrange: float) -> PlanetarySystem:
        system_rng = np.random.default_rng(system_seed)
        e_seed, a_seed, m1_seed, m2_seed = system_rng.integers(0, 2**32 - 1, size=4)

        e = sample_e_init(np.random.default_rng(e_seed))
        a = sample_a_init(np.random.default_rng(a_seed))
        m1 = sample_m1(np.random.default_rng(m1_seed))
        m2 = sample_m2(np.random.default_rng(m2_seed))

        return PlanetarySystem(
            sys={"e_init": e, "a_init": a, "m1": m1, "m2": m2},
            lagrange=lagrange,
            seed=system_seed
        )

    return Parallel(n_jobs=max(1, num_cpus))(
        delayed(_make_system)(seed, lagrange)
        for seed, lagrange in zip(system_seeds, lagrange_radii)
    )
