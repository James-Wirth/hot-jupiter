import math
import logging

import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from joblib import Parallel, delayed, cpu_count
from scipy.optimize import fsolve

from clusters import Cluster
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
from hjmodel import model_utils as default_model_utils
from hjmodel.config import CIRCULARISATION_THRESHOLD_ECCENTRICITY, StopCode

logger = logging.getLogger(__name__)


def check_stopping_conditions(
    e: float,
    a: float,
    t: float,
    R_td: float,
    R_hj: float,
    R_wj: float,
    total_time: int
) -> Optional[StopCode]:

    if e >= 1:
        return StopCode.I
    if a * (1 - e) < R_td:
        return StopCode.TD
    if a < R_hj and (e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY or t >= total_time):
        return StopCode.HJ
    if R_hj < a < R_wj and (e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY or t >= total_time):
        return StopCode.WJ
    if e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY:
        return StopCode.NM
    return None


class EvolutionError(Exception):
    pass


def _resolve_n_jobs(num_cpus: int) -> int:
    if num_cpus is None or num_cpus < 0:
        return max(1, cpu_count() - 1)
    return num_cpus


def _sample_e_init(rng: np.random.Generator) -> float:
    e_val = -1.0
    F_max = 1 - math.exp(-E_INIT_MAX**2 / (2 * E_INIT_RMS**2))
    while e_val < 0.05:
        u = rng.random() * F_max
        e_val = math.sqrt(2 * E_INIT_RMS**2 * math.log(1 / (1 - u)))
    return e_val


def _sample_a_init(rng: np.random.Generator) -> float:
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


def _sample_m1(rng: np.random.Generator) -> float:
    y = rng.random()
    return (M_MIN**0.6 * (1 - y) + y * M_BR**0.6) ** (1 / 0.6)


def _sample_m2(_: Optional[np.random.Generator] = None) -> float:
    return 1e-3


@dataclass
class PlanetarySystem:
    e_init: float
    a_init: float
    m1: float
    m2: float
    lagrange: float
    seed: int

    rng: np.random.Generator = field(init=False, repr=False)
    e: float = field(init=False)
    a: float = field(init=False)
    stopping_condition: StopCode = field(init=False)
    stopping_time: float = field(init=False)
    logger: logging.LoggerAdapter = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.e = float(self.e_init)
        self.a = float(self.a_init)
        self.stopping_condition = StopCode.NM
        self.stopping_time = 0.0
        self.logger = logging.LoggerAdapter(
            logger,
            {
                "system_seed": self.seed,
                "e_init": self.e_init,
                "a_init": self.a_init,
                "m1": self.m1,
                "m2": self.m2,
            },
        )

    @classmethod
    def sample(cls, lagrange: float, system_seed: int) -> "PlanetarySystem":
        system_rng = np.random.default_rng(system_seed)
        e_seed, a_seed, m1_seed, m2_seed = system_rng.integers(0, 2**32 - 1, size=4)

        e_init = _sample_e_init(np.random.default_rng(int(e_seed)))
        a_init = _sample_a_init(np.random.default_rng(int(a_seed)))
        m1 = _sample_m1(np.random.default_rng(int(m1_seed)))
        m2 = _sample_m2()

        return cls(e_init=e_init, a_init=a_init, m1=m1, m2=m2, lagrange=lagrange, seed=int(system_seed))

    @classmethod
    def sample_batch(
        cls,
        n_samples: int,
        cluster: Cluster,
        rng: np.random.Generator,
        num_cpus: int = NUM_CPUS
    ) -> List["PlanetarySystem"]:

        lagrange_radii = cluster.get_lagrange_distribution(n_samples=n_samples, t=0)
        system_seeds = rng.integers(0, 2**32 - 1, size=n_samples)

        def _make_system(system_seed: int, lagrange: float) -> PlanetarySystem:
            return cls.sample(lagrange=lagrange, system_seed=int(system_seed))

        n_jobs = _resolve_n_jobs(num_cpus)
        return Parallel(n_jobs=n_jobs)(
            delayed(_make_system)(int(seed), float(lagrange))
            for seed, lagrange in zip(system_seeds, lagrange_radii)
        )

    def evolve(
        self,
        cluster: Cluster,
        total_time: int,
        hybrid_switch: bool = True,
        max_iters: int = 1_000_000,
        model_utils_module=None
    ) -> None:

        mu = model_utils_module if model_utils_module is not None else default_model_utils
        encounter_sampler = EncounterSampler(sigma_v=0.0, rng=self.rng)

        R_td, R_hj, R_wj = mu.get_critical_radii(m1=self.m1, m2=self.m2)
        t = 0.0
        iterations = 0

        while t < total_time:
            self.stopping_condition = check_stopping_conditions(
                self.e, self.a, t, R_td, R_hj, R_wj, total_time
            )
            if self.stopping_condition is not None:
                break

            r = cluster.get_radius(lagrange=self.lagrange, t=t)
            env = cluster.get_local_environment(r, t)
            encounter_sampler.sigma_v = env["sigma_v"]

            wt_time = encounter_sampler.get_waiting_time(env_vars=env)
            self.e, self.a = mu.tidal_effect(
                e=self.e,
                a=self.a,
                m1=self.m1,
                m2=self.m2,
                time_in_Myr=wt_time
            )
            t = min(t + wt_time, total_time)

            self.stopping_condition = check_stopping_conditions(
                self.e, self.a, t, R_td, R_hj, R_wj, total_time
            )
            if self.stopping_condition is not None:
                break

            rand_params = encounter_sampler.sample_encounter()
            kwargs = {
                **rand_params,
                "e": self.e,
                "a": self.a,
                "m1": self.m1,
                "m2": self.m2,
            }
            if mu.is_analytic_valid(**kwargs, sigma_v=env["sigma_v"]) or not hybrid_switch:
                self.e += mu.de_HR(**kwargs)
            else:
                de, da = mu.de_sim(**kwargs)
                self.e += de
                self.a += da

            iterations += 1
            if iterations >= max_iters:
                self.logger.warning(
                    "Max iterations (%d) reached during evolution; breaking early.", max_iters
                )
                break

        if self.stopping_condition is None:
            self.stopping_condition = StopCode.NM
        self.stopping_time = t

    def run(
        self,
        cluster: Cluster,
        total_time: int,
        hybrid_switch: bool = True,
        model_utils_module=None
    ) -> Dict[str, float]:

        self.evolve(
            cluster=cluster,
            total_time=total_time,
            hybrid_switch=hybrid_switch,
            model_utils_module=model_utils_module,
        )
        return self.to_result_dict(cluster=cluster, radius_time=total_time)

    def to_result_dict(self, cluster: Cluster, radius_time: float) -> Dict[str, float]:
        r = cluster.get_radius(lagrange=self.lagrange, t=radius_time)
        return {
            "r": r,
            "e_init": self.e_init,
            "a_init": self.a_init,
            "m1": self.m1,
            "m2": self.m2,
            "final_e": self.e,
            "final_a": self.a,
            "stopping_condition": self.stopping_condition.value,
            "stopping_time": self.stopping_time,
        }


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
        perts_per_Myr = default_model_utils.get_perts_per_Myr(*env_vars.values())
        return self.rng.exponential(1.0 / perts_per_Myr)
