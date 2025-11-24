import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from joblib import Parallel, cpu_count, delayed

from hjmodel import core
from hjmodel.clusters import Cluster
from hjmodel.config import CIRCULARISATION_THRESHOLD_ECCENTRICITY, NUM_CPUS, StopCode
from hjmodel.sampling import EncounterSampler, SystemSampler

__all__ = ["PlanetarySystem", "check_stopping_conditions"]

logger = logging.getLogger(__name__)


def check_stopping_conditions(
    e: float,
    a: float,
    t: float,
    R_td: float,
    R_hj: float,
    R_wj: float,
    total_time: float,
) -> Optional[StopCode]:

    if e >= 1:
        return StopCode.ION
    if a * (1 - e) < R_td:
        return StopCode.TD
    if a < R_hj and (e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY or t >= total_time):
        return StopCode.HJ
    if R_hj < a < R_wj and (
        e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY or t >= total_time
    ):
        return StopCode.WJ
    if e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY:
        return StopCode.NM
    return None


def _resolve_n_jobs(num_cpus: int) -> int:
    if num_cpus is None or num_cpus < 0:
        return max(1, cpu_count() - 1)
    return num_cpus


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
        e_seed, a_seed, m1_seed = system_rng.integers(0, 2**32 - 1, size=3)

        sampler = SystemSampler(rng=np.random.default_rng(int(e_seed)))
        e_init = sampler.sample_e_init()

        sampler.rng = np.random.default_rng(int(a_seed))
        a_init = sampler.sample_a_init()

        sampler.rng = np.random.default_rng(int(m1_seed))
        m1 = sampler.sample_m1()

        m2 = sampler.sample_m2()

        return cls(
            e_init=e_init,
            a_init=a_init,
            m1=m1,
            m2=m2,
            lagrange=lagrange,
            seed=int(system_seed),
        )

    @classmethod
    def sample_batch(
        cls,
        n_samples: int,
        cluster: Cluster,
        rng: np.random.Generator,
        num_cpus: int = NUM_CPUS,
    ) -> List["PlanetarySystem"]:

        lagrange_radii = cluster.get_lagrange_distribution(
            n_samples=n_samples, t=0, rng=rng
        )
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
        total_time: float,
        hybrid_switch: bool = True,
        max_iters: int = 1_000_000,
    ) -> None:

        encounter_sampler = EncounterSampler(rng=self.rng)

        R_td, R_hj, R_wj = core.get_critical_radii(m1=self.m1, m2=self.m2)
        t = 0.0
        iterations = 0

        while t < total_time:
            self.stopping_condition = check_stopping_conditions(
                self.e, self.a, t, R_td, R_hj, R_wj, total_time
            )
            if self.stopping_condition is not None:
                break

            r = cluster.get_radius(lagrange=self.lagrange, t=t)
            local_env = cluster.get_local_environment(r, t)
            wt_time = encounter_sampler.get_waiting_time(local_env=local_env)

            self.e, self.a = core.tidal_effect(
                e=self.e, a=self.a, m1=self.m1, m2=self.m2, time_in_Myr=wt_time
            )
            t = min(t + wt_time, total_time)

            self.stopping_condition = check_stopping_conditions(
                self.e, self.a, t, R_td, R_hj, R_wj, total_time
            )
            if self.stopping_condition is not None:
                break

            kwargs = {
                **encounter_sampler.sample_encounter(local_env=local_env),
                "e": self.e,
                "a": self.a,
                "m1": self.m1,
                "m2": self.m2,
            }
            if core.is_analytic_valid(**kwargs) or not hybrid_switch:
                self.e += core.de_hr(**kwargs)
            else:
                de, da = core.de_sim(**kwargs, rng=self.rng)
                self.e += de
                self.a += da

            iterations += 1
            if iterations >= max_iters:
                self.logger.warning(
                    "Max iterations (%d) reached during evolution; breaking early.",
                    max_iters,
                )
                break

        if self.stopping_condition is None:
            self.stopping_condition = StopCode.NM
        self.stopping_time = t

    def run(
        self,
        cluster: Cluster,
        total_time: float,
        hybrid_switch: bool = True,
    ) -> Dict[str, float]:

        self.evolve(
            cluster=cluster,
            total_time=total_time,
            hybrid_switch=hybrid_switch,
        )
        return self.to_result_dict(cluster=cluster, total_time=total_time)

    def to_result_dict(self, cluster: Cluster, total_time: float) -> Dict[str, float]:
        r = cluster.get_radius(lagrange=self.lagrange, t=total_time)
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
