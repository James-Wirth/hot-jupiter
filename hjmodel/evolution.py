from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from joblib import Parallel, cpu_count, delayed

from hjmodel import core
from hjmodel.clusters import Cluster
from hjmodel.config import CIRCULARISATION_THRESHOLD_ECCENTRICITY, NUM_CPUS
from hjmodel.sampling import EncounterSampler, SystemSampler

__all__ = ["PlanetarySystem", "StopCode", "check_stopping_conditions"]

logger = logging.getLogger(__name__)


class StopCode(IntEnum):
    """
    Stopping conditions for planetary system evolution:

    - NM: No (significant) migration
    - ION: Ionisation (planet ejected)
    - TD: Tidal disruption
    - HJ: Hot Jupiter formation
    - WJ: Warm Jupiter formation
    """

    NM = 0
    ION = 1
    TD = 2
    HJ = 3
    WJ = 4

    @classmethod
    def from_id(cls, value: int) -> StopCode:
        """
        Create a StopCode from its integer value.

        Args:
            value: Integer value of the stopping condition.

        Returns:
            Corresponding StopCode enum member.
        """
        return cls(value)

    @classmethod
    def from_name(cls, name: str) -> StopCode:
        """
        Create a StopCode from its string name.

        Args:
            name: Name of the stopping condition (e.g., 'HJ', 'ION').

        Returns:
            Corresponding StopCode enum member.

        Raises:
            ValueError: If name is not a valid StopCode name.
        """
        try:
            return getattr(cls, name)
        except AttributeError as err:
            raise ValueError(f"Invalid StopCode name: {name}") from err


def check_stopping_conditions(
    e: float,
    a: float,
    t: float,
    R_td: float,
    R_hj: float,
    R_wj: float,
    time: float,
) -> StopCode | None:
    """
    Evaluate stopping conditions for a planetary system.

    Checks for ionisation, tidal disruption, hot Jupiter formation,
    warm Jupiter formation, or no migration in that priority order.

    Args:
        e: Current orbital eccentricity.
        a: Current semi-major axis (au).
        t: Current simulation time (Myr).
        R_td: Tidal disruption radius (au).
        R_hj: Hot Jupiter threshold radius (au).
        R_wj: Warm Jupiter threshold radius (au).
        time: Total simulation duration (Myr).

    Returns:
        StopCode if a stopping condition is met, None otherwise.
    """
    if e >= 1:
        return StopCode.ION
    if a * (1 - e) < R_td:
        return StopCode.TD
    if a < R_hj and (e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY or t >= time):
        return StopCode.HJ
    if R_hj < a < R_wj and (e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY or t >= time):
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
    """
    Represents a planetary system undergoing dynamical evolution in a cluster.

    Tracks orbital parameters (eccentricity, semi-major axis) as the system
    evolves under stellar encounters and tidal effects. Determines the final
    outcome based on stopping conditions.

    Attributes:
        e_init: Initial orbital eccentricity.
        a_init: Initial semi-major axis (au).
        m1: Host star mass (M_sun).
        m2: Planet mass (M_sun).
        lagrange: Lagrangian mass fraction determining cluster position.
        seed: Random seed for reproducibility.
        e: Current orbital eccentricity.
        a: Current semi-major axis (au).
        stopping_condition: Final outcome of the evolution.
        stopping_time: Time when stopping condition was reached (Myr).
    """

    e_init: float
    a_init: float
    m1: float
    m2: float
    lagrange: float
    seed: int

    rng: np.random.Generator = field(init=False, repr=False)
    e: float = field(init=False)
    a: float = field(init=False)
    R_td: float = field(init=False, repr=False)
    R_hj: float = field(init=False, repr=False)
    R_wj: float = field(init=False, repr=False)
    stopping_condition: StopCode = field(init=False)
    stopping_time: float = field(init=False)
    logger: logging.LoggerAdapter = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.e = float(self.e_init)
        self.a = float(self.a_init)
        self.R_td, self.R_hj, self.R_wj = core.get_critical_radii(
            m1=self.m1, m2=self.m2
        )
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
    def sample(cls, lagrange: float, system_seed: int) -> PlanetarySystem:
        """
        Create a planetary system by sampling initial parameters.

        Args:
            lagrange: Lagrangian mass fraction for cluster position.
            system_seed: Random seed for reproducibility.

        Returns:
            A new PlanetarySystem with sampled initial conditions.
        """
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
    ) -> list[PlanetarySystem]:
        """
        Sample multiple planetary systems in parallel.

        Args:
            n_samples: Number of systems to sample.
            cluster: Cluster for determining Lagrangian positions.
            rng: Random number generator for reproducibility.
            num_cpus: Number of CPUs for parallel processing. -1 uses all but one.

        Returns:
            List of sampled PlanetarySystem instances.
        """
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

    def _check_stop(self, t: float, time: float) -> bool:
        """
        Check if a stopping condition has been reached.

        Args:
            t: Current simulation time (Myr).
            time: Total simulation duration (Myr).

        Returns:
            True if a stopping condition is met, False otherwise.
        """
        self.stopping_condition = check_stopping_conditions(
            self.e, self.a, t, self.R_td, self.R_hj, self.R_wj, time
        )
        return self.stopping_condition is not None

    def _apply_encounter(self, encounter: dict, hybrid_switch: bool) -> None:
        """
        Apply the effect of a stellar encounter on the orbital parameters.

        Uses analytic approximation when valid, otherwise falls back to
        N-body integration if hybrid_switch is enabled.

        Args:
            encounter: Dictionary of encounter parameters.
            hybrid_switch: If True, use N-body when analytic is invalid.
        """
        params = {**encounter, "e": self.e, "a": self.a, "m1": self.m1, "m2": self.m2}

        if core.is_analytic_valid(**params) or not hybrid_switch:
            self.e += core.compute_delta_e_analytic(**params)
        else:
            de, da = core.compute_delta_e_nbody(**params, rng=self.rng)
            self.e += de
            self.a += da

    def evolve(
        self,
        time: float,
        cluster: Cluster,
        hybrid_switch: bool = True,
        max_iters: int = 1_000_000,
    ) -> None:
        """
        Evolve the planetary system through stellar encounters and tidal effects.

        Simulates the dynamical evolution by alternating between waiting for
        encounters, applying tidal circularization, and processing encounters.
        Continues until a stopping condition is met or max_iters is reached.

        Args:
            time: Total simulation duration (Myr).
            cluster: Cluster environment for local density and velocity.
            hybrid_switch: If True, use N-body for close encounters.
            max_iters: Maximum number of iteration cycles.
        """
        encounter_sampler = EncounterSampler(rng=self.rng)
        t = 0.0

        for _ in range(max_iters):
            if self._check_stop(t, time):
                break

            local_env = cluster.get_local_environment(
                r=cluster.get_radius(lagrange=self.lagrange, t=t), t=t
            )

            wt_time = encounter_sampler.get_waiting_time(local_env=local_env)
            t = min(t + wt_time, time)

            self.e, self.a = core.apply_tidal_effect(
                e=self.e, a=self.a, m1=self.m1, m2=self.m2, time_in_Myr=wt_time
            )

            if self._check_stop(t, time):
                break

            self._apply_encounter(
                encounter=encounter_sampler.sample_encounter(local_env=local_env),
                hybrid_switch=hybrid_switch,
            )
        else:
            self.logger.warning(
                "Max iterations (%d) reached during evolution; breaking early.",
                max_iters,
            )

        if self.stopping_condition is None:
            self.stopping_condition = StopCode.NM

        self.stopping_time = t

    def run(
        self,
        time: float,
        cluster: Cluster,
        hybrid_switch: bool = True,
    ) -> dict[str, float]:
        """
        Evolve the system and return results as a dictionary.

        Convenience method that calls evolve() and returns the results
        in a format suitable for DataFrame construction.

        Args:
            time: Total simulation duration (Myr).
            cluster: Cluster environment.
            hybrid_switch: If True, use N-body for close encounters.

        Returns:
            Dictionary of simulation results.
        """
        self.evolve(
            time=time,
            cluster=cluster,
            hybrid_switch=hybrid_switch,
        )
        return self.to_result_dict(time=time, cluster=cluster)

    def to_result_dict(self, time: float, cluster: Cluster) -> dict[str, float]:
        """
        Convert the system state to a results dictionary.

        Args:
            time: Total simulation time for radius calculation.
            cluster: Cluster for computing final radius.

        Returns:
            Dictionary with keys: 'r', 'e_init', 'a_init', 'm1', 'm2',
            'final_e', 'final_a', 'stopping_condition', 'stopping_time'.
        """
        r = cluster.get_radius(lagrange=self.lagrange, t=time)
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
