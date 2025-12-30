import math
from dataclasses import dataclass, field
from typing import ClassVar, Dict

import numpy as np

from hjmodel import core
from hjmodel.clusters import LocalEnvironment
from hjmodel.config import (
    A_BR,
    A_MAX,
    A_MIN,
    B_MAX,
    E_INIT_MAX,
    E_INIT_RMS,
    M_BR,
    M_MAX,
    M_MIN,
    s1,
    s2,
)

__all__ = ["SystemSampler", "EncounterSampler"]


@dataclass(slots=True)
class Sampler:
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)


@dataclass(slots=True)
class SystemSampler(Sampler):
    # Eccentricity sampling (Rayleigh distribution) - class variables computed once
    _e_two_sigma_sq: ClassVar[float] = 2 * E_INIT_RMS**2
    _e_f_max: ClassVar[float] = 1 - math.exp(-(E_INIT_MAX**2) / (2 * E_INIT_RMS**2))

    # Semi-major axis sampling (broken power-law)
    _a_s1_plus_1: ClassVar[float] = s1 + 1
    _a_s2_plus_1: ClassVar[float] = s2 + 1
    _a_i1: ClassVar[float] = (
        np.log(A_BR / A_MIN)
        if s1 == -1
        else (A_BR ** (s1 + 1) - A_MIN ** (s1 + 1)) / (s1 + 1)
    )
    _a_i2: ClassVar[float] = (
        A_BR ** (s1 - s2) * np.log(A_MAX / A_BR)
        if s2 == -1
        else A_BR ** (s1 - s2) * (A_MAX ** (s2 + 1) - A_BR ** (s2 + 1)) / (s2 + 1)
    )
    _a_prob1: ClassVar[float] = _a_i1 / (_a_i1 + _a_i2)

    # Mass sampling (47-Tuc IMF)
    _m1_min_pow: ClassVar[float] = M_MIN**0.6
    _m1_range: ClassVar[float] = M_BR**0.6 - M_MIN**0.6

    def sample_e_init(self) -> float:
        # Rayleigh distribution
        e_val = -1.0
        while e_val < 0.05:
            u = self.rng.random()
            e_val = math.sqrt(
                self._e_two_sigma_sq * math.log(1 / (1 - u * self._e_f_max))
            )
        return e_val

    def sample_a_init(self) -> float:
        # Broken power-law distribution (Fernandes et al., 2019)
        u = self.rng.random()
        if u < self._a_prob1:
            # Sample from first segment
            v = self.rng.random()
            if s1 == -1:
                return A_MIN * (A_BR / A_MIN) ** v
            return (
                A_MIN**self._a_s1_plus_1
                + v * (A_BR**self._a_s1_plus_1 - A_MIN**self._a_s1_plus_1)
            ) ** (1 / self._a_s1_plus_1)
        else:
            # Sample from second segment
            v = self.rng.random()
            if s2 == -1:
                return A_BR * (A_MAX / A_BR) ** v
            return (
                A_BR**self._a_s2_plus_1
                + v * (A_MAX**self._a_s2_plus_1 - A_BR**self._a_s2_plus_1)
            ) ** (1 / self._a_s2_plus_1)

    def sample_m1(self) -> float:
        # 47-Tuc IMF (Giersz and Heggie, 2011)
        y = self.rng.random()
        return (self._m1_min_pow + y * self._m1_range) ** (1 / 0.6)

    def sample_m2(self) -> float:
        return 1e-3


@dataclass(slots=True)
class EncounterSampler(Sampler):
    override_b_max: float = B_MAX

    # Perturbing mass sampling (IMF) - class variables computed once
    _m3_a: ClassVar[float] = 1.8 / (
        4 * M_BR**0.6 - 3 * M_MIN**0.6 - M_BR**2.4 * M_MAX**-1.8
    )
    _m3_b: ClassVar[float] = _m3_a * M_BR**2.4
    _m3_y_crit: ClassVar[float] = (_m3_a / 0.6) * (M_BR**0.6 - M_MIN**0.6)
    _m3_min_pow: ClassVar[float] = M_MIN**0.6
    _m3_br_neg_pow: ClassVar[float] = M_BR**-1.8

    # Orientation sampling
    _two_pi: ClassVar[float] = 2 * math.pi
    _sqrt_two: ClassVar[float] = math.sqrt(2.0)

    def sample_b(self) -> float:
        return self.override_b_max * math.sqrt(self.rng.random())

    def sample_v_infty(self, local_env: LocalEnvironment) -> float:
        sigma_rel = local_env.sigma_v * self._sqrt_two
        if not np.isfinite(sigma_rel) or sigma_rel <= 0.0:
            raise ValueError(f"Invalid sigma_v: {local_env.sigma_v}")

        x, y, z = self.rng.normal(0.0, sigma_rel, size=3)
        v = math.sqrt(x * x + y * y + z * z)
        return float(v)

    def sample_orientation(self) -> Dict[str, float]:
        return {
            "lan_angle": self.rng.random() * self._two_pi,
            "aop_angle": self.rng.random() * self._two_pi,
            "inc_angle": math.acos(1 - 2 * self.rng.random()),
        }

    def sample_m3(self) -> float:
        y = self.rng.random()

        if y <= self._m3_y_crit:
            return ((0.6 * y) / self._m3_a + self._m3_min_pow) ** (1 / 0.6)

        return (self._m3_br_neg_pow + (1.8 / self._m3_b) * (self._m3_y_crit - y)) ** (
            -1 / 1.8
        )

    def sample_encounter(self, local_env: LocalEnvironment) -> Dict[str, float]:
        params = self.sample_orientation()
        params.update(
            {
                "v_infty": self.sample_v_infty(local_env=local_env),
                "b": self.sample_b(),
                "m3": self.sample_m3(),
            }
        )
        return params

    def get_waiting_time(self, local_env: LocalEnvironment) -> float:
        perts_per_Myr = core.get_perturbation_rate(
            local_n_tot=local_env.n_tot, local_sigma_v=local_env.sigma_v
        )
        return self.rng.exponential(1.0 / perts_per_Myr)
