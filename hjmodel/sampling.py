from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

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
    """
    Base class for probabilistic samplers with a random number generator.

    Attributes:
        rng: NumPy random number generator for sampling.
    """

    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)


@dataclass(slots=True)
class SystemSampler(Sampler):
    """
    Sampler for initial planetary system parameters.

    Samples eccentricity from a truncated Rayleigh distribution,
    semi-major axis from a broken power-law, and stellar mass
    from the 47 Tuc initial mass function.
    """

    _e_two_sigma_sq: ClassVar[float] = 2 * E_INIT_RMS**2
    _e_f_max: ClassVar[float] = 1 - math.exp(-(E_INIT_MAX**2) / (2 * E_INIT_RMS**2))

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

    _m1_min_pow: ClassVar[float] = M_MIN**0.6
    _m1_range: ClassVar[float] = M_BR**0.6 - M_MIN**0.6

    def sample_e_init(self) -> float:
        """
        Sample initial orbital eccentricity from a truncated Rayleigh distribution.

        The distribution is truncated to [0.05, E_INIT_MAX] with RMS = E_INIT_RMS.

        Returns:
            Sampled eccentricity value.
        """
        e_val = -1.0
        while e_val < 0.05:
            u = self.rng.random()
            e_val = math.sqrt(
                self._e_two_sigma_sq * math.log(1 / (1 - u * self._e_f_max))
            )
        return e_val

    def sample_a_init(self) -> float:
        """
        Sample initial semi-major axis from a broken power-law distribution.

        Uses the distribution from Fernandes et al. (2019) with break at A_BR,
        slopes s1 and s2, and range [A_MIN, A_MAX].

        Returns:
            Sampled semi-major axis (au).
        """
        u = self.rng.random()
        if u < self._a_prob1:
            v = self.rng.random()
            if s1 == -1:
                return A_MIN * (A_BR / A_MIN) ** v
            return (
                A_MIN**self._a_s1_plus_1
                + v * (A_BR**self._a_s1_plus_1 - A_MIN**self._a_s1_plus_1)
            ) ** (1 / self._a_s1_plus_1)
        else:
            v = self.rng.random()
            if s2 == -1:
                return A_BR * (A_MAX / A_BR) ** v
            return (
                A_BR**self._a_s2_plus_1
                + v * (A_MAX**self._a_s2_plus_1 - A_BR**self._a_s2_plus_1)
            ) ** (1 / self._a_s2_plus_1)

    def sample_m1(self) -> float:
        """
        Sample host star mass from the 47 Tuc initial mass function.

        Uses the IMF from Giersz and Heggie (2011).

        Returns:
            Sampled stellar mass (M_sun).
        """
        y = self.rng.random()
        return (self._m1_min_pow + y * self._m1_range) ** (1 / 0.6)

    def sample_m2(self) -> float:
        """
        Return the planet mass.

        Currently returns a fixed Jupiter-mass value.

        Returns:
            Planet mass (M_sun).
        """
        return 1e-3


@dataclass(slots=True)
class EncounterSampler(Sampler):
    """
    Sampler for stellar encounter parameters.

    Samples perturber mass from the IMF, impact parameter uniformly
    in cross-section, velocity from a Maxwellian distribution, and
    orbital orientation isotropically.

    Attributes:
        override_b_max: Maximum impact parameter for sampling (au).
    """

    override_b_max: float = B_MAX

    _m3_a: ClassVar[float] = 1.8 / (
        4 * M_BR**0.6 - 3 * M_MIN**0.6 - M_BR**2.4 * M_MAX**-1.8
    )
    _m3_b: ClassVar[float] = _m3_a * M_BR**2.4
    _m3_y_crit: ClassVar[float] = (_m3_a / 0.6) * (M_BR**0.6 - M_MIN**0.6)
    _m3_min_pow: ClassVar[float] = M_MIN**0.6
    _m3_br_neg_pow: ClassVar[float] = M_BR**-1.8

    _two_pi: ClassVar[float] = 2 * math.pi
    _sqrt_two: ClassVar[float] = math.sqrt(2.0)

    def sample_b(self) -> float:
        """
        Sample impact parameter uniformly in cross-sectional area.

        The sampling is uniform in b^2 to give uniform coverage in area.

        Returns:
            Sampled impact parameter (au).
        """
        return self.override_b_max * math.sqrt(self.rng.random())

    def sample_v_infty(self, local_env: LocalEnvironment) -> float:
        """
        Sample relative velocity at infinity from a Maxwellian distribution.

        The velocity is sampled from a 3D Gaussian with dispersion
        sqrt(2) * sigma_v to account for relative motion.

        Args:
            local_env: Local cluster environment with velocity dispersion.

        Returns:
            Sampled velocity at infinity (au/yr).

        Raises:
            ValueError: If the velocity dispersion is invalid.
        """
        sigma_rel = local_env.sigma_v * self._sqrt_two
        if not np.isfinite(sigma_rel) or sigma_rel <= 0.0:
            raise ValueError(f"Invalid sigma_v: {local_env.sigma_v}")

        x, y, z = self.rng.normal(0.0, sigma_rel, size=3)
        v = math.sqrt(x * x + y * y + z * z)
        return float(v)

    def sample_orientation(self) -> dict[str, float]:
        """
        Sample orbital orientation angles isotropically.

        Returns:
            Dictionary with keys 'lan_angle' (longitude of ascending node),
            'aop_angle' (argument of periapsis), and 'inc_angle' (inclination),
            all in radians.
        """
        return {
            "lan_angle": self.rng.random() * self._two_pi,
            "aop_angle": self.rng.random() * self._two_pi,
            "inc_angle": math.acos(1 - 2 * self.rng.random()),
        }

    def sample_m3(self) -> float:
        """
        Sample perturber mass from the 47 Tuc initial mass function.

        Uses a two-segment power-law IMF with break at M_BR.

        Returns:
            Sampled perturber mass (M_sun).
        """
        y = self.rng.random()

        if y <= self._m3_y_crit:
            return ((0.6 * y) / self._m3_a + self._m3_min_pow) ** (1 / 0.6)

        return (self._m3_br_neg_pow + (1.8 / self._m3_b) * (self._m3_y_crit - y)) ** (
            -1 / 1.8
        )

    def sample_encounter(self, local_env: LocalEnvironment) -> dict[str, float]:
        """
        Sample all parameters for a single stellar encounter.

        Combines orientation, velocity, impact parameter, and perturber mass
        into a single parameter dictionary.

        Args:
            local_env: Local cluster environment for velocity sampling.

        Returns:
            Dictionary with keys 'lan_angle', 'aop_angle', 'inc_angle',
            'v_infty', 'b', and 'm3'.
        """
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
        """
        Sample the waiting time until the next encounter.

        Draws from an exponential distribution with rate determined by
        the local stellar density and velocity dispersion.

        Args:
            local_env: Local cluster environment.

        Returns:
            Sampled waiting time (Myr).
        """
        perts_per_Myr = core.get_perturbation_rate(
            local_n_tot=local_env.n_tot, local_sigma_v=local_env.sigma_v
        )
        return self.rng.exponential(1.0 / perts_per_Myr)
