import math
from dataclasses import dataclass, field
from typing import Dict

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

# Eccentricity sampling
_E_RAYLEIGH_F_MAX = 1 - math.exp(-(E_INIT_MAX**2) / (2 * E_INIT_RMS**2))
_E_RAYLEIGH_TWO_SIGMA_SQ = 2 * E_INIT_RMS**2

# Semi-major axis sampling
_A_C2_OVER_C1 = A_BR ** (s1 - s2)
_A_S1_PLUS_1 = s1 + 1
_A_S2_PLUS_1 = s2 + 1

if s1 == -1:
    _A_I1 = np.log(A_BR / A_MIN)
else:
    _A_I1 = (A_BR**_A_S1_PLUS_1 - A_MIN**_A_S1_PLUS_1) / _A_S1_PLUS_1

if s2 == -1:
    _A_I2 = _A_C2_OVER_C1 * np.log(A_MAX / A_BR)
else:
    _A_I2 = _A_C2_OVER_C1 * (A_MAX**_A_S2_PLUS_1 - A_BR**_A_S2_PLUS_1) / _A_S2_PLUS_1

_A_PROB1 = _A_I1 / (_A_I1 + _A_I2)

# Mass sampling
_M1_MIN_POW = M_MIN**0.6
_M1_BR_POW = M_BR**0.6
_M1_RANGE = _M1_BR_POW - _M1_MIN_POW


@dataclass(slots=True)
class Sampler:
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)


@dataclass(slots=True)
class SystemSampler(Sampler):

    def sample_e_init(self) -> float:
        # Rayleigh distribution
        e_val = -1.0
        while e_val < 0.05:
            u = self.rng.random()
            e_val = math.sqrt(
                _E_RAYLEIGH_TWO_SIGMA_SQ * math.log(1 / (1 - u * _E_RAYLEIGH_F_MAX))
            )
        return e_val

    def sample_a_init(self) -> float:
        # Broken power-law distribution (Fernandes et al., 2019)
        u = self.rng.random()
        if u < _A_PROB1:
            # Sample from first segment
            v = self.rng.random()
            if s1 == -1:
                return A_MIN * (A_BR / A_MIN) ** v
            return (
                A_MIN**_A_S1_PLUS_1 + v * (A_BR**_A_S1_PLUS_1 - A_MIN**_A_S1_PLUS_1)
            ) ** (1 / _A_S1_PLUS_1)
        else:
            # Sample from second segment
            v = self.rng.random()
            if s2 == -1:
                return A_BR * (A_MAX / A_BR) ** v
            return (
                A_BR**_A_S2_PLUS_1 + v * (A_MAX**_A_S2_PLUS_1 - A_BR**_A_S2_PLUS_1)
            ) ** (1 / _A_S2_PLUS_1)

    def sample_m1(self) -> float:
        # 47-Tuc IMF (Giersz and Heggie, 2011)
        y = self.rng.random()
        return (_M1_MIN_POW + y * _M1_RANGE) ** (1 / 0.6)

    def sample_m2(self) -> float:
        return 1e-3


# Perturbing mass sampling
_M3_A = 1.8 / (4 * M_BR**0.6 - 3 * M_MIN**0.6 - M_BR**2.4 * M_MAX**-1.8)
_M3_B = _M3_A * M_BR**2.4
_M3_Y_CRIT = (_M3_A / 0.6) * (M_BR**0.6 - M_MIN**0.6)
_M3_MIN_POW = M_MIN**0.6
_M3_BR_NEG_POW = M_BR**-1.8

# Orientation sampling
_TWO_PI = 2 * math.pi
_SQRT_TWO = math.sqrt(2.0)


@dataclass(slots=True)
class EncounterSampler(Sampler):

    override_b_max: float = B_MAX

    def sample_b(self) -> float:
        return self.override_b_max * math.sqrt(self.rng.random())

    def sample_v_infty(self, local_env: LocalEnvironment) -> float:
        sigma_rel = local_env.sigma_v * _SQRT_TWO
        if not np.isfinite(sigma_rel) or sigma_rel <= 0.0:
            raise ValueError(f"Invalid sigma_v: {local_env.sigma_v}")

        x, y, z = self.rng.normal(0.0, sigma_rel, size=3)
        v = math.sqrt(x * x + y * y + z * z)
        return float(v)

    def sample_orientation(self) -> Dict[str, float]:
        return {
            "Omega": self.rng.random() * _TWO_PI,
            "omega": self.rng.random() * _TWO_PI,
            "inc": math.acos(1 - 2 * self.rng.random()),
        }

    def sample_m3(self) -> float:
        y = self.rng.random()

        if y <= _M3_Y_CRIT:
            return ((0.6 * y) / _M3_A + _M3_MIN_POW) ** (1 / 0.6)

        return (_M3_BR_NEG_POW + (1.8 / _M3_B) * (_M3_Y_CRIT - y)) ** (-1 / 1.8)

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
        perts_per_Myr = core.get_perts_per_Myr(
            local_n_tot=local_env.n_tot, local_sigma_v=local_env.sigma_v
        )
        return self.rng.exponential(1.0 / perts_per_Myr)
