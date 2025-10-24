import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from hjmodel import core
from hjmodel.clusters import LocalEnvironment
from hjmodel.config import B_MAX, M_BR, M_MAX, M_MIN


@dataclass(slots=True)
class EncounterSampler:
    """
    This helper class generates randomized encounter parameters
    (i.e. orientation, impact parameter, approach speed, perturber mass)
    """

    override_b_max: float = B_MAX
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)

    def sample_b(self) -> float:
        return self.override_b_max * math.sqrt(self.rng.random())

    def sample_v_infty(self, local_env: LocalEnvironment) -> float:
        sigma_rel = local_env.sigma_v * math.sqrt(2.0)
        if not np.isfinite(sigma_rel) or sigma_rel <= 0.0:
            raise ValueError(f"Invalid sigma_v: {local_env.sigma_v}")

        x, y, z = self.rng.normal(0.0, sigma_rel, size=3)
        v = math.sqrt(x * x + y * y + z * z)
        return float(v)

    def sample_orientation(self) -> Dict[str, float]:
        return {
            "Omega": self.rng.random() * 2 * math.pi,
            "omega": self.rng.random() * 2 * math.pi,
            "inc": math.acos(1 - 2 * self.rng.random()),
        }

    def sample_m3(self) -> float:
        y = self.rng.random()
        a = 1.8 / (4 * M_BR**0.6 - 3 * M_MIN**0.6 - M_BR**2.4 * M_MAX**-1.8)
        b = a * M_BR**2.4
        y_crit = (a / 0.6) * (M_BR**0.6 - M_MIN**0.6)

        if y <= y_crit:
            return ((0.6 * y) / a + M_MIN**0.6) ** (1 / 0.6)

        return ((M_BR**-1.8) + (1.8 / b) * (y_crit - y)) ** (-1 / 1.8)

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
