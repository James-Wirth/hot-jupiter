from __future__ import annotations

import math
from dataclasses import dataclass

from hj.config import B_MAX, S_MIN, T_MIN, XI


@dataclass(frozen=True)
class NumericalSettings:
    encounter_xi: float = XI
    ias15_epsilon: float = 1e-9
    tidal_step_fraction: float = 0.05
    tidal_threshold: float = T_MIN
    slow_threshold: float = S_MIN
    b_max: float = B_MAX
    phase_at_pericentre: bool = False

    def __post_init__(self):
        positive = (
            self.encounter_xi,
            self.ias15_epsilon,
            self.tidal_step_fraction,
            self.tidal_threshold,
            self.slow_threshold,
            self.b_max,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Numerical controls must be finite and positive")
        if self.encounter_xi >= 1 or self.tidal_step_fraction >= 1:
            raise ValueError(
                "Encounter truncation and tidal step fraction must be below one"
            )
        if not isinstance(self.phase_at_pericentre, bool):
            raise ValueError("phase_at_pericentre must be a boolean")
