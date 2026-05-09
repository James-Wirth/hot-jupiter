from __future__ import annotations

from typing import Final

__all__ = [
    # General constants
    "G",
    "INIT_PHASES",
    "XI",
    "T_MIN",
    "S_MIN",
    # Unit conversions
    "AU_PER_PSC",
    # Sampling parameters
    "B_MAX",
    "M_MIN",
    "M_MAX",
    "M_BR",
    "A_MIN",
    "A_BR",
    "A_MAX",
    "s1",
    "s2",
    # Tidal circularization
    "K_P",
    "TAU_P",
    "R_P",
    "ETA",
    # Initial eccentricity distribution
    "E_INIT_RMS",
    "E_INIT_MAX",
    # Stopping condition parameters
    "MAX_HJ_PERIOD",
    "MAX_WJ_PERIOD",
    "CIRCULARISATION_THRESHOLD_ECCENTRICITY",
    # Plummer geometry
    "A_OVER_RH",
    # Hardware
    "NUM_CPUS",
]

# General constants
G: Final[float] = 4 * 3.141592653589793**2  # natural units
INIT_PHASES: Final[int] = 5000  # planetary mean-anomaly samples
XI: Final[float] = 1e-4  # encounter integration truncation parameter

# Hybrid-model validity domain
T_MIN: Final[int] = 15
S_MIN: Final[int] = 300

# Unit conversions
AU_PER_PSC: Final[int] = 206265

# Maximum impact parameter / au
B_MAX: Final[int] = 75

# IMF (Giersz & Heggie 2011)
M_MIN: Final[float] = 0.08  # M_sun
M_MAX: Final[int] = 50  # M_sun
M_BR: Final[float] = 0.8  # M_sun

# Initial semi-major axis distribution (Fernandes et al. 2019)
A_MIN: Final[int] = 1  # au
A_BR: Final[float] = 2.5  # au
A_MAX: Final[int] = 30  # au
s1: Final[float] = 0.80
s2: Final[float] = -1.83

# Tidal circularization
K_P: Final[float] = 0.25
TAU_P: Final[float] = 2.1e-14  # Myr^-1
R_P: Final[float] = 4.7e-4  # au
ETA: Final[float] = 2.7

# Initial eccentricity (truncated Rayleigh)
E_INIT_RMS: Final[float] = 0.33
E_INIT_MAX: Final[float] = 0.6

# Stopping conditions
MAX_HJ_PERIOD: Final[float] = 10 / 365  # yr
MAX_WJ_PERIOD: Final[float] = 100 / 365  # yr
CIRCULARISATION_THRESHOLD_ECCENTRICITY: Final[float] = 1e-3

# Plummer scale-radius / half-mass-radius ratio
A_OVER_RH: Final[float] = 0.766

# Hardware
NUM_CPUS: Final[int] = -1
