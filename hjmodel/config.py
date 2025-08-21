import math
from enum import IntEnum

## General constants
G = 4 * math.pi**2  # Gravitational constant / natural units
INIT_PHASES = 5000  # Number of mean anomalies to use in the planetary sampling
XI = 1e-4  # Parameter used to truncate the encounter integration

# Critical hybrid-model domain parameters
T_MIN = 15
S_MIN = 300

# Unit conversions
AU_PER_PSC = 206265

# Sampling parameters
B_MAX = 75  # Maximum impact parameter / au

# IMF (Giersz and Heggie, 2011)
M_MIN = 0.08  # / M_solar
M_MAX = 50  # / M_solar
M_BR = 0.8  # / M_solar

# Initial semi-major axis distribution (Fernandes et al., 2019)
A_MIN = 1  # / au
A_BR = 2.5  # / au
A_MAX = 30  # / au
s1 = 0.80  # (exponents for the broken-power law)
s2 = -1.83

# Tidal circularization
K_P = 0.25
TAU_P = 2.1e-14  # / Myr^-1
R_P = 4.7e-4  # / au
ETA = 2.7

# Initial eccentricity distribution
E_INIT_RMS = 0.33
E_INIT_MAX = 0.6

# Stopping condition parameters
MAX_HJ_PERIOD = 10 / 365  # / yr
MAX_WJ_PERIOD = 100 / 365  # / yr
CIRCULARISATION_THRESHOLD_ECCENTRICITY = 1e-3

# Hardware...
NUM_CPUS = -1


class StopCode(IntEnum):
    """
    This class encapsulates the key stopping conditions for the planetary system:
    (i) No [significant] Migration
    (ii) Ionisation
    (iii) Tidal-Disription
    (iv) Hot-Jupiter Formation
    (v) Warm-Jupiter Formation
    """

    NM = (0, "#D3D3D3")
    ION = (1, "#1b2a49")
    TD = (2, "#769EAA")
    HJ = (3, "#D62728")
    WJ = (4, "#FF7F0E")

    def __new__(cls, value, hexcolor):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.hex = hexcolor
        return obj

    @classmethod
    def from_id(cls, value: int) -> "StopCode":
        return cls(value)

    @classmethod
    def from_name(cls, name: str) -> "StopCode":
        try:
            return getattr(cls, name)
        except AttributeError:
            raise ValueError(f"Invalid StopCode name: {name}")
