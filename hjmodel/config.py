import numpy as np
from enum import IntEnum

# model constants
G = 4 * np.pi ** 2
INIT_PHASES = 5000
XI = 1E-4

# hybrid model critical parameters
T_MIN = 15
S_MIN = 300

# unit conversions
AU_PER_PSC = 206265

# evolution parameters
B_MAX = 75                  # au

M_MIN = 0.08                # M_solar
M_MAX = 50                  # M_solar
M_BR = 0.8                  # M_solar

A_MIN = 1                   # au
A_BR = 2.5                  # au
A_MAX = 30                  # au
s1 = 0.80
s2 = -1.83

K_P = 0.25
TAU_P = 2.1E-14             # per Myr
R_P = 4.7E-4                # au
ETA = 2.7
E_INIT_RMS = 0.33
E_INIT_MAX = 0.6
MAX_HJ_PERIOD = 10/365      # yr
MAX_WJ_PERIOD = 100/365     # yr
CIRCULARISATION_THRESHOLD_ECCENTRICITY = 1e-3

NUM_CPUS = -1

class StopCode(IntEnum):
    NM = (0, "#D3D3D3")
    I  = (1, "#1b2a49")
    TD = (2, "#769EAA")
    HJ = (3, "#D62728")
    WJ = (4, "#FF7F0E")

    def __new__(cls, value, hexcolor):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.hex = hexcolor  # attach extra metadata
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
