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
    # Hardware
    "NUM_CPUS",
]

## General constants
G = 4 * 3.141592653589793**2  # Gravitational constant / natural units
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
