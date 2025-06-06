"""
Constants
"""

import numpy as np

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
K_P = 0.25
TAU_P = 2.1E-14             # per Myr
R_P = 4.7E-4                # au
ETA = 2.7
E_INIT_RMS = 0.33
E_INIT_MAX = 0.6
MAX_HJ_PERIOD = 10/365      # yr
MAX_WJ_PERIOD = 100/365     # yr

SC_DICT = {
    'NM': {'id': 0, 'hex': '#7F7F7F'},
    'I': {'id': 1, 'hex': '#1b2a49'},
    'TD': {'id': 2, 'hex': '#769EAA'},
    'HJ': {'id': 3, 'hex': '#D62728'},
    'WJ': {'id': 4, 'hex': '#FF7F0E'},
}

NUM_CPUS = -1