import numpy as np
import multiprocessing

# model constants
G = 4 * np.pi ** 2
INIT_PHASES = 500
XI = 1E-3

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
    'NM': 0,
    'I': 1,
    'TD': 2,
    'HJ': 3,
    'WJ': 4
}

COLOR_DICT = {
    SC_DICT['NM']: ['magenta', 'm'],
    SC_DICT['I']: ['green', 'g'],
    SC_DICT['TD']: ['blue', 'b'],
    SC_DICT['HJ']: ['red', 'r'],
    SC_DICT['WJ']: ['orange', 'o']
}

NUM_CPUS = -1