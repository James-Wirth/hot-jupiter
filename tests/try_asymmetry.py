import itertools
from itertools import combinations

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy import integrate
import tqdm

from hjmodel.config import *
from hjmodel.model_utils import de_HR, get_pert_orbit_params, get_int_params, get_true_anomaly
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import rebound

CANON = {
    'e': 0.3,
    'a': 1,
    'm1': 1,
    'm2': 1E-3,
    'm3': 1
}
E_THR = 0.05

def diff(b, v_infty, Omega, inc, omega):
    return np.abs(de_HR(v_infty=v_infty, b=b, Omega=Omega, inc=inc, omega=omega,
                  e=CANON['e'], a=CANON['a'], m1=CANON['m1'], m2=CANON['m2'], m3=CANON['m3'])) - E_THR

def b_max(Omega, inc, omega, v_infty):
    return root_scalar(diff, bracket=[0, 1000], method='brentq', args=(v_infty, Omega, inc, omega)).root

def get_crs(v_infty):
    n = 20
    two_pi_range = np.linspace(0, 2*np.pi, n, endpoint=True)
    pi_range = np.linspace(0, np.pi, n, endpoint=False)[1:]
    num_angles = len(two_pi_range)**2 * len(pi_range)
    crs = 0

    for angles in itertools.product(*[two_pi_range, pi_range, two_pi_range]):
        crs += (np.pi ** 2) * np.sin(angles[1]) * (b_max(*angles, v_infty) ** 2) / num_angles
    return crs

def try_asymmetry():
    v_infty_values = np.linspace(0.211, 6.33, 20)
    results = Parallel(n_jobs=NUM_CPUS)(
        delayed(get_crs)(v_infty) for v_infty in tqdm.tqdm(v_infty_values)
    )
    plt.plot(v_infty_values/0.211, results)
    plt.xlim(1, 30)
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    try_asymmetry()