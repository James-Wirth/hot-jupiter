import random
import math
from joblib import delayed, Parallel
from hjmodel.config import *
from scipy.optimize import fsolve

def rand_b() -> float:
    """
    b: float            Random impact parameter (au)
    """
    return np.sqrt(random.random() * B_MAX ** 2)

def rand_v_infty(sigma_v: float) -> float:
    """
    v_infty: float      Random asymptotic relative speed (au per year)
    """
    y = random.random()
    sigma_rel = sigma_v * np.sqrt(2)
    cdf = lambda x: math.erf(x / (np.sqrt(2) * sigma_rel)) - ((np.sqrt(2) * x) / (np.sqrt(np.pi) * sigma_rel)) * np.exp(
        -x ** 2 / (2 * sigma_rel ** 2))
    ans, *info = fsolve(lambda x: cdf(x) - y, np.sqrt(2) * sigma_v)
    return ans

def rand_i() -> float:
    """
    inc: float          Inclination angle
    """
    return np.arccos(1 - 2 * random.random())

def rand_2pi() -> float:
    """
    Returns
    ----------
    inc: float          Angle between 0 and 2pi
    """
    return random.random() * 2 * np.pi

def rand_m3() -> float:
    """
    Returns
    ----------
    m3: float           Mass of perturbing star (M_solar)
    """
    y = random.random()
    a = 1.8 / (4 * (M_BR ** 0.6) - 3 * (M_MIN ** 0.6) - (M_BR ** 2.4) * M_MAX ** (-1.8))
    b = a * (M_BR ** 2.4)
    y_crit = (a / 0.6) * ((M_BR ** 0.6) - (M_MIN ** 0.6))
    if y <= y_crit:
        return ((0.6 * y) / a + (M_MIN ** 0.6)) ** (1 / 0.6)
    else:
        return ((M_BR ** -1.8) + (1.8 / b) * (y_crit - y)) ** (-1 / 1.8)

def random_encounter_params(sigma_v: float) -> dict[str, float]:
    """
    Inputs
    ----------
    sigma_v: float          Isotropic velocity dispersion (au per year)

    Returns
    ----------
    d: dict[str, float]     Dictionary of random encounter parameters for
                            each stochastic kick
    """
    d = {
        'v_infty':  rand_v_infty(sigma_v=sigma_v),
        'b':        rand_b(),
        'Omega':    rand_2pi(),
        'inc':      rand_i(),
        'omega':    rand_2pi(),
        'm3':       rand_m3()
    }
    return d

def rand_e_init() -> float:
    e_init = -1
    F_max = 1 - np.exp(-E_INIT_MAX ** 2 / (2 * E_INIT_RMS ** 2))
    while e_init < 0.05:
        e_init = np.sqrt(2 * E_INIT_RMS**2 * np.log(1/(1-random.random() * F_max)))
    return e_init

def rand_a_init() -> float:
    return 10**(random.random() * np.log10(30))

def rand_m1() -> float:
    y = random.random()
    return (M_MIN**0.6 * (1-y) + y * M_BR**0.6)**(1/0.6)

def rand_m2() -> float:
    return 1E-3

def get_random_system_params(n_samples:int) -> list:
    d = [
        Parallel(n_jobs=NUM_CPUS)(delayed(rand_e_init)() for _ in range(n_samples)),
        Parallel(n_jobs=NUM_CPUS)(delayed(rand_a_init)() for _ in range(n_samples)),
        Parallel(n_jobs=NUM_CPUS)(delayed(rand_m1)() for _ in range(n_samples)),
        Parallel(n_jobs=NUM_CPUS)(delayed(rand_m2)() for _ in range(n_samples))
    ]
    return d

def get_waiting_time(perts_per_Myr: float) -> float:
    return np.random.exponential(1/perts_per_Myr)