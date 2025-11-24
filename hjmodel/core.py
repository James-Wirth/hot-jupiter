import logging
import math
from collections import namedtuple

import numpy as np
import rebound
from numba import njit

from hjmodel.config import (
    B_MAX,
    ETA,
    INIT_PHASES,
    K_P,
    MAX_HJ_PERIOD,
    MAX_WJ_PERIOD,
    R_P,
    S_MIN,
    T_MIN,
    TAU_P,
    XI,
    G,
)

SimResult = namedtuple("SimResult", ["delta_e_sim", "delta_a_sim"])
PerturbingOrbitParams = namedtuple("PerturbingOrbitParams", ["a_pert", "e_pert", "rp"])
IntegrationParams = namedtuple("IntegrationParams", ["t_int", "f0"])
TidalDerivatives = namedtuple("TidalDerivatives", ["de_dt", "da_dt"])
TidalEffectResult = namedtuple("TidalEffectResult", ["e", "a"])
CriticalRadii = namedtuple("CriticalRadii", ["R_td", "R_hj", "R_wj"])


_MEAN_ANOMS_GRID = np.linspace(-np.pi, np.pi, num=INIT_PHASES, endpoint=False)
_XI_CBRT = XI ** (1.0 / 3.0)
_EPS16 = 1e-16
_EPS15 = 1e-15
_EPS14 = 1e-14

logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True, inline="always")
def _make_canonical_angle(x: float) -> float:
    twopi = 2.0 * math.pi
    x = (x + math.pi) % twopi
    if x <= 0.0:
        x += twopi
    return x - math.pi


@njit(cache=True, fastmath=True)
def _solve_kepler_E(M: float, e: float) -> float:
    """
    Halley's method for solving the Kepler equation
    """
    M = _make_canonical_angle(M)
    s = 1.0 if math.sin(M) >= 0.0 else -1.0
    E = M + s * 0.85 * e
    for _ in range(3):
        c = math.cos(E)
        sE = math.sin(E)
        f = E - e * sE - M
        f1 = 1.0 - e * c
        if abs(f1) < _EPS15:
            E -= f
            continue
        r = f / f1
        denom = f1 - 0.5 * (e * sE) * r + (1.0 / 6.0) * (e * c) * r * r
        E -= (f / denom) if abs(denom) > _EPS15 else r
        if abs(f) < _EPS14:
            break
    return E


@njit(cache=True, fastmath=True, inline="always")
def _make_true_anomaly_from_E(E: float, e: float) -> float:
    cE = math.cos(E)
    sE = math.sin(E)
    denom = 1.0 - e * cE
    if abs(denom) < _EPS15:
        denom = _EPS15 if denom >= 0.0 else -_EPS15
    sf = math.sqrt(max(0.0, 1.0 - e * e)) * sE / denom
    cf = (cE - e) / denom
    return _make_canonical_angle(math.atan2(sf, cf))


@njit(cache=True, fastmath=True)
def convert_mean_to_true_anomaly(mean_anom: float, e: float) -> float:
    E = _solve_kepler_E(mean_anom, e)
    return _make_true_anomaly_from_E(E, e)


@njit(cache=True, fastmath=True)
def get_perturber_orbit(
    v_infty: float, b: float, m1: float, m2: float
) -> PerturbingOrbitParams:
    """
    Compute orbital parameters for a hyperbolic perturber.
    """
    _MIN_VINF = 1e-12
    if v_infty <= 0.0 or not math.isfinite(v_infty):
        v_infty = _MIN_VINF

    a_pert = -G * (m1 + m2) / (v_infty * v_infty)
    e_pert = math.sqrt(1.0 + (b / a_pert) ** 2)
    rp = -a_pert * (e_pert - 1.0)

    return PerturbingOrbitParams(a_pert, e_pert, rp)


@njit(cache=True, fastmath=True)
def get_integration_window(
    a_pert: float, e_pert: float, rp: float
) -> IntegrationParams:
    """
    Compute integration time window and initial true anomaly for perturber.
    """
    r_crit = rp / _XI_CBRT
    x = (1.0 / e_pert) * (((a_pert * (1.0 - e_pert * e_pert)) / r_crit) - 1.0)
    x = max(-1.0, min(1.0, x))

    theta_crit = math.acos(x)

    arg = (e_pert + x) / (1.0 + e_pert * x)
    if arg < 1.0:
        arg = 1.0

    F = math.acosh(arg)
    t = (e_pert * math.sinh(F) - F) * (-a_pert) ** 1.5

    return IntegrationParams(2.0 * t, -theta_crit)


@njit(cache=True, fastmath=True)
def de_hr(
    v_infty: float,
    b: float,
    Omega: float,
    inc: float,
    omega: float,
    e: float,
    a: float,
    m1: float,
    m2: float,
    m3: float,
) -> float:
    """
    Eccentricity excitation via Heggie-Rasio (1986) approximation
    """
    m123 = m1 + m2 + m3
    params = get_perturber_orbit(v_infty, b, m1, m2)

    y = e * math.sqrt(max(0.0, 1.0 - e * e)) * (m3 / math.sqrt((m1 + m2) * m123))
    alpha = -1 * (15 / 4) * ((1 + params.e_pert) ** (-3 / 2))
    chi = math.acos(-1 / params.e_pert) + math.sqrt(params.e_pert**2 - 1)
    psi = (1 / 3) * (((params.e_pert**2 - 1) ** (3 / 2)) / (params.e_pert**2))

    cos_inc = math.cos(inc)
    sin_2omega = math.sin(2 * omega)
    cos_2omega = math.cos(2 * omega)
    sin_2Omega = math.sin(2 * Omega)
    cos_2Omega = math.cos(2 * Omega)

    Theta1 = (1.0 - cos_inc * cos_inc) * sin_2Omega
    Theta2 = (1.0 + cos_inc * cos_inc) * cos_2omega * sin_2Omega
    Theta3 = 2.0 * cos_inc * sin_2omega * cos_2Omega

    return (
        alpha
        * y
        * ((a / params.rp) ** (3 / 2))
        * (Theta1 * chi + (Theta2 + Theta3) * psi)
    )


def de_sim(
    v_infty: float,
    b: float,
    Omega: float,
    inc: float,
    omega: float,
    e: float,
    a: float,
    m1: float,
    m2: float,
    m3: float,
    rng: np.random.Generator,
) -> SimResult:
    """
    Eccentricity excitation via full N-body integration (REBOUND)
    """
    a_pert, e_pert, rp = get_perturber_orbit(v_infty=v_infty, b=b, m1=m1, m2=m2)
    t_int, f0 = get_integration_window(a_pert=a_pert, e_pert=e_pert, rp=rp)

    idx = int(rng.integers(0, _MEAN_ANOMS_GRID.size))
    f_phase = convert_mean_to_true_anomaly(float(_MEAN_ANOMS_GRID[idx]), e)

    sim = rebound.Simulation()
    sim.add(m=m1)
    sim.add(m=m2, a=a, e=e, f=f_phase)
    sim.add(m=m3, a=a_pert, e=e_pert, f=f0, Omega=Omega, inc=inc, omega=omega)
    sim.move_to_com()
    sim.integrate(t_int)

    o_binary = sim.particles[1].orbit()
    delta_e_sim = o_binary.e - e
    delta_a_sim = o_binary.a - a

    sim = None
    return SimResult(delta_e_sim, delta_a_sim)


def is_analytic_valid(
    v_infty: float, b: float, a: float, m1: float, m2: float, **kwargs
) -> bool:
    """
    Validate encounter parameters for analytic treatment.

    Returns True if both approximations are satisfied:
    - Tidal parameter (r_p / a) > T_MIN
    - Slow parameter (t_int / t_per) > S_MIN
    """
    params = get_perturber_orbit(v_infty, b, m1, m2)

    if params.rp / a <= T_MIN:
        return False

    t_int, _ = get_integration_window(params.a_pert, params.e_pert, params.rp)
    t_per = math.sqrt(a**3 / (m1 + m2))

    return t_int / t_per > S_MIN


@njit(cache=True, fastmath=True)
def f(e: float) -> float:
    """
    Helper function for de_tid_dt and da_tid_dt
    """
    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e4 * e4
    e10 = e8 * e2

    num = (
        1.0
        + (45.0 / 14.0) * e2
        + 8.0 * e4
        + (685.0 / 224.0) * e6
        + (255.0 / 488.0) * e8
        + (25.0 / 1792.0) * e10
    )
    denom = 1.0 + 3.0 * e2 + (3.0 / 8.0) * e4
    return num / denom


@njit(cache=True, fastmath=True)
def get_tidal_derivatives(e: float, a: float, m1: float, m2: float) -> TidalDerivatives:
    """
    Compute de/dt and da/dt for tidal circularization (in Myr^-1).

    Returns: TidalDerivatives(de_dt, da_dt)
    """
    q = m2 / m1
    n = 10**6 * np.sqrt(G * (1 + q) * m1 / a**3)
    e_sq = e * e
    one_minus_e_sq = 1.0 - e_sq

    common_factor = -21 * K_P * TAU_P * (n * n) / q * (R_P / a) ** 5 * f(e)

    de_dt = common_factor * e / (one_minus_e_sq ** (13 / 2)) / 2
    da_dt = common_factor * a * e_sq / (one_minus_e_sq ** (15 / 2))

    return TidalDerivatives(de_dt, da_dt)


@njit(cache=True, fastmath=True)
def get_dn(
    dedn: float, dadn: float, e: float, a: float, C: float, n_cum: float
) -> float:
    """
    A sufficiently small step size for the tidal circularization integration
    """
    e_safe = e if abs(e) > _EPS16 else _EPS16
    a_safe = a if abs(a) > _EPS16 else _EPS16
    s1 = abs(dedn) / e_safe
    s2 = abs(dadn) / a_safe
    m = s1 if s1 >= s2 else s2
    step = C / m if m > 0.0 else 1.0
    rem = 1.0 - n_cum
    return step if step <= rem else rem


@njit(cache=True, fastmath=True)
def tidal_effect(
    e: float, a: float, m1: float, m2: float, time_in_Myr: float, C: float = 0.01
) -> TidalEffectResult:
    """
    The new eccentricity (e) and semi-major axis (a) after tidal circularization after time_in_Myr
    """
    n_cum = 0.0
    while n_cum < 1.0 and e > 1e-3:
        derivs = get_tidal_derivatives(e, a, m1, m2)
        dedn = derivs.de_dt * time_in_Myr
        dadn = derivs.da_dt * time_in_Myr
        dn = get_dn(dedn, dadn, e, a, C, n_cum)
        n_cum += dn
        e += dedn * dn
        a += dadn * dn
    return TidalEffectResult(e, a)


@njit(cache=True, fastmath=True)
def get_critical_radii(m1: float, m2: float) -> CriticalRadii:
    """
    The critical orbital separations for tidal-disruption, HJ formation and WJ formation
    """
    R_td = ETA * R_P * (m1 / m2) ** (1 / 3)
    R_hj = (MAX_HJ_PERIOD**2 * (m1 + m2)) ** (1 / 3)
    R_wj = (MAX_WJ_PERIOD**2 * (m1 + m2)) ** (1 / 3)
    return CriticalRadii(R_td, R_hj, R_wj)


@njit(cache=True, fastmath=True)
def get_perts_per_Myr(local_n_tot: float, local_sigma_v: float) -> float:
    """
    Average perturbation rate (in inverse Myr)
    """
    _ADJUSTED_GAMMA = 3.21
    return (
        _ADJUSTED_GAMMA
        * local_n_tot
        * ((B_MAX / 75) ** 2)
        * (local_sigma_v * np.sqrt(2))
    )
