from __future__ import annotations

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
PerturbingOrbitParams = namedtuple("PerturbingOrbitParams", ["a_pert", "e_pert", "r_p"])
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
    two_pi = 2.0 * math.pi
    x = (x + math.pi) % two_pi
    if x <= 0.0:
        x += two_pi
    return x - math.pi


@njit(cache=True, fastmath=True)
def _solve_kepler_ecc_anom(mean_anom: float, e: float) -> float:
    """
    Solve the Kepler equation using Halley's method.

    Args:
        mean_anom: Mean anomaly (radians).
        e: Orbital eccentricity.

    Returns:
        Eccentric anomaly (radians).
    """
    mean_anom = _make_canonical_angle(mean_anom)
    sign = 1.0 if math.sin(mean_anom) >= 0.0 else -1.0
    ecc_anom = mean_anom + sign * 0.85 * e
    for _ in range(3):
        cos_ecc = math.cos(ecc_anom)
        sin_ecc = math.sin(ecc_anom)
        residual = ecc_anom - e * sin_ecc - mean_anom
        deriv1 = 1.0 - e * cos_ecc
        if abs(deriv1) < _EPS15:
            ecc_anom -= residual
            continue
        ratio = residual / deriv1
        denom = (
            deriv1
            - 0.5 * (e * sin_ecc) * ratio
            + (1.0 / 6.0) * (e * cos_ecc) * ratio * ratio
        )
        ecc_anom -= (residual / denom) if abs(denom) > _EPS15 else ratio
        if abs(residual) < _EPS14:
            break
    return ecc_anom


@njit(cache=True, fastmath=True, inline="always")
def _make_true_anomaly_from_ecc_anom(ecc_anom: float, e: float) -> float:
    cos_ecc = math.cos(ecc_anom)
    sin_ecc = math.sin(ecc_anom)
    denom = 1.0 - e * cos_ecc
    if abs(denom) < _EPS15:
        denom = _EPS15 if denom >= 0.0 else -_EPS15
    sin_true = math.sqrt(max(0.0, 1.0 - e * e)) * sin_ecc / denom
    cos_true = (cos_ecc - e) / denom
    return _make_canonical_angle(math.atan2(sin_true, cos_true))


@njit(cache=True, fastmath=True)
def convert_mean_to_true_anomaly(mean_anom: float, e: float) -> float:
    """
    Convert mean anomaly to true anomaly.

    Args:
        mean_anom: Mean anomaly (radians).
        e: Orbital eccentricity.

    Returns:
        True anomaly (radians).
    """
    ecc_anom = _solve_kepler_ecc_anom(mean_anom, e)
    return _make_true_anomaly_from_ecc_anom(ecc_anom, e)


@njit(cache=True, fastmath=True)
def get_perturber_orbit(
    v_infty: float, b: float, m1: float, m2: float
) -> PerturbingOrbitParams:
    """
    Compute orbital parameters for a hyperbolic perturber.

    Args:
        v_infty: Velocity at infinity (au/yr).
        b: Impact parameter (au).
        m1: Host star mass (M_sun).
        m2: Planet mass (M_sun).

    Returns:
        PerturbingOrbitParams containing semi-major axis, eccentricity,
        and periapsis distance.
    """
    _MIN_VINF = 1e-12
    if v_infty <= 0.0 or not math.isfinite(v_infty):
        v_infty = _MIN_VINF

    a_pert = -G * (m1 + m2) / (v_infty * v_infty)
    e_pert = math.sqrt(1.0 + (b / a_pert) ** 2)
    r_p = -a_pert * (e_pert - 1.0)

    return PerturbingOrbitParams(a_pert, e_pert, r_p)


@njit(cache=True, fastmath=True, inline="always")
def _get_critical_true_anomaly(a_pert: float, e_pert: float, r_p: float) -> float:
    """
    Compute the critical true anomaly where the perturber enters/exits the interaction region.

    Args:
        a_pert: Perturber semi-major axis (au).
        e_pert: Perturber eccentricity.
        r_p: Periapsis distance (au).

    Returns:
        Critical true anomaly (radians).
    """
    r_crit = r_p / _XI_CBRT
    cos_theta = (1.0 / e_pert) * (((a_pert * (1.0 - e_pert * e_pert)) / r_crit) - 1.0)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


@njit(cache=True, fastmath=True)
def get_integration_time(
    a_pert: float, e_pert: float, r_p: float, m1: float, m2: float
) -> float:
    """
    Compute the total integration time for a perturber flyby.

    Args:
        a_pert: Perturber semi-major axis (au).
        e_pert: Perturber eccentricity.
        r_p: Periapsis distance (au).
        m1: Host star mass (M_sun).
        m2: Planet mass (M_sun).

    Returns:
        Total integration time (yr).
    """
    theta_crit = _get_critical_true_anomaly(a_pert, e_pert, r_p)
    cos_theta = math.cos(theta_crit)

    acosh_arg = (e_pert + cos_theta) / (1.0 + e_pert * cos_theta)
    if acosh_arg < 1.0:
        acosh_arg = 1.0

    hyp_anom = math.acosh(acosh_arg)
    half_time = (e_pert * math.sinh(hyp_anom) - hyp_anom) * math.sqrt(
        (-a_pert) ** 3 / (G * (m1 + m2))
    )

    return 2.0 * half_time


@njit(cache=True, fastmath=True)
def compute_delta_e_analytic(
    v_infty: float,
    b: float,
    lan_angle: float,
    inc_angle: float,
    aop_angle: float,
    e: float,
    a: float,
    m1: float,
    m2: float,
    m3: float,
) -> float:
    """
    Compute eccentricity change using the Heggie-Rasio (1986) analytic approximation.

    Valid for distant, slow encounters where the tidal and slow approximations hold.

    Args:
        v_infty: Velocity at infinity (au/yr).
        b: Impact parameter (au).
        lan_angle: Longitude of ascending node (radians).
        inc_angle: Inclination angle (radians).
        aop_angle: Argument of periapsis (radians).
        e: Current orbital eccentricity.
        a: Current semi-major axis (au).
        m1: Host star mass (M_sun).
        m2: Planet mass (M_sun).
        m3: Perturber mass (M_sun).

    Returns:
        Change in eccentricity (delta_e).
    """
    m_total = m1 + m2 + m3
    params = get_perturber_orbit(v_infty, b, m1, m2)

    mass_factor = (
        e * math.sqrt(max(0.0, 1.0 - e * e)) * (m3 / math.sqrt((m1 + m2) * m_total))
    )
    alpha = -1 * (15 / 4) * ((1 + params.e_pert) ** (-3 / 2))
    chi = math.acos(-1 / params.e_pert) + math.sqrt(params.e_pert**2 - 1)
    psi = (1 / 3) * (((params.e_pert**2 - 1) ** (3 / 2)) / (params.e_pert**2))

    cos_inc = math.cos(inc_angle)
    sin_2aop = math.sin(2 * aop_angle)
    cos_2aop = math.cos(2 * aop_angle)
    sin_2lan = math.sin(2 * lan_angle)
    cos_2lan = math.cos(2 * lan_angle)

    theta1 = (1.0 - cos_inc * cos_inc) * sin_2lan
    theta2 = (1.0 + cos_inc * cos_inc) * cos_2aop * sin_2lan
    theta3 = 2.0 * cos_inc * sin_2aop * cos_2lan

    return (
        alpha
        * mass_factor
        * ((a / params.r_p) ** (3 / 2))
        * (theta1 * chi + (theta2 + theta3) * psi)
    )


def compute_delta_e_nbody(
    v_infty: float,
    b: float,
    lan_angle: float,
    inc_angle: float,
    aop_angle: float,
    e: float,
    a: float,
    m1: float,
    m2: float,
    m3: float,
    rng: np.random.Generator,
) -> SimResult:
    """
    Compute eccentricity and semi-major axis changes via N-body integration.

    Uses REBOUND to perform a full 3-body integration for close encounters
    where the analytic approximation is invalid.

    Args:
        v_infty: Velocity at infinity (au/yr).
        b: Impact parameter (au).
        lan_angle: Longitude of ascending node (radians).
        inc_angle: Inclination angle (radians).
        aop_angle: Argument of periapsis (radians).
        e: Current orbital eccentricity.
        a: Current semi-major axis (au).
        m1: Host star mass (M_sun).
        m2: Planet mass (M_sun).
        m3: Perturber mass (M_sun).
        rng: Random number generator for sampling initial orbital phase.

    Returns:
        SimResult containing delta_e_sim and delta_a_sim.
    """
    a_pert, e_pert, r_p = get_perturber_orbit(v_infty=v_infty, b=b, m1=m1, m2=m2)

    t_int = get_integration_time(a_pert=a_pert, e_pert=e_pert, r_p=r_p, m1=m1, m2=m2)
    f0 = -_get_critical_true_anomaly(a_pert=a_pert, e_pert=e_pert, r_p=r_p)

    idx = int(rng.integers(0, _MEAN_ANOMS_GRID.size))
    f_phase = convert_mean_to_true_anomaly(float(_MEAN_ANOMS_GRID[idx]), e)

    sim = rebound.Simulation()
    sim.G = G
    sim.add(m=m1)
    sim.add(m=m2, a=a, e=e, f=f_phase)
    sim.add(
        m=m3, a=a_pert, e=e_pert, f=f0, Omega=lan_angle, inc=inc_angle, omega=aop_angle
    )
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

    Checks whether both the tidal and slow approximations are satisfied
    for using the Heggie-Rasio analytic formula.

    Args:
        v_infty: Velocity at infinity (au/yr).
        b: Impact parameter (au).
        a: Current semi-major axis (au).
        m1: Host star mass (M_sun).
        m2: Planet mass (M_sun).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        True if both conditions are satisfied:
        - Tidal parameter (r_p / a) > T_MIN
        - Slow parameter (t_int / t_per) > S_MIN
    """
    params = get_perturber_orbit(v_infty, b, m1, m2)

    if params.r_p / a <= T_MIN:
        return False

    t_int = get_integration_time(params.a_pert, params.e_pert, params.r_p, m1, m2)
    t_per = math.sqrt(a**3 / (m1 + m2))

    return t_int / t_per > S_MIN


@njit(cache=True, fastmath=True)
def get_tidal_f_factor(e: float) -> float:
    """
    Compute the f(e) factor for tidal circularization.

    This polynomial approximation captures the eccentricity dependence
    of tidal dissipation rates.

    Args:
        e: Orbital eccentricity.

    Returns:
        Tidal factor f(e).
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
    Compute time derivatives of eccentricity and semi-major axis due to tidal dissipation.

    Args:
        e: Orbital eccentricity.
        a: Semi-major axis (au).
        m1: Host star mass (M_sun).
        m2: Planet mass (M_sun).

    Returns:
        TidalDerivatives containing de_dt and da_dt (both in Myr^-1).
    """
    mass_ratio = m2 / m1
    mean_motion = 10**6 * np.sqrt(G * (1 + mass_ratio) * m1 / a**3)
    e_sq = e * e
    one_minus_e_sq = 1.0 - e_sq

    common_factor = (
        -21
        * K_P
        * TAU_P
        * (mean_motion * mean_motion)
        / mass_ratio
        * (R_P / a) ** 5
        * get_tidal_f_factor(e)
    )

    de_dt = common_factor * e / (one_minus_e_sq ** (13 / 2)) / 2
    da_dt = common_factor * a * e_sq / (one_minus_e_sq ** (15 / 2))

    return TidalDerivatives(de_dt, da_dt)


@njit(cache=True, fastmath=True)
def get_tidal_step_size(
    dedn: float, dadn: float, e: float, a: float, step_factor: float, n_cum: float
) -> float:
    """
    Compute an adaptive step size for tidal circularization integration.

    Ensures the step is small enough to accurately capture the evolution
    while not exceeding the remaining integration interval.

    Args:
        dedn: Derivative of eccentricity with respect to normalized time.
        dadn: Derivative of semi-major axis with respect to normalized time.
        e: Current eccentricity.
        a: Current semi-major axis (au).
        step_factor: Maximum fractional change per step.
        n_cum: Cumulative normalized time elapsed (0 to 1).

    Returns:
        Step size in normalized time units.
    """
    e_safe = e if abs(e) > _EPS16 else _EPS16
    a_safe = a if abs(a) > _EPS16 else _EPS16
    rel_rate_e = abs(dedn) / e_safe
    rel_rate_a = abs(dadn) / a_safe
    max_rel_rate = rel_rate_e if rel_rate_e >= rel_rate_a else rel_rate_a
    step = step_factor / max_rel_rate if max_rel_rate > 0.0 else 1.0
    remaining = 1.0 - n_cum
    return step if step <= remaining else remaining


@njit(cache=True, fastmath=True)
def apply_tidal_effect(
    e: float,
    a: float,
    m1: float,
    m2: float,
    time_in_Myr: float,
    step_factor: float = 0.01,
) -> TidalEffectResult:
    """
    Apply tidal circularization to update orbital parameters.

    Integrates the tidal evolution equations over the specified time interval
    using an adaptive Euler method.

    Args:
        e: Initial orbital eccentricity.
        a: Initial semi-major axis (au).
        m1: Host star mass (M_sun).
        m2: Planet mass (M_sun).
        time_in_Myr: Duration of tidal evolution (Myr).
        step_factor: Maximum fractional change per step. Default: 0.01.

    Returns:
        TidalEffectResult containing updated eccentricity and semi-major axis.
    """
    n_cum = 0.0
    while n_cum < 1.0 and e > 1e-3:
        derivs = get_tidal_derivatives(e, a, m1, m2)
        dedn = derivs.de_dt * time_in_Myr
        dadn = derivs.da_dt * time_in_Myr
        dn = get_tidal_step_size(dedn, dadn, e, a, step_factor, n_cum)
        n_cum += dn
        e += dedn * dn
        a += dadn * dn
    return TidalEffectResult(e, a)


@njit(cache=True, fastmath=True)
def get_critical_radii(m1: float, m2: float) -> CriticalRadii:
    """
    Compute critical orbital radii for stopping condition evaluation.

    Calculates the tidal disruption radius, hot Jupiter threshold,
    and warm Jupiter threshold based on stellar and planetary masses.

    Args:
        m1: Host star mass (M_sun).
        m2: Planet mass (M_sun).

    Returns:
        CriticalRadii containing R_td, R_hj, and R_wj (all in au).
    """
    R_td = ETA * R_P * (m1 / m2) ** (1 / 3)
    R_hj = (MAX_HJ_PERIOD**2 * (m1 + m2)) ** (1 / 3)
    R_wj = (MAX_WJ_PERIOD**2 * (m1 + m2)) ** (1 / 3)
    return CriticalRadii(R_td, R_hj, R_wj)


@njit(cache=True, fastmath=True)
def get_perturbation_rate(local_n_tot: float, local_sigma_v: float) -> float:
    """
    Compute the average stellar encounter rate.

    Args:
        local_n_tot: Local stellar number density (stars per pc^3 per 10^6).
        local_sigma_v: Local velocity dispersion (au/yr).

    Returns:
        Encounter rate (Myr^-1).
    """
    _ADJUSTED_GAMMA = 3.21
    return (
        _ADJUSTED_GAMMA
        * local_n_tot
        * ((B_MAX / 75) ** 2)
        * (local_sigma_v * np.sqrt(2))
    )
