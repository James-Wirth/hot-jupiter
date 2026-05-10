from __future__ import annotations

import math

import numpy as np
import rebound
from numba import njit, prange

from hj.config import (
    A_OVER_RH,
    AU_PER_PSC,
    B_MAX,
    CIRCULARISATION_THRESHOLD_ECCENTRICITY,
    ETA,
    K_P,
    M_BR,
    M_MAX,
    M_MIN,
    MAX_HJ_PERIOD,
    MAX_WJ_PERIOD,
    R_P,
    S_MIN,
    T_MIN,
    TAU_P,
    XI,
    G,
)
from hj.state import STOP_UNSET, StopCode

__all__ = [
    "critical_radii",
    "step",
    "recheck_stop",
    "_analytic_encounter_de",
    "nbody_encounter_de",
    "_convert_mean_to_true_anomaly",
    "plummer_kernel_params",
]

# Numerical constants used by njit kernels
_XI_CBRT: float = XI ** (1.0 / 3.0)
_TWO_PI: float = 2.0 * math.pi
_SQRT_2: float = math.sqrt(2.0)
_E_FLOOR: float = 1.0e-3
_EPS15: float = 1.0e-15
_EPS14: float = 1.0e-14
_EPS16: float = 1.0e-16
_ADJUSTED_GAMMA: float = 3.21
_BMAX_FRAC_SQ: float = (B_MAX / 75.0) ** 2

# Inverse-CDF coefficients for the perturber-mass IMF (47 Tuc, Giersz & Heggie 2011)
_M3_A: float = 1.8 / (4.0 * M_BR**0.6 - 3.0 * M_MIN**0.6 - M_BR**2.4 * M_MAX**-1.8)
_M3_B: float = _M3_A * M_BR**2.4
_M3_Y_CRIT: float = (_M3_A / 0.6) * (M_BR**0.6 - M_MIN**0.6)
_M3_MIN_POW: float = M_MIN**0.6
_M3_BR_NEG_POW: float = M_BR**-1.8


# ------------------------------ Kepler Stuff -------------------------------


@njit(cache=True, fastmath=True, inline="always")
def _wrap_angle(x: float) -> float:
    x = (x + math.pi) % _TWO_PI
    if x <= 0.0:
        x += _TWO_PI
    return x - math.pi


@njit(cache=True, fastmath=True)
def _solve_kepler(mean_anom: float, e: float) -> float:
    mean_anom = _wrap_angle(mean_anom)
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
        if abs(denom) > _EPS15:
            ecc_anom -= residual / denom
        else:
            ecc_anom -= ratio
        if abs(residual) < _EPS14:
            break
    return ecc_anom


@njit(cache=True, fastmath=True)
def _convert_mean_to_true_anomaly(mean_anom: float, e: float) -> float:
    ecc_anom = _solve_kepler(mean_anom, e)
    cos_ecc = math.cos(ecc_anom)
    sin_ecc = math.sin(ecc_anom)
    denom = 1.0 - e * cos_ecc
    if abs(denom) < _EPS15:
        denom = _EPS15 if denom >= 0.0 else -_EPS15
    sin_true = math.sqrt(max(0.0, 1.0 - e * e)) * sin_ecc / denom
    cos_true = (cos_ecc - e) / denom
    return _wrap_angle(math.atan2(sin_true, cos_true))


@njit(cache=True, fastmath=True, inline="always")
def _perturber_orbit(
    v_infty: float, b: float, m1: float, m2: float
) -> tuple[float, float, float]:
    """Hyperbolic perturber orbit (a, e, r_p) from v_inf, b, m1, m2."""
    if v_infty <= 0.0 or not math.isfinite(v_infty):
        v_infty = 1.0e-12
    a_pert = -G * (m1 + m2) / (v_infty * v_infty)
    e_pert = math.sqrt(1.0 + (b / a_pert) ** 2)
    r_p = -a_pert * (e_pert - 1.0)
    return a_pert, e_pert, r_p


@njit(cache=True, fastmath=True, inline="always")
def _critical_true_anomaly(a_pert: float, e_pert: float, r_p: float) -> float:
    """True anomaly at r_crit = r_p / XI^(1/3)."""
    r_crit = r_p / _XI_CBRT
    cos_theta = (1.0 / e_pert) * (((a_pert * (1.0 - e_pert * e_pert)) / r_crit) - 1.0)
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    return math.acos(cos_theta)


@njit(cache=True, fastmath=True, inline="always")
def _integration_time(
    a_pert: float, e_pert: float, r_p: float, m1: float, m2: float
) -> float:
    theta_crit = _critical_true_anomaly(a_pert, e_pert, r_p)
    cos_theta = math.cos(theta_crit)
    acosh_arg = (e_pert + cos_theta) / (1.0 + e_pert * cos_theta)
    if acosh_arg < 1.0:
        acosh_arg = 1.0
    hyp_anom = math.acosh(acosh_arg)
    half_time = (e_pert * math.sinh(hyp_anom) - hyp_anom) * math.sqrt(
        (-a_pert) ** 3 / (G * (m1 + m2))
    )
    return 2.0 * half_time


# --------------------------- Analytic Encounters ---------------------------


@njit(cache=True, fastmath=True)
def _analytic_encounter_de(
    v_infty: float,
    b: float,
    lan: float,
    inc: float,
    aop: float,
    e: float,
    a: float,
    m1: float,
    m2: float,
    m3: float,
) -> tuple[bool, float]:
    """delta_e from a single encounter (Heggie & Rasio 1996)."""
    a_pert, e_pert, r_p = _perturber_orbit(v_infty, b, m1, m2)

    valid = True
    if r_p / a <= T_MIN:
        valid = False
    else:
        t_int = _integration_time(a_pert, e_pert, r_p, m1, m2)
        t_per = math.sqrt(a**3 / (m1 + m2))
        if t_int / t_per <= S_MIN:
            valid = False

    m_total = m1 + m2 + m3
    mass_factor = (
        e * math.sqrt(max(0.0, 1.0 - e * e)) * (m3 / math.sqrt((m1 + m2) * m_total))
    )
    alpha = -1.0 * (15.0 / 4.0) * ((1.0 + e_pert) ** -1.5)
    chi = math.acos(-1.0 / e_pert) + math.sqrt(e_pert * e_pert - 1.0)
    psi = (1.0 / 3.0) * (((e_pert * e_pert - 1.0) ** 1.5) / (e_pert * e_pert))

    cos_inc = math.cos(inc)
    sin_2aop = math.sin(2.0 * aop)
    cos_2aop = math.cos(2.0 * aop)
    sin_2lan = math.sin(2.0 * lan)
    cos_2lan = math.cos(2.0 * lan)

    theta1 = (1.0 - cos_inc * cos_inc) * sin_2lan
    theta2 = (1.0 + cos_inc * cos_inc) * cos_2aop * sin_2lan
    theta3 = 2.0 * cos_inc * sin_2aop * cos_2lan

    delta_e = (
        alpha
        * mass_factor
        * ((a / r_p) ** 1.5)
        * (theta1 * chi + (theta2 + theta3) * psi)
    )
    return valid, delta_e


# -------------------------------- IMF (m3) ---------------------------------


@njit(cache=True, fastmath=True, inline="always")
def _sample_m3(u: float) -> float:
    """47 Tuc IMF (Giersz & Heggie 2011)."""
    if u <= _M3_Y_CRIT:
        return ((0.6 * u) / _M3_A + _M3_MIN_POW) ** (1.0 / 0.6)
    return (_M3_BR_NEG_POW + (1.8 / _M3_B) * (_M3_Y_CRIT - u)) ** (-1.0 / 1.8)


# -------------------------- Tidal Circularisation --------------------------


@njit(cache=True, fastmath=True, inline="always")
def _tidal_f_factor(e: float) -> float:
    """Eccentricity enhancement factor (Hut 1981)."""
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
        + (255.0 / 448.0) * e8
        + (25.0 / 1792.0) * e10
    )
    denom = 1.0 + 3.0 * e2 + (3.0 / 8.0) * e4
    return num / denom


@njit(cache=True, fastmath=True, inline="always")
def _tidal_derivs(e: float, a: float, m1: float, m2: float) -> tuple[float, float]:
    """(de/dt, da/dt) due to tidal circularisation (Hut 1981)."""
    mass_ratio = m2 / m1
    mean_motion = 1.0e6 * math.sqrt(G * (1.0 + mass_ratio) * m1 / a**3)
    e2 = e * e
    one_minus_e2 = 1.0 - e2
    common = (
        -21.0
        * K_P
        * TAU_P
        * (mean_motion * mean_motion)
        / mass_ratio
        * (R_P / a) ** 5
        * _tidal_f_factor(e)
    )
    de_dt = common * e / (one_minus_e2 ** (13.0 / 2.0)) / 2.0
    da_dt = common * a * e2 / (one_minus_e2 ** (15.0 / 2.0))
    return de_dt, da_dt


@njit(cache=True, fastmath=True)
def _apply_tidal(
    e: float, a: float, m1: float, m2: float, dt_total: float
) -> tuple[float, float]:
    """Adaptive RK2 tidal evolution."""
    step_factor = 0.05
    t_remaining = dt_total
    while t_remaining > 0.0 and e > _E_FLOOR:
        de1, da1 = _tidal_derivs(e, a, m1, m2)
        rel_e = abs(de1) / max(abs(e), _EPS16)
        rel_a = abs(da1) / max(abs(a), _EPS16)
        max_rate = rel_e if rel_e > rel_a else rel_a
        if max_rate > 0.0:
            dt = step_factor / max_rate
            if dt > t_remaining:
                dt = t_remaining
        else:
            dt = t_remaining
        de2, da2 = _tidal_derivs(e + 0.5 * de1 * dt, a + 0.5 * da1 * dt, m1, m2)
        e = e + de2 * dt
        a = a + da2 * dt
        t_remaining -= dt
    return e, a


# ----------------------------- Critical Radii ------------------------------


def critical_radii(
    m1: np.ndarray, m2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_td = ETA * R_P * (m1 / m2) ** (1.0 / 3.0)
    R_hj = (MAX_HJ_PERIOD**2 * (m1 + m2)) ** (1.0 / 3.0)
    R_wj = (MAX_WJ_PERIOD**2 * (m1 + m2)) ** (1.0 / 3.0)
    return R_td, R_hj, R_wj


# --------------------------- Stopping Conditions ---------------------------


@njit(cache=True, fastmath=True, inline="always")
def _classify_stop(
    e: float,
    a: float,
    t: float,
    R_td: float,
    R_hj: float,
    R_wj: float,
    time_total: float,
) -> int:
    if e >= 1.0:
        return StopCode.ION.value
    if a * (1.0 - e) < R_td:
        return StopCode.TD.value
    cir = e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY
    timeout = t >= time_total
    if a < R_hj and (cir or timeout):
        return StopCode.HJ.value
    if R_hj < a < R_wj and (cir or timeout):
        return StopCode.WJ.value
    if cir or timeout:
        return StopCode.NM.value
    return STOP_UNSET


# ------------------------------ Plummer (Fast) -----------------------------


@njit(cache=True, fastmath=True, inline="always")
def _plummer_radius_and_env(
    plummer_static: np.ndarray, lagrange: float, t: float
) -> tuple[float, float, float]:
    """(r, n_tot, sigma_v) at Lagrangian mass fraction `lagrange` and time t."""
    rh = (plummer_static[0] ** 1.5 + plummer_static[1] * t) ** (2.0 / 3.0)
    a_t = A_OVER_RH * rh
    u = lagrange ** (2.0 / 3.0)
    r = math.sqrt(u / (1.0 - u)) * a_t
    rho = (3.0 * plummer_static[2] / (4.0 * math.pi * a_t**3)) * (
        1.0 + (r / a_t) ** 2
    ) ** -2.5
    n_tot = rho * plummer_static[3]
    r_au = r * AU_PER_PSC
    a_au = a_t * AU_PER_PSC
    M_t = plummer_static[4] + (plummer_static[5] - plummer_static[4]) * (
        t / plummer_static[6]
    )
    sigma_v = math.sqrt(G * M_t / (6.0 * math.sqrt(r_au * r_au + a_au * a_au)))
    return r, n_tot, sigma_v


# ------------------------------- Step Kernel -------------------------------


@njit(parallel=True, cache=True, fastmath=True)
def step(
    e_arr: np.ndarray,
    a_arr: np.ndarray,
    m1_arr: np.ndarray,
    m2_arr: np.ndarray,
    lagrange_arr: np.ndarray,
    t_arr: np.ndarray,
    stop_code_arr: np.ndarray,
    stop_time_arr: np.ndarray,
    plummer_static: np.ndarray,
    R_td_arr: np.ndarray,
    R_hj_arr: np.ndarray,
    R_wj_arr: np.ndarray,
    time_total: float,
    hybrid_switch: bool,
    u_wt: np.ndarray,
    u_b: np.ndarray,
    u_lan: np.ndarray,
    u_aop: np.ndarray,
    u_inc: np.ndarray,
    u_m3: np.ndarray,
    n_x: np.ndarray,
    n_y: np.ndarray,
    n_z: np.ndarray,
    needs_nbody: np.ndarray,
    enc_v: np.ndarray,
    enc_b: np.ndarray,
    enc_lan: np.ndarray,
    enc_inc: np.ndarray,
    enc_aop: np.ndarray,
    enc_m3: np.ndarray,
) -> None:
    """
    For each system...
      (1) Evaluate n_tot, sigma_v for the cluster at time t.
      (2) Sample waiting time, advance t and apply tidal RK2.
      (3) Check stopping conditions.
      (4) Sample one perturbing encounter.
      (5) If analytic valid (or hybrid_switch=False): commit delta_e, check stopping conditions.
      (6) Else: mark system as needing N-body, store encounter params for the worker.
    """
    N = e_arr.shape[0]
    for i in prange(N):
        if stop_code_arr[i] != STOP_UNSET:
            continue

        t = t_arr[i]
        e = e_arr[i]
        a = a_arr[i]
        m1 = m1_arr[i]
        m2 = m2_arr[i]
        lagrange = lagrange_arr[i]
        R_td = R_td_arr[i]
        R_hj = R_hj_arr[i]
        R_wj = R_wj_arr[i]

        _, n_tot, sigma_v = _plummer_radius_and_env(plummer_static, lagrange, t)

        rate = _ADJUSTED_GAMMA * n_tot * _BMAX_FRAC_SQ * (sigma_v * _SQRT_2)
        wt = -math.log(u_wt[i]) / rate
        new_t = t + wt
        if new_t > time_total:
            new_t = time_total

        e, a = _apply_tidal(e, a, m1, m2, wt)
        t_arr[i] = new_t
        e_arr[i] = e
        a_arr[i] = a

        code = _classify_stop(e, a, new_t, R_td, R_hj, R_wj, time_total)
        if code != STOP_UNSET:
            stop_code_arr[i] = code
            stop_time_arr[i] = new_t
            continue

        sigma_rel = sigma_v * _SQRT_2
        v_inf = (
            math.sqrt(n_x[i] * n_x[i] + n_y[i] * n_y[i] + n_z[i] * n_z[i]) * sigma_rel
        )
        b = B_MAX * math.sqrt(u_b[i])
        lan = u_lan[i] * _TWO_PI
        aop = u_aop[i] * _TWO_PI
        inc = math.acos(1.0 - 2.0 * u_inc[i])
        m3 = _sample_m3(u_m3[i])

        valid, de = _analytic_encounter_de(v_inf, b, lan, inc, aop, e, a, m1, m2, m3)

        if valid or not hybrid_switch:
            e_new = e + de
            e_arr[i] = e_new
            code = _classify_stop(e_new, a, new_t, R_td, R_hj, R_wj, time_total)
            if code != STOP_UNSET:
                stop_code_arr[i] = code
                stop_time_arr[i] = new_t
        else:
            needs_nbody[i] = True
            enc_v[i] = v_inf
            enc_b[i] = b
            enc_lan[i] = lan
            enc_inc[i] = inc
            enc_aop[i] = aop
            enc_m3[i] = m3


@njit(parallel=True, cache=True, fastmath=True)
def recheck_stop(
    idx: np.ndarray,
    e_arr: np.ndarray,
    a_arr: np.ndarray,
    t_arr: np.ndarray,
    stop_code_arr: np.ndarray,
    stop_time_arr: np.ndarray,
    R_td_arr: np.ndarray,
    R_hj_arr: np.ndarray,
    R_wj_arr: np.ndarray,
    time_total: float,
) -> None:
    """For systems whose state was updated by the N-body path."""
    M = idx.shape[0]
    for k in prange(M):
        i = idx[k]
        code = _classify_stop(
            e_arr[i],
            a_arr[i],
            t_arr[i],
            R_td_arr[i],
            R_hj_arr[i],
            R_wj_arr[i],
            time_total,
        )
        if code != STOP_UNSET:
            stop_code_arr[i] = code
            stop_time_arr[i] = t_arr[i]


# --------------------------------- N-body ----------------------------------


def nbody_encounter_de(
    v_infty: float,
    b: float,
    lan: float,
    inc: float,
    aop: float,
    e: float,
    a: float,
    m1: float,
    m2: float,
    m3: float,
    mean_anom: float,
) -> tuple[float, float]:
    """REBOUND (IAS15) integration of one 3-body encounter."""
    a_pert, e_pert, r_p = _perturber_orbit(v_infty, b, m1, m2)
    t_int = _integration_time(a_pert, e_pert, r_p, m1, m2)
    f0 = -_critical_true_anomaly(a_pert, e_pert, r_p)
    f_phase = _convert_mean_to_true_anomaly(mean_anom, e)

    sim = rebound.Simulation()
    sim.G = G
    sim.add(m=m1)
    sim.add(m=m2, a=a, e=e, f=f_phase)
    sim.add(m=m3, a=a_pert, e=e_pert, f=f0, Omega=lan, inc=inc, omega=aop)
    sim.move_to_com()
    sim.integrate(t_int)

    o = sim.particles[1].orbit()
    return o.e - e, o.a - a


def plummer_kernel_params(plummer) -> np.ndarray:
    return np.array(
        [
            plummer.R0,
            plummer.A,
            plummer.M_fixed,
            1.0 / (plummer.M_avg * 1.0e6),
            plummer.M0,
            plummer.M1,
            plummer.T_AGE,
        ],
        dtype=np.float64,
    )
