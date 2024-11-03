from hjmodel.config import *
import numpy as np
import rebound
from scipy.optimize import fsolve

# uniform range of possible starting mean anomalies for planetary orbit
mean_anoms = np.linspace(-np.pi, np.pi, num=INIT_PHASES, endpoint=False)

def kepler(E: float, e: float) -> float:
    """
    Calculates the mean anomaly corresponding to the
    eccentricity anomaly E for an orbit with eccentricity e
    """
    return E - e*np.sin(E)

def true_anomaly_approximation(mean_anom: float, e: float) -> float:
    T1 = mean_anom
    T2 = (2*e - 0.25 * e**3) * np.sin(mean_anom)
    T3 = ((5/4) * e**2) * np.sin(2*mean_anom)
    T4 = ((13/12) * e**3) * np.sin(3*mean_anom)
    return T1 + T2 + T3 + T4

def get_true_anomaly(mean_anom: float, e:float) -> float:
    """
    Calculates the true anomaly corresponding to the
    mean anomaly mean_anom for an orbit with eccentricity e
    """
    if e < 0.3:
        return true_anomaly_approximation(mean_anom, e)
    else:
        ecc_anom, *info = fsolve(lambda E: kepler(E, e) - mean_anom, 0)
        beta = e / (1 + np.sqrt(1 - e ** 2))
        return ecc_anom + 2 * np.arctan((beta * np.sin(ecc_anom)) / (1 - beta * np.cos(ecc_anom)))

def get_pert_orbit_params(v_infty: float, b: float, m1: float, m2: float) -> (float, float, float):
    a_pert = -G * (m1 + m2) / (v_infty ** 2)
    e_pert = np.sqrt(1 + (b / a_pert) ** 2)
    rp = -a_pert * (e_pert - 1)
    return a_pert, e_pert, rp

def get_int_params(a_pert: float, e_pert: float, rp: float):
    # max true anomaly (at infinity)
    max_anomaly = np.arccos(-1 / e_pert)
    # boundary points of perturbing trajectory calculated from parameter XI
    r_crit = rp / (XI ** (1 / 3))
    theta_crit = np.arccos((1 / e_pert) * (((a_pert * (1 - e_pert ** 2)) / r_crit) - 1))
    alpha = theta_crit / max_anomaly
    # t = time until peri-center
    F = np.arccosh((e_pert + np.cos(alpha * max_anomaly)) / (1 + e_pert * np.cos(alpha * max_anomaly)))
    t = (e_pert * np.sinh(F) - F) * (-1 * a_pert) ** (3 / 2)
    # t_int, -f0
    return 2*t, -alpha*max_anomaly

def de_HR(v_infty: float, b: float, Omega: float, inc: float, omega: float,
          e: float, a: float, m1: float, m2: float, m3:float) -> float:
    """
    Calculates the analytic Heggie-Rasio eccentricity excitation for an encounter

    Inputs
    ----------
    v_infty: float              Asymptotic relative speed               au per year
    b: float                    Impact parameter                        au
    Omega: float                Longitude of ascending node
    inc: float                  Inclination
    omega: float                Arguyment of periapsis
    e: float                    Eccentricity before encounter           arbitrary units
    a: float                    Semi-major axis before encounter        au
    m1: float                   Host mass                               M_solar
    m2: float                   Planet mass                             M_solar
    m3: float                   Perturbing mass                         M_solar

    Returns
    ----------
    de: float                   Analytic eccentricity excitation        arbitrary units
    """

    m123 = m1 + m2 + m3
    a_pert, e_pert, rp = get_pert_orbit_params(v_infty, b, m1, m2)
    y = e * np.sqrt(1 - e ** 2) * (m3 / (np.sqrt((m1 + m2) * m123)))
    alpha = -1 * (15/4) * ((1+e_pert)**(-3/2))
    chi = np.arccos(-1/e_pert) + np.sqrt(e_pert**2-1)
    psi = (1/3) * (((e_pert**2-1)**(3/2))/(e_pert**2))
    Theta1 = (np.sin(inc)**2) * np.sin(2 * Omega)
    Theta2 = (1 + (np.cos(inc))**2) * np.cos(2 * omega) * np.sin(2 * Omega)
    Theta3 = 2 * np.cos(inc) * np.sin(2 * omega) * np.cos(2 * Omega)
    return alpha*y*((a / rp) ** (3 / 2))*(Theta1 * chi + (Theta2 + Theta3) * psi)

def de_sim(v_infty: float, b: float, Omega: float, inc: float, omega: float,
           e: float, a: float, m1: float, m2: float, m3: float) -> (float, float):
    """
    Calculates the N-body eccentricity and semi-major axis excitations
    using the REBOUND code

    Inputs
    ----------
    v_infty: float              Asymptotic relative speed               arbitrary units
    b: float                    Impact parameter                        au
    Omega: float                Longitude of ascending node
    inc: float                  Inclination
    omega: float                Arguyment of periapsis
    e: float                    Eccentricity before encounter           arbitrary units
    a: float                    Semi-major axis before encounter        au
    m1: float                   Host mass                               M_solar
    m2: float                   Planet mass                             M_solar
    m3: float                   Perturbing mass                         M_solar

    Returns
    ----------
    de: float                   REBOUND eccentricity excitation         arbitrary units
    da: float                   REBOUND semi-major axis excitation      au
    """

    # calculate orbital parameters of perturbing trajectory
    a_pert, e_pert, rp = get_pert_orbit_params(v_infty=v_infty, b=b, m1=m1, m2=m2)
    t_int, f0 = get_int_params(a_pert=a_pert, e_pert=e_pert, rp=rp)

    f_phase = get_true_anomaly(np.random.choice(mean_anoms), e)

    # configure REBOUND simulation
    sim = rebound.Simulation()
    sim.add(m=m1)
    sim.add(m=m2, a=a, e=e, f=f_phase)
    sim.add(m=m3, a=a_pert, e=e_pert, f=f0, Omega=Omega, inc=inc, omega=omega)
    sim.move_to_com()
    sim.integrate(t_int)

    # calculate final orbital parameters of planetary system
    o_binary = sim.particles[1].orbit()
    delta_e_sim = o_binary.e - e
    delta_a_sim = o_binary.a - a

    # deallocate simulation from memory
    sim = None

    return delta_e_sim, delta_a_sim

def tidal_param(v_infty: float, b: float, a: float, m1: float, m2: float) -> bool:
    _, _, rp = get_pert_orbit_params(v_infty, b, m1, m2)
    return rp/a

def slow_param(v_infty: float, b: float, a: float, m1: float, m2: float) -> bool:
    a_pert, e_pert, rp = get_pert_orbit_params(v_infty, b, m1, m2)
    t_int, _ = get_int_params(a_pert=a_pert, e_pert=e_pert, rp=rp)
    t_per = (a**3 / (m1 + m2))**(1/2)
    return t_int/t_per

def is_analytic_valid(v_infty: float, b: float, Omega: None, inc: None, omega: None,
          e: None, a: float, m1: float, m2: float, m3:float, sigma_v: float) -> bool:
    return True if (tidal_param(v_infty=v_infty, b=b, a=a, m1=m1, m2=m2) > T_MIN
                    and slow_param(v_infty=v_infty, b=b, a=a, m1=m1, m2=m2) > S_MIN) else False

def f(e: float) -> float:
    num_coeffs = np.array([1, 45/14, 8, 685/224, 255/488, 25/1792])
    denom_coeffs = np.array([1, 3, 3/8])
    vec = np.array([e**(2*i) for i in range(6)])
    return np.dot(num_coeffs, vec)/np.dot(denom_coeffs, vec[:3])

# per Myr
def de_tid_dt(e: float, a: float, m1: float, m2: float) -> float:
    q = m2 / m1
    n = 10 ** 6 * np.sqrt(G * (1 + q) * m1 / a ** 3)  # per Myr
    e_dot_tide = -21 / 2 * K_P * TAU_P * (n ** 2) * (q ** -1) * (R_P / a) ** 5 * f(e) * e / (1 - e ** 2) ** (13 / 2)
    return e_dot_tide

# per Myr
def da_tid_dt(e: float, a: float, m1: float, m2: float) -> float:
    q = m2 / m1
    n = 10 ** 6 * np.sqrt(G * (1 + q) * m1 / a ** 3)  # per Myr
    a_dot_tide = -21 * K_P * TAU_P * (n ** 2) * (q ** -1) * (R_P / a) ** 5 * f(e) * (a * e ** 2) / (1 - e ** 2) ** (15 / 2)
    return a_dot_tide

def get_dn(dedn: float, dadn: float, e: float, a: float, C: float, n_cum: float) -> float:
    return min(1 - n_cum, C / max(np.abs(dedn) / e, np.abs(dadn) / a))

def tidal_effect(e: float, a: float, m1: float, m2: float, time_in_Myr: float) -> (float, float):
    # let n be normalised time (i.e. n(0) = 0, n(time_in_Myr) = 1)
    n_cum = 0
    while n_cum < 1 and e > 1E-3:
        dedn = de_tid_dt(e=e, a=a, m1=m1, m2=m2) * time_in_Myr
        dadn = da_tid_dt(e=e, a=a, m1=m1, m2=m2) * time_in_Myr
        dn = get_dn(dedn, dadn, a, e, 0.01, n_cum)
        n_cum += dn
        e += dedn * dn
        a += dadn * dn
    return e, a

def get_critical_radii(m1, m2):
    R_td = ETA * R_P * (m1 / m2) ** (1 / 3)
    R_hj = (MAX_HJ_PERIOD ** 2 * (m1 + m2)) ** (1 / 3)
    R_wj = (MAX_WJ_PERIOD ** 2 * (m1 + m2)) ** (1 / 3)
    return R_td, R_hj, R_wj

def get_perts_per_Myr(local_n_tot, local_sigma_v):
    return 3.21 * local_n_tot * ((B_MAX / 75) ** 2) * (local_sigma_v * np.sqrt(2))