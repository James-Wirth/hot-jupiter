import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from hj import core


def test_unperturbed_high_eccentricity_binary_preserves_orbit():
    delta_e, delta_a = core.nbody_encounter_de(
        10.0, 10.0, 1.0, 1.0, 1.0, 0.99, 1.0, 0.5, 0.001, 0.0, 1e-6
    )
    assert abs(delta_e) < 1e-11
    assert abs(delta_a) < 1e-10


def test_tidal_refinement_converges_to_independent_solver():
    initial = np.array([0.98, 1.0])
    reference = solve_ivp(
        lambda time, orbit: core._tidal_derivs(*orbit, 0.5, 0.001),
        (0.0, 100.0),
        initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    assert reference.success
    errors = []
    angular_errors = []
    for step_fraction in (0.05, 0.01, 0.002):
        actual = np.array(core._apply_tidal(*initial, 0.5, 0.001, 100.0, step_fraction))
        errors.append(np.linalg.norm(actual - reference.y[:, -1]))
        angular_errors.append(abs(actual[1] * (1 - actual[0] ** 2) / (1 - 0.98**2) - 1))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-5
    assert angular_errors[2] < angular_errors[0]


@pytest.mark.parametrize("eccentricity", [0.0, 0.3, 0.98, 0.9999, 0.999999])
@pytest.mark.parametrize("mean_anomaly", [0.0, 1e-10, 1e-6, 0.001, 0.5, np.pi])
def test_kepler_solution_matches_bracketed_reference(eccentricity, mean_anomaly):
    expected = brentq(
        lambda anomaly: anomaly - eccentricity * np.sin(anomaly) - mean_anomaly,
        0.0,
        np.pi,
        xtol=1e-15,
    )
    actual = core._solve_kepler(mean_anomaly, eccentricity)
    assert actual == pytest.approx(expected, abs=2e-11)
    assert core._solve_kepler(-mean_anomaly, eccentricity) == pytest.approx(
        -actual, abs=2e-11
    )


def test_critical_radii_returns_named_bundle():
    m1 = np.array([1.0, 1.2], dtype=np.float64)
    m2 = np.array([0.001, 0.002], dtype=np.float64)

    critical_radii = core.critical_radii(m1, m2)

    assert isinstance(critical_radii, core.CriticalRadii)
    td, hj, wj = critical_radii
    np.testing.assert_array_equal(td, critical_radii.td)
    np.testing.assert_array_equal(hj, critical_radii.hj)
    np.testing.assert_array_equal(wj, critical_radii.wj)
