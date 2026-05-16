import numpy as np

from hj import core


def test_critical_radii_returns_named_bundle():
    m1 = np.array([1.0, 1.2], dtype=np.float64)
    m2 = np.array([0.001, 0.002], dtype=np.float64)

    critical_radii = core.critical_radii(m1, m2)

    assert isinstance(critical_radii, core.CriticalRadii)
    td, hj, wj = critical_radii
    np.testing.assert_array_equal(td, critical_radii.td)
    np.testing.assert_array_equal(hj, critical_radii.hj)
    np.testing.assert_array_equal(wj, critical_radii.wj)
