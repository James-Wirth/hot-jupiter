import pytest

from hjmodel import core
from hjmodel.config import CIRCULARISATION_THRESHOLD_ECCENTRICITY, StopCode
from hjmodel.evolution import PlanetarySystem, check_stopping_conditions


def test_check_stopping_conditions_various():
    m1, m2 = 1.0, 1e-3
    R_td, R_hj, R_wj = core.get_critical_radii(m1=m1, m2=m2)
    total_time = 10.0

    # Ionisation
    code = check_stopping_conditions(
        e=1.0, a=10.0, t=0.0, R_td=R_td, R_hj=R_hj, R_wj=R_wj, total_time=total_time
    )
    assert code == StopCode.ION

    # Tidal disruption
    e = 0.1
    a = (R_td / (1 - e)) * 0.9
    code = check_stopping_conditions(
        e=e, a=a, t=0.0, R_td=R_td, R_hj=R_hj, R_wj=R_wj, total_time=total_time
    )
    assert code == StopCode.TD

    # Hot Jupiter
    e = CIRCULARISATION_THRESHOLD_ECCENTRICITY / 2
    a = R_hj * 0.9
    code = check_stopping_conditions(
        e=e, a=a, t=0.0, R_td=R_td, R_hj=R_hj, R_wj=R_wj, total_time=total_time
    )
    assert code == StopCode.HJ

    # Warm Jupiter
    a = (R_hj + R_wj) / 2
    code = check_stopping_conditions(
        e=e, a=a, t=0.0, R_td=R_td, R_hj=R_hj, R_wj=R_wj, total_time=total_time
    )
    assert code == StopCode.WJ

    # No migration
    a = R_wj * 2
    code = check_stopping_conditions(
        e=e, a=a, t=0.0, R_td=R_td, R_hj=R_hj, R_wj=R_wj, total_time=total_time
    )
    assert code == StopCode.NM


def test_planetary_system_sampling_reproducible():
    lagrange = 1.0
    seed = 12345
    ps1 = PlanetarySystem.sample(lagrange=lagrange, system_seed=seed)
    ps2 = PlanetarySystem.sample(lagrange=lagrange, system_seed=seed)
    assert ps1.e_init == pytest.approx(ps2.e_init)
    assert ps1.a_init == pytest.approx(ps2.a_init)
    assert ps1.m1 == pytest.approx(ps2.m1)
    assert ps1.m2 == pytest.approx(ps2.m2)
    assert ps1.e == pytest.approx(ps2.e)
    assert ps1.a == pytest.approx(ps2.a)
