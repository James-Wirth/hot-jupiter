import pytest

from hjmodel import core

def test_true_anomaly_approximation_matches_for_small_e():
    mean_anom = 0.5
    e = 0.1
    approx = core.true_anomaly_approximation(mean_anom, e)
    true = core.get_true_anomaly(mean_anom, e, e_cutoff=0.3)
    assert pytest.approx(approx, rel=1e-6) == true

def test_is_analytic_valid_thresholds(monkeypatch):
    monkeypatch.setattr(core, "tidal_param", lambda **kwargs: core.T_MIN + 10, raising=False)
    monkeypatch.setattr(core, "slow_param", lambda **kwargs: core.S_MIN + 10, raising=False)
    assert core.is_analytic_valid(v_infty=1, b=1, a=1, m1=1, m2=1) is True

    monkeypatch.setattr(core, "tidal_param", lambda **kwargs: core.T_MIN - 10, raising=False)
    monkeypatch.setattr(core, "slow_param", lambda **kwargs: core.S_MIN - 10, raising=False)
    assert core.is_analytic_valid(v_infty=1, b=1, a=1, m1=1, m2=1) is False

def test_tidal_effect_non_increasing():
    e0 = 0.5
    a0 = 10.0
    m1 = 1.0
    m2 = 1e-3
    e_new, a_new = core.tidal_effect(
        e=e0, a=a0, m1=m1, m2=m2, time_in_Myr=0.1, C=0.01
    )
    assert e_new <= e0 + 1e-12
    assert a_new <= a0 + 1e-12

def test_tidal_effect_strict_decrease_when_derivatives_negative(monkeypatch):
    monkeypatch.setattr(core, "de_tid_dt", lambda e, a, m1, m2: -1.0)
    monkeypatch.setattr(core, "da_tid_dt", lambda e, a, m1, m2: -2.0)
    e0 = 0.5
    a0 = 10.0
    m1 = 1.0
    m2 = 1e-3
    e_new, a_new = core.tidal_effect(
        e=e0, a=a0, m1=m1, m2=m2, time_in_Myr=0.1, C=0.01
    )
    assert e_new < e0
    assert a_new < a0
