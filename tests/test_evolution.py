import numpy as np

from hj.clusters import Plummer
from hj.evolution import StopCode, run_simulation, sample_initial_conditions
from hj.state import STOP_UNSET


def test_sample_initial_conditions_reproducible():
    cluster = Plummer()
    seed = 12345
    s1 = sample_initial_conditions(16, cluster, np.random.default_rng(seed))
    s2 = sample_initial_conditions(16, cluster, np.random.default_rng(seed))
    for col in ("e_init", "a_init", "m1", "m2", "lagrange", "e", "a"):
        np.testing.assert_array_equal(getattr(s1, col), getattr(s2, col))
    assert np.array_equal(s1.e, s1.e_init)
    assert np.array_equal(s1.a, s1.a_init)
    assert (s1.stop_code == STOP_UNSET).all()


def test_run_simulation_terminates_all_systems():
    cluster = Plummer()
    state = sample_initial_conditions(64, cluster, np.random.default_rng(1))
    run_simulation(
        state,
        cluster,
        time_total=200.0,
        rng=np.random.default_rng(2),
        hybrid_switch=False,
        n_jobs=1,
    )
    assert (state.stop_code != STOP_UNSET).all()
    assert set(state.stop_code.tolist()).issubset({sc.value for sc in StopCode})


def test_run_simulation_hybrid_mode_terminates():
    cluster = Plummer()
    state = sample_initial_conditions(16, cluster, np.random.default_rng(3))
    run_simulation(
        state,
        cluster,
        time_total=200.0,
        rng=np.random.default_rng(4),
        hybrid_switch=True,
        n_jobs=1,
    )
    assert (state.stop_code != STOP_UNSET).all()
    assert set(state.stop_code.tolist()).issubset({sc.value for sc in StopCode})


def test_run_simulation_deterministic_under_same_seed():
    cluster = Plummer()
    seed = 7

    s_a = sample_initial_conditions(64, cluster, np.random.default_rng(seed))
    run_simulation(
        s_a,
        cluster,
        time_total=200.0,
        rng=np.random.default_rng(seed + 1),
        hybrid_switch=False,
        n_jobs=1,
    )

    s_b = sample_initial_conditions(64, cluster, np.random.default_rng(seed))
    run_simulation(
        s_b,
        cluster,
        time_total=200.0,
        rng=np.random.default_rng(seed + 1),
        hybrid_switch=False,
        n_jobs=1,
    )

    for col in ("e", "a", "stop_code", "stop_time", "t"):
        np.testing.assert_array_equal(getattr(s_a, col), getattr(s_b, col))
