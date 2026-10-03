import json

import numpy as np
import pytest

from hj import HJModel, NumericalSettings, Plummer, core, evolution


@pytest.mark.parametrize(
    "field,value",
    [
        ("encounter_xi", 0),
        ("encounter_xi", 1),
        ("ias15_epsilon", np.nan),
        ("tidal_step_fraction", -0.1),
        ("tidal_step_fraction", 1),
        ("b_max", 0),
        ("slow_threshold", np.inf),
        ("phase_at_pericentre", "yes"),
    ],
)
def test_invalid_numerical_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        NumericalSettings(**{field: value})


def test_impact_cutoff_changes_rate_and_encounter_distribution_together():
    cluster = Plummer()
    variates = evolution._sample_encounter_variates(1, np.random.default_rng(42))
    times, impacts = [], []
    for cutoff in (37.5, 75.0):
        state = evolution.sample_initial_conditions(
            1, cluster, np.random.default_rng(2)
        )
        state.lagrange[:] = 0.2
        params = evolution._empty_encounter_params(1)
        core.step(
            state.e,
            state.a,
            state.m1,
            state.m2,
            state.lagrange,
            state.t,
            state.stop_code,
            state.stop_time,
            core.plummer_kernel_params(cluster),
            core.critical_radii(state.m1, state.m2),
            12000.0,
            True,
            variates,
            params,
            b_max=cutoff,
            tidal_threshold=1e12,
        )
        assert params.needs_nbody[0]
        times.append(state.t[0])
        impacts.append(params.b[0])
    assert times[0] == pytest.approx(4 * times[1])
    assert impacts[1] == pytest.approx(2 * impacts[0])


def test_switching_threshold_changes_validity_without_changing_analytic_kick():
    args = (2.0, 60.0, 1.0, 1.0, 1.0, 0.3, 1.0, 0.5, 0.001, 0.5)
    valid, kick = core._analytic_encounter_de(*args)
    invalid, same_kick = core._analytic_encounter_de(*args, tidal_threshold=1e12)
    assert valid
    assert not invalid
    assert kick == same_kick


def test_nbody_worker_receives_requested_numerics(monkeypatch):
    captured = {}

    def integrate(*args, **kwargs):
        captured.update(kwargs)
        return 0.0, 0.0

    monkeypatch.setattr(core, "nbody_encounter_de", integrate)
    numerics = NumericalSettings(
        encounter_xi=1e-7, ias15_epsilon=1e-12, phase_at_pericentre=True
    )
    evolution._nbody_one((1.0,) * 11, numerics)
    assert captured == {"xi": 1e-7, "epsilon": 1e-12, "phase_at_pericentre": True}


def test_population_tidal_refinement_improves_angular_momentum_conservation():
    errors = []
    for fraction in (0.05, 0.005):
        cluster = Plummer()
        state = evolution.sample_initial_conditions(
            1, cluster, np.random.default_rng(1)
        )
        state.e[:] = 0.98
        state.a[:] = 1.0
        state.m1[:] = 0.5
        state.m2[:] = 0.001
        state.lagrange[:] = 0.99
        evolution.run_simulation(
            state,
            cluster,
            100.0,
            np.random.default_rng(2),
            hybrid_switch=False,
            n_jobs=1,
            numerics=NumericalSettings(tidal_step_fraction=fraction),
        )
        errors.append(abs(state.a[0] * (1 - state.e[0] ** 2) / (1 - 0.98**2) - 1))
    assert errors[1] < errors[0] / 10


def test_population_metadata_records_custom_controls(tmp_path):
    model = HJModel("custom", tmp_path)
    numerics = NumericalSettings(
        encounter_xi=1e-6, tidal_step_fraction=0.01, b_max=100.0
    )
    model.run(0.0, 4, Plummer(), seed=42, n_jobs=1, numerics=numerics)
    metadata = json.loads((tmp_path / "custom/run_000/metadata.json").read_text())
    assert NumericalSettings(**metadata["numerics"]) == numerics
