from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hj import core
from hj.validation import (
    EncounterSettings,
    benchmark_encounters,
    encounter_convergence,
    integrate_encounter,
    summarize_convergence,
)


def fast_encounter():
    return benchmark_encounters(phases=1)[1]


def test_benchmark_is_reproducible_and_names_are_unique():
    first = benchmark_encounters(42, 3)
    assert first == benchmark_encounters(42, 3)
    assert first != benchmark_encounters(43, 3)
    assert len({e.name for e in first}) == len(first) == 18


def test_increasing_phase_count_preserves_existing_encounters():
    small = {e.name: e for e in benchmark_encounters(42, 2)}
    large = {e.name: e for e in benchmark_encounters(42, 8)}
    assert all(large[name] == encounter for name, encounter in small.items())


def test_convergence_summary_reports_tails_and_zero_event_uncertainty():
    results = pd.DataFrame(
        [
            {
                "name": f"penetrating_{index:03d}",
                "xi": 1e-4,
                "epsilon": 1e-9,
                "endpoint_class_disagreement": False,
                "host_bound_disagreement": False,
                "host_bound": index < 3,
                "reference_host_bound": index < 3,
                "endpoint_energy_error_scaled": 1e-14,
                "abs_delta_e_error": error,
                "abs_q_error_over_initial_q": error,
            }
            for index, error in enumerate([0.0, 0.01, 0.02, 0.5])
        ]
    )
    summary = summarize_convergence(results).iloc[0]
    assert summary.phases == 4
    assert summary.abs_delta_e_error_median == pytest.approx(0.015)
    assert summary.abs_delta_e_error_max == 0.5
    assert summary.surviving_pairs == 3
    assert summary.ionised_pairs == 1
    assert summary.surviving_abs_delta_e_error_max == 0.02
    assert summary.class_disagreement_lower == 0
    assert summary.class_disagreement_upper > 0
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_convergence(pd.concat([results, results]))


def test_encounter_phase_is_anchored_at_pericentre():
    encounter = fast_encounter()
    for xi in (1e-3, 1e-5):
        sim, duration = core._create_encounter_simulation(
            *encounter.arguments(),
            encounter.mean_anomaly_at_pericentre,
            xi=xi,
            phase_at_pericentre=True,
        )
        initial_orbit = sim.particles[1].orbit(primary=sim.particles[0])
        phase_at_pericentre = initial_orbit.M + initial_orbit.n * duration / 2
        difference = np.angle(
            np.exp(1j * (phase_at_pericentre - encounter.mean_anomaly_at_pericentre))
        )
        assert abs(difference) < 1e-10


def test_ias15_agrees_with_independent_cartesian_integration():
    encounter = fast_encounter()
    ias15 = integrate_encounter(encounter)
    dop853 = integrate_encounter(encounter, integrator="dop853")
    assert ias15["final_e"] == pytest.approx(dop853["final_e"], abs=1e-8)
    assert ias15["final_a"] == pytest.approx(dop853["final_a"], abs=1e-8)
    assert ias15["final_q"] == pytest.approx(dop853["final_q"], abs=1e-8)
    assert ias15["host_bound"] == dop853["host_bound"]
    assert ias15["endpoint_energy_error_scaled"] < 1e-12
    assert ias15["endpoint_angular_momentum_error_scaled"] < 1e-12
    assert dop853["endpoint_energy_error_scaled"] < 1e-9


def test_production_and_diagnostic_encounters_share_integration():
    encounter = fast_encounter()
    settings = EncounterSettings()
    _, duration = core._create_encounter_simulation(
        *encounter.arguments(),
        0.0,
    )
    mean_motion = np.sqrt(core.G * (encounter.m1 + encounter.m2) / encounter.a**3)
    start_phase = encounter.mean_anomaly_at_pericentre - mean_motion * duration / 2
    delta_e, delta_a = core.nbody_encounter_de(*encounter.arguments(), start_phase)
    diagnostics = integrate_encounter(encounter, settings)
    assert delta_e == pytest.approx(diagnostics["delta_e"], abs=1e-14)
    assert delta_a == pytest.approx(diagnostics["delta_a"], abs=1e-14)


def test_convergence_against_identical_reference_has_zero_error():
    settings = EncounterSettings()
    result = encounter_convergence([fast_encounter()], [settings], settings).iloc[0]
    assert result.abs_delta_e_error == 0
    assert result.abs_q_error_over_initial_q == 0
    assert not result.host_bound_disagreement


@pytest.mark.parametrize(
    "xi,epsilon", [(0, 1e-9), (1, 1e-9), (1e-4, 0), (np.nan, 1e-9)]
)
def test_invalid_numerical_settings_are_rejected(xi, epsilon):
    with pytest.raises(ValueError):
        EncounterSettings(xi, epsilon)


def test_invalid_encounter_is_rejected():
    with pytest.raises(ValueError):
        replace(fast_encounter(), e=1.0)


def test_less_resolved_reference_is_rejected():
    with pytest.raises(ValueError):
        encounter_convergence(
            [fast_encounter()], [EncounterSettings(1e-6)], EncounterSettings()
        )
