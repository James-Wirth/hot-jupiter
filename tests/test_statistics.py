import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, t

from hj.statistics import (
    binomial_sample_size,
    outcome_intervals,
    paired_replicate_comparison,
    replicate_intervals,
    replicate_probabilities,
)


def population(hj_counts):
    return pd.DataFrame(
        [
            {
                "replicate_id": str(batch),
                "system_id": system,
                "stop_code": 3 if system < count else 0,
            }
            for batch, count in enumerate(hj_counts)
            for system in range(10)
        ]
    )


@pytest.mark.parametrize("method", ["wilson", "exact"])
def test_outcome_intervals_match_binomial_reference(method):
    codes = np.array([3] * 58 + [0] * (9982 - 58))
    actual = outcome_intervals(codes, method=method).loc["HJ"]
    expected = binomtest(58, 9982).proportion_ci(method=method)
    assert actual["count"] == 58
    assert actual["total"] == 9982
    assert actual.lower == expected.low
    assert actual.upper == expected.high


def test_zero_events_have_a_positive_upper_limit():
    row = outcome_intervals(np.zeros(100, dtype=int), method="exact").loc["HJ"]
    assert row.lower == 0
    assert row.upper == pytest.approx(1 - 0.025**0.01)


def test_empty_selection_is_not_zero_probability():
    result = outcome_intervals([])
    assert result.probability.isna().all()
    assert result.lower.isna().all()
    assert (result.total == 0).all()


@pytest.mark.parametrize("codes", [[-1, 0], [5], [np.nan], [[0, 1]]])
def test_incomplete_or_invalid_outcomes_are_rejected(codes):
    with pytest.raises(ValueError):
        outcome_intervals(codes)


def test_replicate_intervals_use_between_run_variance():
    row = replicate_intervals(population([1, 2, 3, 4])).loc["HJ"]
    rates = np.array([0.1, 0.2, 0.3, 0.4])
    error = rates.std(ddof=1) / 2
    assert row.probability == pytest.approx(0.25)
    assert row.standard_error == pytest.approx(error)
    assert row.lower == pytest.approx(0.25 - t.ppf(0.975, 3) * error)
    assert row.upper == pytest.approx(0.25 + t.ppf(0.975, 3) * error)


def test_paired_comparison_aligns_replicates_before_differencing():
    treatment = population([2, 5, 5, 8]).sample(frac=1, random_state=42)
    baseline = population([1, 2, 3, 4])
    row = paired_replicate_comparison(treatment, baseline).loc["HJ"]
    differences = np.array([0.1, 0.3, 0.2, 0.4])
    assert row.difference == pytest.approx(differences.mean())
    assert row.standard_error == pytest.approx(differences.std(ddof=1) / 2)


def test_reusing_a_seed_does_not_create_independent_replicates():
    duplicate = pd.concat([population([1, 2]), population([1, 2])])
    with pytest.raises(ValueError, match="Repeated systems"):
        replicate_probabilities(duplicate)


def test_unmatched_replicates_are_rejected():
    with pytest.raises(ValueError, match="same replicate"):
        paired_replicate_comparison(population([1, 2]), population([1, 2, 3]))


def test_single_replicate_cannot_estimate_between_run_variance():
    with pytest.raises(ValueError, match="At least two"):
        replicate_intervals(population([1]))


def test_no_observed_variation_does_not_imply_certainty():
    row = replicate_intervals(population([0, 0, 0])).loc["HJ"]
    assert row.status == "unresolved"
    assert (row.lower, row.upper) == (0, 1)


def test_roundoff_does_not_create_replicate_variance():
    row = paired_replicate_comparison(population([2, 3]), population([1, 2])).loc["HJ"]
    assert row.difference == pytest.approx(0.1)
    assert row.status == "unresolved"
    assert (row.lower, row.upper) == (-1, 1)


def test_sample_size_scales_with_requested_precision():
    assert 250000 < binomial_sample_size(0.006) < 260000
    assert binomial_sample_size(0.006, 0.025) == pytest.approx(
        4 * binomial_sample_size(0.006), abs=4
    )


def test_paired_transition_counts_align_systems():
    from hj.statistics import paired_outcome_transitions

    baseline = population([1, 2])
    treatment = population([3, 2]).sample(frac=1, random_state=42)
    result = paired_outcome_transitions(treatment, baseline)
    assert result.loc["NM", "HJ"] == 2
    assert result.loc["HJ", "HJ"] == 3
    assert result.loc["NM", "NM"] == 15
    assert result.to_numpy().sum() == 20


def test_transition_counts_reject_unmatched_or_duplicate_systems():
    from hj.statistics import paired_outcome_transitions

    baseline = population([1, 2])
    with pytest.raises(ValueError, match="identical paired"):
        paired_outcome_transitions(baseline.iloc[:-1], baseline)
    with pytest.raises(ValueError, match="unique"):
        paired_outcome_transitions(pd.concat([baseline, baseline]), baseline)


def test_empty_paired_transitions_have_zero_counts():
    from hj.statistics import paired_outcome_transitions

    empty = population([1]).iloc[:0]
    assert paired_outcome_transitions(empty, empty).to_numpy().sum() == 0
