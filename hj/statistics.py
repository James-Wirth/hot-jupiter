from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t

from hj.state import StopCode

_OUTCOMES = {code.name: code.value for code in StopCode}


def _validate_confidence(confidence):
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")


def _validate_codes(stop_codes):
    codes = np.asarray(stop_codes)
    if codes.ndim != 1 or not np.isin(codes, list(_OUTCOMES.values())).all():
        raise ValueError(
            "Outcomes must be a one-dimensional array of completed stop codes"
        )
    return codes


def outcome_intervals(
    stop_codes, confidence: float = 0.95, method: str = "wilson"
) -> pd.DataFrame:
    _validate_confidence(confidence)
    if method not in {"wilson", "exact"}:
        raise ValueError("method must be wilson or exact")
    codes = _validate_codes(stop_codes)
    total = len(codes)
    rows = []
    for label, code in _OUTCOMES.items():
        count = int(np.count_nonzero(codes == code))
        if total:
            interval = binomtest(count, total).proportion_ci(confidence, method=method)
            lower, upper = interval.low, interval.high
        else:
            lower, upper = np.nan, np.nan
        rows.append(
            {
                "outcome": label,
                "count": count,
                "total": total,
                "probability": count / total if total else np.nan,
                "lower": lower,
                "upper": upper,
                "confidence": confidence,
                "method": method,
            }
        )
    return pd.DataFrame(rows).set_index("outcome")


def replicate_probabilities(
    df: pd.DataFrame, replicate_column: str = "replicate_id"
) -> pd.DataFrame:
    if df.empty or df[replicate_column].isna().any():
        raise ValueError("Replicates must be nonempty and have complete identifiers")
    _validate_codes(df.stop_code)
    if "system_id" in df and df.duplicated([replicate_column, "system_id"]).any():
        raise ValueError(
            "Repeated systems within a replicate; check for reused seeds or duplicate runs"
        )
    counts = pd.crosstab(df[replicate_column], df.stop_code).reindex(
        columns=list(_OUTCOMES.values()), fill_value=0
    )
    counts.columns = list(_OUTCOMES)
    return counts.div(counts.sum(axis=1), axis=0)


def _mean_interval(values, confidence, bounds):
    n = len(values)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(n))
    if np.ptp(values) <= 8 * np.finfo(float).eps * max(1.0, abs(mean)):
        return mean, 0.0, *bounds, "unresolved"
    half_width = float(t.ppf((1 + confidence) / 2, n - 1) * standard_error)
    return (
        mean,
        standard_error,
        max(bounds[0], mean - half_width),
        min(bounds[1], mean + half_width),
        "estimated",
    )


def replicate_intervals(
    df: pd.DataFrame,
    confidence: float = 0.95,
    replicate_column: str = "replicate_id",
) -> pd.DataFrame:
    _validate_confidence(confidence)
    probabilities = replicate_probabilities(df, replicate_column)
    if len(probabilities) < 2:
        raise ValueError("At least two independent replicates are required")
    rows = []
    for label in _OUTCOMES:
        mean, error, lower, upper, status = _mean_interval(
            probabilities[label].to_numpy(), confidence, (0.0, 1.0)
        )
        rows.append(
            {
                "outcome": label,
                "probability": mean,
                "standard_error": error,
                "lower": lower,
                "upper": upper,
                "replicates": len(probabilities),
                "confidence": confidence,
                "status": status,
                "method": "replicate_student_t",
            }
        )
    return pd.DataFrame(rows).set_index("outcome")


def paired_replicate_comparison(
    treatment: pd.DataFrame,
    baseline: pd.DataFrame,
    confidence: float = 0.95,
    replicate_column: str = "replicate_id",
) -> pd.DataFrame:
    _validate_confidence(confidence)
    treatment_rates = replicate_probabilities(treatment, replicate_column)
    baseline_rates = replicate_probabilities(baseline, replicate_column)
    if set(treatment_rates.index) != set(baseline_rates.index):
        raise ValueError(
            "Treatment and baseline must contain the same replicate identifiers"
        )
    baseline_rates = baseline_rates.loc[treatment_rates.index]
    n = len(treatment_rates)
    if n < 2:
        raise ValueError("At least two independent replicate pairs are required")
    rows = []
    for label in _OUTCOMES:
        values = (treatment_rates[label] - baseline_rates[label]).to_numpy()
        difference, error, lower, upper, status = _mean_interval(
            values, confidence, (-1.0, 1.0)
        )
        rows.append(
            {
                "outcome": label,
                "treatment_probability": treatment_rates[label].mean(),
                "baseline_probability": baseline_rates[label].mean(),
                "difference": difference,
                "standard_error": error,
                "lower": lower,
                "upper": upper,
                "replicate_pairs": n,
                "confidence": confidence,
                "status": status,
                "method": "paired_replicate_student_t",
            }
        )
    return pd.DataFrame(rows).set_index("outcome")


def binomial_sample_size(
    probability: float, relative_half_width: float = 0.05, confidence: float = 0.95
) -> int:
    from scipy.stats import norm

    _validate_confidence(confidence)
    if (
        not 0 < probability < 1
        or not math.isfinite(relative_half_width)
        or relative_half_width <= 0
    ):
        raise ValueError(
            "Require 0 < probability < 1 and a finite positive relative half width"
        )
    return math.ceil(
        norm.ppf((1 + confidence) / 2) ** 2
        * (1 - probability)
        / (probability * relative_half_width**2)
    )


def paired_outcome_transitions(
    treatment: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    keys = ["replicate_id", "system_id"]
    for df in (treatment, baseline):
        if not set(keys) <= set(df.columns):
            raise ValueError("Paired transitions require replicate_id and system_id")
        if df[keys].isna().any().any() or df.duplicated(keys).any():
            raise ValueError("Paired system identifiers must be complete and unique")
        _validate_codes(df.stop_code)
    left = baseline.set_index(keys).stop_code
    right = treatment.set_index(keys).stop_code
    if len(left) != len(right) or not left.index.isin(right.index).all():
        raise ValueError(
            "Treatment and baseline must contain identical paired system identifiers"
        )
    right = right.reindex(left.index)
    counts = pd.crosstab(left.to_numpy(), right.to_numpy()).reindex(
        index=list(_OUTCOMES.values()),
        columns=list(_OUTCOMES.values()),
        fill_value=0,
    )
    counts.index = pd.Index(list(_OUTCOMES), name="baseline")
    counts.columns = pd.Index(list(_OUTCOMES), name="treatment")
    return counts.astype(int)
