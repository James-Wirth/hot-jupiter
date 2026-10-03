import numpy as np
import pandas as pd
import pytest

from hj.diagnostics import outcome_diagnostics
from hj.results import Results


def test_result_diagnostics_use_the_requested_radius_selection():
    df = pd.DataFrame(
        {"r": [1.0, 200.0], "e": [-0.2, 0.3], "a": [5.0, 1.0], "stop_code": [0, 0]}
    )
    results = Results(df)
    assert results.compute_outcome_diagnostics().loc["ALL", "total"] == 1
    assert results.compute_outcome_diagnostics(r_max=300).loc["ALL", "total"] == 2


def test_diagnostics_expose_negative_analytic_eccentricities():
    df = pd.DataFrame(
        {"e": [-0.2, 0.3, 1.2], "a": [5.0, 1.0, -2.0], "stop_code": [0, 0, 1]}
    )
    result = outcome_diagnostics(df)
    assert result.loc["NM", "negative_eccentricity"] == 1
    assert result.loc["NM", "flagged_fraction"] == 0.5
    assert result.loc["ION", "flagged_system"] == 0
    assert result.loc["ALL", "total"] == 3
    assert result.loc["ALL", "flagged_system"] == 1
    assert df.e.iloc[0] == -0.2


def test_diagnostics_count_each_flagged_system_once():
    df = pd.DataFrame(
        {
            "e": [-0.2, np.nan, 1.2, 0.2],
            "a": [-1.0, 1.0, 1.0, 1.0],
            "stop_code": [0, 0, 3, -1],
        }
    )
    result = outcome_diagnostics(df)
    assert result.loc["NM", "negative_eccentricity"] == 1
    assert result.loc["NM", "nonpositive_bound_semimajor_axis"] == 1
    assert result.loc["NM", "nonfinite_orbit"] == 1
    assert result.loc["HJ", "unbound_eccentricity_in_bound_outcome"] == 1
    assert result.loc["UNCLASSIFIED", "invalid_stop_code"] == 1
    assert result.loc["ALL", "flagged_system"] == 4


def test_diagnostics_require_orbital_elements():
    with pytest.raises(ValueError, match="Missing columns"):
        outcome_diagnostics(pd.DataFrame({"stop_code": [0]}))


def test_empty_diagnostics_are_undefined_instead_of_clean():
    result = outcome_diagnostics(pd.DataFrame(columns=["e", "a", "stop_code"]))
    assert result.loc["ALL", "total"] == 0
    assert np.isnan(result.loc["ALL", "flagged_fraction"])
