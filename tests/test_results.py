import pandas as pd
import pytest

from hjmodel.config import StopCode
from hjmodel.results import Results


def test_compute_outcome_probabilities_normalization():
    hj_rows = [{"r": 10.0, "stopping_condition": StopCode.HJ.value}] * 5
    td_rows = [{"r": 10.0, "stopping_condition": StopCode.TD.value}] * 3
    nm_rows = [{"r": 10.0, "stopping_condition": StopCode.NM.value}] * 2
    df = pd.DataFrame(hj_rows + td_rows + nm_rows)
    results = Results(df)
    probs = results.compute_outcome_probabilities()
    assert pytest.approx(probs["HJ"], rel=1e-3) == 0.5
    assert pytest.approx(probs["TD"], rel=1e-3) == 0.3
    assert pytest.approx(probs["NM"], rel=1e-3) == 0.2
    for label in ["ION", "WJ"]:
        assert pytest.approx(probs[label], abs=1e-6) == 0.0


def test_filter_and_sample_frac_consistency():
    hj_rows = [{"r": 1.0, "stopping_condition": StopCode.HJ.value}] * 10
    nm_rows = [{"r": 1.0, "stopping_condition": StopCode.NM.value}] * 90
    df = pd.DataFrame(hj_rows + nm_rows)
    results = Results(df)
    filtered = results.filter_outcomes(include=["HJ"], sample_frac={"NM": 0.5})
    assert all(
        filtered["stopping_condition"].isin([StopCode.HJ.value, StopCode.NM.value])
    )
