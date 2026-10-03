import json
from dataclasses import replace

import pandas as pd
import pytest

from hj import HJModel, NumericalSettings, Plummer, evolution
from hj.experiments import run_paired_experiment


def run_study(path, **kwargs):
    return run_paired_experiment(
        path, num_replicates=2, num_systems=4, time_myr=0.0, seed=42, **kwargs
    )


def test_paired_study_resumes_without_duplicating_completed_runs(tmp_path):
    path = tmp_path / "study"
    first = run_study(path)
    before = {p: p.read_bytes() for p in path.glob("*/run_*/results.parquet")}
    second = run_study(path, resume=True)
    pd.testing.assert_frame_equal(first, second)
    assert len(before) == 4
    assert before == {p: p.read_bytes() for p in path.glob("*/run_*/results.parquet")}
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["completed_pairs"] == 2
    assert manifest["status"] == "complete"
    diagnostics = pd.read_csv(path / "baseline_diagnostics.csv", index_col="outcome")
    assert diagnostics.loc["ALL", "flagged_system"] == 0
    transitions = pd.read_csv(path / "transition_counts.csv", index_col="baseline")
    assert transitions.loc["NM", "NM"] == 8


def test_resume_rejects_changed_numerics(tmp_path):
    path = tmp_path / "study"
    run_study(path)
    with pytest.raises(ValueError, match="different configuration"):
        run_study(
            path,
            resume=True,
            treatment_numerics=NumericalSettings(
                encounter_xi=1e-6, phase_at_pericentre=True
            ),
        )


def test_resume_can_extend_replicate_prefix(tmp_path):
    path = tmp_path / "study"
    run_study(path)
    old = json.loads((path / "manifest.json").read_text())["replicate_seeds"]
    run_paired_experiment(
        path, num_replicates=3, num_systems=4, time_myr=0.0, seed=42, resume=True
    )
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["replicate_seeds"][:2] == old
    assert manifest["completed_pairs"] == 3
    assert len(list(path.glob("*/run_*/results.parquet"))) == 6


def test_interrupted_pair_is_resumed_without_repeating_its_baseline(
    tmp_path, monkeypatch
):
    original = HJModel.run
    failed = False

    def interrupt(self, *args, **kwargs):
        nonlocal failed
        if self.name == "treatment" and not failed:
            failed = True
            raise RuntimeError("interrupted treatment")
        return original(self, *args, **kwargs)

    path = tmp_path / "study"
    monkeypatch.setattr(HJModel, "run", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_study(path)
    baseline = next(path.glob("baseline/run_*/results.parquet"))
    before = baseline.read_bytes()
    run_study(path, resume=True)
    assert baseline.read_bytes() == before
    assert len(list(path.glob("*/run_*/results.parquet"))) == 4


def test_cutoff_pair_requires_shared_pericentre_phase(tmp_path):
    baseline = NumericalSettings()
    treatment = replace(baseline, encounter_xi=1e-6)
    with pytest.raises(ValueError, match="align phases"):
        run_study(
            tmp_path / "invalid",
            treatment_numerics=treatment,
            baseline_numerics=baseline,
        )


def test_step_limit_failure_is_recorded_instead_of_counted_as_no_migration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(evolution, "_MAX_STEPS", 0)
    model = HJModel("failed", tmp_path)
    with pytest.raises(RuntimeError, match="active systems"):
        model.run(100.0, 4, Plummer(), seed=42, n_jobs=1)
    metadata = json.loads((tmp_path / "failed/run_000/metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert model.df.empty


def test_incomplete_results_are_excluded_from_aggregation(tmp_path):
    path = tmp_path / "failed/run_000"
    path.mkdir(parents=True)
    pd.DataFrame({"r": [1.0], "stop_code": [0]}).to_parquet(path / "results.parquet")
    (path / "metadata.json").write_text(json.dumps({"status": "failed"}))
    assert HJModel("failed", tmp_path).df.empty
