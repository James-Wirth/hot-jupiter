import os
import pandas as pd
import pytest

from hjmodel.__init__ import HJModel
from hjmodel.config import StopCode

def test_caching_and_invalidation(tmp_path):
    base_dir = tmp_path / "data"
    exp_name = "exp1"
    run_dir = base_dir / exp_name / "run_000"
    run_dir.mkdir(parents=True)
    results_path = run_dir / "results.parquet"
    df_orig = pd.DataFrame([{"r": 1.0, "stopping_condition": StopCode.HJ.value}])
    df_orig.to_parquet(results_path, engine="pyarrow")

    model = HJModel(name=exp_name, base_dir=base_dir)
    df1 = model.df
    assert not df1.empty
    _ = model.results  # cache
    model.invalidate_cache()
    assert model._df is None and model._results_cached is None
    df2 = model.df
    pd.testing.assert_frame_equal(df1, df2)
    assert df2 is not df1

def test_allocate_new_run_dir_increments(tmp_path):
    base_dir = tmp_path / "data"
    model = HJModel(name="exp2", base_dir=base_dir)
    (model.exp_path / "run_000").mkdir(parents=True)
    (model.exp_path / "run_001").mkdir(parents=True)
    next_dir = model._allocate_new_run_dir()
    assert next_dir.name == "run_002"
    assert next_dir.exists()

def test_hjmodel_run_output_structure(tmp_path, monkeypatch, dummy_cluster):
    from hjmodel.evolution import PlanetarySystem

    def fake_run(self, cluster, total_time, hybrid_switch=True):
        return {
            "r": 3.14,
            "e_init": self.e_init,
            "a_init": self.a_init,
            "m1": self.m1,
            "m2": self.m2,
            "final_e": self.e,
            "final_a": self.a,
            "stopping_condition": StopCode.HJ.value,
            "stopping_time": total_time,
        }

    monkeypatch.setattr(PlanetarySystem, "run", fake_run)

    model = HJModel(name="exp_run", base_dir=tmp_path / "data")
    model.run(time=5, num_systems=2, cluster=dummy_cluster, num_batches=1, hybrid_switch=True, seed=42)
    df = model.results.df
    expected_cols = {"r", "e_init", "a_init", "m1", "m2", "final_e", "final_a", "stopping_condition", "stopping_time"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 2
    assert set(df["stopping_condition"].unique()) == {StopCode.HJ.value}
