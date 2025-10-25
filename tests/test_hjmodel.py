import pandas as pd

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
    _ = model.results
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
