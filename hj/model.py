from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hj.clusters import Cluster
from hj.evolution import run_simulation, sample_initial_conditions
from hj.results import Results

__all__ = ["HJModel"]

logger = logging.getLogger(__name__)


class HJModel:
    def __init__(self, name: str, base_dir: Path | None = None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent / "data"
        self.name = name
        self.base_dir = Path(base_dir)
        self.exp_path = self.base_dir / self.name
        self.exp_path.mkdir(parents=True, exist_ok=True)

        self.path: str | None = None
        self._df: pd.DataFrame | None = None
        self._results_cached: Results | None = None

        logger.info("Initialized HJModel for experiment '%s'.", self.name)

    def _load_runs(self) -> pd.DataFrame:
        """Concatenate all run_*/results.parquet files into a single DataFrame."""
        files = sorted(self.exp_path.glob("run_*/results.parquet"))
        if not files:
            return pd.DataFrame()
        frames = []
        for f in files:
            try:
                df = pd.read_parquet(f, engine="pyarrow")
                df["run_id"] = f.parent.name
                frames.append(df)
            except Exception as exc:
                logger.warning("Couldn't read %s: %s", f, exc)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @property
    def df(self) -> pd.DataFrame:
        """Lazy-loaded DataFrame across all runs."""
        if self._df is None:
            self._df = self._load_runs()
        return self._df

    def invalidate_cache(self) -> None:
        """Clear cached DataFrame and Results view."""
        self._df = None
        self._results_cached = None

    @property
    def results(self) -> Results:
        """Lazy-loaded Results analysis view."""
        if self._results_cached is None:
            self._results_cached = Results(self.df)
        return self._results_cached

    def _allocate_new_run_dir(self) -> Path:
        """Create the next run_NNN directory under the experiment path."""
        existing = [
            d
            for d in self.exp_path.iterdir()
            if d.is_dir() and re.match(r"run_(\d+)$", d.name)
        ]
        if existing:
            indices = sorted(int(re.search(r"\d+", d.name).group()) for d in existing)
            next_index = indices[-1] + 1
        else:
            next_index = 0
        run_dir = self.exp_path / f"run_{next_index:03d}"
        run_dir.mkdir(parents=True, exist_ok=False)
        logger.info("Created new run directory: %s", run_dir)
        return run_dir

    def run(
        self,
        time: float,
        num_systems: int,
        cluster: Cluster,
        hybrid_switch: bool = True,
        seed: int | None = None,
    ) -> None:
        """Execute one simulation run; write `<exp>/run_XXX/results.parquet`."""
        if time < 0:
            raise ValueError("time must be >= 0")
        if num_systems <= 0:
            raise ValueError("num_systems must be >= 0")

        run_dir = self._allocate_new_run_dir()
        self.path = str(run_dir / "results.parquet")

        logger.info(
            "Evaluating %d systems (t = %s Myr) for experiment %s",
            num_systems,
            time,
            self.name,
        )

        rng = np.random.default_rng(seed)
        state = sample_initial_conditions(num_systems, cluster, rng)
        run_simulation(state, cluster, float(time), rng, hybrid_switch=hybrid_switch)

        r = cluster.radius(state.lagrange, float(time))

        table = pa.Table.from_pydict(
            {
                "r": np.asarray(r, dtype=np.float64),
                "e_init": state.e_init,
                "a_init": state.a_init,
                "m1": state.m1,
                "m2": state.m2,
                "lagrange": state.lagrange,
                "e": state.e,
                "a": state.a,
                "stop_code": state.stop_code.astype(np.int32),
                "stop_time": state.stop_time,
            }
        )
        pq.write_table(table, self.path, compression="snappy")

        self.invalidate_cache()
