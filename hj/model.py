from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hj import config
from hj.clusters import Cluster
from hj.evolution import run_simulation, sample_initial_conditions
from hj.numerics import NumericalSettings
from hj.provenance import runtime_metadata
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
        files = sorted(self.exp_path.glob("run_*/results.parquet"))
        if not files:
            return pd.DataFrame()
        frames = []
        for f in files:
            try:
                metadata_path = f.with_name("metadata.json")
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    if metadata.get("status") != "complete":
                        logger.warning("Skipping incomplete run %s", f.parent)
                        continue
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
        if self._df is None:
            self._df = self._load_runs()
        return self._df

    def invalidate_cache(self) -> None:
        self._df = None
        self._results_cached = None

    @property
    def results(self) -> Results:
        if self._results_cached is None:
            self._results_cached = Results(self.df)
        return self._results_cached

    def _allocate_new_run_dir(self) -> Path:
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
        *,
        n_jobs: int = -1,
        numerics: NumericalSettings | None = None,
    ) -> None:

        if not np.isfinite(time) or time < 0:
            raise ValueError("time must be >= 0")
        if num_systems <= 0:
            raise ValueError("num_systems must be >= 0")
        numerics = NumericalSettings() if numerics is None else numerics

        run_dir = self._allocate_new_run_dir()
        self.path = str(run_dir / "results.parquet")

        logger.info(
            "Evaluating %d systems (t = %s Myr) for experiment %s",
            num_systems,
            time,
            self.name,
        )

        seed_sequence = np.random.SeedSequence(seed)
        streams = seed_sequence.spawn(3)
        initial_rng, encounter_rng, phase_rng = [
            np.random.default_rng(s) for s in streams
        ]
        metadata = {
            **runtime_metadata(),
            "time_myr": float(time),
            "num_systems": num_systems,
            "hybrid_switch": hybrid_switch,
            "seed_entropy": seed_sequence.entropy,
            "random_stream_scheme": "initial_encounter_phase_v1",
            "random_stream_spawn_keys": [list(s.spawn_key) for s in streams],
            "bit_generator": type(encounter_rng.bit_generator).__name__,
            "cluster_type": type(cluster).__name__,
            "cluster": vars(cluster),
            "constants": {name: getattr(config, name) for name in config.__all__},
            "integrator": "ias15",
            "ias15_epsilon": numerics.ias15_epsilon,
            "ias15_adaptive_mode": 2,
            "ias15_min_dt": 0.0,
            "tidal_step_fraction": numerics.tidal_step_fraction,
            "numerics": asdict(numerics),
            "n_jobs": n_jobs,
            "status": "running",
        }
        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            state = sample_initial_conditions(num_systems, cluster, initial_rng)
            run_simulation(
                state,
                cluster,
                float(time),
                encounter_rng,
                hybrid_switch=hybrid_switch,
                n_jobs=n_jobs,
                phase_rng=phase_rng,
                numerics=numerics,
            )

            r = cluster.radius(state.lagrange, float(time))

            table = pa.Table.from_pydict(
                {
                    "system_id": np.arange(num_systems, dtype=np.int64),
                    "replicate_id": [str(seed_sequence.entropy)] * num_systems,
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
            metadata["status"] = "complete"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            raise

        self.invalidate_cache()
