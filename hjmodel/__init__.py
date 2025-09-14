import gc
import logging
import math
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import dask.dataframe as dd
import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from more_itertools import chunked

from hjmodel.clusters import Cluster
from hjmodel.evolution import PlanetarySystem
from hjmodel.results import Results

logger = logging.getLogger(__name__)


class HJModel:
    """
    This class orchestrates the simulations, by:

    (i) Creating an output directory for the new model run
    (ii) Dispatching the planetary system evolution to the PlanetarySystem class in evolution.py
    (iii) Saving the batched results
    """

    def __init__(self, name: str, base_dir: Path = None):

        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent / "data"

        self.name = name
        self.base_dir = base_dir
        self.exp_path = self.base_dir / self.name
        self.exp_path.mkdir(parents=True, exist_ok=True)

        self.time: float = 0.0
        self.num_systems: int = 0
        self.path: Optional[str] = None

        self._df: Optional[pd.DataFrame] = None
        self._results_cached: Optional[Results] = None

        logger.info("Initialized HJModel for experiment '%s'.", self.name)

    def _load_all_runs_df(self) -> pd.DataFrame:
        result_files = sorted(self.exp_path.glob("run_*/results.parquet"))
        if not result_files:
            return pd.DataFrame()

        dfs = []
        for file in result_files:
            try:
                df = pd.read_parquet(file, engine="pyarrow")
                df["run_id"] = file.parent.name
                dfs.append(df)
            except Exception as e:
                logger.warning("Couldn't read file %s: %s", file, e)
        if dfs:
            concatenated = pd.concat(dfs, ignore_index=True)
            logger.info(
                "Loaded combined DataFrame with %d rows from %d runs.",
                len(concatenated),
                len(dfs),
            )
            return concatenated
        else:
            return pd.DataFrame()

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load_all_runs_df()
        return self._df

    def invalidate_cache(self):
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
        run_indices = (
            sorted([int(re.search(r"\d+", d.name).group()) for d in existing])
            if existing
            else []
        )
        next_index = (max(run_indices) + 1) if run_indices else 0
        run_dir = self.exp_path / f"run_{next_index:03d}"
        run_dir.mkdir(parents=True, exist_ok=False)
        logger.info("Created new run directory: %s", run_dir)
        return run_dir

    def run(
        self,
        time: float,
        num_systems: int,
        cluster: Cluster,
        num_batches: int = 10,
        hybrid_switch: bool = True,
        seed: Optional[int] = None,
    ) -> None:

        if time < 0:
            raise ValueError("time must be >= 0")
        if num_systems <= 0:
            raise ValueError("num_systems must be >= 0")
        if num_batches <= 0:
            raise ValueError("num_batches must be > 0")

        self.time = time
        self.num_systems = num_systems

        run_dir = self._allocate_new_run_dir()
        self.path = str(run_dir / "results.parquet")

        logger.info(
            "Evaluating %d systems over %d partitions (t = %s Myr) for experiment %s",
            self.num_systems,
            num_batches,
            self.time,
            self.name,
        )

        output_dir = Path(self.path).parent / "parquet_batches"
        if output_dir.exists():
            logger.info("Cleaning existing directory: %s", output_dir)
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)
        planetary_systems = PlanetarySystem.sample_batch(
            n_samples=num_systems,
            cluster=cluster,
            rng=rng,
        )
        batch_size = math.ceil(num_systems / num_batches)
        n_jobs = max(1, cpu_count() - 1)

        try:
            for batch_idx, batch in enumerate(chunked(planetary_systems, batch_size)):
                logger.info(
                    "Processing partition %d/%d",
                    batch_idx + 1,
                    math.ceil(num_systems / batch_size),
                )
                results = Parallel(
                    n_jobs=n_jobs,
                    prefer="processes",
                    batch_size="auto",
                )(
                    delayed(PlanetarySystem.run)(ps, cluster, self.time, hybrid_switch)
                    for ps in batch
                )
                partition_df = pd.DataFrame(results)
                partition_path = output_dir / f"partition_{batch_idx + 1}.parquet"
                partition_df.to_parquet(
                    partition_path, engine="pyarrow", compression="snappy"
                )
                del results, partition_df
                gc.collect()

            logger.info("All partitions processed. Combining results with Dask...")
            ddf = dd.read_parquet(str(output_dir / "*.parquet"))
            logger.info("Saving combined dataset to %s", self.path)

            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            ddf.to_parquet(self.path, engine="pyarrow", write_index=False)
        except Exception as exc:
            logger.error("run_dynamic failed during processing or saving: %s", exc)
            raise
        finally:
            for partition_file in output_dir.glob("partition_*.parquet"):
                try:
                    partition_file.unlink()
                except Exception:
                    logger.warning("Couldn't delete partition file %s", partition_file)
            try:
                output_dir.rmdir()
            except Exception:
                logger.debug(
                    "Couldn't remove output directory: %s (might not be empty)",
                    output_dir,
                )

        self.invalidate_cache()

        if not self.path or not os.path.exists(self.path):
            raise RuntimeError(
                f"Expected output file {self.path} not found after run_dynamic."
            )
