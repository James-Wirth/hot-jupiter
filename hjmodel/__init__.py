from __future__ import annotations

import gc
import logging
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from more_itertools import chunked

from hjmodel.clusters import Cluster
from hjmodel.evolution import PlanetarySystem
from hjmodel.io import DaskProcessor
from hjmodel.results import Results

__all__ = ["HJModel"]

logger = logging.getLogger(__name__)


class HJModel:
    """
    Main orchestrator for Hot Jupiter formation simulations.

    Manages simulation runs, persists results to disk, and provides
    access to analysis utilities via the results property.

    Attributes:
        name: Experiment name (used for directory naming).
        base_dir: Base directory for storing experiment data.
        exp_path: Full path to the experiment directory.
        time: Simulation duration of the most recent run (Myr).
        num_systems: Number of systems in the most recent run.
        path: Path to the most recent results file.
    """

    def __init__(self, name: str, base_dir: Path | None = None):
        """
        Initialize the HJModel with an experiment name.

        Args:
            name: Unique name for this experiment.
            base_dir: Base directory for data storage. Defaults to ../data.
        """
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent / "data"

        self.name = name
        self.base_dir = base_dir
        self.exp_path = self.base_dir / self.name
        self.exp_path.mkdir(parents=True, exist_ok=True)

        self.time: float = 0.0
        self.num_systems: int = 0
        self.path: str | None = None

        self._df: pd.DataFrame | None = None
        self._results_cached: Results | None = None

        logger.info("Initialized HJModel for experiment '%s'.", self.name)

    def _load_all_runs(self) -> pd.DataFrame:
        """
        Load and concatenate results from all runs in the experiment.

        Returns:
            DataFrame containing all results with 'run_id' column added.
        """
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
                "Loaded combined data with %d rows from %d runs.",
                len(concatenated),
                len(dfs),
            )
            return concatenated
        else:
            return pd.DataFrame()

    @property
    def df(self) -> pd.DataFrame:
        """
        Get the combined DataFrame of all simulation results.

        Lazily loads and caches results from disk on first access.
        """
        if self._df is None:
            self._df = self._load_all_runs()
        return self._df

    def invalidate_cache(self) -> None:
        """
        Clear cached results, forcing reload on next access.

        Call after adding new runs to see updated results.
        """
        self._df = None
        self._results_cached = None

    @property
    def results(self) -> Results:
        """
        Get the Results analysis object.

        Lazily creates and caches a Results instance on first access.
        """
        if self._results_cached is None:
            self._results_cached = Results(self.df)
        return self._results_cached

    def _allocate_new_run_dir(self) -> Path:
        """
        Create a new numbered run directory.

        Returns:
            Path to the newly created run directory.
        """
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
        seed: int | None = None,
    ) -> None:
        """
        Execute a simulation run with the specified parameters.

        Samples planetary systems, evolves them through stellar encounters
        and tidal effects, and saves results to disk.

        Args:
            time: Total simulation duration (Myr).
            num_systems: Number of planetary systems to simulate.
            cluster: Cluster environment for the simulation.
            num_batches: Number of batches for parallel processing.
            hybrid_switch: If True, use N-body for close encounters.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If time, num_systems, or num_batches are invalid.
            RuntimeError: If the output file is not created.
        """
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

        dask_processor = DaskProcessor(Path(self.path))
        dask_processor.prepare_dir()

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
                dask_processor.write_partition(partition_df)
                del results, partition_df
                gc.collect()

            dask_processor.save_all_partitions()
        except Exception as exc:
            logger.error("run_dynamic failed during processing or saving: %s", exc)
            raise
        finally:
            dask_processor.clean_partitions()

        self.invalidate_cache()

        if not self.path or not os.path.exists(self.path):
            raise RuntimeError(
                f"Expected output file {self.path} not found after run_dynamic."
            )
