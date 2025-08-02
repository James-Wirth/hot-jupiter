import os
import shutil
import re
import gc
import math
import logging

import numpy as np
import pandas as pd
import dask.dataframe as dd

from pathlib import Path
from typing import Dict, Optional
from joblib import Parallel, delayed, cpu_count
from more_itertools import chunked

from clusters.cluster import Cluster

from hjmodel.results import Results
from hjmodel.config import StopCode, CIRCULARISATION_THRESHOLD_ECCENTRICITY
from hjmodel import model_utils
from hjmodel.random_sampler import PlanetarySystem, sample_planetary_systems, EncounterSampler

logger = logging.getLogger(__name__)


def check_stopping_conditions(
    e: float,
    a: float,
    t: float,
    R_td: float,
    R_hj: float,
    R_wj: float,
    total_time: int
) -> int | None:
    if e >= 1:
        return StopCode.I
    if a * (1 - e) < R_td:
        return StopCode.TD
    if a < R_hj and (e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY or t >= total_time):
        return StopCode.HJ
    if R_hj < a < R_wj and (e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY or t >= total_time):
        return StopCode.WJ
    if e <= CIRCULARISATION_THRESHOLD_ECCENTRICITY:
        return StopCode.NM
    return None


def _eval_system_dynamic(
    ps: PlanetarySystem,
    cluster: Cluster,
    total_time: int,
    hybrid_switch: bool = True
) -> Dict[str, float]:
    """
    Simulate the evolution of the orbital parameters for a single planetary system.
    """

    system_rng = np.random.default_rng(ps.seed)
    encounter_sampler = EncounterSampler(sigma_v=0.0, rng=system_rng)

    R_td, R_hj, R_wj = model_utils.get_critical_radii(m1=ps.sys["m1"], m2=ps.sys["m2"])
    e, a = ps.sys["e_init"], ps.sys["a_init"]
    t = 0.0
    stopping_condition = None

    while t < total_time:
        stopping_condition = check_stopping_conditions(e, a, t, R_td, R_hj, R_wj, total_time)
        if stopping_condition is not None:
            break

        r = cluster.get_radius(lagrange=ps.lagrange, t=t)
        env = cluster.get_local_environment(r, t)
        encounter_sampler.sigma_v = env["sigma_v"]

        wt_time = encounter_sampler.get_waiting_time(env_vars=env)
        e, a = model_utils.tidal_effect(
            e=e,
            a=a,
            m1=ps.sys["m1"],
            m2=ps.sys["m2"],
            time_in_Myr=wt_time
        )
        t = min(t + wt_time, total_time)

        stopping_condition = check_stopping_conditions(e, a, t, R_td, R_hj, R_wj, total_time)
        if stopping_condition is not None:
            break

        rand_params = encounter_sampler.sample_encounter()
        kwargs = rand_params | {
            "e": e,
            "a": a,
            "m1": ps.sys["m1"],
            "m2": ps.sys["m2"]
        }
        if model_utils.is_analytic_valid(**kwargs, sigma_v=env["sigma_v"]) or not hybrid_switch:
            e += model_utils.de_HR(**kwargs)
        else:
            de, da = model_utils.de_sim(**kwargs)
            e += de
            a += da

    return {
        "final_e": e,
        "final_a": a,
        "stopping_condition": (stopping_condition or StopCode.NM).value,
        "stopping_time": t
    }


class HJModel:
    """
    Orchestrator class for running and saving simulations.
    """

    def __init__(
        self,
        name: str,
        base_dir: Path = None
    ):

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
            logger.info("Loaded combined DataFrame with %d rows from %d runs.",
                        len(concatenated), len(dfs))
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
            d for d in self.exp_path.iterdir()
            if d.is_dir() and re.match(r"run_(\d+)$", d.name)
        ]
        run_indices = sorted([
            int(re.search(r"\d+", d.name).group())
            for d in existing
        ]) if existing else []
        next_index = (max(run_indices) + 1) if run_indices else 0
        run_dir = self.exp_path / f"run_{next_index:03d}"
        run_dir.mkdir(parents=True, exist_ok=False)
        logger.info("Created new run directory: %s", run_dir)
        return run_dir

    def run(
        self,
        time: int,
        num_systems: int,
        cluster: Cluster,
        num_batches: int = 10,
        hybrid_switch: bool = True,
        seed: Optional[int] = None
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

        logger.info("Evaluating %d systems over %d partitions (t = %s Myr) for experiment %s",
                    self.num_systems, num_batches, self.time, self.name)

        output_dir = Path(self.path).parent / "parquet_batches"
        if output_dir.exists():
            logger.info("Cleaning existing directory: %s", output_dir)
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)
        planetary_systems = sample_planetary_systems(n_samples=num_systems, cluster=cluster, rng=rng)
        batch_size = math.ceil(num_systems / num_batches)

        n_jobs = max(1, cpu_count() - 1)

        try:
            for batch_idx, batch in enumerate(chunked(planetary_systems, batch_size)):
                logger.info("Processing partition %d/%d", batch_idx + 1, math.ceil(num_systems / batch_size))
                results = Parallel(
                    n_jobs=n_jobs, prefer="processes", batch_size="auto",
                )(
                    delayed(_eval_system_dynamic)(
                        ps=ps, cluster=cluster, total_time=self.time, hybrid_switch=hybrid_switch
                    )
                    for ps in batch
                )
                partition_df = pd.DataFrame([
                    {"r": cluster.get_radius(lagrange=ps.lagrange, t=self.time), **ps.sys, **res}
                    for ps, res in zip(batch, results)
                ])
                partition_path = output_dir / f"partition_{batch_idx + 1}.parquet"
                partition_df.to_parquet(partition_path, engine="pyarrow", compression="snappy")

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
                logger.debug("Couldn't remove output directory: %s (might not be empty)", output_dir)

        self.invalidate_cache()

        if not self.path or not os.path.exists(self.path):
            raise RuntimeError(f"Expected output file {self.path} not found after run_dynamic.")
