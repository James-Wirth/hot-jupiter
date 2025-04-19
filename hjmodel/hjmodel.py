import os
import shutil
import glob
import gc
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from pathlib import Path
import dask.dataframe as dd
from typing import Dict
import tqdm

from hjmodel.config import *
from hjmodel import model_utils
from hjmodel.random_sampler import EncounterSampler, PlanetarySystem, PlanetarySystemList
from clusters.cluster import Cluster

pd.options.mode.chained_assignment = None

def eval_system_dynamic(ps: PlanetarySystem, cluster: Cluster,
                        total_time: int, hybrid_switch: bool = True) -> Dict:

    # get critical radii and initialize running variables
    R_td, R_hj, R_wj = model_utils.get_critical_radii(m1=ps.sys["m1"], m2=ps.sys["m2"])
    e, a = ps.sys["e_init"], ps.sys["a_init"]

    t = 0
    stopping_condition = None

    def _check_stopping_conditions(_e, _a, _t):
        if _e >= 1:
            return SC_DICT['I']['id']
        if _a * (1 - _e) < R_td:
            return SC_DICT['TD']['id']
        if _a < R_hj and (_e <= 1E-3 or _t >= total_time):
            return SC_DICT['HJ']['id']
        if R_hj < _a < R_wj and (_e <= 1E-3 or _t >= total_time):
            return SC_DICT['WJ']['id']
        if _e <= 1E-3:
            return SC_DICT['NM']['id']
        return None

    while t < total_time:
        stopping_condition = _check_stopping_conditions(_e=e, _a=a, _t=t)
        if stopping_condition is not None:
            break
        else:
            pass

        # get local cluster environment
        r = cluster.get_radius(lagrange=ps.lagrange, t=t)
        env = cluster.get_local_environment(r, t)
        encounter_sampler = EncounterSampler(sigma_v=env["sigma_v"])

        # tidal evolution between encounters
        wt_time = encounter_sampler.get_waiting_time(env_vars=env)
        e, a = model_utils.tidal_effect(e=e, a=a, m1=ps.sys["m1"], m2=ps.sys["m2"], time_in_Myr=wt_time)
        t = min(t + wt_time, total_time)

        # check stopping conditions after tidal evolution
        stopping_condition = _check_stopping_conditions(_e=e, _a=a, _t=t)
        if stopping_condition is not None:
            break
        else:
            pass

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
        "stopping_condition": stopping_condition if stopping_condition is not None else SC_DICT['NM']['id'],
        "stopping_time": t
    }

class HJModel:
    def __init__(self, res_path: str):
        self.time = self.num_systems = 0
        self.path = res_path
        print(self.path)

        base_stem = res_path.replace(".pq", "").rsplit("_RUN", 1)[0]
        file_pattern = f"{base_stem}_RUN*.pq"
        matching_files = glob.glob(file_pattern)

        if os.path.exists(res_path):
            matching_files.append(res_path)

        self.df = None
        if matching_files:
            dataframes = [pd.read_parquet(f, engine='pyarrow') for f in matching_files]
            self.df = pd.concat(dataframes, ignore_index=True)


    def run_dynamic(self, time: int, num_systems: int, cluster: Cluster,
                    num_batches: int = 250, hybrid_switch: bool = True):
        self.time = time
        self.num_systems = num_systems
        batch_size = num_systems // num_batches
        print(f"Evaluating {self.num_systems} systems over {num_batches} partitions (t = {self.time} Myr)")

        output_dir = Path(self.path).parent / "parquet_batches"
        if output_dir.exists():
            print(f"Cleaning existing directory: {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        planetary_systems = PlanetarySystemList(n_samples=batch_size, cluster=cluster)

        def process_and_write_partition(batch_idx: int):
            print(f"Processing partition {batch_idx + 1}/{num_batches}")
            results = Parallel(n_jobs=cpu_count() - 1, prefer="threads", batch_size="auto", require="sharedmem")(
                delayed(eval_system_dynamic)(ps=ps, cluster=cluster, total_time=self.time, hybrid_switch=hybrid_switch)
                for ps in tqdm.tqdm(planetary_systems)
            )
            partition_df = pd.DataFrame([
                {"r": cluster.get_radius(lagrange=ps.lagrange, t=self.time), **ps.sys, **res}
                for ps, res in zip(planetary_systems, results)
            ])
            partition_path = output_dir / f"partition_{batch_idx + 1}.parquet"
            partition_df.to_parquet(partition_path, engine="pyarrow", compression="snappy")
            del results, partition_df
            gc.collect()

        for i in range(num_batches):
            process_and_write_partition(i)

        print("All partitions processed. Combining results with Dask...")
        ddf = dd.read_parquet(str(output_dir / "*.parquet"))
        print(f"Saving combined dataset to {self.path}")
        ddf.to_parquet(self.path, engine="pyarrow", write_index=False)
        for partition_file in output_dir.glob("partition_*.parquet"):
            partition_file.unlink()
        output_dir.rmdir()

        self.df = pd.read_parquet(self.path)
        gc.collect()
