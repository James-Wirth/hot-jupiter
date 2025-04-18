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

from hjmodel.model_utils import eval_system_dynamic
from hjmodel.random_sampler import PlanetarySystemList
from clusters import Plummer

pd.options.mode.chained_assignment = None

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


    def run_dynamic(self, time: int, num_systems: int, cluster: Plummer,
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
