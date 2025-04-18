import os
import shutil
import glob
import gc
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from tqdm import contrib
from pathlib import Path
import dask.dataframe as dd

from hjmodel.config import *
from hjmodel import model_utils, rand_utils
from clusters.plummer import Plummer

pd.options.mode.chained_assignment = None

def eval_system_dynamic(e_init: float, a_init: float, m1: float, m2: float,
                        lagrange: float, cluster: Plummer, total_time: int,
                        hybrid_switch: bool = True) -> list:
    """
    Calculates outcome for a given (randomised) system in the cluster

    Inputs
    ----------
    e_init: float               Initial eccentricity (Rayleigh)
    a_init: float               Initial semi-major axis                     au
    m1: float                   Host star mass                              M_solar
    m2: float                   Planet mass                                 M_solar
    cluster: DynamicPlummer     Cluster profile                             ---
    total_time: int             Time duration of simulation                 Myr

    Returns
    ----------
    [e, a, stopping_condition, stopping_time]: list
    """

    # Get critical radii and initialize running variables
    R_td, R_hj, R_wj = model_utils.get_critical_radii(m1=m1, m2=m2)
    e, a, current_time, stopping_time = e_init, a_init, 0, 0
    stopping_condition = None

    def check_stopping_conditions(e, a, current_time):
        if e >= 1:
            return SC_DICT['I']['id']
        if a * (1 - e) < R_td:
            return SC_DICT['TD']['id']
        if a < R_hj and (e <= 1E-3 or current_time >= total_time):
            return SC_DICT['HJ']['id']
        if R_hj < a < R_wj and (e <= 1E-3 or current_time >= total_time):
            return SC_DICT['WJ']['id']
        if e <= 1E-3:
            return SC_DICT['NM']['id']
        return None

    while current_time < total_time:

        # check stopping conditions before tidal evolution
        stopping_condition = check_stopping_conditions(e, a, current_time)
        if stopping_condition is not None:
            break

        r = cluster.map_lagrange_to_radius(lagrange, current_time)
        env_vars = cluster.env_vars(r, current_time)
        perts_per_Myr = model_utils.get_perts_per_Myr(*env_vars.values())
        wt_time = rand_utils.get_waiting_time(perts_per_Myr=perts_per_Myr)

        # tidal evolution
        e, a = model_utils.tidal_effect(e=e, a=a, m1=m1, m2=m2, time_in_Myr=wt_time)
        current_time = min(current_time + wt_time, total_time)

        # check stopping conditions after tidal evolution
        stopping_condition = check_stopping_conditions(e, a, current_time)
        if stopping_condition is not None:
            break

        if stopping_condition is None and current_time < total_time:
            rand_params = rand_utils.random_encounter_params(sigma_v=env_vars['sigma_v'])
            args = {
                'v_infty': rand_params['v_infty'],
                'b': rand_params['b'],
                'Omega': rand_params['Omega'],
                'inc': rand_params['inc'],
                'omega': rand_params['omega'],
                'e_init': e,
                'a_init': a,
                'm1': m1,
                'm2': m2,
                'm3': rand_params['m3']
            }

            if model_utils.is_analytic_valid(*args.values(), sigma_v=env_vars['sigma_v']) or not hybrid_switch:
                e += model_utils.de_HR(*args.values())
            else:
                de, da = model_utils.de_sim(*args.values())
                e += de
                a += da

    return [e, a, stopping_condition if stopping_condition is not None else SC_DICT['NM']['id'], current_time]

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


    def check_overwrite(self):
        ans = input(f'Data already exists at {self.path}.\nDo you want to overwrite? (Y/n): ')
        while not ans.lower() in ['y', 'yes', 'n', 'no']:
            ans = input('Invalid response. Please re-enter (Y/n): ')
        return ans.lower() in ['y', 'yes']

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

        lagrange = cluster.get_lagrange_distribution(n_samples=batch_size, t=0)

        def process_and_write_partition(batch_idx: int):
            print(f"Processing partition {batch_idx + 1}/{num_batches}")
            sys = rand_utils.get_random_system_params(n_samples=batch_size)
            results = Parallel(n_jobs=cpu_count() - 1, prefer="threads", batch_size="auto", require="sharedmem")(
                delayed(eval_system_dynamic)(*args, cluster, self.time, hybrid_switch) for args in contrib.tzip(*sys, lagrange)
            )
            present_r_vals = np.vectorize(cluster.map_lagrange_to_radius)(lagrange, t=time)
            partition_df = pd.DataFrame({
                "r": present_r_vals,
                "final_e": np.array([row[0] for row in results], dtype=np.float32),
                "final_a": np.array([row[1] for row in results], dtype=np.float32),
                "stopping_condition": np.array([row[2] for row in results], dtype=np.int8),
                "stopping_time": np.array([row[3] for row in results], dtype=np.float32),
                "e_init": sys[0],
                "a_init": sys[1],
                "m1": sys[2],
            })
            partition_path = output_dir / f"partition_{batch_idx + 1}.parquet"
            partition_df.to_parquet(partition_path, engine="pyarrow", compression="snappy")

            del sys, results, partition_df, present_r_vals
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
