import os
import shutil

import matplotlib
import pandas as pd
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed, cpu_count
from hjmodel.config import *
from hjmodel import model_utils, rand_utils
from hjmodel.cluster import DynamicPlummer
from tqdm import contrib, tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import dask.dataframe as dd
import gc

pd.options.mode.chained_assignment = None

def eval_system_dynamic(e_init: float, a_init: float, m1: float, m2: float,
                        lagrange: float, cluster: DynamicPlummer, total_time: int,
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
            return SC_DICT['I']
        if a * (1 - e) < R_td:
            return SC_DICT['TD']
        if a < R_hj and (e <= 1E-3 or current_time >= total_time):
            return SC_DICT['HJ']
        if R_hj < a < R_wj and (e <= 1E-3 or current_time >= total_time):
            return SC_DICT['WJ']
        if e <= 1E-3:
            return SC_DICT['NM']
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

    return [e, a, stopping_condition if stopping_condition is not None else SC_DICT['NM'], current_time]

class HJModel:
    """
    HJModel class
    Performs the Monte-Carlo simulation for given cluster parameters

    Inputs
    ----------
    time: int               Time duration of simulation                             Myr
    num_systems: int        Number of experiments in Monte-Carlo simulation         arbitrary units

    Methods
    ----------
    run:                    Generate representative sample of radial coordinates (n=num_systems)
                            within the cluster, and determine local density and dispersion for each.
                            Simulate the evolution of each system; parallelize over systems.

    show_results:           Output the outcome statistics
    """

    def __init__(self, res_path: str):
        self.time = self.num_systems = 0
        self.path = res_path
        print(self.path)
        if os.path.exists(self.path):
            self.df = pd.read_parquet(self.path, engine='pyarrow')
        else:
            self.df = None

    def check_overwrite(self):
        ans = input(f'Data already exists at {self.path}.\nDo you want to overwrite? (Y/n): ')
        while not ans.lower() in ['y', 'yes', 'n', 'no']:
            ans = input('Invalid response. Please re-enter (Y/n): ')
        return ans.lower() in ['y', 'yes']

    def run_dynamic(self, time: int, num_systems: int, cluster: DynamicPlummer,
                    num_batches: int = 250, hybrid_switch: bool = True):
        """
        Executes a Monte Carlo simulation for planetary systems in a cluster environment.

        Parameters
        ----------
        time : int
            Total simulation time in Myr.
        num_systems : int
            Total number of systems to simulate.
        cluster : DynamicPlummer
            The cluster model used for environmental parameters.
        num_batches : int, optional
            Number of partitions to write to the Parquet file (default is 100).
        hybrid_switch : bool, optional
            Toggle hybrid model
        """
        if self.df is not None:
            if not self.check_overwrite():
                print("Run interrupted.")
                return

        self.time = time
        self.num_systems = num_systems

        batch_size = num_systems // num_batches
        print(f"Evaluating {self.num_systems} systems over {num_batches} partitions (t = {self.time} Myr)")

        output_dir = Path(self.path).parent / "parquet_batches"
        if output_dir.exists():
            print(f"Cleaning existing directory: {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Precompute shared Lagrange distribution
        lagrange = cluster.get_lagrange_distribution(n_samples=batch_size, t=0)

        def process_and_write_partition(batch_idx: int):
            """
            Processes a single partition and writes it directly to a Parquet file.
            """
            print(f"Processing partition {batch_idx + 1}/{num_batches}")
            sys = rand_utils.get_random_system_params(n_samples=batch_size)

            # Parallel computation of system dynamics
            results = Parallel(n_jobs=cpu_count() - 1, prefer="threads", batch_size="auto", require="sharedmem")(
                delayed(eval_system_dynamic)(*args, cluster, self.time, hybrid_switch) for args in contrib.tzip(*sys, lagrange)
            )

            # Compute radii
            present_r_vals = np.vectorize(cluster.map_lagrange_to_radius)(lagrange, t=time)

            # Create DataFrame for the partition
            partition_df = pd.DataFrame({
                "r": present_r_vals,
                "final_e": [row[0] for row in results],
                "final_a": [row[1] for row in results],
                "stopping_condition": [row[2] for row in results],
                "stopping_time": [row[3] for row in results],
                "e_init": sys[0],
                "a_init": sys[1],
                "m1": sys[2],
            })

            # Write partition directly to Parquet file
            partition_path = output_dir / f"partition_{batch_idx + 1}.parquet"
            partition_df.to_parquet(partition_path, engine="pyarrow", compression="snappy")

        # Process and write all partitions
        for i in range(num_batches):
            process_and_write_partition(i)

        print("All partitions processed. Combining results with Dask...")

        # Combine all partitions using Dask
        ddf = dd.read_parquet(str(output_dir / "*.parquet"))
        print(f"Saving combined dataset to {self.path}")
        ddf.to_parquet(self.path, engine="pyarrow", write_index=False)

        # Optionally clean up individual partition files
        for partition_file in output_dir.glob("partition_*.parquet"):
            partition_file.unlink()
        output_dir.rmdir()

        # Load the combined dataset into memory
        self.df = pd.read_parquet(self.path)

    def plot_outcomes(self, ax):
        df_filt = self.df.loc[self.df['stopping_condition'].isin([1])==False]
        df_filt['x'] = df_filt['final_a']
        df_filt['y'] = 1/(1-df_filt['final_e'])
        df_filt = df_filt.drop(df_filt[(df_filt['stopping_condition'] == 0)].sample(frac=0.5).index)
        df_filt = df_filt.drop(df_filt[df_filt['stopping_condition'] == 2].sample(frac=0.5).index)

        palette_copy = PALETTE.copy()
        palette_copy[0] = 'lightgray'

        cmap = matplotlib.colors.ListedColormap(list(palette_copy.values()))
        df_filt.plot.scatter(x='x', y='y', ax=ax, color=cmap(df_filt['stopping_condition']), s=0.1)
        for i in ['NM', 'TD', 'HJ', 'WJ']:
             ax.scatter(np.nan, np.nan, color=palette_copy[SC_DICT[i]], s=10, label=i, rasterized=True)
        ax.legend(frameon=True)

    def get_outcome_probabilities(self) -> dict[str, float]:
        return {key: self.df.loc[(self.df['stopping_condition'] == SC_DICT[key])
                                & (self.df['r'] <= 100)].shape[0]/self.df.shape[0]
                for key in SC_DICT}

    def get_statistics_for_outcome(self, outcomes: list[str], feature: str) -> list[float]:
        return self.df.loc[(self.df['stopping_condition'].isin(SC_DICT[x] for x in outcomes))
                           & (self.df['r'] <= 100)][feature].to_list()

    def get_projected_distribution(self):
        self.df['stoc_r_proj'] = self.df['r'].apply(lambda r: r * np.sin(rand_utils.rand_i()))

    def get_radius_histogram(self, label='stoc_r_proj', num_bins=60):
        bins = np.geomspace(0.99*self.df[label].min(), 1.01*self.df[label].max(), num_bins)
        self.df['binned'] = pd.cut(self.df[label], bins)
        is_multi = self.df["binned"].value_counts() > 1500
        filtered = self.df.loc[self.df["binned"].isin(is_multi[is_multi].index)].reset_index(drop=True)
        ret = filtered.groupby(['binned'])['stopping_condition'].value_counts(normalize=True)
        return ret, filtered['binned'].min().left, filtered['binned'].max().left

    def ax_config(self, ax, xrange: tuple[float, float], yrange: tuple[float, float],
                  xlabel: str, ylabel: str, xscale='log', yscale='log', rasterized=False):
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend([], [], frameon=False)
        if rasterized:
            ax.collections[0].set_rasterized(True)

    def get_summary_figure(self):
        fig = plt.figure(layout='constrained')
        fig.set_size_inches(4, 5)
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax3 = fig.add_subplot(gs[1, :])

        sns.histplot(data=self.df, x='r', hue='stopping_condition', ax=ax1, palette=PALETTE,
                     element='step', fill=False, common_norm=False, stat='density', cumulative=True, log_scale=True,
                     bins=200)
        self.ax_config(ax=ax1, xrange=(10**(-1.5), 20), yrange=(1E-4, 1), xlabel='r / pc', ylabel='CDF')

        df_filt = self.df.loc[self.df['stopping_condition'].isin([SC_DICT[outcome] for outcome in ['I', 'TD', 'HJ']])]
        df_filt['stopping_time_Gyr'] = df_filt['stopping_time'] / 1E3
        sns.histplot(data=df_filt, x='stopping_time_Gyr', hue='stopping_condition', ax=ax2, palette=PALETTE,
                     element='step', fill=False, common_norm=False, stat='density', cumulative=True, log_scale=True,
                     bins=400)
        self.ax_config(ax=ax2, xrange=(1E-3, 11.99), yrange=(1E-4, 1), xlabel='$T_{\\mathrm{stop}}$ / Gyr', ylabel='CDF')

        self.plot_outcomes(ax3)
        self.ax_config(ax=ax3, xrange=(1E-2, 1E3), yrange=(1, 1E5), xlabel='$a$', ylabel='$1/(1-e)$')
        points = ax3.collections[0]
        points.set_rasterized(True)
        fig.legend(loc='upper right', labels=list(SC_DICT.keys())[::-1], reverse=True, bbox_to_anchor=(0.9, 1.05),
                   ncols=5, frameon=True)
        return fig

    def get_projected_probability_figure(self):
        self.get_projected_distribution()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(4, 2.5)
        res, llim, rlim = self.get_radius_histogram()
        res = res.reset_index()
        res['binned'] = res['binned'].apply(lambda x: x.left)
        for key, grp in res.groupby(['stopping_condition']):
            print(grp)
            ax.step(grp['binned'], grp['proportion'], color=PALETTE[key[0]], label=list(SC_DICT.keys())[key[0]])
        self.ax_config(ax=ax, xrange=(llim, rlim), yrange=(1E-4, 1),
                  xlabel='Projected $r_{\\perp}$ / pc', ylabel='Probability $P_{\\mathrm{oc}}$')
        fig.legend(loc='upper right', labels=list(SC_DICT.keys()), reverse=False, bbox_to_anchor=(0.86, 1.01),
                   ncols=5, frameon=True)
        return fig
    
