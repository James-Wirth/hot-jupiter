import os

import matplotlib
import pandas as pd
pd.options.mode.chained_assignment = None
from joblib import Parallel, delayed
from hjmodel.config import *
from hjmodel import model_utils, rand_utils
from hjmodel.cluster import Plummer, DynamicPlummer
from tqdm import contrib
import matplotlib.pyplot as plt
import seaborn as sns

def get_p_oc(x, y_rel, d):
    Q = np.array([d/(x[i]*np.sqrt(x[i]**2-d**2)) if x[i] > d else 0 for i in range(len(x))])
    Q_num = Q * y_rel
    return np.nansum(Q_num)/np.nansum(Q)

def eval_system_dynamic(e_init: float, a_init: float, m1: float, m2: float,
                r: float, cluster: DynamicPlummer, total_time: int) -> list:

    R_td, R_hj, R_wj = model_utils.get_critical_radii(m1=m1, m2=m2)

    # initialise running variables
    e, a = e_init, a_init
    current_time = stopping_time = 0
    stopping_condition = SC_DICT['NM']

    while current_time < total_time:

        # get environment vars from cluster object
        local_n_tot = cluster.number_density(r, current_time)
        local_sigma_v = cluster.isotropic_velocity_dispersion(r, current_time)
        perts_per_Myr = model_utils.get_perts_per_Myr(local_n_tot=local_n_tot, local_sigma_v=local_sigma_v)

        # get random encounter parameters and random wait time until the next stochastic kick
        rand_params = rand_utils.random_encounter_params(sigma_v=local_sigma_v)
        wt_time = rand_utils.get_waiting_time(perts_per_Myr=perts_per_Myr)

        # check if there is sufficient time for the next stochastic kick
        current_time = min(current_time + wt_time, total_time)
        enough_time_for_next_pert = current_time < total_time

        # check stopping conditions
        if e >= 1:
            stopping_condition = SC_DICT['I']                         # ionisation
            stopping_time = current_time - wt_time
            break
        elif a * (1 - e) < R_td:
            stopping_condition = SC_DICT['TD']                        # tidal disruption
            stopping_time = current_time - wt_time
            break
        else:
            # tidal evolution step
            e, a = model_utils.tidal_effect(e=e, a=a, m1=m1, m2=m2, time_in_Myr=wt_time)
            if a < R_hj and (not enough_time_for_next_pert or e <= 1E-3):    # hot Jupiter formation
                stopping_condition = SC_DICT['HJ']
                stopping_time = current_time
                break
            elif R_hj < a < R_wj and (not enough_time_for_next_pert or e <= 1E-3):  # warm Jupiter formation
                stopping_condition = SC_DICT['WJ']
                stopping_time = current_time
                break
            elif e <= 1E-3:
                stopping_time = current_time                          # circularised but no migration
                break

        # apply stochastic kick
        if stopping_condition == SC_DICT['NM'] and enough_time_for_next_pert and e >= 0:
            args = (rand_params['v_infty'], rand_params['b'], rand_params['Omega'], rand_params['inc'],
                    rand_params['omega'], e, a, m1, m2, rand_params['m3'])
            if model_utils.is_analytic_valid(*args, sigma_v=local_sigma_v):
                e += model_utils.de_HR(*args)
            else:
                de, da = model_utils.de_sim(*args)
                e += de
                a += da

    return [e, a, stopping_condition, stopping_time]

def eval_system(local_n_tot: float, local_sigma_v: float,
                e_init: float, a_init: float, m1: float, m2: float,
                total_time: int) -> list:
    """
    Calculates outcome for a given (randomised) system in the cluster

    Inputs
    ----------
    local_n_tot: float          Local stellar number density                per 10^6 cubic parsecs
    local_sigma_v: float        Local isotropic velocity dispersion         au per year
    e_init: float               Initial eccentricity (Rayleigh)
    a_init: float               Initial semi-major axis                     au
    m1: float                   Host star mass                              M_solar
    m2: float                   Planet mass                                 M_solar
    total_time: int             Time duration of simulation                 Myr

    Returns
    ----------
    [e,                         Final eccentricity                          arbitrary units
    a,                          Final semi-major axis                       au
    stopping_condition,         Stopping condition {NM, I, TD, HJ, WJ}      N/A
    stopping_time]              Stopping time                               Myr
    : list
    """

    R_td, R_hj, R_wj = model_utils.get_critical_radii(m1=m1, m2=m2)
    perts_per_Myr = model_utils.get_perts_per_Myr(local_n_tot=local_n_tot, local_sigma_v = local_sigma_v)

    # initialise running variables
    e, a = e_init, a_init
    current_time = stopping_time = 0
    stopping_condition = SC_DICT['NM']

    while current_time < total_time:
        # get random encounter parameters and random wait time until the next stochastic kick
        rand_params = rand_utils.random_encounter_params(sigma_v=local_sigma_v)
        wt_time = rand_utils.get_waiting_time(perts_per_Myr=perts_per_Myr)

        # check if there is sufficient time for the next stochastic kick
        current_time = min(current_time + wt_time, total_time)
        enough_time_for_next_pert = current_time < total_time

        # check stopping conditions
        if e >= 1:
            stopping_condition = SC_DICT['I']                         # ionisation
            stopping_time = current_time - wt_time
            break
        elif a * (1 - e) < R_td:
            stopping_condition = SC_DICT['TD']                        # tidal disruption
            stopping_time = current_time - wt_time
            break
        else:
            # tidal evolution step
            e, a = model_utils.tidal_effect(e=e, a=a, m1=m1, m2=m2, time_in_Myr=wt_time)
            if a < R_hj and (not enough_time_for_next_pert or e <= 1E-3):    # hot Jupiter formation
                stopping_condition = SC_DICT['HJ']
                stopping_time = current_time
                break
            elif R_hj < a < R_wj and (not enough_time_for_next_pert or e <= 1E-3):  # warm Jupiter formation
                stopping_condition = SC_DICT['WJ']
                stopping_time = current_time
                break
            elif e <= 1E-3:
                stopping_time = current_time                          # circularised but no migration
                break

        # apply stochastic kick
        if stopping_condition == SC_DICT['NM'] and enough_time_for_next_pert and e >= 0:
            args = (rand_params['v_infty'], rand_params['b'], rand_params['Omega'], rand_params['inc'],
                    rand_params['omega'], e, a, m1, m2, rand_params['m3'])
            if model_utils.is_analytic_valid(*args, sigma_v=local_sigma_v):
                e += model_utils.de_HR(*args)
            else:
                de, da = model_utils.de_sim(*args)
                e += de
                a += da

    return [e, a, stopping_condition, stopping_time]

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
        if os.path.exists(self.path):
            self.df = pd.read_parquet(self.path, engine='pyarrow')
        else:
            self.df = None

    def check_overwrite(self):
        ans = input(f'Data already exists at {self.path}.\nDo you want to overwrite? (Y/n): ')
        while not ans.lower() in ['y', 'yes', 'n', 'no']:
            ans = input('Invalid response. Please re-enter (Y/n): ')
        return ans.lower() in ['y', 'yes']

    def run(self, time: int, num_systems: int):
        if self.df is not None:
            if not self.check_overwrite():
                print('Run interrupted.')
                return
        self.time = time
        self.num_systems = num_systems
        print(f'Evaluating N = {self.num_systems} systems (for t = {self.time} Myr)')

        plummer = Plummer(M0=1.64E6, rt=86, rh=1.91, N=2E6)
        r_vals = plummer.get_radial_distribution(n_samples=self.num_systems)

        # cluster args
        cls = [Parallel(n_jobs=NUM_CPUS)(delayed(plummer.number_density)(r) for r in r_vals),
               Parallel(n_jobs=NUM_CPUS)(delayed(plummer.isotropic_velocity_dispersion)(r) for r in r_vals)]
        # sys args
        sys = rand_utils.get_random_system_params(n_samples=self.num_systems)

        results = Parallel(n_jobs=NUM_CPUS)(
            delayed(eval_system)(*args, self.time) for args in contrib.tzip(*cls, *sys)
        )
        d = {
            'r': r_vals,
            'final_e': [row[0] for row in results],
            'final_a': [row[1] for row in results],
            'stopping_condition': [row[2] for row in results],
            'stopping_time': [row[3] for row in results],
            'e_init': sys[0],
            'a_init': sys[1],
            'm1': sys[2]
        }
        df = pd.DataFrame(data=d)
        df.to_parquet(self.path, engine='pyarrow')
        self.df = df

    def run_dynamic(self, time: int, num_systems: int):
        if self.df is not None:
            if not self.check_overwrite():
                print('Run interrupted.')
                return
        self.time = time
        self.num_systems = num_systems
        print(f'Evaluating N = {self.num_systems} systems (for t = {self.time} Myr)')

        # 47 Tuc
        plummer = DynamicPlummer(M0=(1.64E6, 0.9E6),
                                 rt=(86, 70),
                                 rh=(1.91, 4.96),
                                 N=(2E6, 1.85E6),
                                 total_time=12000)

        r_vals = plummer.get_radial_distribution(n_samples=self.num_systems)
        # sys args
        sys = rand_utils.get_random_system_params(n_samples=self.num_systems)

        results = Parallel(n_jobs=NUM_CPUS)(
            delayed(eval_system_dynamic)(*args, plummer, self.time) for args in contrib.tzip(*sys, r_vals)
        )

        d = {
            'r': r_vals,
            'final_e': [row[0] for row in results],
            'final_a': [row[1] for row in results],
            'stopping_condition': [row[2] for row in results],
            'stopping_time': [row[3] for row in results],
            'e_init': sys[0],
            'a_init': sys[1],
            'm1': sys[2]
        }
        df = pd.DataFrame(data=d)
        df.to_parquet(self.path, engine='pyarrow')
        self.df = df

    def plot_outcomes(self, ax):
        df_filt = self.df.loc[self.df['stopping_condition'].isin([1])==False]
        df_filt['x'] = df_filt['final_a']
        df_filt['y'] = 1/(1-df_filt['final_e'])

        indexes = df_filt[(df_filt['stopping_condition'] == 0)].sample(frac=0.5).index
        df_filt = df_filt.drop(indexes)
        indexes = df_filt[df_filt['stopping_condition'] == 2].sample(frac=0.5).index
        df_filt = df_filt.drop(indexes)

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

    def get_a_init_histogram(self, label='a_init', num_bins=30):
        bins = np.geomspace(1, 30, num_bins)
        self.df['a_init_binned'] = pd.cut(self.df[label], bins)
        is_multi = self.df["a_init_binned"].value_counts() > 1000
        filtered = self.df.loc[self.df["a_init_binned"].isin(is_multi[is_multi].index)].reset_index(drop=True)
        ret = filtered.groupby(['a_init_binned'])['stopping_condition'].value_counts(normalize=True)
        return ret, filtered['a_init_binned'].min().left, filtered['a_init_binned'].max().left
