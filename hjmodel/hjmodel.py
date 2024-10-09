import pandas as pd
from joblib import Parallel, delayed
from hjmodel.config import *
from hjmodel import model_utils, rand_utils
from hjmodel.cluster import Plummer
from tqdm import contrib
import matplotlib.pyplot as plt

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
        enough_time_for_next_pert = current_time <= total_time

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
            if (a < R_hj and
                    (not enough_time_for_next_pert or e <= 1E-3)):    # hot Jupiter formation
                stopping_condition = SC_DICT['HJ']
                stopping_time = current_time
                break
            elif (R_hj < a < R_wj
                  and (not enough_time_for_next_pert or e <= 1E-3)):  # warm Jupiter formation
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
                de, da = model_utils.de_SIM_rand_phase(*args)
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

    def __init__(self, time: int, num_systems: int, res_path: str, res_name: str):
        self.time = time
        self.num_systems = num_systems
        self.res_path = res_path
        self.res_name = res_name

    def run(self):
        plummer = Plummer(M0=1.64E6, rt=86, rh=1.91, N=2E6)
        r_vals = plummer.get_radial_distribution(n_samples=self.num_systems)

        # cluster args
        cls = [Parallel(n_jobs=NUM_CPUS)(delayed(plummer.number_density)(r) for r in r_vals),
               Parallel(n_jobs=NUM_CPUS)(delayed(plummer.isotropic_velocity_dispersion)(r) for r in r_vals)]
        # sys args
        sys = rand_utils.get_random_system_params(n_samples=self.num_systems)

        results = Parallel(n_jobs=-1)(
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
        df.to_parquet(f'{self.res_path}{self.res_name}.pq', engine='pyarrow')

    def show_results(self):
        df = pd.read_parquet(f'{self.res_path}{self.res_name}.pq', engine='pyarrow')
        fracs = [df.loc[df['stopping_condition'] == i].shape[0] for i in range(len(SC_DICT))]
        for key in SC_DICT:
            print(f'{key}: {fracs[SC_DICT[key]] / df.shape[0]}')
        for index, row in df.iterrows():
            if row['stopping_condition'] != 1:
                plt.scatter(np.log10(row['final_a']), -np.log10(1 - row['final_e']),
                            color=COLOR_DICT[row['stopping_condition']][0], s=3)
        plt.show()
