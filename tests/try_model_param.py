import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tqdm
from matplotlib import pyplot as plt
from hjmodel.config import *
from hjmodel import model_utils, rand_utils
from test_data.test_model_param_data.analytic_results import Analytic

import scienceplots
plt.style.use(['science','nature'])

# canonical values (unless the parameter in question is being varied)
CANON = {
    'n_tot': 2.5E-3,
    'sigma_v': 1.266,
    'e_init': 0.3,
    'a_init': 1,
    'm1': 1,
    'm2': 1E-3,
    'total_time': 10000
}

def eval_system(local_n_tot: float, local_sigma_v: float,
                e_init: float, a_init: float, m1: float, m2: float,
                total_time: int) -> list:
    """
    A modified version of the eval_system_dynamic method
    in the HJModel class,
    with fixed local_n_tot and local_sigma_v

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

def get_outcome_dict(num_systems, *args):
    outcomes = np.array(Parallel(n_jobs=NUM_CPUS)(
        delayed(eval_system)(*args) for _ in tqdm.tqdm(range(num_systems))
    )).T
    unique, counts = np.unique(outcomes[2], return_counts=True)
    return {x:0 for x in list(SC_DICT.values())} | dict(zip(unique.astype(int), counts/num_systems))

def save_res_for_param(param_name, param_values, num_systems):
    d = []
    copy = CANON.copy()
    for i in range(len(param_values)):
        copy[param_name] = param_values[i]
        outcome_dict = get_outcome_dict(num_systems[i], *copy.values())
        error = 1/np.sqrt(num_systems[i])
        d.append(
            {
                param_name: param_values[i],
                'error': error,
            } | outcome_dict
        )
    df = pd.DataFrame(data=d)
    df.to_parquet(f'test_data/test_model_param_data/test_{param_name}.pq', engine='pyarrow')

def vary_param(param_name: str, param_values: np.ndarray, num_systems: list[int]):
    assert param_values.shape[0] == len(num_systems)
    save_res_for_param(param_name=param_name, param_values=param_values, num_systems=num_systems)

def add_fig_to_ax(ax, param_name: str,
                  xrange: list[float], yrange: list[float],
                  xlabel: str,
                  clip: dict[str: int]):
    df = pd.read_parquet(f'test_data/test_model_param_data/test_{param_name}.pq', engine='pyarrow').sort_values(by=param_name)
    if param_name == 'sigma_v':
        df[param_name] = df[param_name]/0.211
    elif param_name == 'n_tot':
        df[param_name] = df[param_name] * 10**6
    for outcome in SC_DICT.values():
        if clip[outcome] == 0:
            ax.plot(df[param_name], df[f'{outcome}'],
                    color=COLOR_DICT[outcome][0],
                    label={v: k for k, v in SC_DICT.items()}[outcome],
                    zorder=10 - outcome)
        else:
            ax.plot(df[param_name][0:-clip[outcome]], df[f'{outcome}'][0:-clip[outcome]],
                    color=COLOR_DICT[outcome][0],
                    label={v: k for k, v in SC_DICT.items()}[outcome],
                    zorder=10-outcome)

    # get analytic predictions
    param_values = np.geomspace(xrange[0], xrange[1], 1000)
    copy = CANON.copy()
    analytic_res = []
    for i in range(len(param_values)):
        copy[param_name] = param_values[i]
        analytic_res.append(Analytic(*copy.values()).get_analytic_probabilities())
    analytic_res_T = np.array(analytic_res).T

    if param_name == 'sigma_v':
        param_values = param_values/0.211
        xrange = (xrange[0] / 0.211, xrange[1] / 0.211)
    elif param_name == 'n_tot':
        param_values = param_values * 10**6
        xrange = (xrange[0] * 10**6, xrange[1] * 10**6)
    for i in range(analytic_res_T.shape[0]):
        ax.plot(param_values, analytic_res_T[i], linestyle='dotted', color=COLOR_DICT[i][0])
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)

    if xlabel != '':
        ax.set_xlabel(xlabel)

def save_fig():
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 3), gridspec_kw={'height_ratios':[1,1],
                                                                           'hspace':0.15,
                                                                           'wspace':0.05})
    # density plot
    clip = {SC_DICT['NM']: 3,SC_DICT['I']: 0,SC_DICT['TD']: 0,SC_DICT['HJ']: 0,SC_DICT['WJ']: 3}
    add_fig_to_ax(axs[0,0], param_name='n_tot',
                  xrange=[1E-4, 1], yrange=[0, 1],
                  xlabel='', clip=clip)
    axs[0,0].set_xscale('log')
    axs[0,0].set_xticks([])
    add_fig_to_ax(axs[1, 0], param_name='n_tot',
                  xrange=[1E-4, 1], yrange=[1E-3, 1E-1],
                  xlabel='$n$ / $\\mathrm{pc}^{-3}$', clip=clip)
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')

    # velocity dispersion plot
    clip = {SC_DICT['NM']: 0, SC_DICT['I']: 0, SC_DICT['TD']: 0, SC_DICT['HJ']: 0, SC_DICT['WJ']: 0}
    add_fig_to_ax(axs[0,1], param_name='sigma_v',
                  xrange=[0.211, 6.33], yrange=[0, 1],
                  xlabel='', clip=clip)
    axs[0, 1].set_xticks([])
    add_fig_to_ax(axs[1, 1], param_name='sigma_v',
                  xrange=[0.211, 6.33], yrange=[1E-3, 1E-1],
                  xlabel='$\\sigma$ / $\\mathrm{km} \\mathrm{s}^{-1}$', clip=clip)
    axs[1, 1].set_yscale('log')
    axs[0, 1].set_yticks([])
    axs[1, 1].set_yticks([])

    # initial semi-major axis plot
    clip = {SC_DICT['NM']: 0, SC_DICT['I']: 0, SC_DICT['TD']: 0, SC_DICT['HJ']: 0, SC_DICT['WJ']: 0}
    add_fig_to_ax(axs[0,2], param_name='a_init',
                  xrange=[1E0, 1E1], yrange=[0, 1],
                  xlabel='', clip=clip)
    axs[0,2].set_xticks([])
    add_fig_to_ax(axs[1, 2], param_name='a_init',
                  xrange=[1E0, 1E1], yrange=[1E-3, 1E-1],
                  xlabel='$a_0$ / au', clip=clip)
    axs[1, 2].set_yscale('log')
    axs[0, 2].set_yticks([])
    axs[1, 2].set_yticks([])

    # style figure
    axs[0,0].plot(np.nan, np.nan, linestyle='dotted', color='black', label='Analytic')
    axs[0,0].set_ylabel('Probability $P_{\\mathrm{oc}}$')
    h, l = axs[0,0].get_legend_handles_labels()
    fig.legend(h, l, ncols=6, bbox_to_anchor=(0.72, 0.98), frameon=True)
    fig.tight_layout()
    plt.savefig('test_data/test_model_param_data/test_model_param_plot.pdf', format='pdf', bbox_inches="tight")


if __name__ == '__main__':
    vary_param(param_name='n_tot', param_values=np.geomspace(1E-4, 1, 20), num_systems=[40000]*20)
    vary_param(param_name='sigma_v', param_values=np.linspace(0.211, 6.33, 20), num_systems=[40000]*20)
    vary_param(param_name='a_init', param_values=np.linspace(1, 10, 20), num_systems=[40000]*20)
    save_fig()