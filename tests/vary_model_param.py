import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tqdm
from matplotlib import pyplot as plt

from hjmodel.config import *
from hjmodel.hjmodel import eval_system
from hjmodel import model_utils, rand_utils
from test_data.test_model_param_data.analytic_results import Analytic

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

def get_outcome_dict(num_systems, *args):
    outcomes = np.array(Parallel(n_jobs=NUM_CPUS)(
        delayed(eval_system)(*args) for _ in tqdm.tqdm(range(num_systems))
    ))[:, 2]
    unique, counts = np.unique(outcomes, return_counts=True)
    return {x:0 for x in list(SC_DICT.values())} | dict(zip(unique.astype(int), counts/num_systems))

def save_res_for_param(param_name, param_values, num_systems):
    d = []
    for i in range(len(param_values)):
        CANON[param_name] = param_values[i]
        outcome_dict = get_outcome_dict(num_systems[i], *CANON.values())
        error = 1/np.sqrt(num_systems[i])
        d.append(
            {
                param_name: param_values[i],
                'error': error,
            } | outcome_dict
        )
    df = pd.DataFrame(data=d)
    df.to_parquet(f'test_data/test_model_param_data/test_{param_name}.pq', engine='pyarrow')

def vary_density():
    param_values = np.geomspace(1E-4, 1, 10)
    def num_systems_map(param_value):
        if 1E-4 <= param_value < 1E-3:
            return 2000
        elif 1E-3 <= param_value < 1E-2:
            return 2000
        elif 1E-2 <= param_value < 1E-1:
            return 2000
        else:
            return 2000
    num_systems = np.vectorize(num_systems_map)(param_values)
    save_res_for_param(param_name='n_tot', param_values=param_values, num_systems=num_systems)
    df = pd.read_parquet(f'test_data/test_model_param_data/test_n_tot.pq', engine='pyarrow')
    print(df.head())

def add_fig_to_ax(ax, param_name: str,
                  xrange: list[float], yrange: list[float],
                  xlabel: str,
                  clip: dict[str: int]):
    df = pd.read_parquet(f'test_data/test_model_param_data/test_{param_name}.pq', engine='pyarrow')
    for outcome in SC_DICT:
        if clip[outcome] == 0:
            ax.plot(df[param_name], df[outcome],
                    '-s', color=COLOR_DICT[SC_DICT[outcome]][0],
                    label=outcome,
                    zorder=10 - SC_DICT[outcome], markersize=3)
        else:
            ax.plot(df[param_name][0:-clip[outcome]], df[outcome][0:-clip[outcome]],
                    '-s', color=COLOR_DICT[SC_DICT[outcome]][0],
                    label=outcome,
                    zorder=10-SC_DICT[outcome], markersize=3)

    # get analytic predictions
    param_values = np.geomspace(xrange[0], xrange[1], 1000)
    copy = CANON.copy()
    analytic_res = []
    for i in range(len(param_values)):
        copy[param_name] = param_values[i]
        analytic_res.append(Analytic(*copy.values()).get_analytic_probabilities())
    analytic_res_T = np.array(analytic_res).T
    for i in range(analytic_res_T.shape[0]):
        ax.plot(param_values, analytic_res_T[i], linestyle='dotted', color=COLOR_DICT[i])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_xlabel(xlabel)

def save_fig():
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharey=True)

    clip = {'NM': 3,'I': 0,'TD': 0,'HJ': 0,'WJ': 3}
    add_fig_to_ax(ax1, param_name='n_tot',
                  xrange=[1E-4, 1], yrange=[1E-4, 1],
                  xlabel='$n$ / $\\mathrm{pc}^{-3}$', clip=clip)
    add_fig_to_ax(ax2, param_name='n_tot',
                  xrange=[1E-4, 1], yrange=[1E-4, 1],
                  xlabel='$n$ / $\\mathrm{pc}^{-3}$', clip=clip)
    add_fig_to_ax(ax3, param_name='n_tot',
                  xrange=[1E-4, 1], yrange=[1E-4, 1],
                  xlabel='$n$ / $\\mathrm{pc}^{-3}$', clip=clip)

    # style figure
    ax1.plot(np.nan, np.nan, linestyle='dotted', color='black', label='Analytic')
    ax1.set_ylabel('Outcome probability')
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, ncols=6, bbox_to_anchor=(0.77, 1.07), frameon=True)
    fig.tight_layout()
    plt.savefig('test_data/test_model_param_data/test_model_param_plot.pdf', format='pdf', bbox_inches="tight")

if __name__ == '__main__':
    vary_density()
    # save_fig()