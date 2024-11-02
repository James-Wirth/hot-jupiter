import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hjmodel.config import *
from joblib import Parallel, delayed
import tqdm
import hjmodel.model_utils

import scienceplots
plt.style.use(['science','nature'])

def eval_pert(*args):
    return [hjmodel.model_utils.de_sim(*args)[0],
            hjmodel.model_utils.de_HR(*args)]

def eval_hybrid_params(*args):
    return [hjmodel.model_utils.tidal_param(*args),
            hjmodel.model_utils.slow_param(*args)]

def get_v_infty_dependence():
    args = {
        'b': 30,
        'Omega': 1,
        'inc': 1,
        'omega': 1,
        'e_init': 0.3,
        'a_init': 1,
        'm1': 1,
        'm2': 1E-3,
        'm3': 1
    }
    v_infty_values = np.geomspace(1E-1, 1E2, 10000)
    pert_results = np.array(Parallel(n_jobs=NUM_CPUS)(
        delayed(eval_pert)(v_infty, *args.values()) for v_infty in tqdm.tqdm(v_infty_values)
    )).T
    hybrid_params_results = np.array(Parallel(n_jobs=NUM_CPUS)(
        delayed(eval_hybrid_params)(v_infty, *(args[x] for x in ['b', 'a_init', 'm1', 'm2'])) for v_infty in tqdm.tqdm(v_infty_values)
    )).T
    d = {
        'v_infty':     v_infty_values,
        'de_sim':      np.abs(pert_results[0]),
        'de_hr':       np.abs(pert_results[1]),
        'tidal_param': hybrid_params_results[0],
        'slow_param':  hybrid_params_results[1]
    }
    df = pd.DataFrame(data=d)
    df.to_parquet('test_data/test_v_infty_dependence_data/test_v_infty_dependence.pq', engine='pyarrow')

def ax_config(ax, xrange, yrange, xlabel, ylabel, ax_type=None):
    if ax_type is not None:
        ax.set_xscale(ax_type)
        ax.set_yscale(ax_type)
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticks([])
    if ylabel is not None:
        ax.set_ylabel(ylabel)

def test_v_infty_dependence():
    path = 'test_data/test_v_infty_dependence_data/test_v_infty_dependence.pq'
    if not os.path.exists(path):
        get_v_infty_dependence()
    df = pd.read_parquet(path, engine='pyarrow')

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 0.6]})
    fig.subplots_adjust(hspace=0.1)

    # excitation against v_infty plot
    # ax1.scatter(df['v_infty'], df['de_sim'],
    #             label='$\\epsilon_{\\mathrm{sim}}$', s=1, color='xkcd:light grey')
    ax1.plot(df['v_infty'], df['de_sim'].rolling(50, center=True).mean(),
                label='$\\langle \\epsilon_{\\mathrm{sim}} \\rangle_{\\mathrm{\\phi}}$', color='xkcd:black', linestyle='solid')
    ax1.plot(df['v_infty'], df['de_hr'],
                label='$\\epsilon_{\\mathrm{hr}}$', color='xkcd:red')
    ax1.axhline(0.7,
                label='$\\epsilon_{\\mathrm{ion}}=1-e_0$', linestyle='dashed', color='green')
    ax_config(ax1,
              xrange=(1E-1, 1E2),
              yrange=(1E-5, 1E2),
              xlabel=None,
              ylabel='Eccentricity excitation $|\\epsilon|$',
              ax_type='log')
    ax1.legend(frameon=True)

    ax1.axvspan(1.4, 25, color="gray", alpha=0.1)

    # T and S against v_infty plot
    ln2 = ax2.plot(df['v_infty'], df['tidal_param']/15,
             label='$T/T_{\\mathrm{min}}$', linestyle=(0, (5, 1)), color='xkcd:eggplant purple')
    ln3 = ax2.plot(df['v_infty'], df['slow_param']/300,
             label='$S/S_{\\mathrm{min}}$', linestyle=(0, (1, 1)), color='xkcd:eggplant purple')
    ax2.axvspan(1.4, 25, color="gray", alpha=0.1)
    # ax2.axhline(y=1, color='green')
    ax2.annotate("", (1.4, 0.3), (25, 0.3), arrowprops={'arrowstyle': '<->'})
    ax2.annotate('$\\mathcal{D}$', xy=(5, 0.4), textcoords='data', fontsize=14, color='black')
    ax_config(ax2, xrange=(1E-1, 1E2),
              yrange=(1E-1, 14),
              xlabel='$v_{\\infty}$ / $\\mathrm{au} \\ \\mathrm{yr}^{-1}$',
              ylabel=None, ax_type='log')
    ax2.legend(frameon=True)

    fig.set_size_inches(4, 4)
    fig.tight_layout()
    fig.savefig('test_data/test_v_infty_dependence_data/test_v_infty_dependence.pdf', format='pdf')

def test_x():
    get_v_infty_dependence()
