import os
import pandas as pd
from matplotlib import pyplot as plt
from hjmodel.config import *
from joblib import Parallel, delayed
import tqdm
import hjmodel.model_utils

import scienceplots
plt.style.use(['science','nature'])

def mean_de_sim(*args):
    return np.mean([hjmodel.model_utils.de_sim(*args)[0] for _ in range(100)])

def eval_pert(*args):
    return [mean_de_sim(*args),
            hjmodel.model_utils.de_HR(*args)]

def eval_hybrid_params(*args):
    return [hjmodel.model_utils.tidal_param(*args),
            hjmodel.model_utils.slow_param(*args)]

def get_v_infty_dependence():
    args = {
        'b': 50,
        'Omega': 1,
        'inc': 1,
        'omega': 1,
        'e_init': 0.3,
        'a_init': 1,
        'm1': 1,
        'm2': 1E-3,
        'm3': 1
    }
    v_infty_values = np.geomspace(1E-1, 1E3, 1000)
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

def try_v_infty_dependence():
    path = 'test_data/test_v_infty_dependence_data/test_v_infty_dependence.pq'
    if not os.path.exists(path):
        get_v_infty_dependence()
    df = pd.read_parquet(path, engine='pyarrow').sort_values(by='v_infty')

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 0.6]})
    fig.subplots_adjust(hspace=0.1)

    ax1.plot(df['v_infty'], df['de_sim'],
                label='$\\langle \\epsilon_{\\mathrm{sim}} \\rangle_{\\mathrm{\\phi}}$', color='xkcd:black',
                linestyle='solid', linewidth=0.75)
    ax1.plot(df['v_infty'], df['de_hr'], label='$\\epsilon_{\\mathrm{hr}}$', color='xkcd:red')
    ax1.axhline(0.7, label='$\\epsilon_{\\mathrm{ion}}=1-e_0$', linestyle='dashed', color='green')
    ax_config(ax1,
              xrange=(10**(-1), 1E3),
              yrange=(10**(-6.7), 20),
              xlabel=None,
              ylabel='Eccentricity excitation $|\\epsilon|$',
              ax_type='log')
    ax1.legend(frameon=True, loc='upper right')

    ax1.axvspan(1E-1, 0.73, color="gray", alpha=0.1)
    ax1.axvspan(46, 1E3, color="gray", alpha=0.1)

    # T and S against v_infty plot
    ln2 = ax2.plot(df['v_infty'], df['tidal_param']/15,
             label='$T/T_{\\mathrm{min}}$', linestyle=(0, (5, 1)), color='xkcd:black')
    ln3 = ax2.plot(df['v_infty'], df['slow_param']/300,
             label='$S/S_{\\mathrm{min}}$', linestyle=(0, (1, 1)), color='xkcd:black')
    ax2.axvspan(1E-1, 0.73, color="gray", alpha=0.1)
    ax2.axvspan(46, 1E3, color="gray", alpha=0.1)
    ax2.annotate("", (0.73, 0.3), (46, 0.3), arrowprops={'arrowstyle': '<->'}, color='xkcd:dark grey')
    ax2.annotate('$\\mathcal{D}$', xy=(5, 0.4), textcoords='data', fontsize=14, color='xkcd:dark grey')
    ax_config(ax2, xrange=(10**(-1), 1E3),
              yrange=(1E-1, 30),
              xlabel='$v_{\\infty}$ / $\\mathrm{au} \\ \\mathrm{yr}^{-1}$',
              ylabel=None, ax_type='log')
    ax2.legend(frameon=True, loc='upper right')

    fig.set_size_inches(4, 3)
    fig.tight_layout()
    fig.savefig('test_data/test_v_infty_dependence_data/test_v_infty_dependence.pdf', format='pdf')

if __name__ == '__main__':
    try_v_infty_dependence()