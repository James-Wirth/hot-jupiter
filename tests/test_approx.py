import os
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d
from tqdm import contrib
from hjmodel.config import *
from hjmodel import model_utils, rand_utils

def eval_pert(sigma_v: float, e: float, a: float, m1: float, m2: float):
    rand_params = rand_utils.random_encounter_params(sigma_v=sigma_v)
    args = {
        'v_infty':  rand_params['v_infty'],
        'b':        rand_params['b'],
        'Omega':    rand_params['omega'],
        'inc':      rand_params['inc'],
        'omega':    rand_params['omega'],
        'e_init':   e,
        'a_init':   a,
        'm1':       m1,
        'm2':       m2,
        'm3':       rand_params['m3']
    }
    tidal_param = model_utils.tidal_param(*(args[x] for x in ['v_infty', 'b', 'a', 'm1', 'm2']))
    slow_param = model_utils.slow_param(*(args[x] for x in ['v_infty', 'b', 'a', 'm1', 'm2']))

    de_sim = np.abs(model_utils.de_SIM_rand_phase(*args.values())[0])
    de_hr = np.abs(model_utils.de_HR(*args.values()))

    err = np.abs(de_sim - de_hr) / de_hr
    is_ionising = (e + de_sim >= 1) or (e + de_hr >= 1)

    return [tidal_param, slow_param, err, is_ionising]

def get_approx(sigma_v: float, num_systems: int):
    sys = rand_utils.get_random_system_params(n_samples=num_systems)
    results = Parallel(n_jobs=NUM_CPUS)(
        delayed(eval_pert)(sigma_v, *args) for args in contrib.tzip(*sys)
    )
    d = {
        'tidal_param':  [row[0] for row in results],
        'slow_param':   [row[1] for row in results],
        'err':          [row[2] for row in results],
        'is_ionising':  [row[3] for row in results]
    }
    df = pd.DataFrame(data=d)
    df.to_parquet(f'test_data/test_approx_data/test_approx_{sigma_v}.pq', engine='pyarrow')

def thresh_mean(arr):
    return np.mean(arr) if arr.shape[0] > 5 else np.nan

def test_approx(sigma_v=1.266, num_systems=500000):
    path = f'test_data/test_approx_data/test_approx_{sigma_v}.pq'
    if not os.path.exists(path):
        get_approx(sigma_v=sigma_v, num_systems=num_systems)
    df = pd.read_parquet(path, engine='pyarrow')

    max_t, max_s = 60, 1000
    t_bins = np.linspace(0, max_t, 25)
    s_bins = np.linspace(0, max_s, 25)
    ret = binned_statistic_2d(df['tidal_param'], df['slow_param'], np.log10(df['err']),
                              statistic=thresh_mean, bins=[t_bins, s_bins])

    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 0.05]})
    fig.set_size_inches(2.5, 2)
    fig.subplots_adjust(wspace=0.25)

    img = axs[0].imshow(ret.statistic.T, origin='lower', extent=(0, max_t, 0, max_s),
                        cmap='RdYlBu_r', aspect=max_t / max_s, vmin=-2, vmax=0.5)
    axs[0].set_xlabel('$T$')
    axs[0].set_ylabel('$S$')
    axs[0].set_xlim(0, max_t)
    axs[0].set_ylim(0, max_s)
    axs[0].legend(frameon=True)
    axs[0].set_facecolor('lightgray')

    cb2 = plt.colorbar(img, cax=axs[1], ax=[axs[0]])
    cb2.set_label(label='$\log \Delta$')

    plt.savefig(f'test_data/test_approx_data/test_approx_{sigma_v}.pdf', format='pdf', bbox_inches="tight")
