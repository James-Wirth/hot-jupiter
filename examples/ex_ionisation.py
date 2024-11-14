import numpy as np
from hjmodel import model_utils, rand_utils
from hjmodel.config import *
from joblib import Parallel, delayed
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import scienceplots
plt.style.use(['science','nature'])

CANON = {
    'sigma_v': 1.266,
    'e_init': 0.3,
    'a_init': 1,
    'm1': 1,
    'm2': 1E-3
}

def eval_system(num_perts):
    e = CANON['e_init']
    a = CANON['a_init']
    for i in range(num_perts):
        rand_params = rand_utils.random_encounter_params(sigma_v=CANON['sigma_v'])
        args = {
            'v_infty':      rand_params['v_infty'],
            'b':            rand_params['b'],
            'Omega':        rand_params['Omega'],
            'inc':          rand_params['inc'],
            'omega':        rand_params['omega'],
            'e_init':       e,
            'a_init':       a,
            'm1':           CANON['m1'],
            'm2':           CANON['m2'],
            'm3':           rand_params['m3']
        }
        de, da = model_utils.de_sim(*[args[x] for x in args])
        e += de
        a += da
        if e > 1:
            return i
    return np.nan

def evaluate_density(n_tot, num_systems, total_time):
    num_perts = int(total_time * model_utils.get_perts_per_Myr(n_tot, CANON['sigma_v']))
    results = Parallel(n_jobs=NUM_CPUS)(
        delayed(eval_system)(num_perts) for _ in tqdm.tqdm(range(num_systems))
    )
    return results / model_utils.get_perts_per_Myr(n_tot, CANON['sigma_v'])

def run_diffusion(n_tots):
    d = {n_tot: evaluate_density(n_tot=n_tot, num_systems=200, total_time=10000) for n_tot in n_tots}
    df = pd.DataFrame(data=d)
    df.to_parquet('test_data/try_diffusion_data/try_ionisation.pq', engine='pyarrow')

def plot_diffusion(n_tots):
    df = pd.read_parquet('test_data/try_diffusion_data/try_ionisation.pq', engine='pyarrow')
    fig, ax = plt.subplots()
    for i in range(len(n_tots)):
        sns.histplot(x=df[n_tots[i]][df[n_tots[i]].notnull()] / 1000, cumulative=True, bins=100,
                     stat='probability', fill=None, element='step', ax=ax, label=n_tots[i])
    ax.legend()
    plt.show()

if __name__ == '__main__':
    n_tots = np.geomspace(5E-3, 5E-2, 3)
    run_diffusion(n_tots)
    plot_diffusion(n_tots)