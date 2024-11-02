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

# perts per Myr with b_max dependence
def get_perts_per_Myr(local_n_tot, local_sigma_v, b_max):
    return 3.21 * local_n_tot * ((b_max / 75) ** 2) * (local_sigma_v * np.sqrt(2))

CANON = {
    'n_tot': 2.5E-3,
    'sigma_v': 1.266,
    'e_init': 0.5,
    'a_init': 1,
    'm1': 1,
    'm2': 1E-3
}
GAMMA = 0.046 * CANON['n_tot'] * 0.7 * ((CANON['a_init'])/5)**(3/2)

def eval_system(num_perts, b_max):
    e = CANON['e_init']
    a = CANON['a_init']
    for _ in range(num_perts):
        rand_params = rand_utils.random_encounter_params(sigma_v=CANON['sigma_v'], override_b_max=b_max)
        args = (rand_params['v_infty'], rand_params['b'], rand_params['Omega'], rand_params['inc'],
                rand_params['omega'], e, a, CANON['m1'], CANON['m2'], rand_params['m3'])
        de, da = model_utils.de_sim(*args)
        e += de
        a += da
    return e

def try_diffusion(b_max, num_systems, total_time):
    num_perts = int(total_time * get_perts_per_Myr(CANON['n_tot'], CANON['sigma_v'], b_max))
    results = Parallel(n_jobs=NUM_CPUS)(
        delayed(eval_system)(num_perts, b_max) for _ in tqdm.tqdm(range(num_systems))
    )
    return results

def run_diffusion(b_vals):
    results = [try_diffusion(b_max=b_max, num_systems=10000, total_time=4000) for b_max in b_vals]
    np.save('test_data/try_diffusion_data/try_diffusion2.npy', np.array(results, dtype=object), allow_pickle=True)

def plot_diffusion(b_vals):
    b = np.load('test_data/try_diffusion_data/try_diffusion2.npy', allow_pickle=True)
    d = {
        f'{b_vals[i]}': b[i] for i in range(len(b_vals))
    }
    df_sz = pd.DataFrame(d).melt().rename(columns={"value": "e", "variable": "$b_{max}$"})

    N = 200
    eps = 1 / N
    e = np.linspace(-eps / 2, 1 + eps / 2, N + 2)
    d_analytic = {
        f'{b_vals[i]}': get_analytic(b_vals[i], total_time=4000) for i in range(len(b_vals))
    }
    df_sz_analytic = pd.DataFrame(d_analytic).melt().rename(columns={"value": "p", "variable": "$b_{max}$"})

    fig, ax = plt.subplots(figsize=(4, 4))
    g = sns.histplot(data=df_sz, x='e', hue='$b_{max}$', ax=ax, element='step', fill=False,
                     common_norm=False, bins=100, binrange=(0, 1), stat='density')

    h = sns.lineplot(data=df_sz_analytic, x=np.tile(e,len(b_vals)), y='p', hue='$b_{max}$', ax=ax)

    ax.set_xlim(0, 0.995)
    ax.set_ylim(1E-2, 10**1.6)
    ax.set_yscale('log')
    ax.set_ylabel('Probability density (PDF)')
    ax.legend(frameon=True, title='$b_{\\mathrm{max}}$')
    plt.savefig('test_data/try_diffusion_data/try_diffusion2.pdf', format='pdf')
    plt.show()

def get_analytic(b_max, total_time):
    N = 200
    eps = 1 / N

    def gauss(e):
        sigma = 1E-3
        e0 = CANON['e_init']
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-1 * ((e - e0) ** 2) / (2 * sigma ** 2))

    def Gamma(e, eps):
        return GAMMA * e * np.sqrt(1 - e ** 2) * (1 / eps)

    def eps_ij(ei, ej):
        return np.abs(ei - ej)

    # two "phantom points" at -eps/2 and 1+eps/2 to enforce the boundary condition
    e = np.linspace(-eps / 2, 1 + eps / 2, N + 2)
    p = np.zeros(N + 2)
    for i in range(0, p.size):
        p[i] = gauss(e[i])
    num_iterations = int(total_time * get_perts_per_Myr(CANON['n_tot'], CANON['sigma_v'], b_max))
    print(num_iterations)
    dt = total_time / num_iterations

    for _ in range(num_iterations):
        for i in range(0, N + 2):
            if i == 0:
                p[0] = 0
            elif i == N + 1:
                 p[N + 1] = p[N]
            else:
                T1 = 0.5 * (p[i + 1] - p[i]) * Gamma(e[i] + eps / 2, eps / 2)
                T2 = -0.5 * (p[i] - p[i - 1]) * Gamma(e[i] - eps / 2, eps / 2)
                T3 = 0
                for j in range(i + 1, N - 1):
                    T3 += 0.5 * Gamma(e[j] + eps / 2, eps_ij(e[i], e[j]) + eps / 2) * (p[j + 1] - p[j])
                T4 = 0
                for j in range(1, i):
                    T4 += 0.5 * Gamma(e[j] - eps / 2, eps_ij(e[i], e[j]) + eps / 2) * (p[j - 1] - p[j])
                p[i] += (T1 + T2 + T3 + T4) * dt
    integral = p.sum() * eps
    p_norm = p / integral
    return p_norm

if __name__ == '__main__':
    b_vals = [50, 75, 100]
    run_diffusion(b_vals)
    plot_diffusion(b_vals)