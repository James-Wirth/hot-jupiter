from joblib import Parallel, delayed
import tqdm
from hjmodel.config import *
from hjmodel.hjmodel import eval_system, model_utils
import matplotlib.pyplot as plt

CANON = {
    'n_tot': 2.5E-3,
    'sigma_v': 1.266,
    'e_init': 0.3,
    'a_init': 1,
    'm1': 1,
    'm2': 1E-3,
    'total_time': 10000
}

def show_paths():
    num_systems = 2000
    result = np.array(Parallel(n_jobs=NUM_CPUS)(
        delayed(eval_system)(*CANON.values()) for _ in tqdm.tqdm(range(num_systems))
    )).T

    plt.hist(result[2])
    plt.show()

    R_td, R_hj, R_wj = model_utils.get_critical_radii(m1=CANON['m1'], m2=CANON['m2'])

    plt.scatter(np.log10(result[1]), -np.log10(1-result[0]))
    plt.axvline(x=np.log10(R_hj), color='xkcd:red')
    plt.axvline(x=np.log10(R_wj), color='xkcd:orange')

    contour_loga_values = np.linspace(-2, 1, 100)
    contour_y_values = contour_loga_values - np.log10(R_td)
    plt.plot(contour_loga_values, contour_y_values, color='xkcd:blue')

    plt.xlim(-2, 1)
    plt.ylim(0, 2.5)
    plt.show()

if __name__ == '__main__':
    show_paths()