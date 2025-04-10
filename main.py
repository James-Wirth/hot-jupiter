from hjmodel import HJModel
import os
import matplotlib.pyplot as plt
from hjmodel.fixed_cluster import Plummer
import scienceplots

import dask.dataframe as dd

from pathlib import Path
plt.style.use(['science','nature'])

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cluster_name = '47tuc_FINAL_BATCHED_ANALYTIC'
    res_path = os.path.join(dir_path, 'data', f'exp_data_{cluster_name}.pq')

    # load a time-dependent Plummer profile instance with
    # cluster parameters defined as per Giersz and Heggie (2011)
    plummer = Plummer(M0=(1.64E6, 0.9E6),
                             rt=(86, 70),
                             rh=(1.91, 4.96),
                             N=(2E6, 1.85E6),
                             total_time=12000)

    model = HJModel(res_path=res_path)

    # run model for 12 Gyr, with 5E5 Monte-Carlo systems
    model.run_dynamic(time=12000, num_systems=58000, cluster=plummer, hybrid_switch=False)


if __name__ == '__main__':
    main()


