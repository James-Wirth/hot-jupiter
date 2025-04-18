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

    for i in range(0, 1):
        cluster_name = f'47_TUC_HYBRID_RUN{i}'
        res_path = os.path.join(dir_path, 'data', f'exp_data_{cluster_name}.pq')

        plummer = Plummer()
        model = HJModel(res_path=res_path)
        model.run_dynamic(time=12000, num_systems=50000, cluster=plummer, hybrid_switch=True)


if __name__ == '__main__':
    main()


