"""
An example simulation run
"""

import os

from hjmodel import HJModel

from clusters.cluster import Cluster
from clusters.plummer import Plummer

from hjmodel.processor import Processor

def _get_res_path(exp_name: str) -> str:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, 'data', f'exp_data_{exp_name}.pq')

def run(exp_name: str):

    cluster = Cluster(
        profile=Plummer(N0=2e6, R0=1.91, A=6.991e-4),
        r_max=100
    )

    for i in range(0, 10):
        res_path = _get_res_path(exp_name=f'{exp_name}_RUN{i}')
        model = HJModel(res_path=res_path)
        model.run_dynamic(time=12000, num_systems=1000, cluster=cluster, hybrid_switch=True)
        pass

def plot(exp_name: str):
    res_path = _get_res_path(exp_name=exp_name)
    model = HJModel(res_path=res_path)

    # e.g. output outcome probabilities
    processor = Processor(data=model.df)
    print(processor.compute_outcome_probabilities())



if __name__ == '__main__':
    plot(exp_name="TEST")


