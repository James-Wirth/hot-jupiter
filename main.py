import os
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','nature'])

def get_res_path(exp_name: str) -> str:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    res_path = os.path.join(dir_path, 'data', f'exp_data_{exp_name}.pq')
    return res_path

def run(exp_name: str):
    from hjmodel import HJModel
    from clusters.cluster import Cluster
    from clusters.plummer import Plummer

    for i in range(0, 10):
        run_name = f'{exp_name}_RUN{i}'
        res_path = get_res_path(exp_name=run_name)

        plummer_profile = Plummer(N0=2e6, R0=1.91, A=6.991e-4)
        cluster = Cluster(profile=plummer_profile, r_max=100)

        model = HJModel(res_path=res_path)
        model.run_dynamic(time=12000, num_systems=1000, cluster=cluster, hybrid_switch=False)

def plot(exp_name: str):
    from hjmodel import HJModel
    from hjmodel.processor import Processor

    res_path = get_res_path(exp_name=exp_name)
    model = HJModel(res_path=res_path)

    # example to show probability vs projected radius
    fig, ax = plt.subplots()
    processor = Processor(data=model.df)
    print(processor.compute_outcome_probabilities())
    plt.show()



if __name__ == '__main__':
    plot(exp_name="TEST")


