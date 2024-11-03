from hjmodel import HJModel
import os
import matplotlib.pyplot as plt
from hjmodel.cluster import DynamicPlummer
import scienceplots
plt.style.use(['science','nature'])

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cluster_name = '47tuc_new'
    res_path = os.path.join(dir_path, 'data', f'exp_data_{cluster_name}.pq')

    # load a time-dependent Plummer profile instance with
    # cluster parameters defined as per Giersz and Heggie (2011)
    plummer = DynamicPlummer(M0=(1.64E6, 0.9E6),
                             rt=(86, 70),
                             rh=(1.91, 4.96),
                             N=(2E6, 1.85E6),
                             total_time=12000)

    model = HJModel(res_path=res_path)

    # run model for 12 Gyr, with 5E5 Monte-Carlo systems
    model.run_dynamic(time=12000, num_systems=500000, cluster=plummer)

    # summary figures
    fig = model.get_summary_figure()
    plt.savefig(f'data/{cluster_name}_overall.pdf', format='pdf', dpi=1000)

    # outcome probability against projected radius
    fig = model.get_projected_probability_figure()
    plt.savefig(f'data/{cluster_name}_r_proj_override.pdf', format='pdf')

    # overall outcome probabilities
    print(model.get_outcome_probabilities())

if __name__ == '__main__':
    main()

