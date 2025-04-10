from matplotlib.gridspec import GridSpec
from hjmodel.config import SC_DICT

from hjmodel import HJModel
import os
import matplotlib.pyplot as plt
from hjmodel.cluster import DynamicPlummer
import scienceplots
plt.style.use(['science','nature'])

def get_res_path(cluster_name: str):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    res_path = os.path.join(dir_path, 'data', f'exp_data_{cluster_name}.pq')
    return res_path

def comparison_fig():
    analytic_model = HJModel(res_path=get_res_path('47tuc_FINAL_BATCHED_ANALYTIC'))
    hybrid_model = HJModel(res_path=get_res_path('47tuc_FINAL_BATCHED'))

    fig = plt.figure()
    fig.set_size_inches(8, 5)
    gs = GridSpec(2, 2, figure=fig)

    ax20 = fig.add_subplot(gs[0, 0])
    ax21 = fig.add_subplot(gs[0, 1], sharey=ax20)
    ax30 = fig.add_subplot(gs[1, 0])
    ax31 = fig.add_subplot(gs[1, 1], sharey=ax30)

    analytic_model.plot_phase_plane_fig(ax20)
    hybrid_model.plot_phase_plane_fig(ax21)
    analytic_model.plot_projected_probability_fig(ax30)
    hybrid_model.plot_projected_probability_fig(ax31)

    lines = ax31.get_lines()  # Get all line objects from the axis
    labels = [line.get_label() for line in lines if line.get_label() != "_nolegend_"]

    ax20.set_title('Analytic', fontsize=11, fontweight='bold')
    ax21.set_title('Hybrid', fontsize=11, fontweight='bold')

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, hspace=0.3, wspace=0.1)
    fig.legend(lines, labels, loc='lower center', ncol=len(labels), fontsize=9, frameon=True)
    plt.savefig(f'data/comparison_fig3.pdf', format='pdf', dpi=1000)
    return


def stopping_time_fig():
    hybrid_model = HJModel(res_path=get_res_path('47tuc_FINAL_BATCHED'))
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 3)

    hybrid_model.plot_stopping_time_cdf_fig(ax)
    fig.legend(labels=["I", "TD", "HJ"][::-1], loc="upper center", reverse=True, ncols=3, frameon=True, bbox_to_anchor=(0.5, 0.98))
    plt.savefig(f'data/stopping_time_fig.pdf', format='pdf', dpi=1000)


def semi_major_axis_fig():
    hybrid_model = HJModel(res_path=get_res_path('47tuc_FINAL_BATCHED'))

    fig, axes = plt.subplots(1, 2, figsize=(4, 3), gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.1})
    hybrid_model.plot_sma_fig(axes[0])
    hybrid_model.plot_sma_histogram_fig(axes[1])
    points = axes[0].collections[0]
    points.set_rasterized(True)
    legend = fig.legend(
        loc="upper center",
        reverse=True,
        ncols=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.98),
        markerscale=2,  # Increases the legend marker size (adjust as needed)
        handletextpad=0.5,  # Adjusts spacing between marker and text
        handlelength=1.5  # Ensures handles are not stretched
    )
    plt.savefig(f'data/semi_major_axis_fig.pdf', format='pdf', dpi=300)



def print_probas():
    hybrid_model = HJModel(res_path=get_res_path('47tuc_FINAL_BATCHED'))
    print(hybrid_model.get_outcome_probabilities())
    print(hybrid_model.get_outcome_probabilities_for_range(0, 5))
    print(hybrid_model.get_outcome_probabilities_for_range(5, 100))

    analytic_model = HJModel(res_path=get_res_path('47tuc_FINAL_BATCHED_ANALYTIC'))
    print(analytic_model.get_outcome_probabilities())
    print(analytic_model.get_outcome_probabilities_for_range(0, 5))
    print(analytic_model.get_outcome_probabilities_for_range(5, 100))


if __name__ == '__main__':
    print_probas()
    comparison_fig()
    stopping_time_fig()
    semi_major_axis_fig()