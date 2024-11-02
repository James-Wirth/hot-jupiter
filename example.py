from scipy.signal import savgol_filter

from hjmodel import HJModel
from hjmodel.config import SC_DICT, COLOR_DICT, PALETTE
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from hjmodel.cluster import DynamicPlummer

from matplotlib.gridspec import GridSpec

import scienceplots
plt.style.use(['science','nature'])

def run():
    outcome_probs = model.get_outcome_probabilities()
    df = model.df
    print(outcome_probs)

    # --------------
    fig = plt.figure(layout='constrained')
    fig.set_size_inches(4, 5)

    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[1,:])

    """
    # --------------
    outcomes = list(SC_DICT.keys())
    res, llim, rlim = model.get_radius_histogram()
    res = res.reset_index()
    res['r_binned'] = res['r_binned'].apply(lambda x: x.left)
    for key, grp in res.groupby(['stopping_condition']):
        print(grp)
        ax.step(grp['r_binned'], grp['proportion'], color=PALETTE[key[0]], label=outcomes[key[0]])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(llim, rlim)
    ax.set_xlabel('r / pc')
    ax.set_ylabel('Probability $P_{\\mathrm{oc}}$')
    ax.set_title('$P_{\\mathrm{oc}}$ vs radius', y=0.9, x=0.15, pad=-14, fontsize=8.5)
    ax.legend([], [], frameon=False)
    """

    # --------------
    outcomes = list(SC_DICT.keys())
    kde = sns.histplot(data=df, x='r', hue='stopping_condition', ax=ax1, palette=PALETTE,
                       element='step', fill=False, common_norm=True, stat='density', cumulative=True, log_scale=True,
                       bins=200)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(0.1, 20)
    ax1.set_ylim(1E-5, 1)
    ax1.set_xlabel('r / pc')
    ax1.set_ylabel('CDF')
    # ax1.set_title('r\n(CDF)', y=0.9, x=0.15, pad=-14, fontsize=8.5)
    ax1.legend([], [], frameon=False)

    # --------------
    outcomes = ['I', 'TD', 'HJ']
    df_filt = df.loc[df['stopping_condition'].isin([SC_DICT[outcome] for outcome in outcomes])]
    df_filt['stopping_time_Gyr'] = df_filt['stopping_time'] / 1E3
    kde = sns.histplot(data=df_filt, x='stopping_time_Gyr', hue='stopping_condition', ax=ax2, palette=PALETTE,
                       element='step', fill=False, common_norm=True, stat='density', cumulative=True, log_scale=True,
                       bins=400)
    ax2.set_xlabel('$T_{\\mathrm{stop}}$ / Gyr')
    ax2.set_ylabel('CDF')
    ax2.set_ylim(1E-5, 1)
    ax2.set_xlim(1E-3, 11.99)
    ax2.set_yscale('log')
    # ax2.set_title('$T_{\\mathrm{stop}}$\n(CDF)', y=0.9, x=0.15, pad=-14, fontsize=8.5)
    ax2.legend([], [], frameon=False)

    # --------------
    model.plot_outcomes(ax3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim(1E-2, 1E3)
    ax3.set_ylim(1, 1E5)
    ax3.set_xlabel('$a$')
    ax3.set_ylabel('$1/(1-e)$')
    points = ax3.collections[0]
    points.set_rasterized(True)
    ax3.legend([], [], frameon=False)

    # --------------
    outcomes = list(SC_DICT.keys())
    fig.legend(loc='upper right', labels=outcomes[::-1], reverse=True, bbox_to_anchor=(0.9, 1.05), ncols=5,
               frameon=True)
    plt.savefig(f'data/{cluster_name}_overall.pdf', format='pdf', dpi=1000)

    # stochastic r proj
    #---------------
    model.get_projected_distribution()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(4, 2.5)
    outcomes = list(SC_DICT.keys())
    res, llim, rlim = model.get_radius_histogram()
    res = res.reset_index()
    res['binned'] = res['binned'].apply(lambda x: x.left)
    for key, grp in res.groupby(['stopping_condition']):
        print(grp)
        ax.step(grp['binned'], grp['proportion'], color=PALETTE[key[0]], label=outcomes[key[0]])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(llim, rlim)
    ax.set_xlabel('Projected $r_{\\perp}$ / pc')
    ax.set_ylabel('Probability $P_{\\mathrm{oc}}$')
    ax.legend([], [], frameon=False)
    fig.legend(loc='upper right', labels=outcomes, reverse=False, bbox_to_anchor=(0.86, 1.01), ncols=5,
               frameon=True)
    plt.savefig(f'data/{cluster_name}_r_proj_override.pdf', format='pdf')

    # --------------
    outcomes = list(SC_DICT.keys())
    features = ['e_init', 'a_init', 'm1']
    titles = ['$e_0$', '$a_0$', '$m_{\\star}$']
    xlabels = ['$e_0$', '$a_0$ / au', '$m_{\\star}$ / $M_{\\odot}$']
    xlims = [(0.06, 0.6), (1, 30), (0.1, 0.8)]
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True)
    fig.set_size_inches(8, 2.5)
    fig.subplots_adjust(wspace=0.1)
    for i, ax in enumerate(fig.axes):
        # kde = sns.kdeplot(data=df, x=features[i], hue='stopping_condition', ax=ax, palette=PALETTE)
        kde = sns.histplot(data=df, x=features[i], hue='stopping_condition', ax=ax, palette=PALETTE,
                           log_scale=True if i == 1 else False,
                           element='step', fill=False, common_norm=True, stat='density', bins=25)
        kde.legend_.remove()
        ax.set_xlabel(xlabels[i])
        ax.set_xlim(xlims[i])
        if i == 0:
            ax.set_ylabel('PDF')
        ax.set_yscale('log')
        ax.set_ylim(1E-4, 1E1)
        ax.set_title(titles[i], y=1.0, pad=-14, fontsize=10)
    fig.legend(loc='upper right', labels=outcomes[::-1], reverse=True, bbox_to_anchor=(0.71, 1.08), ncols=5,
               frameon=True)
    plt.savefig(f'data/{cluster_name}_hj_properties.pdf', format='pdf', dpi=1000)


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cluster_name = '47tuc'
    res_path = os.path.join(dir_path, 'data', f'exp_data_{cluster_name}.pq')
    model = HJModel(res_path=res_path)
    # model.run_dynamic(time=12000, num_systems=500000)
    run()

