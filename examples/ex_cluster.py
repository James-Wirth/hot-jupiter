from matplotlib.animation import FuncAnimation

from hjmodel.cluster import DynamicPlummer
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'nature'])


def test_density_evolution():
    plummer = DynamicPlummer(M0=(1.64E6, 0.9E6),
                             rt=(86, 70),
                             rh=(1.91, 4.96),
                             N=(2E6, 1.85E6),
                             total_time=12000)

    r_values = np.geomspace(0.001, 86, 100)

    fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
    fig.set_size_inches(4, 5)
    fig.subplots_adjust(hspace=0.05)
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=12)

    for t_value in range(12):
        y = [plummer.number_density(r, t_value * 1000)* 10**6 for r in r_values]
        axs[0].plot(r_values, y, label=t_value, color=cmap(norm(t_value)))
    axs[0].set_xscale('symlog')
    axs[0].set_yscale('log')
    axs[0].set_ylim(1, 10 ** 5.5)
    axs[0].set_ylabel('Number density / $\\mathrm{pc}^{-3}$')
    axs[0].set_xticks([])

    for t_value in range(12):
        y = [plummer.isotropic_velocity_dispersion(r, t_value * 1000)/0.211 for r in r_values]
        axs[1].plot(r_values, y, label=t_value, color=cmap(norm(t_value)))
    axs[1].set_xscale('symlog')
    axs[1].set_xlabel('$r / \\mathrm{pc}$')
    axs[1].set_ylabel('1d velocity dispersion / km/s')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), aspect=30)
    cbar.set_label('Time / Gyr')

    # plt.tight_layout()
    plt.savefig('test_data/test_cluster_data/test_density_evolution.pdf', format='pdf')

def test_lagrange_radii():
    plummer = DynamicPlummer(M0=(1.64E6, 0.9E6),
                             rt=(86, 70),
                             rh=(1.91, 4.96),
                             N=(2E6, 1.85E6),
                             total_time=12000)

    n_samples = 1000
    t_values = np.linspace(0, 12000, 200)  # 200 frames

    d = plummer.get_lagrange_distribution(n_samples=1000, t=0)
    map = np.vectorize(plummer.map_lagrange_to_radius)

    fig, ax = plt.subplots()
    histogram, edges, _ = ax.hist([], bins=200, color='blue', alpha=0.7)
    ax.set_xlim(0, 200)
    ax.set_xlabel("Radial Distance")
    ax.set_ylabel("Frequency")
    title = ax.set_title("Radial Distribution at t=0")

    def update(frame):
        t = t_values[frame]
        data = map(d, t=t)
        ax.clear()
        ax.hist(data, bins=200, color='blue', alpha=0.7)
        ax.set_xlim(0, 200)
        ax.set_xlabel("Radial Distance")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Radial Distribution at t={int(t)}")

    anim = FuncAnimation(fig, update, frames=len(t_values), interval=100)
    plt.show()
