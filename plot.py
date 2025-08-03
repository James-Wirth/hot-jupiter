#!/usr/bin/env python3
import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines

from hjmodel import HJModel
from hjmodel.results import Results

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

PLOT_STYLES = {
    "mnras": {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
    }
}


def apply_plot_style(style: str = "mnras", disable_tex: bool = False) -> None:
    if style not in PLOT_STYLES:
        raise ValueError(f"Unknown plotting style: {style}")
    rc = copy.deepcopy(PLOT_STYLES[style])
    if disable_tex:
        rc["text.usetex"] = False
        rc.pop("text.latex.preamble", None)
    try:
        plt.rcParams.update(rc)
    except Exception as e:
        logger.warning(
            "Failed to apply plotting style '%s' (disable_tex=%s): %s. Falling back to defaults.",
            style,
            disable_tex,
            e,
        )
        fallback = copy.deepcopy(rc)
        fallback["text.usetex"] = False
        fallback.pop("text.latex.preamble", None)
        try:
            plt.rcParams.update(fallback)
        except Exception:
            logger.warning(
                "Fallback style application also failed; using matplotlib defaults."
            )


def load_results(name: str) -> Results:
    model = HJModel(name=name)
    return model.results


def hide_ticks(ax, hide_x=False, hide_y=False):
    if hide_x:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if hide_y:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)


def save_fig(fig, filename: str, dpi_override: int = 1000):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, format="pdf", dpi=dpi_override)
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)


def comparison_fig(
    analytic_exp: str,
    hybrid_exp: str,
    style: str = "mnras",
    disable_tex: bool = False,
):
    analytic = load_results(analytic_exp)
    hybrid = load_results(hybrid_exp)

    apply_plot_style(style, disable_tex=disable_tex)
    fig = plt.figure(figsize=(7.086, 4))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)

    analytic.plot_phase_plane(ax1)
    hybrid.plot_phase_plane(ax2)
    analytic.plot_projected_probability(ax3)
    hybrid.plot_projected_probability(ax4)

    ax1.text(0.03, 0.95, "Analytic", transform=ax1.transAxes, ha="left", va="top")
    ax2.text(0.03, 0.95, "Hybrid", transform=ax2.transAxes, ha="left", va="top")

    hide_ticks(ax2, hide_y=True)
    hide_ticks(ax4, hide_y=True)
    hide_ticks(ax1, hide_x=True)
    hide_ticks(ax2, hide_x=True)

    lines = [ln for ln in ax4.get_lines() if ln.get_label() != "_nolegend_"]
    labels = [ln.get_label() for ln in lines]
    if lines:
        fig.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(labels),
            fontsize=8,
            frameon=False,
        )

    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.9, bottom=0.15, hspace=0.12, wspace=0.12
    )

    fname = f"comparison_{analytic_exp}_vs_{hybrid_exp}.pdf"
    save_fig(fig, fname, dpi_override=1000)


def stopping_time_fig(name: str, style: str = "mnras", disable_tex: bool = False):
    res = load_results(name)
    apply_plot_style(style, disable_tex=disable_tex)

    fig, ax = plt.subplots(figsize=(3.32, 3.85))
    res.plot_stopping_cdf(ax)

    fig.legend(
        ["I", "TD", "HJ"],
        loc="upper center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.97),
    )

    fname = f"stopping_time_{name}.pdf"
    save_fig(fig, fname, dpi_override=1000)


def semi_major_axis_fig(name: str, style: str = "mnras", disable_tex: bool = False):
    res = load_results(name)
    apply_plot_style(style, disable_tex=disable_tex)

    fig, axes = plt.subplots(
        1, 2, figsize=(3.32, 4), gridspec_kw={"width_ratios": [2, 1], "wspace": 0.12}
    )

    res.plot_sma_scatter(axes[0])
    res.plot_sma_distribution(axes[1])

    hide_ticks(axes[1], hide_y=True)

    outcomes = ["NM", "TD", "HJ", "WJ"]
    collections = axes[0].collections
    legend_handles = [
        mlines.Line2D([], [], color=col.get_facecolor()[0], label=label)
        for col, label in zip(collections, outcomes)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncols=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.97),
    )

    fname = f"sma_{name}.pdf"
    save_fig(fig, fname, dpi_override=1000)


def print_outcome_probabilities(name: str):
    res = load_results(name)
    header = f"Results for {name} (aggregated)"
    print(header)
    print("-" * len(header))
    for label, val in res.compute_outcome_probabilities().items():
        print(f"{label}: {val:.4f}")
    for desc, r_range in [("0 < r < 0.5", (0, 0.5)), ("8 < r < 100", (8, 100))]:
        print(f"  {desc}:")
        sub_probs = res.compute_outcome_probabilities(r_range=r_range)
        for label, val in sub_probs.items():
            print(f"    {label}: {val:.4f}")


def main():
    # e.g.
    EXPERIMENT = "EXAMPLE"
    DISABLE_TEX = True
    STYLE = "mnras"

    print_outcome_probabilities(EXPERIMENT)
    # stopping_time_fig(EXPERIMENT, style=STYLE, disable_tex=DISABLE_TEX)
    # semi_major_axis_fig(EXPERIMENT, style=STYLE, disable_tex=DISABLE_TEX)


if __name__ == "__main__":
    main()
