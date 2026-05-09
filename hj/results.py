from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hj.evolution import StopCode

__all__ = ["Results", "STOPCODE_COLORS"]

logger = logging.getLogger(__name__)
LabelOrEnum = str | StopCode

STOPCODE_COLORS: dict[StopCode, str] = {
    StopCode.NM: "#D3D3D3",
    StopCode.ION: "#1b2a49",
    StopCode.TD: "#769EAA",
    StopCode.HJ: "#D62728",
    StopCode.WJ: "#FF7F0E",
}

_ID_TO_LABEL: dict[int, str] = {sc.value: sc.name for sc in StopCode}
_LABEL_TO_ID: dict[str, int] = {sc.name: sc.value for sc in StopCode}
_HEX_BY_ID: dict[int, str] = {sc.value: STOPCODE_COLORS[sc] for sc in StopCode}
_VALID_LABELS: list[str] = list(_LABEL_TO_ID)


def _to_code(label: LabelOrEnum) -> int:
    if isinstance(label, StopCode):
        return label.value
    try:
        return StopCode[label].value
    except KeyError as err:
        raise ValueError(
            f"Invalid outcome label: {label}. Valid keys: {_VALID_LABELS}"
        ) from err


def _codes(labels: Iterable[LabelOrEnum]) -> list[int]:
    return [_to_code(lbl) for lbl in labels]


class Results:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def filter_outcomes(
        self,
        include: list[LabelOrEnum] | None = None,
        exclude: list[LabelOrEnum] | None = None,
        sample_frac: dict[LabelOrEnum, float] | None = None,
        r_range: tuple[float, float] | None = None,
    ) -> pd.DataFrame:
        df = self.df
        if include:
            df = df[df["stop_code"].isin(_codes(include))]
        if exclude:
            df = df[~df["stop_code"].isin(_codes(exclude))]
        if r_range:
            df = df[df["r"].between(*r_range)]
        if sample_frac:
            frac_by_code = {_to_code(lbl): frac for lbl, frac in sample_frac.items()}
            parts = [
                g.sample(frac=frac, random_state=0)
                if (frac := frac_by_code.get(code)) is not None
                else g
                for code, g in df.groupby("stop_code", sort=False)
            ]
            df = pd.concat(parts, ignore_index=True)
        return df

    def _iter_outcome_slices(
        self, df: pd.DataFrame, outcomes: Iterable[LabelOrEnum]
    ) -> Iterator[tuple[int, str, str, pd.DataFrame]]:
        groups = dict(tuple(df.groupby("stop_code", sort=False)))
        empty = df.iloc[0:0]
        for lbl in outcomes:
            code = _to_code(lbl)
            yield (
                code,
                _HEX_BY_ID[code],
                _ID_TO_LABEL[code],
                groups.get(code, empty),
            )

    def plot_phase_plane(
        self,
        ax: plt.Axes,
        exclude: tuple[LabelOrEnum, ...] = ("ION",),
        xrange: tuple[float, float] = (1e-2, 1e3),
        yrange: tuple[float, float] = (1, 1e5),
    ) -> None:
        df = self.filter_outcomes(exclude=list(exclude))
        cmap = mcolors.ListedColormap([_HEX_BY_ID[i] for i in sorted(_HEX_BY_ID)])
        ax.scatter(
            df["a"],
            1.0 / (1.0 - df["e"]),
            c=df["stop_code"],
            cmap=cmap,
            s=1,
            rasterized=True,
            edgecolors="none",
        )
        for sc in StopCode:
            ax.scatter([], [], color=STOPCODE_COLORS[sc], label=sc.name, s=10)
        ax.set(
            xscale="log",
            yscale="log",
            xlim=xrange,
            ylim=yrange,
            xlabel="$a$",
            ylabel="$1/(1-e)$",
        )

    def plot_stopping_cdf(
        self,
        ax: plt.Axes,
        include: tuple[LabelOrEnum, ...] = ("ION", "TD", "HJ"),
        xrange: tuple[float, float] = (1e-3, 11.99),
        yrange: tuple[float, float] = (0.01, 1),
    ) -> None:
        df = self.filter_outcomes(include=list(include)).assign(
            stopping_time_Gyr=lambda d: d["stop_time"] / 1e3,
            stopping_label=lambda d: d["stop_code"].map(_ID_TO_LABEL),
        )
        bins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), 1000)
        sns.histplot(
            data=df,
            x="stopping_time_Gyr",
            hue="stopping_label",
            ax=ax,
            hue_order=["HJ", "TD", "ION"],
            palette={sc.name: STOPCODE_COLORS[sc] for sc in StopCode},
            element="step",
            fill=False,
            common_norm=False,
            stat="density",
            cumulative=True,
            bins=bins,
        )
        ax.set(
            xscale="log",
            yscale="linear",
            xlim=xrange,
            ylim=yrange,
            xlabel=r"$T_{\mathrm{stop}}$ / Gyr",
            ylabel="CDF",
        )
        ax.legend().remove()

    def plot_sma_distribution(
        self,
        ax: plt.Axes,
        conditions: tuple[LabelOrEnum, ...] = ("NM", "TD", "HJ", "WJ"),
        cumulative: bool = True,
        drop_frac: float = 0,
    ) -> None:
        df = self.filter_outcomes(
            include=list(conditions), sample_frac={"NM": 1 - drop_frac}
        )
        bins = np.logspace(np.log10(0.01), np.log10(1000), 200)
        for _, hex_color, _, sub in self._iter_outcome_slices(df, conditions):
            sns.histplot(
                data=sub,
                y="a",
                ax=ax,
                color=hex_color,
                cumulative=cumulative,
                stat="density",
                element="step",
                bins=bins,
                fill=False,
                rasterized=True,
            )
        ax.set(yscale="log", ylim=(0.01, 100), xlabel="CDF")
        ax.tick_params(left=False)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.margins(x=0)
        ax.set_yticks([])

    def plot_sma_scatter(
        self,
        ax: plt.Axes,
        conditions: tuple[LabelOrEnum, ...] = ("NM", "TD", "HJ", "WJ"),
        drop_frac: float = 0.95,
        xrange: tuple[float, float] = (0.5, 15),
        yrange: tuple[float, float] = (0.01, 100),
    ) -> None:
        df = self.filter_outcomes(
            include=list(conditions), sample_frac={"NM": 1 - drop_frac}
        )
        for z, (_, hex_color, name, sub) in enumerate(
            self._iter_outcome_slices(df, conditions)
        ):
            ax.scatter(
                sub["r"],
                sub["a"],
                s=3,
                linewidth=0,
                color=hex_color,
                label=name,
                zorder=z,
                rasterized=True,
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlim=xrange,
            ylim=yrange,
            xlabel="$r$ / pc",
            ylabel="$a$ / au",
        )
        ax.legend().remove()

    def compute_outcome_probabilities(
        self,
        r_range: tuple[float, float] | None = None,
        r_max: float = 100.0,
    ) -> dict[str, float]:
        df = self.df
        df = df[df["r"].between(*r_range)] if r_range else df[df["r"] <= r_max]
        counts = df["stop_code"].value_counts()
        total = int(counts.sum())
        if total == 0:
            logger.warning("[compute_outcome_probabilities] empty filtered dataset.")
            return {label: 0.0 for label in _LABEL_TO_ID}
        return {
            label: float(counts.get(code, 0)) / total
            for label, code in _LABEL_TO_ID.items()
        }

    def get_statistics_for_outcomes(
        self,
        outcomes: list[LabelOrEnum],
        feature: str,
        r_max: float = 100.0,
    ) -> list[float]:
        mask = self.df["stop_code"].isin(_codes(outcomes)) & (self.df["r"] <= r_max)
        return self.df.loc[mask, feature].tolist()

    def project_radius(self, random_seed: int | None = None) -> None:
        rng = np.random.default_rng(random_seed)
        self.df = self.df.assign(
            r_proj=lambda d: (
                d["r"] * np.sqrt(1 - rng.uniform(-1, 1, size=d.shape[0]) ** 2)
            )
        )

    def radius_histogram(
        self,
        label: str = "r_proj",
        bins: int = 100,
        min_counts: int = 1000,
    ) -> tuple[pd.Series, float, float]:
        empty = (pd.Series(dtype=float), 0.0, 0.0)
        data = self.df[label].abs()
        data = data[data > 0]
        if data.empty:
            logger.warning("[radius_histogram] no valid data for label %s", label)
            return empty

        bin_edges = np.geomspace(data.min() * 0.99, data.max() * 1.01, bins)
        binned = pd.cut(data, bin_edges)
        valid_bins = binned.value_counts().loc[lambda s: s > min_counts].index
        if valid_bins.empty:
            logger.warning(
                "[radius_histogram] no bins exceed min_counts=%d", min_counts
            )
            return empty

        sub = self.df.loc[binned.isin(valid_bins)]
        grouped = sub.groupby(pd.cut(sub[label].abs(), bin_edges))[
            "stop_code"
        ].value_counts(normalize=True)
        if grouped.empty:
            return empty
        lefts = [iv.left for iv in grouped.index.get_level_values(0)]
        return grouped, min(lefts), max(lefts)

    def plot_projected_probability(
        self,
        ax: plt.Axes,
        linestyle: str = "solid",
        xrange: tuple[float, float] = (1e-1, 1e1),
        yrange: tuple[float, float] = (1e-4, 1),
    ) -> None:
        self.project_radius()
        hist, _, _ = self.radius_histogram()
        if hist.empty:
            logger.warning("[plot_projected_probability] empty histogram, skipping.")
            return
        df = hist.rename("proportion").reset_index()
        df = df.rename(columns={hist.index.names[0]: "binned"})
        df["bin_left"] = df["binned"].apply(lambda iv: iv.left)
        for code, grp in df.groupby("stop_code"):
            ax.step(
                grp["bin_left"],
                grp["proportion"],
                label=_ID_TO_LABEL.get(code, str(code)),
                linestyle=linestyle,
                color=_HEX_BY_ID.get(code, "#000000"),
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlim=xrange,
            ylim=yrange,
            xlabel="Projected $r_{\\bot}$ / pc",
            ylabel="Probability",
        )
