import logging

from typing import Dict, List, Optional, Tuple, Union
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from hjmodel.config import StopCode

logger = logging.getLogger(__name__)
LabelOrEnum = Union[str, StopCode]


class Results:
    """
    Wrapper for the outputting/plotting results of the HJModel run.
    An instance of this class can be accessed via the HJModel.results property in the HJModel class

    Contains methods for outputting the outcome probabilites and plotting useful figures.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.id2label: Dict[int, str] = {sc.value: sc.name for sc in StopCode}
        self.hex_by_id: Dict[int, str] = {sc.value: sc.hex for sc in StopCode}
        self.valid_labels: List[str] = [sc.name for sc in StopCode]

    def _normalize_label(self, label: LabelOrEnum) -> StopCode:
        if isinstance(label, StopCode):
            return label
        try:
            return StopCode.from_name(label)
        except ValueError:
            raise ValueError(
                f"Invalid outcome label: {label}. Valid keys: {self.valid_labels}"
            )

    def _normalize_labels(self, labels: List[LabelOrEnum]) -> Tuple[StopCode, ...]:
        return tuple(self._normalize_label(l) for l in labels)

    @lru_cache(maxsize=128)
    def _filter_cached(
        self,
        include: Optional[Tuple[LabelOrEnum, ...]],
        exclude: Optional[Tuple[LabelOrEnum, ...]],
        sample_frac_items: Optional[Tuple[Tuple[str, float], ...]],
        r_range: Optional[Tuple[float, float]],
    ) -> pd.DataFrame:
        df = self.df
        if include:
            stopcodes = tuple(self._normalize_label(l) for l in include)
            codes = [sc.value for sc in stopcodes]
            df = df[df["stopping_condition"].isin(codes)]
        if exclude:
            stopcodes = tuple(self._normalize_label(l) for l in exclude)
            codes = [sc.value for sc in stopcodes]
            df = df[~df["stopping_condition"].isin(codes)]
        if r_range:
            r_min, r_max = r_range
            df = df[df["r"].between(r_min, r_max)]
        if sample_frac_items:
            sample_frac = dict(sample_frac_items)
            parts = []
            for label, frac in sample_frac.items():
                sc = self._normalize_label(label)
                code = sc.value
                group = df[df["stopping_condition"] == code]
                parts.append(group.sample(frac=frac, random_state=0))
            excluded_codes = [
                self._normalize_label(l).value for l in sample_frac.keys()
            ]
            others = df[~df["stopping_condition"].isin(excluded_codes)]
            df = pd.concat(parts + [others], ignore_index=True)
        return df.copy()

    def filter_outcomes(
        self,
        include: Optional[List[LabelOrEnum]] = None,
        exclude: Optional[List[LabelOrEnum]] = None,
        sample_frac: Optional[Dict[LabelOrEnum, float]] = None,
        r_range: Optional[Tuple[float, float]] = None,
    ) -> pd.DataFrame:

        include_key = tuple(include) if include else None
        exclude_key = tuple(exclude) if exclude else None
        sample_frac_items = (
            tuple(
                sorted(
                    (
                        (l.name if isinstance(l, StopCode) else l, f)
                        for l, f in (sample_frac or {}).items()
                    )
                )
            )
            if sample_frac
            else None
        )
        return self._filter_cached(include_key, exclude_key, sample_frac_items, r_range)

    def _cmap(self) -> mcolors.ListedColormap:
        colors = [self.hex_by_id[i] for i in sorted(self.hex_by_id)]
        return mcolors.ListedColormap(colors)

    def plot_phase_plane(
        self,
        ax: plt.Axes,
        exclude: Tuple[LabelOrEnum, ...] = ("I",),
        xrange: Tuple[float, float] = (1e-2, 1e3),
        yrange: Tuple[float, float] = (1, 1e5),
    ) -> None:

        df = self.filter_outcomes(exclude=list(exclude)).assign(
            x=lambda d: d["final_a"], y=lambda d: 1 / (1 - d["final_e"])
        )
        cmap = self._cmap()
        ax.scatter(
            df["x"],
            df["y"],
            c=df["stopping_condition"],
            cmap=cmap,
            s=1,
            rasterized=True,
            edgecolors="none",
        )
        for sc in StopCode:
            ax.scatter(
                [],
                [],
                color=sc.hex,
                label=sc.name,
                s=10,
                rasterized=True,
            )
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
        include: Tuple[LabelOrEnum, ...] = ("I", "TD", "HJ"),
        xrange: Tuple[float, float] = (1e-3, 11.99),
        yrange: Tuple[float, float] = (0.01, 1),
    ) -> None:
        import seaborn as sns

        df = self.filter_outcomes(include=list(include)).assign(
            stopping_time_Gyr=lambda d: d["stopping_time"] / 1e3,
            stopping_label=lambda d: d["stopping_condition"].map(self.id2label),
        )
        bins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), 1000)
        palette = {sc.name: sc.hex for sc in StopCode}

        sns.histplot(
            data=df,
            x="stopping_time_Gyr",
            hue="stopping_label",
            ax=ax,
            hue_order=["HJ", "TD", "I"],
            palette=palette,
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
        conditions: Tuple[LabelOrEnum, ...] = ("NM", "TD", "HJ", "WJ"),
        cumulative: bool = True,
        drop_frac: float = 0,
    ) -> None:
        import seaborn as sns

        df = self.filter_outcomes(
            include=list(conditions), sample_frac={"NM": 1 - drop_frac}
        )
        for cond in conditions:
            sc = self._normalize_label(cond)
            code = sc.value
            dsub = df[df["stopping_condition"] == code]
            sns.histplot(
                data=dsub,
                y="final_a",
                ax=ax,
                color=self.hex_by_id[code],
                cumulative=cumulative,
                stat="density",
                element="step",
                bins=np.logspace(np.log10(0.01), np.log10(1000), 200),
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
        conditions: Tuple[LabelOrEnum, ...] = ("NM", "TD", "HJ", "WJ"),
        drop_frac: float = 0.95,
        xrange: Tuple[float, float] = (0.5, 15),
        yrange: Tuple[float, float] = (0.01, 100),
    ) -> None:

        df = self.filter_outcomes(
            include=list(conditions), sample_frac={"NM": 1 - drop_frac}
        )
        for z, cond in enumerate(conditions):
            sc = self._normalize_label(cond)
            code = sc.value
            dsub = df[df["stopping_condition"] == code]
            ax.scatter(
                dsub["r"],
                dsub["final_a"],
                s=3,
                linewidth=0,
                color=self.hex_by_id[code],
                label=sc.name,
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
        r_range: Optional[Tuple[float, float]] = None,
        r_max: float = 100.0,
    ) -> Dict[str, float]:

        df = self.df
        if r_range:
            df = df[df["r"].between(r_range[0], r_range[1])]
        else:
            df = df[df["r"] <= r_max]
        total = len(df)
        label_to_id = {sc.name: sc.value for sc in StopCode}
        if total == 0:
            logger.warning("[compute_outcome_probabilities] empty filtered dataset.")
            return {label: 0.0 for label in label_to_id.keys()}
        return {
            label: len(df[df["stopping_condition"] == code]) / total
            for label, code in label_to_id.items()
        }

    def get_statistics_for_outcomes(
        self,
        outcomes: List[LabelOrEnum],
        feature: str,
        r_max: float = 100.0,
    ) -> List[float]:

        stopcodes = self._normalize_labels(outcomes)
        codes = [sc.value for sc in stopcodes]
        df = self.df[
            (self.df["stopping_condition"].isin(codes)) & (self.df["r"] <= r_max)
        ]
        return df[feature].tolist()

    def project_radius(self, random_seed: Optional[int] = None) -> None:
        rng = np.random.default_rng(random_seed)
        self.df = self.df.assign(
            r_proj=lambda d: d["r"] * np.sin(rng.random(d.shape[0]) * 2 * np.pi)
        )

    def radius_histogram(
        self,
        label: str = "r_proj",
        bins: int = 100,
        min_counts: int = 1000,
    ) -> Tuple[pd.Series, float, float]:

        data = self.df[label].abs()
        data = data[data > 0]
        if data.empty:
            logger.warning("[radius_histogram] no valid data for label %s", label)
            empty = pd.Series(dtype=float)
            return empty, 0.0, 0.0

        bin_edges = np.geomspace(data.min() * 0.99, data.max() * 1.01, bins)
        binned = pd.cut(data, bin_edges)
        counts = binned.value_counts()
        valid_bins = counts[counts > min_counts].index
        if valid_bins.empty:
            logger.warning(
                "[radius_histogram] no bins exceed min_counts=%d", min_counts
            )
            empty = pd.Series(dtype=float)
            return empty, 0.0, 0.0

        filtered = self.df.loc[binned.isin(valid_bins)].copy()
        grouped = filtered.groupby(pd.cut(filtered[label].abs(), bin_edges))[
            "stopping_condition"
        ].value_counts(normalize=True)
        if grouped.empty:
            return grouped, 0.0, 0.0
        lefts = [interval.left for interval in grouped.index.get_level_values(0)]
        return grouped, min(lefts), max(lefts)

    def plot_projected_probability(
        self,
        ax: plt.Axes,
        linestyle: str = "solid",
        xrange: Tuple[float, float] = (1e-1, 1e1),
        yrange: Tuple[float, float] = (1e-4, 1),
    ) -> None:

        self.project_radius()
        hist, _, _ = self.radius_histogram()
        if hist.empty:
            logger.warning("[plot_projected_probability] empty histogram, skipping.")
            return
        hist = hist.rename("proportion")
        df = hist.reset_index()
        bin_col = hist.index.names[0]
        df = df.rename(columns={bin_col: "binned"})
        df["bin_left"] = df["binned"].apply(lambda iv: iv.left)
        for cond, grp in df.groupby("stopping_condition"):
            ax.step(
                grp["bin_left"],
                grp["proportion"],
                label=self.id2label.get(cond, str(cond)),
                linestyle=linestyle,
                color=self.hex_by_id.get(cond, "#000000"),
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlim=xrange,
            ylim=yrange,
            xlabel="Projected $r_{\\bot}$ / pc",
            ylabel="Probability",
        )
