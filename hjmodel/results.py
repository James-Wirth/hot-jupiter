from __future__ import annotations

import logging
from functools import lru_cache

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hjmodel.evolution import StopCode

__all__ = ["Results"]

logger = logging.getLogger(__name__)
LabelOrEnum = str | StopCode

STOPCODE_COLORS: dict[StopCode, str] = {
    StopCode.NM: "#D3D3D3",
    StopCode.ION: "#1b2a49",
    StopCode.TD: "#769EAA",
    StopCode.HJ: "#D62728",
    StopCode.WJ: "#FF7F0E",
}


class Results:
    """
    Wrapper for analyzing and plotting HJModel simulation results.

    Provides filtering, statistical analysis, and visualization methods
    for simulation outcomes. Accessible via the HJModel.results property.

    Attributes:
        df: DataFrame containing simulation results.
        id2label: Mapping from StopCode values to names.
        hex_by_id: Mapping from StopCode values to hex colors.
        valid_labels: List of valid StopCode names.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize Results with a simulation DataFrame.

        Args:
            df: DataFrame containing simulation results with columns including
                'stopping_condition', 'r', 'final_e', 'final_a', etc.
        """
        self.df = df.copy()
        self.id2label: dict[int, str] = {sc.value: sc.name for sc in StopCode}
        self.hex_by_id: dict[int, str] = {
            sc.value: STOPCODE_COLORS[sc] for sc in StopCode
        }
        self.valid_labels: list[str] = [sc.name for sc in StopCode]

    def _normalize_label(self, label: LabelOrEnum) -> StopCode:
        if isinstance(label, StopCode):
            return label
        try:
            return StopCode.from_name(label)
        except ValueError as err:
            raise ValueError(
                f"Invalid outcome label: {label}. Valid keys: {self.valid_labels}"
            ) from err

    def _normalize_labels(self, labels: list[LabelOrEnum]) -> tuple[StopCode, ...]:
        return tuple(self._normalize_label(lbl) for lbl in labels)

    @lru_cache(maxsize=128)
    def _filter_cached(
        self,
        include: tuple[LabelOrEnum, ...] | None,
        exclude: tuple[LabelOrEnum, ...] | None,
        sample_frac_items: tuple[tuple[str, float], ...] | None,
        r_range: tuple[float, float] | None,
    ) -> pd.DataFrame:
        _df = self.df.copy()
        if include:
            stopcodes = tuple(self._normalize_label(lbl) for lbl in include)
            codes = [sc.value for sc in stopcodes]
            _df = _df[_df["stopping_condition"].isin(codes)]
        if exclude:
            stopcodes = tuple(self._normalize_label(lbl) for lbl in exclude)
            codes = [sc.value for sc in stopcodes]
            _df = _df[~_df["stopping_condition"].isin(codes)]
        if r_range:
            r_min, r_max = r_range
            _df = _df[_df["r"].between(r_min, r_max)]
        if sample_frac_items:
            sample_frac = dict(sample_frac_items)
            parts = []
            for label, frac in sample_frac.items():
                sc = self._normalize_label(label)
                code = sc.value
                group = _df[_df["stopping_condition"] == code]
                parts.append(group.sample(frac=frac, random_state=0))
            excluded_codes = [self._normalize_label(lbl).value for lbl in sample_frac]
            others = _df[~_df["stopping_condition"].isin(excluded_codes)]
            _df = pd.concat(parts + [others], ignore_index=True)
        return _df

    def filter_outcomes(
        self,
        include: list[LabelOrEnum] | None = None,
        exclude: list[LabelOrEnum] | None = None,
        sample_frac: dict[LabelOrEnum, float] | None = None,
        r_range: tuple[float, float] | None = None,
    ) -> pd.DataFrame:
        """
        Filter simulation results by outcome type, radius, or sampling fraction.

        Args:
            include: List of outcomes to include (e.g., ['HJ', 'WJ']).
            exclude: List of outcomes to exclude.
            sample_frac: Dictionary mapping outcomes to sampling fractions.
            r_range: Tuple (r_min, r_max) to filter by radius.

        Returns:
            Filtered DataFrame.
        """
        include_key = tuple(include) if include else None
        exclude_key = tuple(exclude) if exclude else None
        sample_frac_items = (
            tuple(
                sorted(
                    (
                        (lbl.name if isinstance(lbl, StopCode) else lbl, f)
                        for lbl, f in (sample_frac or {}).items()
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
        exclude: tuple[LabelOrEnum, ...] = ("ION",),
        xrange: tuple[float, float] = (1e-2, 1e3),
        yrange: tuple[float, float] = (1, 1e5),
    ) -> None:
        """
        Plot the phase plane (a vs 1/(1-e)) colored by outcome.

        Args:
            ax: Matplotlib axes to plot on.
            exclude: Outcomes to exclude from the plot.
            xrange: X-axis limits (semi-major axis in au).
            yrange: Y-axis limits (1/(1-e)).
        """
        _df = self.filter_outcomes(exclude=list(exclude)).assign(
            x=lambda d: d["final_a"], y=lambda d: 1 / (1 - d["final_e"])
        )
        cmap = self._cmap()
        ax.scatter(
            _df["x"],
            _df["y"],
            c=_df["stopping_condition"],
            cmap=cmap,
            s=1,
            rasterized=True,
            edgecolors="none",
        )
        for sc in StopCode:
            ax.scatter(
                [],
                [],
                color=STOPCODE_COLORS[sc],
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
        include: tuple[LabelOrEnum, ...] = ("ION", "TD", "HJ"),
        xrange: tuple[float, float] = (1e-3, 11.99),
        yrange: tuple[float, float] = (0.01, 1),
    ) -> None:
        """
        Plot cumulative distribution of stopping times by outcome.

        Args:
            ax: Matplotlib axes to plot on.
            include: Outcomes to include in the plot.
            xrange: X-axis limits (stopping time in Gyr).
            yrange: Y-axis limits (CDF).
        """
        import seaborn as sns

        _df = self.filter_outcomes(include=list(include)).assign(
            stopping_time_Gyr=lambda d: d["stopping_time"] / 1e3,
            stopping_label=lambda d: d["stopping_condition"].map(self.id2label),
        )
        bins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), 1000)
        palette = {sc.name: STOPCODE_COLORS[sc] for sc in StopCode}

        sns.histplot(
            data=_df,
            x="stopping_time_Gyr",
            hue="stopping_label",
            ax=ax,
            hue_order=["HJ", "TD", "ION"],
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
        conditions: tuple[LabelOrEnum, ...] = ("NM", "TD", "HJ", "WJ"),
        cumulative: bool = True,
        drop_frac: float = 0,
    ) -> None:
        """
        Plot the semi-major axis distribution for specified outcomes.

        Args:
            ax: Matplotlib axes to plot on.
            conditions: Outcomes to include in the plot.
            cumulative: If True, plot cumulative distribution.
            drop_frac: Fraction of NM outcomes to drop for clarity.
        """
        import seaborn as sns

        _df = self.filter_outcomes(
            include=list(conditions), sample_frac={"NM": 1 - drop_frac}
        )
        for cond in conditions:
            sc = self._normalize_label(cond)
            code = sc.value
            dsub = _df[_df["stopping_condition"] == code]
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
        conditions: tuple[LabelOrEnum, ...] = ("NM", "TD", "HJ", "WJ"),
        drop_frac: float = 0.95,
        xrange: tuple[float, float] = (0.5, 15),
        yrange: tuple[float, float] = (0.01, 100),
    ) -> None:
        """
        Plot scatter of semi-major axis vs cluster radius by outcome.

        Args:
            ax: Matplotlib axes to plot on.
            conditions: Outcomes to include in the plot.
            drop_frac: Fraction of NM outcomes to drop for clarity.
            xrange: X-axis limits (radius in pc).
            yrange: Y-axis limits (semi-major axis in au).
        """
        _df = self.filter_outcomes(
            include=list(conditions), sample_frac={"NM": 1 - drop_frac}
        )
        for z, cond in enumerate(conditions):
            sc = self._normalize_label(cond)
            code = sc.value
            dsub = _df[_df["stopping_condition"] == code]
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
        r_range: tuple[float, float] | None = None,
        r_max: float = 100.0,
    ) -> dict[str, float]:
        """
        Compute the probability of each outcome type.

        Args:
            r_range: Tuple (r_min, r_max) to filter by radius.
            r_max: Maximum radius if r_range not specified.

        Returns:
            Dictionary mapping outcome names to probabilities.
        """
        _df = self.df.copy()
        if r_range:
            _df = _df[_df["r"].between(r_range[0], r_range[1])]
        else:
            _df = _df[_df["r"] <= r_max]
        total = len(_df)
        label_to_id = {sc.name: sc.value for sc in StopCode}
        if total == 0:
            logger.warning("[compute_outcome_probabilities] empty filtered dataset.")
            return {label: 0.0 for label in label_to_id}
        return {
            label: len(_df[_df["stopping_condition"] == code]) / total
            for label, code in label_to_id.items()
        }

    def get_statistics_for_outcomes(
        self,
        outcomes: list[LabelOrEnum],
        feature: str,
        r_max: float = 100.0,
    ) -> list[float]:
        """
        Get values of a feature for specified outcomes.

        Args:
            outcomes: List of outcomes to include.
            feature: Column name to extract.
            r_max: Maximum radius for filtering.

        Returns:
            List of feature values.
        """
        stopcodes = self._normalize_labels(outcomes)
        codes = [sc.value for sc in stopcodes]
        _df = self.df[
            (self.df["stopping_condition"].isin(codes)) & (self.df["r"] <= r_max)
        ]
        return _df[feature].tolist()

    def project_radius(self, random_seed: int | None = None) -> None:
        """
        Project 3D radii onto a random 2D plane.

        Adds an 'r_proj' column to the DataFrame representing projected radius.

        Args:
            random_seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(random_seed)
        self.df = self.df.assign(
            r_proj=lambda d: d["r"]
            * np.sqrt(1 - rng.uniform(-1, 1, size=d.shape[0]) ** 2)
        )

    def radius_histogram(
        self,
        label: str = "r_proj",
        bins: int = 100,
        min_counts: int = 1000,
    ) -> tuple[pd.Series, float, float]:
        """
        Compute outcome probabilities binned by radius.

        Args:
            label: Column to bin by (default 'r_proj').
            bins: Number of bins.
            min_counts: Minimum counts per bin to include.

        Returns:
            Tuple of (grouped Series, min bin edge, max bin edge).
        """
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
        xrange: tuple[float, float] = (1e-1, 1e1),
        yrange: tuple[float, float] = (1e-4, 1),
    ) -> None:
        """
        Plot outcome probability as a function of projected radius.

        Args:
            ax: Matplotlib axes to plot on.
            linestyle: Line style for the plot.
            xrange: X-axis limits (projected radius in pc).
            yrange: Y-axis limits (probability).
        """
        self.project_radius()
        hist, _, _ = self.radius_histogram()
        if hist.empty:
            logger.warning("[plot_projected_probability] empty histogram, skipping.")
            return
        hist = hist.rename("proportion")
        _df = hist.reset_index()
        bin_col = hist.index.names[0]
        _df = _df.rename(columns={bin_col: "binned"})
        _df["bin_left"] = _df["binned"].apply(lambda iv: iv.left)
        for cond, grp in _df.groupby("stopping_condition"):
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
