import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional

from hjmodel.config import SC_DICT

class Processor:
    def __init__(
        self,
        data: pd.DataFrame,
        sc_dict: Dict = SC_DICT,
    ):
        self.df = data.copy()
        self.sc_dict = sc_dict.copy()
        self.id2label = {info['id']: label for label, info in self.sc_dict.items()}
        self.hex_by_id = {info['id']: info['hex'] for info in self.sc_dict.values()}

    def _filter_outcomes(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        sample_frac: Optional[Dict[str, float]] = None,
        r_range: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        df = self.df
        if include:
            codes = [self.sc_dict[o]['id'] for o in include]
            df = df[df['stopping_condition'].isin(codes)]
        if exclude:
            codes = [self.sc_dict[o]['id'] for o in exclude]
            df = df[~df['stopping_condition'].isin(codes)]
        if r_range:
            r_min, r_max = r_range
            df = df[df['r'].between(r_min, r_max)]
        if sample_frac:
            parts = []
            for label, frac in sample_frac.items():
                code = self.sc_dict[label]['id']
                group = df[df['stopping_condition'] == code]
                parts.append(group.sample(frac=frac, random_state=0))
            others = df[~df['stopping_condition'].isin(
                [self.sc_dict[l]['id'] for l in sample_frac]
            )]
            df = pd.concat(parts + [others], ignore_index=True)
        return df.copy()

    def _setup_cmap(self) -> mcolors.ListedColormap:
        colors = [self.hex_by_id[i] for i in sorted(self.hex_by_id)]
        return mcolors.ListedColormap(colors)

    def plot_phase_plane(
        self,
        ax: plt.Axes,
        xrange: Tuple[float, float] = (1e-2, 1e3),
        yrange: Tuple[float, float] = (1, 1e5)
    ):
        df = (
            self._filter_outcomes(exclude=['I'])
            .assign(
                x=lambda d: d['final_a'],
                y=lambda d: 1 / (1 - d['final_e'])
            )
        )
        cmap = self._setup_cmap()
        ax.scatter(
            df['x'], df['y'],
            c=df['stopping_condition'],
            cmap=cmap, s=0.1, rasterized=True
        )
        for label in self.sc_dict:
            ax.scatter([], [], color=self.sc_dict[label]['hex'],
                       label=label, s=10, rasterized=True)
        ax.set(
            xscale='log', yscale='log',
            xlim=xrange, ylim=yrange,
            xlabel='$a$', ylabel='$1/(1-e)$'
        )
        ax.legend(frameon=True)

    def plot_stopping_cdf(
        self,
        ax: plt.Axes,
        include: List[str] = ['I', 'TD', 'HJ'],
        xrange: Tuple[float, float] = (1e-3, 11.99),
        yrange: Tuple[float, float] = (0.01, 1)
    ):
        import seaborn as sns

        df = (
            self._filter_outcomes(include=include)
            .assign(
                stopping_time_Gyr=lambda d: d['stopping_time'] / 1e3,
                stopping_label=lambda d: d['stopping_condition'].map(self.id2label)
            )
        )
        bins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), 1000)
        palette = {label: info['hex'] for label, info in self.sc_dict.items()}
        sns.histplot(
            data=df,
            x='stopping_time_Gyr',
            hue='stopping_label',
            ax=ax,
            palette=palette,
            element='step', fill=False,
            common_norm=False, stat='density',
            cumulative=True, bins=bins
        )
        ax.set(
            xscale='log', yscale='linear',
            xlim=xrange, ylim=yrange,
            xlabel=r'$T_{\mathrm{stop}}$ / Gyr',
            ylabel='CDF'
        )
        ax.legend().remove()

    def plot_sma_distribution(
        self,
        ax: plt.Axes,
        conditions: List[str] = ['NM', 'TD', 'WJ', 'HJ'],
        cumulative: bool = True,
        drop_frac: float = 0.9
    ):
        import seaborn as sns

        df = self._filter_outcomes(include=conditions,
                                   sample_frac={'NM': 1-drop_frac})
        for cond in conditions:
            code = self.sc_dict[cond]['id']
            dsub = df[df['stopping_condition'] == code]
            sns.histplot(
                data=dsub,
                y='final_a', ax=ax,
                color=self.hex_by_id[code],
                cumulative=cumulative, stat='density',
                element='step',
                bins=np.logspace(np.log10(0.01), np.log10(1000), 200),
                fill=False, rasterized=True
            )
        ax.set(
            yscale='log', ylim=(0.01, 100),
            xlabel='Cumulative Density'
        )
        ax.tick_params(left=False)
        ax.set_yticks([])

    def plot_sma_scatter(
        self,
        ax: plt.Axes,
        conditions: List[str] = ['NM', 'TD', 'WJ', 'HJ'],
        drop_frac: float = 0.9,
        xrange: Tuple[float, float] = (0.5, 20),
        yrange: Tuple[float, float] = (0.01, 100)
    ):
        df = self._filter_outcomes(include=conditions,
                                   sample_frac={'NM': 1-drop_frac})
        for z, cond in enumerate(conditions):
            code = self.sc_dict[cond]['id']
            dsub = df[df['stopping_condition'] == code]
            ax.scatter(
                dsub['r'], dsub['final_a'],
                s=3, linewidth=0,
                color=self.hex_by_id[code],
                label=cond, zorder=z,
                rasterized=True
            )
        ax.set(
            xscale='linear', yscale='log',
            xlim=xrange, ylim=yrange,
            xlabel='r / pc', ylabel='a / au'
        )
        ax.legend().remove()

    def compute_outcome_probabilities(
        self,
        r_range: Optional[Tuple[float, float]] = None,
        r_max: Optional[float] = 100.0
    ) -> Dict[str, float]:
        df = self.df
        if r_range:
            df = df[df['r'].between(r_range[0], r_range[1])]
        else:
            df = df[df['r'] <= r_max]
        total = len(df)
        label_to_id = {label: info['id'] for label, info in self.sc_dict.items()}
        return {
            label: len(df[df['stopping_condition'] == code]) / total
            for label, code in label_to_id.items()
        }

    def get_statistics_for_outcomes(
        self,
        outcomes: List[str],
        feature: str,
        r_max: Optional[float] = 100.0
    ) -> List[float]:
        codes = [self.sc_dict[o]['id'] for o in outcomes]
        df = self.df[(self.df['stopping_condition'].isin(codes)) & (self.df['r'] <= r_max)]
        return df[feature].tolist()

    def project_radius(self, random_seed: Optional[int] = None) -> None:
        rng = np.random.default_rng(random_seed)
        self.df = self.df.assign(
            r_proj=lambda d: d['r'] * np.sin(rng.random(d.shape[0]) * 2 * np.pi)
        )

    def radius_histogram(
        self,
        label: str = 'r_proj',
        bins: int = 100,
        min_counts: int = 1000
    ):
        data = self.df[label].abs()
        data = data[data > 0]
        bin_edges = np.geomspace(data.min() * 0.99, data.max() * 1.01, bins)

        binned = pd.cut(data, bin_edges)
        counts = binned.value_counts()
        valid_bins = counts[counts > min_counts].index
        filtered = self.df.loc[binned.isin(valid_bins)].copy()
        grouped = (
            filtered
            .groupby(pd.cut(filtered[label].abs(), bin_edges))['stopping_condition']
            .value_counts(normalize=True)
        )
        lefts = [interval.left for interval in grouped.index.get_level_values(0)]
        return grouped, min(lefts), max(lefts)

    def plot_projected_probability(
        self,
        ax: plt.Axes,
        linestyle: str = 'solid',
        xrange: Optional[Tuple[float, float]] = [1e-1, 1e1],
        yrange: Tuple[float, float] = (1e-4, 1)
    ):
        self.project_radius()
        hist, lmin, lmax = self.radius_histogram()
        hist = hist.rename('proportion')
        df = hist.reset_index()

        bin_col = hist.index.names[0]
        df = df.rename(columns={bin_col: 'binned'})

        df['bin_left'] = df['binned'].apply(lambda iv: iv.left)
        for cond, grp in df.groupby('stopping_condition'):
            label = self.id2label[cond]
            ax.step(
                grp['bin_left'], grp['proportion'],
                label=label, linestyle=linestyle,
                color=self.hex_by_id[cond]
            )
        ax.set(
            xscale='log', yscale='log',
            xlim=(xrange if isinstance(xrange, tuple) else tuple(xrange)),
            ylim=yrange,
            xlabel='Projected $r_p$ / pc',
            ylabel='Probability'
        )
        ax.legend()
