"""Matplotlib plots for regime analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Default color palette for regimes
REGIME_COLORS = [
    "#d62728",  # red (crisis)
    "#ff7f0e",  # orange (quiet bear)
    "#2ca02c",  # green (recovery)
    "#1f77b4",  # blue (steady bull)
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]


class RegimeVisualizer:
    """Matplotlib plots for regime analysis."""

    def plot_regime_timeline(
        self,
        prices: pd.Series,
        regime_labels: pd.Series,
        regime_names: dict[int, str] | None = None,
    ) -> plt.Figure:
        """Price chart with background colored by regime.

        Args:
            prices: Price series with DatetimeIndex.
            regime_labels: Regime labels aligned with prices.
            regime_names: Optional mapping of regime int to name.

        Returns:
            matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Align data
        common = prices.index.intersection(regime_labels.index)
        p = prices.reindex(common)
        r = regime_labels.reindex(common)

        # Plot price
        ax.plot(p.index, p.values, color="black", linewidth=0.8)

        # Color background by regime
        unique_regimes = sorted(r.unique())
        for regime in unique_regimes:
            mask = r == regime
            name = (
                regime_names.get(regime, f"Regime {regime}")
                if regime_names
                else f"Regime {regime}"
            )
            color = REGIME_COLORS[regime % len(REGIME_COLORS)]

            # Find contiguous blocks
            changes = mask.astype(int).diff().fillna(0)
            starts = r.index[changes == 1]
            ends = r.index[changes == -1]

            # Handle edge cases
            if mask.iloc[0]:
                starts = starts.insert(0, r.index[0])
            if mask.iloc[-1]:
                ends = ends.append(pd.Index([r.index[-1]]))

            for start, end in zip(starts, ends, strict=False):
                ax.axvspan(
                    start, end, alpha=0.2, color=color, label=name
                )

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles, strict=False))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left")

        ax.set_title("Price with Regime Overlay")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_regime_scatter(
        self,
        pca_components: pd.DataFrame,
        regime_labels: pd.Series,
        regime_names: dict[int, str] | None = None,
    ) -> plt.Figure:
        """2D scatter of PC1 vs PC2, colored by regime.

        Args:
            pca_components: DataFrame with at least 2 PCA components.
            regime_labels: Aligned regime labels.
            regime_names: Optional regime name mapping.

        Returns:
            matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        common = pca_components.index.intersection(regime_labels.index)
        pc = pca_components.reindex(common)
        r = regime_labels.reindex(common)

        cols = pc.columns[:2]
        unique_regimes = sorted(r.unique())

        for regime in unique_regimes:
            mask = r == regime
            name = (
                regime_names.get(regime, f"Regime {regime}")
                if regime_names
                else f"Regime {regime}"
            )
            color = REGIME_COLORS[regime % len(REGIME_COLORS)]
            ax.scatter(
                pc.loc[mask, cols[0]],
                pc.loc[mask, cols[1]],
                c=color,
                label=name,
                alpha=0.5,
                s=10,
            )

        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_title("PCA Component Scatter by Regime")
        ax.legend()
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_transition_matrix(
        self,
        transition_matrix: np.ndarray,
        regime_names: list[str] | None = None,
    ) -> plt.Figure:
        """Heatmap of transition probabilities.

        Args:
            transition_matrix: Square matrix of transition probs.
            regime_names: Optional list of regime names.

        Returns:
            matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        n = transition_matrix.shape[0]
        names = regime_names or [f"Regime {i}" for i in range(n)]

        im = ax.imshow(transition_matrix, cmap="YlOrRd", vmin=0, vmax=1)

        # Add text annotations
        for i in range(n):
            for j in range(n):
                text = f"{transition_matrix[i, j]:.2f}"
                color = "white" if transition_matrix[i, j] > 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_xlabel("To Regime")
        ax.set_ylabel("From Regime")
        ax.set_title("Regime Transition Probabilities")
        fig.colorbar(im)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_regime_performance(
        self,
        regime_stats: dict,
    ) -> plt.Figure:
        """Bar chart comparing return/vol/Sharpe per regime.

        Args:
            regime_stats: Dict from RegimeAnalyzer.characterize_regimes().

        Returns:
            matplotlib Figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        regimes = sorted(regime_stats.keys())
        labels = [
            regime_stats[r].get("suggested_label", r) for r in regimes
        ]
        colors = [
            REGIME_COLORS[i % len(REGIME_COLORS)]
            for i in range(len(regimes))
        ]

        # Return
        returns = [
            regime_stats[r].get("annualized_return", 0) for r in regimes
        ]
        axes[0].bar(labels, returns, color=colors)
        axes[0].set_title("Annualized Return")
        axes[0].set_ylabel("Return")
        axes[0].tick_params(axis="x", rotation=45)

        # Volatility
        vols = [
            regime_stats[r].get("annualized_vol", 0) for r in regimes
        ]
        axes[1].bar(labels, vols, color=colors)
        axes[1].set_title("Annualized Volatility")
        axes[1].set_ylabel("Volatility")
        axes[1].tick_params(axis="x", rotation=45)

        # Sharpe
        sharpes = [regime_stats[r].get("sharpe", 0) for r in regimes]
        axes[2].bar(labels, sharpes, color=colors)
        axes[2].set_title("Sharpe Ratio")
        axes[2].set_ylabel("Sharpe")
        axes[2].tick_params(axis="x", rotation=45)

        fig.suptitle("Regime Performance Comparison")
        fig.tight_layout()
        plt.close(fig)
        return fig
