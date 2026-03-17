"""Regime characterization and analysis utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from meridian.config.constants import TRADING_DAYS_PER_YEAR


class RegimeAnalyzer:
    """Characterize detected regimes with statistics and labels."""

    def characterize_regimes(
        self,
        regime_labels: pd.Series,
        feature_matrix: pd.DataFrame,
        returns: pd.Series | None = None,
        volatility: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Per regime: annualized_return, annualized_vol, sharpe,
        avg_duration_days, dominant_features, suggested_label.

        Args:
            regime_labels: Series with regime labels per date.
            feature_matrix: Feature matrix aligned with labels.
            returns: Optional returns series for performance stats.
            volatility: Optional volatility series.

        Returns:
            Dict keyed by regime label (str) with summary stats.
        """
        stats: dict[str, Any] = {}
        unique_regimes = sorted(regime_labels.unique())

        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_key = f"regime_{regime}"
            count = int(mask.sum())

            regime_info: dict[str, Any] = {
                "count": count,
                "pct_of_total": count / len(regime_labels),
            }

            # Duration analysis
            durations = self._compute_durations(regime_labels, regime)
            regime_info["avg_duration_days"] = (
                float(np.mean(durations)) if durations else 0.0
            )
            regime_info["max_duration_days"] = (
                int(np.max(durations)) if durations else 0
            )

            # Performance stats if returns provided
            if returns is not None:
                aligned_returns = returns.reindex(regime_labels.index)
                regime_returns = aligned_returns[mask].dropna()
                if len(regime_returns) > 0:
                    ann_return = float(
                        regime_returns.mean() * TRADING_DAYS_PER_YEAR
                    )
                    ann_vol = float(
                        regime_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                    )
                    sharpe = (
                        ann_return / ann_vol if ann_vol > 0 else 0.0
                    )
                    regime_info["annualized_return"] = ann_return
                    regime_info["annualized_vol"] = ann_vol
                    regime_info["sharpe"] = sharpe

            # Dominant features (highest mean absolute value in regime)
            regime_features = feature_matrix[mask]
            if len(regime_features) > 0:
                mean_abs = regime_features.abs().mean()
                top_features = mean_abs.nlargest(5)
                regime_info["dominant_features"] = {
                    str(k): float(v) for k, v in top_features.items()
                }

            # Suggested label heuristic
            regime_info["suggested_label"] = self._suggest_label(
                regime, len(unique_regimes), regime_info
            )

            stats[regime_key] = regime_info

        return stats

    def _compute_durations(
        self, labels: pd.Series, target_regime: int
    ) -> list[int]:
        """Compute consecutive-day durations for a given regime."""
        durations = []
        current_run = 0
        for label in labels.values:
            if label == target_regime:
                current_run += 1
            elif current_run > 0:
                durations.append(current_run)
                current_run = 0
        if current_run > 0:
            durations.append(current_run)
        return durations

    def _suggest_label(
        self,
        regime: int,
        n_regimes: int,
        info: dict[str, Any],
    ) -> str:
        """Assign a human-readable label based on stats."""
        if "annualized_return" in info and "annualized_vol" in info:
            high_ret = info["annualized_return"] > 0
            high_vol = info["annualized_vol"] > 0.20
            if high_ret and not high_vol:
                return "Steady Bull"
            elif high_ret and high_vol:
                return "Recovery"
            elif not high_ret and not high_vol:
                return "Quiet Bear"
            else:
                return "Crisis"
        # Fallback: position-based
        labels = ["Crisis", "Quiet Bear", "Recovery", "Steady Bull"]
        if n_regimes <= len(labels):
            return labels[regime % len(labels)]
        return f"Regime {regime}"

    def find_transitions(
        self, regime_labels: pd.Series
    ) -> pd.DataFrame:
        """Find dates where regime changed.

        Returns:
            DataFrame with columns: date, from_regime, to_regime,
            return_before_5d, return_after_5d (NaN if no returns).
        """
        transitions = []
        values = regime_labels.values
        dates = regime_labels.index

        for i in range(1, len(values)):
            if values[i] != values[i - 1]:
                transitions.append(
                    {
                        "date": dates[i],
                        "from_regime": int(values[i - 1]),
                        "to_regime": int(values[i]),
                    }
                )

        if not transitions:
            return pd.DataFrame(
                columns=["date", "from_regime", "to_regime"]
            )

        return pd.DataFrame(transitions)

    def regime_conditional_performance(
        self,
        regime_labels: pd.Series,
        strategy_returns: pd.Series,
    ) -> dict[str, Any]:
        """Strategy performance broken down by active regime."""
        aligned = strategy_returns.reindex(regime_labels.index).dropna()
        aligned_labels = regime_labels.reindex(aligned.index)

        results: dict[str, Any] = {}
        for regime in sorted(aligned_labels.unique()):
            mask = aligned_labels == regime
            r = aligned[mask]
            if len(r) > 0:
                results[f"regime_{regime}"] = {
                    "mean_daily_return": float(r.mean()),
                    "annualized_return": float(
                        r.mean() * TRADING_DAYS_PER_YEAR
                    ),
                    "annualized_vol": float(
                        r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                    ),
                    "sharpe": float(
                        r.mean()
                        / r.std()
                        * np.sqrt(TRADING_DAYS_PER_YEAR)
                    )
                    if r.std() > 0
                    else 0.0,
                    "n_days": int(mask.sum()),
                    "hit_rate": float((r > 0).mean()),
                }
        return results
