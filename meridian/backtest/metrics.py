"""Performance metrics: Sharpe, Sortino, drawdown, Calmar, etc."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from meridian.config.constants import TRADING_DAYS_PER_YEAR


class PerformanceMetrics:
    """Compute comprehensive performance metrics from an equity curve.

    All annualization assumes 252 trading days per year.
    """

    @staticmethod
    def compute_all(
        returns: pd.Series,
        risk_free_rate: float = 0.04,
    ) -> dict[str, Any]:
        """Compute all metrics from a daily return series.

        Args:
            returns: Daily portfolio returns.
            risk_free_rate: Annual risk-free rate (default 4%).

        Returns:
            Dict with comprehensive metrics.
        """
        if len(returns) == 0:
            return {"error": "No returns data"}

        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return {"error": "All returns are NaN"}

        n_days = len(returns_clean)
        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

        # Return metrics
        total_return = float((1 + returns_clean).prod() - 1)
        n_years = n_days / TRADING_DAYS_PER_YEAR
        annualized_return = float(
            (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        )

        # Risk metrics
        daily_vol = float(returns_clean.std())
        annualized_volatility = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

        dd_series = PerformanceMetrics.compute_drawdown_series(returns_clean)
        max_drawdown = float(dd_series["drawdown"].min())
        max_drawdown_duration = PerformanceMetrics._max_dd_duration(dd_series)

        var_95 = float(returns_clean.quantile(0.05))
        cvar_95 = float(returns_clean[returns_clean <= var_95].mean())

        # Risk-adjusted metrics
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = (
            excess_return / annualized_volatility if annualized_volatility > 0 else 0.0
        )

        downside_returns = returns_clean[returns_clean < daily_rf]
        downside_vol = (
            float(downside_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if len(downside_returns) > 0
            else 0.0
        )
        sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0.0

        calmar_ratio = (
            annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        )

        # Trade metrics
        winning_days = returns_clean[returns_clean > 0]
        losing_days = returns_clean[returns_clean < 0]

        win_rate = len(winning_days) / n_days if n_days > 0 else 0.0
        avg_win = float(winning_days.mean()) if len(winning_days) > 0 else 0.0
        avg_loss = float(losing_days.mean()) if len(losing_days) > 0 else 0.0

        gross_profit = float(winning_days.sum()) if len(winning_days) > 0 else 0.0
        gross_loss = abs(float(losing_days.sum())) if len(losing_days) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float("inf")

        # Distribution metrics
        skewness = float(returns_clean.skew())
        kurtosis = float(returns_clean.kurtosis())

        # Stability metrics
        rolling_sharpe = PerformanceMetrics._rolling_sharpe(
            returns_clean, window=63, risk_free_rate=risk_free_rate
        )
        rolling_sharpe_clean = rolling_sharpe.dropna()

        # Monthly returns
        monthly = returns_clean.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        pct_positive_months = float((monthly > 0).mean()) if len(monthly) > 0 else 0.0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "max_drawdown": max_drawdown,
            "max_drawdown_duration_days": max_drawdown_duration,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "best_day": float(returns_clean.max()),
            "worst_day": float(returns_clean.min()),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "rolling_sharpe_mean": float(rolling_sharpe_clean.mean())
            if len(rolling_sharpe_clean) > 0
            else 0.0,
            "rolling_sharpe_std": float(rolling_sharpe_clean.std())
            if len(rolling_sharpe_clean) > 0
            else 0.0,
            "pct_positive_months": pct_positive_months,
            "n_days": n_days,
        }

    @staticmethod
    def compute_drawdown_series(returns: pd.Series) -> pd.DataFrame:
        """Compute drawdown time series.

        Returns DataFrame with:
            - cumulative_return
            - running_max
            - drawdown (negative values)
            - in_drawdown (bool)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        return pd.DataFrame(
            {
                "cumulative_return": cumulative,
                "running_max": running_max,
                "drawdown": drawdown,
                "in_drawdown": drawdown < 0,
            }
        )

    @staticmethod
    def compute_rolling_metrics(
        returns: pd.Series,
        window: int = 63,
        risk_free_rate: float = 0.04,
    ) -> pd.DataFrame:
        """Compute rolling performance metrics."""
        rolling_ret = returns.rolling(window).apply(
            lambda x: (1 + x).prod() ** (TRADING_DAYS_PER_YEAR / len(x)) - 1,
            raw=False,
        )
        rolling_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        rolling_sharpe = PerformanceMetrics._rolling_sharpe(
            returns, window, risk_free_rate
        )

        return pd.DataFrame(
            {
                "rolling_return": rolling_ret,
                "rolling_volatility": rolling_vol,
                "rolling_sharpe": rolling_sharpe,
            }
        )

    @staticmethod
    def regime_conditional_metrics(
        returns: pd.Series,
        regime_labels: pd.Series,
        risk_free_rate: float = 0.04,
    ) -> dict[int, dict]:
        """Compute metrics separately for each market regime."""
        aligned_labels = regime_labels.reindex(returns.index)
        results: dict[int, dict] = {}

        for regime in sorted(aligned_labels.dropna().unique()):
            mask = aligned_labels == regime
            regime_returns = returns[mask].dropna()
            if len(regime_returns) > 5:
                results[int(regime)] = PerformanceMetrics.compute_all(
                    regime_returns, risk_free_rate
                )
        return results

    @staticmethod
    def _rolling_sharpe(
        returns: pd.Series,
        window: int = 63,
        risk_free_rate: float = 0.04,
    ) -> pd.Series:
        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
        excess = returns - daily_rf
        rolling_mean = excess.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        sharpe = (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        return sharpe

    @staticmethod
    def _max_dd_duration(dd_series: pd.DataFrame) -> int:
        """Max consecutive days spent in drawdown."""
        in_dd = dd_series["in_drawdown"].values
        max_dur = 0
        current = 0
        for v in in_dd:
            if v:
                current += 1
                max_dur = max(max_dur, current)
            else:
                current = 0
        return max_dur
