"""Backtest report generation: text summaries and plots.

Generates human-readable reports from BacktestResult objects.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from meridian.backtest.engine import BacktestResult


class BacktestReport:
    """Generate text and visual reports from backtest results."""

    def __init__(self, result: BacktestResult) -> None:
        self.result = result

    def text_summary(self) -> str:
        """Generate a comprehensive text report."""
        buf = StringIO()
        r = self.result
        m = r.metrics

        buf.write("=" * 60 + "\n")
        buf.write(f"  BACKTEST REPORT: {r.strategy_name}\n")
        buf.write("=" * 60 + "\n\n")

        # Overview
        buf.write("--- Overview ---\n")
        buf.write(f"  Period:          {r.backtest_start} to {r.backtest_end}\n")
        buf.write(f"  Initial capital: ${r.config.initial_capital:,.0f}\n")
        train = r.config.train_days
        test = r.config.test_days
        step = r.config.step_days
        buf.write(
            f"  Walk-forward:    {train}d train"
            f" / {test}d test / {step}d step\n"
        )
        buf.write(f"  Windows:         {r.total_windows}\n")
        buf.write(f"  Total trades:    {r.total_trades}\n\n")

        if "error" in m:
            buf.write(f"  ERROR: {m['error']}\n")
            return buf.getvalue()

        # Performance
        buf.write("--- Performance ---\n")
        buf.write(f"  Total return:        {m.get('total_return', 0):.2%}\n")
        buf.write(f"  Annualized return:   {m.get('annualized_return', 0):.2%}\n")
        buf.write(f"  Annualized vol:      {m.get('annualized_volatility', 0):.2%}\n")
        buf.write(f"  Max drawdown:        {m.get('max_drawdown', 0):.2%}\n")
        buf.write(
            f"  Max DD duration:     {m.get('max_drawdown_duration_days', 0)} days\n\n"
        )

        # Risk-adjusted
        buf.write("--- Risk-Adjusted ---\n")
        buf.write(f"  Sharpe ratio:        {m.get('sharpe_ratio', 0):.3f}\n")
        buf.write(f"  Sortino ratio:       {m.get('sortino_ratio', 0):.3f}\n")
        buf.write(f"  Calmar ratio:        {m.get('calmar_ratio', 0):.3f}\n")
        buf.write(f"  VaR (95%):           {m.get('var_95', 0):.4f}\n")
        buf.write(f"  CVaR (95%):          {m.get('cvar_95', 0):.4f}\n\n")

        # Trade statistics
        buf.write("--- Trade Statistics ---\n")
        buf.write(f"  Win rate:            {m.get('win_rate', 0):.1%}\n")
        buf.write(f"  Profit factor:       {m.get('profit_factor', 0):.2f}\n")
        buf.write(f"  Avg win:             {m.get('avg_win', 0):.4f}\n")
        buf.write(f"  Avg loss:            {m.get('avg_loss', 0):.4f}\n")
        buf.write(f"  Best day:            {m.get('best_day', 0):.4f}\n")
        buf.write(f"  Worst day:           {m.get('worst_day', 0):.4f}\n\n")

        # Distribution
        buf.write("--- Distribution ---\n")
        buf.write(f"  Skewness:            {m.get('skewness', 0):.3f}\n")
        buf.write(f"  Kurtosis:            {m.get('kurtosis', 0):.3f}\n")
        buf.write(f"  % positive months:   {m.get('pct_positive_months', 0):.1%}\n\n")

        # Window results
        if r.windows:
            buf.write("--- Walk-Forward Windows ---\n")
            for w in r.windows:
                buf.write(
                    f"  Window {w.window_idx}: "
                    f"{w.test_start} to {w.test_end} | "
                    f"Return: {w.window_return:+.2%} | "
                    f"Trades: {w.n_trades}\n"
                )
            buf.write("\n")

        # Validation
        if r.validation_results:
            buf.write("--- Validation ---\n")
            status = "PASSED" if r.validation_results.get("passed") else "FAILED"
            buf.write(f"  Result: {status}\n")
            for check in r.validation_results.get("checks", []):
                status = "PASS" if check["passed"] else "FAIL"
                buf.write(f"  [{status}] {check['name']}: {check['message']}\n")
            buf.write("\n")

        # Transaction costs
        if r.trades:
            total_costs = sum(t.get("total_cost", 0.0) for t in r.trades)
            total_gross = sum(abs(t.get("gross_value", 0.0)) for t in r.trades)
            cost_pct = total_costs / total_gross if total_gross > 0 else 0.0
            buf.write("--- Transaction Costs ---\n")
            buf.write(f"  Total costs:         ${total_costs:,.2f}\n")
            buf.write(f"  Cost as % of traded: {cost_pct:.4%}\n\n")

        buf.write("=" * 60 + "\n")
        return buf.getvalue()

    def comparison_table(self, other_results: list[BacktestResult]) -> str:
        """Generate a comparison table across multiple strategies."""
        all_results = [self.result] + other_results
        buf = StringIO()

        # Header
        names = [r.strategy_name for r in all_results]
        max_name = max(len(n) for n in names)
        col_width = 14

        buf.write(" " * (max_name + 2))
        for name in names:
            buf.write(f"{name:>{col_width}}")
        buf.write("\n")
        buf.write("-" * (max_name + 2 + col_width * len(names)) + "\n")

        # Metrics rows
        metric_rows = [
            ("Total Return", "total_return", ".2%"),
            ("Ann. Return", "annualized_return", ".2%"),
            ("Ann. Vol", "annualized_volatility", ".2%"),
            ("Sharpe", "sharpe_ratio", ".3f"),
            ("Sortino", "sortino_ratio", ".3f"),
            ("Max DD", "max_drawdown", ".2%"),
            ("Calmar", "calmar_ratio", ".3f"),
            ("Win Rate", "win_rate", ".1%"),
            ("Profit Factor", "profit_factor", ".2f"),
        ]

        for label, key, fmt in metric_rows:
            buf.write(f"{label:<{max_name + 2}}")
            for r in all_results:
                val = r.metrics.get(key, 0.0)
                if isinstance(val, int | float):
                    buf.write(f"{val:{col_width}{fmt}}")
                else:
                    buf.write(f"{'N/A':>{col_width}}")
            buf.write("\n")

        return buf.getvalue()

    def plot_equity_curve(self, save_path: Path | None = None) -> Any:
        """Plot equity curve with drawdown overlay.

        Returns matplotlib figure if matplotlib is available.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        if not self.result.equity_curve:
            return None

        dates = [ep["date"] for ep in self.result.equity_curve]
        values = [ep["portfolio_value"] for ep in self.result.equity_curve]
        returns = [ep["daily_return"] for ep in self.result.equity_curve]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Equity curve
        ax1.plot(dates, values, linewidth=1.5, color="#2196F3")
        ax1.set_title(f"{self.result.strategy_name} — Equity Curve")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(
            y=self.result.config.initial_capital,
            color="gray",
            linestyle="--",
            alpha=0.5,
        )

        # Drawdown
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        ax2.fill_between(dates, drawdown, 0, alpha=0.3, color="#F44336")
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_monthly_returns(self, save_path: Path | None = None) -> Any:
        """Plot monthly returns heatmap.

        Returns matplotlib figure if matplotlib is available.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        if not self.result.equity_curve:
            return None

        # Build daily returns series
        dates = pd.to_datetime([ep["date"] for ep in self.result.equity_curve])
        returns = pd.Series(
            [ep["daily_return"] for ep in self.result.equity_curve],
            index=dates,
        )

        # Monthly returns
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

        if len(monthly) == 0:
            return None

        # Build year x month matrix
        years = sorted(set(monthly.index.year))
        months = range(1, 13)
        data = np.full((len(years), 12), np.nan)

        for i, year in enumerate(years):
            for j, month in enumerate(months):
                mask = (monthly.index.year == year) & (monthly.index.month == month)
                if mask.any():
                    data[i, j] = monthly[mask].values[0]

        fig, ax = plt.subplots(figsize=(14, max(4, len(years))))
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)

        ax.set_xticks(range(12))
        ax.set_xticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        ax.set_title(f"{self.result.strategy_name} — Monthly Returns")

        # Annotate cells
        for i in range(len(years)):
            for j in range(12):
                if not np.isnan(data[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{data[i, j]:.1%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        fig.colorbar(im, ax=ax, label="Monthly Return")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
