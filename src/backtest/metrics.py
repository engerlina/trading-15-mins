"""
Backtest performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
import logging

from ..risk.position import Trade
from ..utils.logger import get_logger

logger = get_logger("backtest_metrics")


class BacktestMetrics:
    """
    Calculate comprehensive backtest metrics.

    Metrics:
    - CAGR (Compound Annual Growth Rate)
    - Sharpe Ratio
    - Sortino Ratio
    - Max Drawdown
    - Calmar Ratio
    - Profit Factor
    - Win Rate
    - Exposure
    """

    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 365 * 48):
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of trading periods per year (48 = 30m bars per day)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate_all(
        self,
        equity_curve: pd.Series,
        trades: List[Trade],
        initial_capital: float,
        prices: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate all backtest metrics.

        Args:
            equity_curve: Series of equity values indexed by timestamp
            trades: List of completed trades
            initial_capital: Starting capital
            prices: Optional price series for buy-and-hold comparison

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics["initial_capital"] = initial_capital
        metrics["final_equity"] = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
        metrics["total_return"] = (metrics["final_equity"] / initial_capital) - 1
        metrics["total_return_pct"] = f"{metrics['total_return']:.2%}"

        # Time-based metrics
        if len(equity_curve) > 1:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            metrics["trading_days"] = days

            # CAGR
            metrics["cagr"] = self._calculate_cagr(initial_capital, metrics["final_equity"], days)

            # Returns
            returns = equity_curve.pct_change().dropna()

            # Sharpe Ratio
            metrics["sharpe"] = self._calculate_sharpe(returns)

            # Sortino Ratio
            metrics["sortino"] = self._calculate_sortino(returns)

            # Max Drawdown
            metrics["max_drawdown"] = self._calculate_max_drawdown(equity_curve)
            metrics["max_drawdown_pct"] = f"{metrics['max_drawdown']:.2%}"

            # Calmar Ratio
            metrics["calmar"] = self._calculate_calmar(metrics["cagr"], metrics["max_drawdown"])

            # Volatility
            metrics["volatility"] = returns.std() * np.sqrt(self.periods_per_year)

        # Trade metrics
        if trades:
            trade_metrics = self._calculate_trade_metrics(trades)
            metrics.update(trade_metrics)
        else:
            metrics["total_trades"] = 0
            metrics["win_rate"] = 0.0
            metrics["profit_factor"] = 0.0

        # Buy-and-hold benchmark comparison
        if prices is not None and len(prices) > 1:
            benchmark_metrics = self._calculate_benchmark(prices, initial_capital, metrics)
            metrics.update(benchmark_metrics)

        return metrics

    def _calculate_benchmark(
        self,
        prices: pd.Series,
        initial_capital: float,
        strategy_metrics: Dict
    ) -> Dict:
        """Calculate buy-and-hold benchmark metrics."""
        benchmark = {}

        # Buy-and-hold return
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        bh_return = (end_price / start_price) - 1
        benchmark["benchmark_return"] = bh_return
        benchmark["benchmark_return_pct"] = f"{bh_return:.2%}"

        # Buy-and-hold final equity
        bh_shares = initial_capital / start_price
        bh_final_equity = bh_shares * end_price
        benchmark["benchmark_final_equity"] = bh_final_equity

        # Strategy vs benchmark (alpha)
        strategy_return = strategy_metrics.get("total_return", 0)
        benchmark["alpha"] = strategy_return - bh_return
        benchmark["alpha_pct"] = f"{benchmark['alpha']:.2%}"

        # Outperformance
        benchmark["beats_benchmark"] = strategy_return > bh_return

        # Buy-and-hold risk metrics
        bh_equity_curve = (prices / start_price) * initial_capital
        bh_returns = bh_equity_curve.pct_change().dropna()

        if len(bh_returns) > 1:
            benchmark["benchmark_sharpe"] = self._calculate_sharpe(bh_returns)
            benchmark["benchmark_max_drawdown"] = self._calculate_max_drawdown(bh_equity_curve)
            benchmark["benchmark_volatility"] = bh_returns.std() * np.sqrt(self.periods_per_year)

            # Risk-adjusted alpha (strategy sharpe - benchmark sharpe)
            benchmark["sharpe_alpha"] = strategy_metrics.get("sharpe", 0) - benchmark["benchmark_sharpe"]

        return benchmark

    def _calculate_cagr(
        self,
        initial_capital: float,
        final_equity: float,
        days: int
    ) -> float:
        """Calculate Compound Annual Growth Rate."""
        if days <= 0 or initial_capital <= 0:
            return 0.0

        years = days / 365.0
        if years <= 0:
            return 0.0

        return (final_equity / initial_capital) ** (1 / years) - 1

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe Ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / self.periods_per_year)
        return np.sqrt(self.periods_per_year) * excess_returns.mean() / returns.std()

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate annualized Sortino Ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / self.periods_per_year)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0

        downside_std = downside_returns.std()
        return np.sqrt(self.periods_per_year) * excess_returns.mean() / downside_std

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0

        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max

        return abs(drawdowns.min())

    def _calculate_calmar(self, cagr: float, max_drawdown: float) -> float:
        """Calculate Calmar Ratio (CAGR / Max Drawdown)."""
        if max_drawdown == 0:
            return float("inf") if cagr > 0 else 0.0
        return cagr / max_drawdown

    def _calculate_trade_metrics(self, trades: List[Trade]) -> Dict:
        """Calculate trade-based metrics."""
        metrics = {}

        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]

        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        metrics["total_trades"] = len(trades)
        metrics["winning_trades"] = len(winning_trades)
        metrics["losing_trades"] = len(losing_trades)
        metrics["win_rate"] = len(winning_trades) / len(trades) if trades else 0

        # PnL metrics
        metrics["total_pnl"] = sum(pnls)
        metrics["avg_pnl"] = np.mean(pnls) if pnls else 0
        metrics["avg_pnl_pct"] = np.mean(pnl_pcts) if pnl_pcts else 0
        metrics["pnl_std"] = np.std(pnls) if pnls else 0

        # Best/worst trades
        metrics["best_trade"] = max(pnls) if pnls else 0
        metrics["worst_trade"] = min(pnls) if pnls else 0
        metrics["best_trade_pct"] = max(pnl_pcts) if pnl_pcts else 0
        metrics["worst_trade_pct"] = min(pnl_pcts) if pnl_pcts else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average win/loss
        metrics["avg_win"] = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        metrics["avg_loss"] = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # Expectancy
        metrics["expectancy"] = (
            metrics["win_rate"] * metrics["avg_win"] +
            (1 - metrics["win_rate"]) * metrics["avg_loss"]
        )

        # Trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
        metrics["avg_trade_duration_hours"] = np.mean(durations) if durations else 0

        # Fees
        metrics["total_fees"] = sum(t.fees for t in trades)
        metrics["total_funding"] = sum(t.funding for t in trades)

        # Exit reason breakdown
        exit_reasons = {}
        for trade in trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "pnl": 0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["pnl"] += trade.pnl
        metrics["exit_reasons"] = exit_reasons

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)

        metrics["max_consecutive_wins"] = max_consec_wins
        metrics["max_consecutive_losses"] = max_consec_losses

        return metrics

    def format_report(self, metrics: Dict) -> str:
        """Format metrics as a readable report."""
        report = []
        report.append("=" * 60)
        report.append("BACKTEST RESULTS")
        report.append("=" * 60)

        # Performance
        report.append("\n--- Performance ---")
        report.append(f"Initial Capital:     ${metrics.get('initial_capital', 0):,.2f}")
        report.append(f"Final Equity:        ${metrics.get('final_equity', 0):,.2f}")
        report.append(f"Total Return:        {metrics.get('total_return', 0):.2%}")
        report.append(f"CAGR:                {metrics.get('cagr', 0):.2%}")
        report.append(f"Trading Days:        {metrics.get('trading_days', 0)}")

        # Risk Metrics
        report.append("\n--- Risk Metrics ---")
        report.append(f"Sharpe Ratio:        {metrics.get('sharpe', 0):.2f}")
        report.append(f"Sortino Ratio:       {metrics.get('sortino', 0):.2f}")
        report.append(f"Calmar Ratio:        {metrics.get('calmar', 0):.2f}")
        report.append(f"Max Drawdown:        {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Volatility:          {metrics.get('volatility', 0):.2%}")

        # Trade Statistics
        report.append("\n--- Trade Statistics ---")
        report.append(f"Total Trades:        {metrics.get('total_trades', 0)}")
        report.append(f"Winning Trades:      {metrics.get('winning_trades', 0)}")
        report.append(f"Losing Trades:       {metrics.get('losing_trades', 0)}")
        report.append(f"Win Rate:            {metrics.get('win_rate', 0):.2%}")
        report.append(f"Profit Factor:       {metrics.get('profit_factor', 0):.2f}")
        report.append(f"Expectancy:          ${metrics.get('expectancy', 0):.2f}")

        # PnL
        report.append("\n--- PnL Breakdown ---")
        report.append(f"Total PnL:           ${metrics.get('total_pnl', 0):,.2f}")
        report.append(f"Avg Trade PnL:       ${metrics.get('avg_pnl', 0):.2f}")
        report.append(f"Avg Win:             ${metrics.get('avg_win', 0):.2f}")
        report.append(f"Avg Loss:            ${metrics.get('avg_loss', 0):.2f}")
        report.append(f"Best Trade:          ${metrics.get('best_trade', 0):.2f}")
        report.append(f"Worst Trade:         ${metrics.get('worst_trade', 0):.2f}")

        # Costs
        report.append("\n--- Costs ---")
        report.append(f"Total Fees:          ${metrics.get('total_fees', 0):.2f}")
        report.append(f"Total Funding:       ${metrics.get('total_funding', 0):.2f}")

        # Streaks
        report.append("\n--- Streaks ---")
        report.append(f"Max Consec. Wins:    {metrics.get('max_consecutive_wins', 0)}")
        report.append(f"Max Consec. Losses:  {metrics.get('max_consecutive_losses', 0)}")

        # Benchmark comparison (if available)
        if "benchmark_return" in metrics:
            report.append("\n--- Buy & Hold Benchmark ---")
            report.append(f"Benchmark Return:    {metrics.get('benchmark_return', 0):.2%}")
            report.append(f"Benchmark Equity:    ${metrics.get('benchmark_final_equity', 0):,.2f}")
            report.append(f"Benchmark Sharpe:    {metrics.get('benchmark_sharpe', 0):.2f}")
            report.append(f"Benchmark Max DD:    {metrics.get('benchmark_max_drawdown', 0):.2%}")
            report.append(f"Strategy Alpha:      {metrics.get('alpha', 0):.2%}")
            report.append(f"Sharpe Alpha:        {metrics.get('sharpe_alpha', 0):+.2f}")
            beats = "YES" if metrics.get('beats_benchmark', False) else "NO"
            report.append(f"Beats Benchmark:     {beats}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


def monte_carlo_simulation(
    trades: List[Trade],
    n_simulations: int = 1000,
    initial_capital: float = 10000.0,
    confidence_level: float = 0.95
) -> Dict:
    """
    Run Monte Carlo simulation on trade results.

    Args:
        trades: List of historical trades
        n_simulations: Number of simulations
        initial_capital: Starting capital
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary with simulation results
    """
    if not trades:
        return {}

    pnls = np.array([t.pnl for t in trades])
    n_trades = len(trades)

    # Run simulations
    final_equities = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Random sample with replacement
        sampled_pnls = np.random.choice(pnls, size=n_trades, replace=True)

        # Calculate equity curve
        equity = initial_capital + np.cumsum(sampled_pnls)

        # Final equity
        final_equities.append(equity[-1])

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        max_drawdowns.append(np.max(drawdowns))

    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)

    # Calculate confidence intervals
    alpha = (1 - confidence_level) / 2

    return {
        "mean_final_equity": np.mean(final_equities),
        "median_final_equity": np.median(final_equities),
        "std_final_equity": np.std(final_equities),
        "equity_ci_lower": np.percentile(final_equities, alpha * 100),
        "equity_ci_upper": np.percentile(final_equities, (1 - alpha) * 100),
        "mean_max_drawdown": np.mean(max_drawdowns),
        "median_max_drawdown": np.median(max_drawdowns),
        "max_drawdown_ci_lower": np.percentile(max_drawdowns, alpha * 100),
        "max_drawdown_ci_upper": np.percentile(max_drawdowns, (1 - alpha) * 100),
        "prob_profit": (final_equities > initial_capital).mean(),
        "prob_double": (final_equities > 2 * initial_capital).mean(),
        "worst_case_equity": np.percentile(final_equities, 5),
        "best_case_equity": np.percentile(final_equities, 95),
    }
