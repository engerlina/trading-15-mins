"""
Short Timeframe Strategy Backtester

Implements and backtests:
1. RSI Mean Reversion (5m with 1h filter)
2. RSI Divergence (15m)

Usage:
    python scripts/backtest_short_tf.py --symbol BTCUSDT --strategy rsi_mean_reversion
    python scripts/backtest_short_tf.py --symbol BTCUSDT --strategy rsi_divergence
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger("short_tf_backtest")


class ShortTFBacktester:
    """Backtester for short timeframe strategies."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0005,  # 0.05% taker fee
        slippage_bps: float = 5,   # 5 bps slippage
        leverage: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps / 10000
        self.leverage = leverage

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 5m data to higher timeframe."""
        df = df.set_index('timestamp')
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return resampled.reset_index()

    def run_rsi_mean_reversion(
        self,
        df_5m: pd.DataFrame,
        rsi_period: int = 7,
        rsi_oversold: float = 25,         # More extreme oversold
        rsi_overbought: float = 75,       # More extreme overbought
        htf_filter: bool = True,
        profit_target_pct: float = 0.02,  # 2% target (was 1%)
        stop_loss_pct: float = 0.01,      # 1% stop (was 0.5%)
        max_holding_bars: int = 96,       # Max 8 hours (was 4)
        long_only: bool = False,          # Only take long trades
        exit_on_rsi_neutral: bool = False, # Don't exit on RSI crossing 50
    ) -> Dict:
        """
        RSI Mean Reversion Strategy

        - Buy when 5m RSI < oversold AND 1h RSI < 50 (if filter enabled)
        - Sell when 5m RSI > overbought AND 1h RSI > 50 (if filter enabled)
        - Exit at profit target, stop loss, or max holding time
        """
        logger.info(f"Running RSI Mean Reversion: period={rsi_period}, oversold={rsi_oversold}, overbought={rsi_overbought}")

        # Calculate 5m RSI
        df = df_5m.copy()
        df['rsi_5m'] = self.calculate_rsi(df['close'], rsi_period)

        # Calculate 1h RSI for higher timeframe filter
        if htf_filter:
            df_1h = self.resample_to_timeframe(df_5m.copy(), '1h')
            df_1h['rsi_1h'] = self.calculate_rsi(df_1h['close'], 14)
            # Merge back to 5m
            df_1h = df_1h[['timestamp', 'rsi_1h']]
            df['timestamp_1h'] = df['timestamp'].dt.floor('1h')
            df = df.merge(df_1h, left_on='timestamp_1h', right_on='timestamp', how='left', suffixes=('', '_1h'))
            df['rsi_1h'] = df['rsi_1h'].ffill()
        else:
            df['rsi_1h'] = 50  # Neutral, no filter

        # Simulation
        equity = self.initial_capital
        position = 0  # 1 = long, -1 = short, 0 = flat
        entry_price = 0
        entry_bar = 0
        trades = []
        equity_curve = [equity]

        for i in range(1, len(df)):
            current_price = df.iloc[i]['close']
            rsi_5m = df.iloc[i]['rsi_5m']
            rsi_1h = df.iloc[i]['rsi_1h']
            timestamp = df.iloc[i]['timestamp']

            if pd.isna(rsi_5m) or pd.isna(rsi_1h):
                equity_curve.append(equity)
                continue

            # Check exits first
            if position != 0:
                bars_held = i - entry_bar

                if position == 1:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price

                    # Exit conditions
                    exit_reason = None
                    if pnl_pct >= profit_target_pct:
                        exit_reason = 'profit_target'
                    elif pnl_pct <= -stop_loss_pct:
                        exit_reason = 'stop_loss'
                    elif bars_held >= max_holding_bars:
                        exit_reason = 'max_time'
                    elif exit_on_rsi_neutral and rsi_5m > 50:  # RSI crossed back to neutral
                        exit_reason = 'rsi_neutral'

                    if exit_reason:
                        # Close long
                        exit_price = current_price * (1 - self.slippage_bps)
                        fee = abs(equity * self.leverage) * self.fee_rate
                        pnl = (exit_price - entry_price) / entry_price * equity * self.leverage - fee
                        equity += pnl
                        trades.append({
                            'entry_time': df.iloc[entry_bar]['timestamp'],
                            'exit_time': timestamp,
                            'side': 'long',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl / self.initial_capital * 100,
                            'bars_held': bars_held,
                            'exit_reason': exit_reason
                        })
                        position = 0

                elif position == -1:  # Short position
                    pnl_pct = (entry_price - current_price) / entry_price

                    exit_reason = None
                    if pnl_pct >= profit_target_pct:
                        exit_reason = 'profit_target'
                    elif pnl_pct <= -stop_loss_pct:
                        exit_reason = 'stop_loss'
                    elif bars_held >= max_holding_bars:
                        exit_reason = 'max_time'
                    elif exit_on_rsi_neutral and rsi_5m < 50:
                        exit_reason = 'rsi_neutral'

                    if exit_reason:
                        # Close short
                        exit_price = current_price * (1 + self.slippage_bps)
                        fee = abs(equity * self.leverage) * self.fee_rate
                        pnl = (entry_price - exit_price) / entry_price * equity * self.leverage - fee
                        equity += pnl
                        trades.append({
                            'entry_time': df.iloc[entry_bar]['timestamp'],
                            'exit_time': timestamp,
                            'side': 'short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl / self.initial_capital * 100,
                            'bars_held': bars_held,
                            'exit_reason': exit_reason
                        })
                        position = 0

            # Check entries (only if flat)
            if position == 0:
                # Long entry: RSI oversold
                if rsi_5m < rsi_oversold and (not htf_filter or rsi_1h > 50):
                    entry_price = current_price * (1 + self.slippage_bps)
                    fee = abs(equity * self.leverage) * self.fee_rate
                    equity -= fee
                    position = 1
                    entry_bar = i

                # Short entry: RSI overbought (skip if long_only)
                elif not long_only and rsi_5m > rsi_overbought and (not htf_filter or rsi_1h < 50):
                    entry_price = current_price * (1 - self.slippage_bps)
                    fee = abs(equity * self.leverage) * self.fee_rate
                    equity -= fee
                    position = -1
                    entry_bar = i

            equity_curve.append(equity)

        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve, df, 'RSI Mean Reversion')

    def run_rsi_divergence(
        self,
        df_5m: pd.DataFrame,
        rsi_period: int = 14,
        pivot_bars: int = 5,
        min_divergence_bars: int = 10,
        max_divergence_bars: int = 50,
        profit_target_pct: float = 0.015,  # 1.5% target
        stop_loss_pct: float = 0.01,       # 1% stop
        max_holding_bars: int = 96,        # Max 8 hours
        long_only: bool = False,           # Only take long trades
        use_trailing_stop: bool = False,   # Use trailing stop instead of fixed
        trailing_stop_pct: float = 0.02,   # 2% trailing stop
        input_timeframe: str = '5m',       # Input data timeframe
    ) -> Dict:
        """
        RSI Divergence Strategy (on 15m/30m timeframe)

        - Bullish divergence: Price lower low + RSI higher low -> BUY
        - Bearish divergence: Price higher high + RSI lower high -> SELL
        """
        logger.info(f"Running RSI Divergence: period={rsi_period}, pivot_bars={pivot_bars}, trailing={use_trailing_stop}")

        # Resample based on input timeframe
        if input_timeframe == '30m':
            # Use 30m directly, no resampling needed
            df = df_5m.copy()
        else:
            # Resample 5m to 15m for divergence detection
            df = self.resample_to_timeframe(df_5m.copy(), '15min')
        df['rsi'] = self.calculate_rsi(df['close'], rsi_period)

        # Find pivot highs and lows
        df['pivot_high'] = False
        df['pivot_low'] = False

        for i in range(pivot_bars, len(df) - pivot_bars):
            # Check for pivot high
            if all(df.iloc[i]['high'] >= df.iloc[i-j]['high'] for j in range(1, pivot_bars+1)) and \
               all(df.iloc[i]['high'] >= df.iloc[i+j]['high'] for j in range(1, pivot_bars+1)):
                df.iloc[i, df.columns.get_loc('pivot_high')] = True

            # Check for pivot low
            if all(df.iloc[i]['low'] <= df.iloc[i-j]['low'] for j in range(1, pivot_bars+1)) and \
               all(df.iloc[i]['low'] <= df.iloc[i+j]['low'] for j in range(1, pivot_bars+1)):
                df.iloc[i, df.columns.get_loc('pivot_low')] = True

        # Simulation
        equity = self.initial_capital
        position = 0
        entry_price = 0
        entry_bar = 0
        best_price = 0  # Track best price since entry for trailing stop
        trades = []
        equity_curve = [equity]

        # Track recent pivots for divergence detection
        recent_pivot_lows = []  # [(bar_idx, price, rsi)]
        recent_pivot_highs = []

        for i in range(pivot_bars, len(df)):
            current_price = df.iloc[i]['close']
            current_rsi = df.iloc[i]['rsi']
            timestamp = df.iloc[i]['timestamp']

            if pd.isna(current_rsi):
                equity_curve.append(equity)
                continue

            # Update pivot tracking
            if df.iloc[i]['pivot_low']:
                recent_pivot_lows.append((i, df.iloc[i]['low'], current_rsi))
                # Keep only recent pivots
                recent_pivot_lows = [(idx, p, r) for idx, p, r in recent_pivot_lows
                                     if i - idx <= max_divergence_bars]

            if df.iloc[i]['pivot_high']:
                recent_pivot_highs.append((i, df.iloc[i]['high'], current_rsi))
                recent_pivot_highs = [(idx, p, r) for idx, p, r in recent_pivot_highs
                                      if i - idx <= max_divergence_bars]

            # Update best price for trailing stop
            if position == 1:
                best_price = max(best_price, current_price)
            elif position == -1:
                best_price = min(best_price, current_price) if best_price > 0 else current_price

            # Check exits
            if position != 0:
                bars_held = i - entry_bar

                if position == 1:
                    pnl_pct = (current_price - entry_price) / entry_price

                    exit_reason = None
                    if use_trailing_stop:
                        # Trailing stop: exit if price drops trailing_stop_pct from best
                        trailing_stop_price = best_price * (1 - trailing_stop_pct)
                        if current_price <= trailing_stop_price and pnl_pct > 0:
                            exit_reason = 'trailing_stop'
                        elif pnl_pct <= -stop_loss_pct:
                            exit_reason = 'stop_loss'
                        elif bars_held >= max_holding_bars:
                            exit_reason = 'max_time'
                    else:
                        if pnl_pct >= profit_target_pct:
                            exit_reason = 'profit_target'
                        elif pnl_pct <= -stop_loss_pct:
                            exit_reason = 'stop_loss'
                        elif bars_held >= max_holding_bars:
                            exit_reason = 'max_time'

                    if exit_reason:
                        exit_price = current_price * (1 - self.slippage_bps)
                        fee = abs(equity * self.leverage) * self.fee_rate
                        pnl = (exit_price - entry_price) / entry_price * equity * self.leverage - fee
                        equity += pnl
                        trades.append({
                            'entry_time': df.iloc[entry_bar]['timestamp'],
                            'exit_time': timestamp,
                            'side': 'long',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl / self.initial_capital * 100,
                            'bars_held': bars_held,
                            'exit_reason': exit_reason
                        })
                        position = 0

                elif position == -1:
                    pnl_pct = (entry_price - current_price) / entry_price

                    exit_reason = None
                    if pnl_pct >= profit_target_pct:
                        exit_reason = 'profit_target'
                    elif pnl_pct <= -stop_loss_pct:
                        exit_reason = 'stop_loss'
                    elif bars_held >= max_holding_bars:
                        exit_reason = 'max_time'

                    if exit_reason:
                        exit_price = current_price * (1 + self.slippage_bps)
                        fee = abs(equity * self.leverage) * self.fee_rate
                        pnl = (entry_price - exit_price) / entry_price * equity * self.leverage - fee
                        equity += pnl
                        trades.append({
                            'entry_time': df.iloc[entry_bar]['timestamp'],
                            'exit_time': timestamp,
                            'side': 'short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl / self.initial_capital * 100,
                            'bars_held': bars_held,
                            'exit_reason': exit_reason
                        })
                        position = 0

            # Check for divergence entries (only if flat)
            if position == 0 and len(recent_pivot_lows) >= 2:
                # Check for bullish divergence (price lower low, RSI higher low)
                for j in range(len(recent_pivot_lows) - 1):
                    prev_idx, prev_price, prev_rsi = recent_pivot_lows[j]
                    curr_idx, curr_price, curr_rsi = recent_pivot_lows[-1]

                    bars_between = curr_idx - prev_idx
                    if min_divergence_bars <= bars_between <= max_divergence_bars:
                        # Bullish divergence: price made lower low, RSI made higher low
                        if curr_price < prev_price and curr_rsi > prev_rsi and curr_rsi < 40:
                            entry_price = current_price * (1 + self.slippage_bps)
                            best_price = entry_price  # Reset best price for trailing stop
                            fee = abs(equity * self.leverage) * self.fee_rate
                            equity -= fee
                            position = 1
                            entry_bar = i
                            break

            if not long_only and position == 0 and len(recent_pivot_highs) >= 2:
                # Check for bearish divergence (price higher high, RSI lower high)
                for j in range(len(recent_pivot_highs) - 1):
                    prev_idx, prev_price, prev_rsi = recent_pivot_highs[j]
                    curr_idx, curr_price, curr_rsi = recent_pivot_highs[-1]

                    bars_between = curr_idx - prev_idx
                    if min_divergence_bars <= bars_between <= max_divergence_bars:
                        # Bearish divergence: price made higher high, RSI made lower high
                        if curr_price > prev_price and curr_rsi < prev_rsi and curr_rsi > 60:
                            entry_price = current_price * (1 - self.slippage_bps)
                            fee = abs(equity * self.leverage) * self.fee_rate
                            equity -= fee
                            position = -1
                            entry_bar = i
                            break

            equity_curve.append(equity)

        return self._calculate_metrics(trades, equity_curve, df, 'RSI Divergence')

    def _calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[float],
        df: pd.DataFrame,
        strategy_name: str
    ) -> Dict:
        """Calculate performance metrics."""
        equity_series = pd.Series(equity_curve)

        # Basic metrics
        final_equity = equity_curve[-1] if equity_curve else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100

        # Calculate drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100

        # Trade statistics
        n_trades = len(trades)
        if n_trades > 0:
            winners = [t for t in trades if t['pnl'] > 0]
            losers = [t for t in trades if t['pnl'] <= 0]
            win_rate = len(winners) / n_trades * 100

            avg_win = np.mean([t['pnl'] for t in winners]) if winners else 0
            avg_loss = np.mean([t['pnl'] for t in losers]) if losers else 0

            total_profit = sum(t['pnl'] for t in winners)
            total_loss = abs(sum(t['pnl'] for t in losers))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            avg_bars_held = np.mean([t['bars_held'] for t in trades])

            # Exit reason breakdown
            exit_reasons = {}
            for t in trades:
                r = t['exit_reason']
                exit_reasons[r] = exit_reasons.get(r, 0) + 1
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_bars_held = 0
            exit_reasons = {}

        # Calculate Sharpe (daily returns)
        returns = equity_series.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 12) if returns.std() > 0 else 0  # Annualized for 5m

        # Buy and hold benchmark
        start_price = df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        benchmark_return = (end_price - start_price) / start_price * 100

        metrics = {
            'strategy': strategy_name,
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'alpha': total_return - benchmark_return,
            'beats_benchmark': total_return > benchmark_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_bars_held': avg_bars_held,
            'exit_reasons': exit_reasons,
            'trades': trades,
        }

        return metrics

    def print_results(self, metrics: Dict):
        """Print formatted results."""
        print("\n" + "=" * 60)
        print(f"BACKTEST RESULTS: {metrics['strategy']}")
        print("=" * 60)

        print(f"\n--- Performance ---")
        print(f"Initial Capital:     ${metrics['initial_capital']:,.2f}")
        print(f"Final Equity:        ${metrics['final_equity']:,.2f}")
        print(f"Total Return:        {metrics['total_return']:.2f}%")
        print(f"Benchmark Return:    {metrics['benchmark_return']:.2f}%")
        print(f"Alpha:               {metrics['alpha']:.2f}%")
        print(f"Beats Benchmark:     {'YES' if metrics['beats_benchmark'] else 'NO'}")

        print(f"\n--- Risk Metrics ---")
        print(f"Max Drawdown:        {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")

        print(f"\n--- Trade Statistics ---")
        print(f"Total Trades:        {metrics['n_trades']}")
        print(f"Win Rate:            {metrics['win_rate']:.2f}%")
        print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"Avg Win:             ${metrics['avg_win']:.2f}")
        print(f"Avg Loss:            ${metrics['avg_loss']:.2f}")
        print(f"Avg Bars Held:       {metrics['avg_bars_held']:.1f}")

        if metrics['exit_reasons']:
            print(f"\n--- Exit Reasons ---")
            for reason, count in sorted(metrics['exit_reasons'].items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Short Timeframe Strategy Backtester')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--strategy', type=str, default='both',
                       choices=['rsi_mean_reversion', 'rsi_divergence', 'both'],
                       help='Strategy to backtest')
    parser.add_argument('--timeframe', type=str, default='5m', choices=['5m', '30m'],
                       help='Data timeframe (5m or 30m)')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--leverage', type=float, default=1.0, help='Leverage')
    parser.add_argument('--long-only', action='store_true', help='Only take long trades')
    parser.add_argument('--trailing-stop', action='store_true', help='Use trailing stop instead of fixed profit target')
    parser.add_argument('--trailing-pct', type=float, default=0.05, help='Trailing stop percentage (default 5%)')

    args = parser.parse_args()

    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / f'{args.symbol}_{args.timeframe}.parquet'
    logger.info(f"Loading data from {data_path}")

    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter by date
    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

    logger.info(f"Data loaded: {len(df)} bars from {df.timestamp.min()} to {df.timestamp.max()}")

    # Run backtester
    backtester = ShortTFBacktester(
        initial_capital=10000.0,
        fee_rate=0.0005,
        slippage_bps=5,
        leverage=args.leverage
    )

    results = []

    if args.strategy in ['rsi_mean_reversion', 'both']:
        logger.info(f"Running RSI Mean Reversion strategy (long_only={args.long_only})...")
        metrics = backtester.run_rsi_mean_reversion(df, long_only=args.long_only)
        backtester.print_results(metrics)
        results.append(metrics)

    if args.strategy in ['rsi_divergence', 'both']:
        logger.info(f"Running RSI Divergence strategy (long_only={args.long_only}, trailing={args.trailing_stop})...")
        metrics = backtester.run_rsi_divergence(
            df,
            long_only=args.long_only,
            use_trailing_stop=args.trailing_stop,
            trailing_stop_pct=args.trailing_pct,
            input_timeframe=args.timeframe
        )
        backtester.print_results(metrics)
        results.append(metrics)

    # Summary comparison if both
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("STRATEGY COMPARISON")
        print("=" * 60)
        print(f"{'Metric':<25} {'RSI Mean Rev':<15} {'RSI Divergence':<15}")
        print("-" * 55)
        print(f"{'Total Return':<25} {results[0]['total_return']:>12.2f}% {results[1]['total_return']:>12.2f}%")
        print(f"{'Benchmark':<25} {results[0]['benchmark_return']:>12.2f}% {results[1]['benchmark_return']:>12.2f}%")
        print(f"{'Alpha':<25} {results[0]['alpha']:>12.2f}% {results[1]['alpha']:>12.2f}%")
        print(f"{'Max Drawdown':<25} {results[0]['max_drawdown']:>12.2f}% {results[1]['max_drawdown']:>12.2f}%")
        print(f"{'Sharpe Ratio':<25} {results[0]['sharpe_ratio']:>12.2f} {results[1]['sharpe_ratio']:>12.2f}")
        print(f"{'Trades':<25} {results[0]['n_trades']:>12} {results[1]['n_trades']:>12}")
        print(f"{'Win Rate':<25} {results[0]['win_rate']:>12.2f}% {results[1]['win_rate']:>12.2f}%")
        print(f"{'Beats Benchmark':<25} {'YES' if results[0]['beats_benchmark'] else 'NO':>12} {'YES' if results[1]['beats_benchmark'] else 'NO':>12}")
        print("=" * 60)


if __name__ == '__main__':
    main()
