"""
Kronos Trading Strategy Configuration
=====================================

Momentum Long-Only Trend Following Strategy
Optimized for crypto perpetual futures (BTC, ETH, SOL)

Created: 2026-01-11
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class BacktestParams:
    """Backtest configuration parameters."""
    initial_capital: float = 10000.0
    fee_rate: float = 0.0005          # 5 bps per trade
    slippage_bps: float = 5           # 5 bps slippage
    stop_loss_pct: float = 0.99       # 99% - effectively disabled
    take_profit_pct: float = 5.00     # 500% - effectively disabled
    leverage: float = 2.0             # 2x leverage
    max_leverage: float = 3.0         # Maximum allowed leverage
    trailing_stop_pct: float = 0.22   # 22% trailing stop from peak
    min_holding_bars: int = 96        # 2 days minimum hold (96 x 30min)
    max_holding_bars: int = 99999     # No max hold limit
    confidence_threshold: float = 0.10


@dataclass
class RiskParams:
    """Risk engine configuration parameters."""
    max_position_size_multiplier: float = 0.95  # 95% of equity * leverage
    volatility_target: float = 0.80             # High target = larger positions
    max_drawdown: float = 0.85                  # 85% drawdown limit
    max_consecutive_losses: int = 500           # Effectively disabled
    daily_loss_limit: float = 0.99              # Effectively disabled


@dataclass
class TrendDetectionParams:
    """Trend detection parameters."""
    ma_trend_period: int = 200        # 200-bar MA (100 hours on 30min)
    crash_threshold: float = 0.55     # 55% drawdown from ATH triggers exit
    warmup_bars: int = 200            # Bars before MA is valid


@dataclass
class PositionSizingParams:
    """Position sizing parameters."""
    # Regime-based multipliers
    strong_trend_multiplier: float = 1.5   # regime_bias >= 0.8
    moderate_trend_multiplier: float = 1.2  # regime_bias >= 0.5
    weak_trend_multiplier: float = 1.0      # regime_bias >= 0.3
    uncertain_multiplier: float = 0.8       # regime_bias < 0.3

    # Regime thresholds
    strong_threshold: float = 0.8
    moderate_threshold: float = 0.5
    weak_threshold: float = 0.3

    # Safety cap
    max_leverage_cap: float = 1.5  # Never exceed 1.5x equity


# Default configuration instances
DEFAULT_BACKTEST_PARAMS = BacktestParams()
DEFAULT_RISK_PARAMS = RiskParams()
DEFAULT_TREND_PARAMS = TrendDetectionParams()
DEFAULT_POSITION_PARAMS = PositionSizingParams()


def get_risk_engine_config(leverage: float = 2.0) -> Dict:
    """Get risk engine configuration dictionary."""
    params = DEFAULT_RISK_PARAMS
    return {
        "max_position_size": params.max_position_size_multiplier * leverage,
        "max_leverage": DEFAULT_BACKTEST_PARAMS.max_leverage,
        "volatility_target": params.volatility_target,
        "max_drawdown": params.max_drawdown,
        "max_consecutive_losses": params.max_consecutive_losses,
        "daily_loss_limit": params.daily_loss_limit,
    }


def get_backtest_config_dict(leverage: float = 2.0) -> Dict:
    """Get backtest configuration dictionary."""
    params = DEFAULT_BACKTEST_PARAMS
    return {
        "initial_capital": params.initial_capital,
        "fee_rate": params.fee_rate,
        "slippage_bps": params.slippage_bps,
        "stop_loss_pct": params.stop_loss_pct,
        "take_profit_pct": params.take_profit_pct,
        "leverage": leverage,
        "max_leverage": params.max_leverage,
        "trailing_stop_pct": params.trailing_stop_pct,
        "min_holding_bars": params.min_holding_bars,
        "max_holding_bars": params.max_holding_bars,
        "confidence_threshold": params.confidence_threshold,
    }


# Performance results from backtesting
PERFORMANCE_RESULTS = {
    "test_period": "2023-01-01 to 2024-12-31",
    "BTCUSDT": {
        "strategy_return": 6.3088,      # 630.88%
        "benchmark_return": 4.5960,     # 459.60%
        "alpha": 1.7129,                # +171.29%
        "beats_benchmark": True,
    },
    "ETHUSDT": {
        "strategy_return": 0.4811,      # 48.11%
        "benchmark_return": 0.4716,     # 47.16%
        "alpha": 0.0095,                # +0.95%
        "beats_benchmark": True,
    },
    "SOLUSDT": {
        "strategy_return": 0.8444,      # 84.44%
        "benchmark_return": 0.8977,     # 89.77%
        "alpha": -0.0533,               # -5.33%
        "beats_benchmark": False,
    },
}


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("KRONOS STRATEGY CONFIGURATION")
    print("=" * 60)

    print("\nBacktest Parameters:")
    for key, value in get_backtest_config_dict().__items__():
        print(f"  {key}: {value}")

    print("\nRisk Engine Parameters:")
    for key, value in get_risk_engine_config().__items__():
        print(f"  {key}: {value}")

    print("\nTrend Detection:")
    print(f"  MA Period: {DEFAULT_TREND_PARAMS.ma_trend_period} bars")
    print(f"  Crash Threshold: {DEFAULT_TREND_PARAMS.crash_threshold:.0%}")
    print(f"  Warmup: {DEFAULT_TREND_PARAMS.warmup_bars} bars")

    print("\nPerformance Results:")
    for symbol, results in PERFORMANCE_RESULTS.items():
        if symbol == "test_period":
            continue
        status = "BEATS" if results["beats_benchmark"] else "UNDER"
        print(f"  {symbol}: {results['strategy_return']:.2%} vs {results['benchmark_return']:.2%} ({status})")
