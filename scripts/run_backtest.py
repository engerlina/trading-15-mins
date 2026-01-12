#!/usr/bin/env python3
"""
Standalone Backtest Script

Run backtest with a trained model.

Usage:
    python scripts/run_backtest.py --model models/BTCUSDT_signal_model.pkl --data data/processed/BTCUSDT_processed.parquet
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.signal_model import SignalModel
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.metrics import BacktestMetrics, monte_carlo_simulation
from src.utils.logger import setup_logger

logger = setup_logger("backtest", log_dir="logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Run backtest")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to processed data"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Signal confidence threshold"
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.02,
        help="Stop loss percentage"
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.04,
        help="Take profit percentage"
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Run Monte Carlo simulation"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Backtest Runner")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = SignalModel.load(args.model)

    # Load data
    logger.info(f"Loading data from {args.data}")
    data = pd.read_parquet(args.data)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    logger.info(f"Data: {len(data)} rows, {data['timestamp'].min()} to {data['timestamp'].max()}")

    # Get feature columns from model
    feature_cols = model.feature_names
    valid_features = [c for c in feature_cols if c in data.columns]
    logger.info(f"Using {len(valid_features)} features")

    # Configure backtest
    config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=0.0005,
        slippage_bps=5,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        confidence_threshold=args.confidence
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(
        data=data,
        model=model,
        feature_columns=valid_features
    )

    # Print results
    metrics_calc = BacktestMetrics()
    report = metrics_calc.format_report(result.metrics)
    print(report)

    # Monte Carlo simulation
    if args.monte_carlo and result.trades:
        logger.info("\nRunning Monte Carlo simulation...")
        mc_results = monte_carlo_simulation(
            trades=result.trades,
            n_simulations=1000,
            initial_capital=args.capital
        )

        print("\n" + "=" * 60)
        print("MONTE CARLO SIMULATION (1000 runs)")
        print("=" * 60)
        print(f"Mean Final Equity:     ${mc_results['mean_final_equity']:,.2f}")
        print(f"95% CI:                ${mc_results['equity_ci_lower']:,.2f} - ${mc_results['equity_ci_upper']:,.2f}")
        print(f"Probability of Profit: {mc_results['prob_profit']:.1%}")
        print(f"Probability of 2x:     {mc_results['prob_double']:.1%}")
        print(f"Worst Case (5%):       ${mc_results['worst_case_equity']:,.2f}")
        print(f"Best Case (95%):       ${mc_results['best_case_equity']:,.2f}")
        print(f"Mean Max Drawdown:     {mc_results['mean_max_drawdown']:.1%}")
        print("=" * 60)

    # Save equity curve
    equity_file = Path("results") / "equity_curve.csv"
    equity_file.parent.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(equity_file)
    logger.info(f"Equity curve saved to {equity_file}")


if __name__ == "__main__":
    main()
