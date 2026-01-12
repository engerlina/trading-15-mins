#!/usr/bin/env python3
"""
Data Collection Script

Collects historical data from Binance for backtesting.

Usage:
    python scripts/collect_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --start 2023-01-01 --end 2024-12-31
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collector import BinanceCollector, collect_historical_data
from src.utils.logger import setup_logger

logger = setup_logger("collect_data", log_dir="logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect historical market data")

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Symbols to collect"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["5m", "30m", "4h"],
        help="Timeframes to collect"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory"
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Data Collection")
    logger.info("=" * 60)
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)

    # Collect data
    results = await collect_historical_data(
        symbols=args.symbols,
        timeframes=args.timeframes,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Collection Summary")
    logger.info("=" * 60)

    for key, df in results.items():
        logger.info(f"{key}: {len(df)} rows")

    logger.info("Data collection complete!")


if __name__ == "__main__":
    asyncio.run(main())
