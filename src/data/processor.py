"""
Data processing and normalization for Kronos Trading System.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..utils.logger import get_logger

logger = get_logger("data_processor")


class DataProcessor:
    """
    Process and normalize market data for the trading system.

    Responsibilities:
    - Gap filling
    - Resampling across timeframes
    - Feature normalization
    - Multi-timeframe alignment
    """

    def __init__(self):
        self.timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }

    def fill_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str,
        method: str = "ffill"
    ) -> pd.DataFrame:
        """
        Fill gaps in OHLCV data.

        Args:
            df: DataFrame with timestamp, open, high, low, close, volume
            timeframe: Expected timeframe (e.g., '5m', '30m')
            method: Fill method ('ffill', 'interpolate')

        Returns:
            Gap-free DataFrame
        """
        if df.empty:
            return df

        df = df.copy()
        df = df.set_index("timestamp")

        # Create complete time index
        freq = self._timeframe_to_freq(timeframe)
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )

        # Reindex and fill
        df = df.reindex(full_index)

        if method == "ffill":
            # Forward fill OHLC, zero fill volume
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
            df["volume"] = df["volume"].fillna(0)
        elif method == "interpolate":
            df = df.interpolate(method="linear")
            df["volume"] = df["volume"].fillna(0)

        df = df.reset_index()
        df = df.rename(columns={"index": "timestamp"})

        logger.info(f"Filled {len(full_index) - len(df)} gaps in {timeframe} data")
        return df

    def resample(
        self,
        df: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to a different timeframe.

        Args:
            df: Source DataFrame
            target_timeframe: Target timeframe (e.g., '4h')

        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df

        df = df.copy()
        df = df.set_index("timestamp")

        freq = self._timeframe_to_freq(target_timeframe)

        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        resampled = resampled.reset_index()
        logger.info(f"Resampled to {target_timeframe}: {len(resampled)} candles")
        return resampled

    def calculate_returns(
        self,
        df: pd.DataFrame,
        periods: List[int] = [1, 3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Calculate forward and backward returns.

        Args:
            df: OHLCV DataFrame
            periods: List of periods for return calculation

        Returns:
            DataFrame with return columns added
        """
        df = df.copy()

        for period in periods:
            # Backward returns (for features)
            df[f"return_{period}"] = df["close"].pct_change(period)

            # Forward returns (for targets)
            df[f"fwd_return_{period}"] = df["close"].shift(-period) / df["close"] - 1

        return df

    def calculate_volatility(
        self,
        df: pd.DataFrame,
        windows: List[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility measures.

        Args:
            df: OHLCV DataFrame
            windows: Rolling window sizes

        Returns:
            DataFrame with volatility columns added
        """
        df = df.copy()

        for window in windows:
            # Returns-based volatility
            returns = df["close"].pct_change()
            df[f"volatility_{window}"] = returns.rolling(window).std() * np.sqrt(252 * 24)  # Annualized

            # ATR (Average True Range)
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift(1)).abs()
            low_close = (df["low"] - df["close"].shift(1)).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f"atr_{window}"] = true_range.rolling(window).mean()

        return df

    def calculate_momentum(
        self,
        df: pd.DataFrame,
        periods: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Calculate momentum indicators.

        Args:
            df: OHLCV DataFrame
            periods: Momentum periods

        Returns:
            DataFrame with momentum columns added
        """
        df = df.copy()

        for period in periods:
            # Price momentum
            df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1

            # RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        return df

    def merge_funding_rates(
        self,
        ohlcv: pd.DataFrame,
        funding: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge funding rate data with OHLCV.

        Args:
            ohlcv: OHLCV DataFrame
            funding: Funding rate DataFrame

        Returns:
            Merged DataFrame with forward-filled funding rates
        """
        if funding.empty:
            ohlcv["funding_rate"] = 0.0
            return ohlcv

        ohlcv = ohlcv.copy()
        funding = funding.copy()

        # Merge on nearest timestamp
        ohlcv = pd.merge_asof(
            ohlcv.sort_values("timestamp"),
            funding.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        ohlcv["funding_rate"] = ohlcv["funding_rate"].ffill().fillna(0)
        return ohlcv

    def align_timeframes(
        self,
        data: Dict[str, pd.DataFrame],
        base_timeframe: str = "30m"
    ) -> pd.DataFrame:
        """
        Align multiple timeframe data to a base timeframe.

        Args:
            data: Dict of {timeframe: DataFrame}
            base_timeframe: Base timeframe to align to

        Returns:
            DataFrame with features from all timeframes
        """
        if base_timeframe not in data:
            raise ValueError(f"Base timeframe {base_timeframe} not in data")

        base = data[base_timeframe].copy()

        for tf, df in data.items():
            if tf == base_timeframe:
                continue

            # Prefix columns with timeframe
            df = df.copy()
            df = df.rename(columns={
                col: f"{tf}_{col}" for col in df.columns if col != "timestamp"
            })

            # Merge as-of
            base = pd.merge_asof(
                base.sort_values("timestamp"),
                df.sort_values("timestamp"),
                on="timestamp",
                direction="backward"
            )

        return base

    def prepare_kronos_input(
        self,
        df: pd.DataFrame,
        context_length: int = 512
    ) -> np.ndarray:
        """
        Prepare data for Kronos model input.

        Args:
            df: OHLCV DataFrame
            context_length: Number of bars for context

        Returns:
            NumPy array of shape (n_samples, context_length, 5) for OHLCV
        """
        if len(df) < context_length:
            raise ValueError(f"Need at least {context_length} rows, got {len(df)}")

        # Extract OHLCV columns
        ohlcv = df[["open", "high", "low", "close", "volume"]].values

        # Create rolling windows
        n_samples = len(ohlcv) - context_length + 1
        samples = np.array([
            ohlcv[i:i + context_length]
            for i in range(n_samples)
        ])

        return samples

    def normalize_features(
        self,
        df: pd.DataFrame,
        method: str = "zscore",
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Normalize feature columns.

        Args:
            df: Feature DataFrame
            method: Normalization method ('zscore', 'minmax', 'robust')
            window: Rolling window for normalization (None for full history)

        Returns:
            Normalized DataFrame
        """
        df = df.copy()

        # Identify numeric columns (exclude timestamp)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if window:
                if method == "zscore":
                    mean = df[col].rolling(window).mean()
                    std = df[col].rolling(window).std()
                    df[col] = (df[col] - mean) / std.replace(0, 1)
                elif method == "minmax":
                    min_val = df[col].rolling(window).min()
                    max_val = df[col].rolling(window).max()
                    df[col] = (df[col] - min_val) / (max_val - min_val).replace(0, 1)
            else:
                if method == "zscore":
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
                elif method == "minmax":
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        return df

    def _timeframe_to_freq(self, timeframe: str) -> str:
        """Convert timeframe string to pandas frequency."""
        if timeframe.endswith("m"):
            return f"{timeframe[:-1]}min"
        elif timeframe.endswith("h"):
            return f"{timeframe[:-1]}h"
        elif timeframe.endswith("d"):
            return f"{timeframe[:-1]}D"
        else:
            return timeframe


def process_symbol_data(
    symbol: str,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    timeframes: List[str] = ["5m", "30m", "4h"]
) -> pd.DataFrame:
    """
    Process all data for a single symbol.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        raw_dir: Directory with raw data
        processed_dir: Directory for processed data
        timeframes: List of timeframes to process

    Returns:
        Processed and aligned DataFrame
    """
    processor = DataProcessor()
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    data = {}

    for tf in timeframes:
        file_path = raw_path / f"{symbol}_{tf}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df = processor.fill_gaps(df, tf)
            df = processor.calculate_returns(df)
            df = processor.calculate_volatility(df)
            df = processor.calculate_momentum(df)
            data[tf] = df
            logger.info(f"Processed {symbol} {tf}: {len(df)} rows")

    # Load funding data
    funding_path = raw_path / f"{symbol}_funding.parquet"
    if funding_path.exists():
        funding = pd.read_parquet(funding_path)
        # Merge funding with base timeframe
        if "30m" in data:
            data["30m"] = processor.merge_funding_rates(data["30m"], funding)

    # Align timeframes
    if data:
        aligned = processor.align_timeframes(data, base_timeframe="30m")

        # Save processed data
        output_path = processed_path / f"{symbol}_processed.parquet"
        aligned.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

        return aligned

    return pd.DataFrame()
