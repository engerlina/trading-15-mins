"""
Data collection from Binance and Hyperliquid.

Binance: Used for historical data (deep history)
Hyperliquid: Used for live trading and validation
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import logging

from ..utils.logger import get_logger

logger = get_logger("data_collector")


class DataCollector(ABC):
    """Abstract base class for data collectors."""

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch OHLCV data."""
        pass

    @abstractmethod
    async def fetch_funding_rate(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch funding rate history."""
        pass


class BinanceCollector(DataCollector):
    """
    Binance Futures data collector.
    Used for historical data collection due to deep history availability.
    """

    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://fapi.binance.com"
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(f"Initialized BinanceCollector with base_url: {base_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            headers = {}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
                logger.info("Using API key for authenticated requests (higher rate limits)")
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
            logger.debug("Created new aiohttp session with 30s timeout")
        return self.session

    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance Futures.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Candlestick interval (e.g., '5m', '30m', '4h')
            start_time: Start datetime
            end_time: End datetime (default: now)
            limit: Max candles per request (max 1500)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if end_time is None:
            end_time = datetime.utcnow()

        interval = self.TIMEFRAME_MAP.get(timeframe, timeframe)
        session = await self._get_session()

        all_data = []
        current_start = start_time
        batch_num = 0
        total_days = (end_time - start_time).days

        logger.info(f"Starting fetch: {symbol} {timeframe} from {start_time} to {end_time} ({total_days} days)")

        while current_start < end_time:
            batch_num += 1
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": int(current_start.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000),
                "limit": limit
            }

            url = f"{self.base_url}/fapi/v1/klines"
            logger.debug(f"Batch {batch_num}: Requesting {url} for {current_start.strftime('%Y-%m-%d %H:%M')}")

            try:
                async with session.get(url, params=params) as response:
                    logger.debug(f"Batch {batch_num}: Response status {response.status}")

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Binance API error (status {response.status}): {error_text}")
                        break

                    data = await response.json()

                    if not data:
                        logger.info(f"Batch {batch_num}: No more data available")
                        break

                    all_data.extend(data)

                    # Update start time for next batch
                    last_timestamp = data[-1][0]
                    new_start = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)

                    # Progress logging every 10 batches
                    if batch_num % 10 == 0:
                        progress = min((current_start - start_time).days / max(total_days, 1) * 100, 100)
                        logger.info(f"Batch {batch_num}: Fetched {len(data)} candles, total {len(all_data)}, progress ~{progress:.1f}%")

                    # Check if we've reached the end (got fewer candles than limit, or not advancing)
                    if len(data) < limit:
                        logger.info(f"Batch {batch_num}: Received {len(data)} < {limit} candles, reached end of data")
                        break

                    # Check if timestamp is not advancing (stuck in loop)
                    if new_start <= current_start:
                        logger.warning(f"Batch {batch_num}: Timestamp not advancing, breaking loop")
                        break

                    current_start = new_start

                    # Rate limiting
                    await asyncio.sleep(0.1)

            except asyncio.TimeoutError:
                logger.error(f"Batch {batch_num}: Request timed out, retrying...")
                await asyncio.sleep(1)
                continue
            except aiohttp.ClientError as e:
                logger.error(f"Batch {batch_num}: Network error: {e}")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                logger.error(f"Batch {batch_num}: Unexpected error: {type(e).__name__}: {e}")
                break

        logger.info(f"Fetch complete: {len(all_data)} total candles in {batch_num} batches")

        if not all_data:
            return pd.DataFrame()

        # Parse response
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        # Select and sort - keep quote_volume for Kronos (as 'amount')
        df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]
        df = df.drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        return df

    async def fetch_funding_rate(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch funding rate history from Binance Futures.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_time: Start datetime
            end_time: End datetime (default: now)
            limit: Max records per request

        Returns:
            DataFrame with columns: timestamp, funding_rate
        """
        if end_time is None:
            end_time = datetime.utcnow()

        session = await self._get_session()
        all_data = []
        current_start = start_time
        batch_num = 0
        total_days = (end_time - start_time).days

        logger.info(f"Starting funding rate fetch: {symbol} from {start_time} to {end_time} ({total_days} days)")

        while current_start < end_time:
            batch_num += 1
            params = {
                "symbol": symbol,
                "startTime": int(current_start.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000),
                "limit": limit
            }

            url = f"{self.base_url}/fapi/v1/fundingRate"
            logger.debug(f"Funding batch {batch_num}: Requesting for {current_start.strftime('%Y-%m-%d %H:%M')}")

            try:
                async with session.get(url, params=params) as response:
                    logger.debug(f"Funding batch {batch_num}: Response status {response.status}")

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Binance API error (status {response.status}): {error_text}")
                        break

                    data = await response.json()

                    if not data:
                        logger.info(f"Funding batch {batch_num}: No more data available")
                        break

                    all_data.extend(data)

                    # Update start time for next batch
                    last_timestamp = data[-1]["fundingTime"]
                    new_start = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)

                    # Progress logging every 5 batches
                    if batch_num % 5 == 0:
                        progress = min((current_start - start_time).days / max(total_days, 1) * 100, 100)
                        logger.info(f"Funding batch {batch_num}: Fetched {len(data)} records, total {len(all_data)}, progress ~{progress:.1f}%")

                    # Check if we've reached the end (got fewer records than limit)
                    if len(data) < limit:
                        logger.info(f"Funding batch {batch_num}: Received {len(data)} < {limit} records, reached end of data")
                        break

                    # Check if timestamp is not advancing (stuck in loop)
                    if new_start <= current_start:
                        logger.warning(f"Funding batch {batch_num}: Timestamp not advancing, breaking loop")
                        break

                    current_start = new_start
                    await asyncio.sleep(0.1)

            except asyncio.TimeoutError:
                logger.error(f"Funding batch {batch_num}: Request timed out, retrying...")
                await asyncio.sleep(1)
                continue
            except aiohttp.ClientError as e:
                logger.error(f"Funding batch {batch_num}: Network error: {e}")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                logger.error(f"Funding batch {batch_num}: Unexpected error: {type(e).__name__}: {e}")
                break

        logger.info(f"Funding fetch complete: {len(all_data)} total records in {batch_num} batches")

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df[["timestamp", "funding_rate"]]
        df = df.drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Fetched {len(df)} unique funding rates for {symbol}")
        return df

    async def fetch_open_interest(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch open interest history."""
        if end_time is None:
            end_time = datetime.utcnow()

        session = await self._get_session()
        interval = self.TIMEFRAME_MAP.get(timeframe, timeframe)

        params = {
            "symbol": symbol,
            "period": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": 500
        }

        try:
            async with session.get(
                f"{self.base_url}/futures/data/openInterestHist",
                params=params
            ) as response:
                if response.status != 200:
                    return pd.DataFrame()

                data = await response.json()

                if not data:
                    return pd.DataFrame()

                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["open_interest"] = df["sumOpenInterest"].astype(float)
                df = df[["timestamp", "open_interest"]]

                return df

        except Exception as e:
            logger.error(f"Error fetching open interest: {e}")
            return pd.DataFrame()


class HyperliquidCollector(DataCollector):
    """
    Hyperliquid data collector.
    Used for live trading and real-time data.
    """

    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://api.hyperliquid.xyz"
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 5000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Hyperliquid.

        Args:
            symbol: Asset name (e.g., 'BTC', 'ETH')
            timeframe: Candlestick interval
            start_time: Start datetime
            end_time: End datetime (default: now)
            limit: Max candles per request

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if end_time is None:
            end_time = datetime.utcnow()

        session = await self._get_session()

        # Convert symbol format (BTCUSDT -> BTC)
        asset = symbol.replace("USDT", "").replace("USD", "")

        # Hyperliquid uses a different API structure
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": asset,
                "interval": timeframe,
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000)
            }
        }

        try:
            async with session.post(
                f"{self.base_url}/info",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Hyperliquid API error: {error_text}")
                    return pd.DataFrame()

                data = await response.json()

                if not data:
                    return pd.DataFrame()

                # Parse Hyperliquid response format
                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                df["open"] = df["o"].astype(float)
                df["high"] = df["h"].astype(float)
                df["low"] = df["l"].astype(float)
                df["close"] = df["c"].astype(float)
                df["volume"] = df["v"].astype(float)

                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                df = df.sort_values("timestamp").reset_index(drop=True)

                logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe} from Hyperliquid")
                return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV from Hyperliquid: {e}")
            return pd.DataFrame()

    async def fetch_funding_rate(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch funding rate from Hyperliquid.

        Returns:
            DataFrame with columns: timestamp, funding_rate
        """
        session = await self._get_session()
        asset = symbol.replace("USDT", "").replace("USD", "")

        payload = {
            "type": "fundingHistory",
            "req": {
                "coin": asset,
                "startTime": int(start_time.timestamp() * 1000)
            }
        }

        try:
            async with session.post(
                f"{self.base_url}/info",
                json=payload
            ) as response:
                if response.status != 200:
                    return pd.DataFrame()

                data = await response.json()

                if not data:
                    return pd.DataFrame()

                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
                df["funding_rate"] = df["fundingRate"].astype(float)
                df = df[["timestamp", "funding_rate"]]

                return df

        except Exception as e:
            logger.error(f"Error fetching funding rate from Hyperliquid: {e}")
            return pd.DataFrame()

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current mid price for a symbol."""
        session = await self._get_session()
        asset = symbol.replace("USDT", "").replace("USD", "")

        payload = {"type": "allMids"}

        try:
            async with session.post(
                f"{self.base_url}/info",
                json=payload
            ) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                return float(data.get(asset, 0))

        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return None


def _check_existing_data(file_path: Path, end_time: datetime) -> Tuple[Optional[pd.DataFrame], Optional[datetime]]:
    """
    Check if data file exists and determine resume point.

    Returns:
        Tuple of (existing_df or None, resume_start_time or None)
        If file is complete (last timestamp near end_time), returns (df, None) to skip
        If file needs update, returns (df, resume_time)
        If no file exists, returns (None, None)
    """
    if not file_path.exists():
        return None, None

    try:
        existing_df = pd.read_parquet(file_path)
        if existing_df.empty:
            return None, None

        last_timestamp = existing_df["timestamp"].max()
        if isinstance(last_timestamp, pd.Timestamp):
            last_timestamp = last_timestamp.to_pydatetime()

        # If last timestamp is within 1 day of end_time, consider it complete
        if (end_time - last_timestamp).days <= 1:
            return existing_df, None  # Complete, skip

        # Resume from last timestamp
        return existing_df, last_timestamp + timedelta(milliseconds=1)

    except Exception as e:
        logger.warning(f"Could not read existing file {file_path}: {e}")
        return None, None


async def collect_historical_data(
    symbols: List[str],
    timeframes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw"
) -> Dict[str, pd.DataFrame]:
    """
    Collect historical data for multiple symbols and timeframes.
    Resumes from existing data if files already exist.

    Args:
        symbols: List of symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
        timeframes: List of timeframes (e.g., ['5m', '30m', '4h'])
        start_date: Start date string (e.g., '2023-01-01')
        end_date: End date string (e.g., '2024-12-31')
        output_dir: Directory to save data

    Returns:
        Dictionary of DataFrames by symbol_timeframe key
    """
    import os

    logger.info("=" * 50)
    logger.info("Starting data collection (with resume support)")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 50)

    # Use API key if available for higher rate limits
    api_key = os.getenv("BINANCE_API_KEY", "")
    collector = BinanceCollector(api_key=api_key)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = datetime.strptime(start_date, "%Y-%m-%d")
    end_time = datetime.strptime(end_date, "%Y-%m-%d")

    results = {}
    total_tasks = len(symbols) * (len(timeframes) + 1)  # +1 for funding
    completed = 0
    skipped = 0

    try:
        for symbol_idx, symbol in enumerate(symbols):
            logger.info(f"\n>>> Processing symbol {symbol_idx + 1}/{len(symbols)}: {symbol}")

            for tf_idx, timeframe in enumerate(timeframes):
                completed += 1
                key = f"{symbol}_{timeframe}"
                file_path = output_path / f"{key}.parquet"

                # Check for existing data
                existing_df, resume_time = _check_existing_data(file_path, end_time)

                if existing_df is not None and resume_time is None:
                    # File is complete, skip
                    logger.info(f"[{completed}/{total_tasks}] SKIP {key} - already complete ({len(existing_df)} rows)")
                    results[key] = existing_df
                    skipped += 1
                    continue

                if resume_time is not None:
                    logger.info(f"[{completed}/{total_tasks}] RESUME {key} from {resume_time.strftime('%Y-%m-%d %H:%M')}...")
                    fetch_start = resume_time
                else:
                    logger.info(f"[{completed}/{total_tasks}] Collecting {symbol} {timeframe}...")
                    fetch_start = start_time

                # Fetch OHLCV
                ohlcv = await collector.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=fetch_start,
                    end_time=end_time
                )

                if not ohlcv.empty:
                    # Merge with existing data if resuming
                    if existing_df is not None:
                        ohlcv = pd.concat([existing_df, ohlcv], ignore_index=True)
                        ohlcv = ohlcv.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                        logger.info(f"Merged with existing data: {len(existing_df)} + {len(ohlcv) - len(existing_df)} = {len(ohlcv)} rows")

                    results[key] = ohlcv

                    # Save to parquet
                    ohlcv.to_parquet(file_path, index=False)
                    logger.info(f"SUCCESS: Saved {len(ohlcv)} rows to {file_path}")
                else:
                    if existing_df is not None:
                        results[key] = existing_df
                        logger.info(f"No new data, keeping existing {len(existing_df)} rows")
                    else:
                        logger.warning(f"WARNING: No data returned for {symbol} {timeframe}")

                await asyncio.sleep(0.5)  # Rate limiting

            # Fetch funding rate (only once per symbol)
            completed += 1
            key = f"{symbol}_funding"
            file_path = output_path / f"{key}.parquet"

            # Check for existing funding data
            existing_df, resume_time = _check_existing_data(file_path, end_time)

            if existing_df is not None and resume_time is None:
                logger.info(f"[{completed}/{total_tasks}] SKIP {key} - already complete ({len(existing_df)} rows)")
                results[key] = existing_df
                skipped += 1
                continue

            if resume_time is not None:
                logger.info(f"[{completed}/{total_tasks}] RESUME {key} from {resume_time.strftime('%Y-%m-%d %H:%M')}...")
                fetch_start = resume_time
            else:
                logger.info(f"[{completed}/{total_tasks}] Collecting {symbol} funding rates...")
                fetch_start = start_time

            funding = await collector.fetch_funding_rate(
                symbol=symbol,
                start_time=fetch_start,
                end_time=end_time
            )

            if not funding.empty:
                # Merge with existing data if resuming
                if existing_df is not None:
                    funding = pd.concat([existing_df, funding], ignore_index=True)
                    funding = funding.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                    logger.info(f"Merged with existing data: {len(existing_df)} + {len(funding) - len(existing_df)} = {len(funding)} rows")

                results[key] = funding

                funding.to_parquet(file_path, index=False)
                logger.info(f"SUCCESS: Saved {len(funding)} funding rates to {file_path}")
            else:
                if existing_df is not None:
                    results[key] = existing_df
                    logger.info(f"No new data, keeping existing {len(existing_df)} rows")
                else:
                    logger.warning(f"WARNING: No funding data returned for {symbol}")

        logger.info(f"\nSkipped {skipped} files that were already complete")

    except Exception as e:
        logger.error(f"FATAL ERROR during collection: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("\nClosing collector session...")
        await collector.close()
        logger.info("Session closed")

    logger.info("\n" + "=" * 50)
    logger.info(f"Collection complete. {len(results)} datasets saved.")
    logger.info("=" * 50)

    return results


def run_collection(
    symbols: List[str],
    timeframes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw"
):
    """Synchronous wrapper for data collection."""
    return asyncio.run(
        collect_historical_data(symbols, timeframes, start_date, end_date, output_dir)
    )
