"""Real-time data feed for trading engine."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import httpx
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """OHLCV candle data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class DataFeed:
    """
    Real-time data feed from Hyperliquid.

    Fetches OHLCV candles and maintains a rolling buffer for strategy calculations.
    """

    # Hyperliquid API endpoints
    MAINNET_URL = "https://api.hyperliquid.xyz/info"
    TESTNET_URL = "https://api.hyperliquid-testnet.xyz/info"

    # Timeframe mapping (Hyperliquid uses specific interval strings)
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
        symbol: str = "BTC",
        timeframe: str = "15m",
        buffer_size: int = 200,  # Keep 200 candles for RSI calculation
        network: str = "mainnet",
    ):
        """
        Initialize data feed.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH", "SOL")
            timeframe: Candle timeframe
            buffer_size: Number of candles to keep in buffer
            network: "mainnet" or "testnet"
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.buffer_size = buffer_size
        self.network = network

        self.base_url = self.MAINNET_URL if network == "mainnet" else self.TESTNET_URL

        # Data buffer
        self._candles: List[Candle] = []
        self._last_update: Optional[datetime] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        """Start the data feed."""
        self._client = httpx.AsyncClient(timeout=30.0)
        await self._fetch_initial_data()
        logger.info(f"DataFeed started: {self.symbol} {self.timeframe}")

    async def stop(self) -> None:
        """Stop the data feed."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("DataFeed stopped")

    async def _fetch_initial_data(self) -> None:
        """Fetch initial historical data to fill the buffer."""
        try:
            candles = await self._fetch_candles(limit=self.buffer_size)
            self._candles = candles
            if candles:
                self._last_update = candles[-1].timestamp
            logger.info(f"Loaded {len(self._candles)} initial candles")
        except Exception as e:
            logger.error(f"Error fetching initial data: {e}")
            raise

    async def _fetch_candles(
        self,
        limit: int = 100,
        start_time: Optional[int] = None,
    ) -> List[Candle]:
        """
        Fetch candles from Hyperliquid API.

        Args:
            limit: Number of candles to fetch
            start_time: Start timestamp in milliseconds

        Returns:
            List of Candle objects
        """
        if not self._client:
            raise RuntimeError("DataFeed not started")

        # Hyperliquid API expects specific format
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": self.symbol,
                "interval": self.TIMEFRAME_MAP.get(self.timeframe, "15m"),
                "startTime": start_time or int((datetime.utcnow() - timedelta(days=7)).timestamp() * 1000),
                "endTime": int(datetime.utcnow().timestamp() * 1000),
            }
        }

        try:
            response = await self._client.post(self.base_url, json=payload)
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return []

            data = response.json()

            candles = []
            for c in data[-limit:]:  # Take last `limit` candles
                candle = Candle(
                    timestamp=datetime.fromtimestamp(c["t"] / 1000),
                    open=float(c["o"]),
                    high=float(c["h"]),
                    low=float(c["l"]),
                    close=float(c["c"]),
                    volume=float(c["v"]),
                )
                candles.append(candle)

            return candles

        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return []

    async def update(self) -> Optional[Candle]:
        """
        Update data feed with latest candle.

        Returns:
            New candle if available, None otherwise
        """
        try:
            # Fetch recent candles
            recent = await self._fetch_candles(limit=5)

            if not recent:
                return None

            new_candle = None

            for candle in recent:
                # Check if this is a new candle
                if not self._candles or candle.timestamp > self._candles[-1].timestamp:
                    self._candles.append(candle)
                    new_candle = candle
                elif candle.timestamp == self._candles[-1].timestamp:
                    # Update the latest candle (still forming)
                    self._candles[-1] = candle

            # Trim buffer to size
            if len(self._candles) > self.buffer_size:
                self._candles = self._candles[-self.buffer_size:]

            self._last_update = datetime.utcnow()
            return new_candle

        except Exception as e:
            logger.error(f"Error updating data feed: {e}")
            return None

    def get_dataframe(self) -> pd.DataFrame:
        """Get candle data as DataFrame."""
        if not self._candles:
            return pd.DataFrame()

        data = [c.to_dict() for c in self._candles]
        df = pd.DataFrame(data)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_latest_price(self) -> Optional[float]:
        """Get the latest close price."""
        if self._candles:
            return self._candles[-1].close
        return None

    def get_latest_candle(self) -> Optional[Candle]:
        """Get the latest candle."""
        if self._candles:
            return self._candles[-1]
        return None

    @property
    def is_ready(self) -> bool:
        """Check if data feed has enough data for strategy."""
        return len(self._candles) >= 50  # Minimum for RSI calculation

    @property
    def candle_count(self) -> int:
        """Get number of candles in buffer."""
        return len(self._candles)


class MultiSymbolDataFeed:
    """Data feed for multiple symbols."""

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        buffer_size: int = 200,
        network: str = "mainnet",
    ):
        self.feeds: Dict[str, DataFeed] = {}
        for symbol in symbols:
            self.feeds[symbol] = DataFeed(
                symbol=symbol,
                timeframe=timeframe,
                buffer_size=buffer_size,
                network=network,
            )

    async def start(self) -> None:
        """Start all feeds."""
        await asyncio.gather(*[feed.start() for feed in self.feeds.values()])

    async def stop(self) -> None:
        """Stop all feeds."""
        await asyncio.gather(*[feed.stop() for feed in self.feeds.values()])

    async def update_all(self) -> Dict[str, Optional[Candle]]:
        """Update all feeds."""
        results = {}
        for symbol, feed in self.feeds.items():
            results[symbol] = await feed.update()
        return results

    def get_feed(self, symbol: str) -> Optional[DataFeed]:
        """Get feed for specific symbol."""
        return self.feeds.get(symbol)
