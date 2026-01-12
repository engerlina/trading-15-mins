"""RSI Divergence strategy for live trading."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal."""

    signal_type: str  # 'bullish_divergence', 'bearish_divergence', 'no_signal'
    symbol: str
    price: float
    rsi: float
    timestamp: datetime
    confidence: float = 0.0
    pivot_price: float = 0.0
    pivot_rsi: float = 0.0

    @property
    def is_entry(self) -> bool:
        return self.signal_type in ['bullish_divergence', 'bearish_divergence']

    @property
    def side(self) -> Optional[str]:
        if self.signal_type == 'bullish_divergence':
            return 'long'
        elif self.signal_type == 'bearish_divergence':
            return 'short'
        return None


@dataclass
class ExitSignal:
    """Exit signal for managing positions."""

    should_exit: bool
    exit_reason: str
    current_price: float
    pnl_pct: float
    bars_held: int


class RSIDivergenceStrategy:
    """
    RSI Divergence Strategy for live trading.

    Detects bullish and bearish divergences between price and RSI.
    Adapted from the backtesting logic in scripts/backtest_short_tf.py.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        pivot_bars: int = 5,
        min_divergence_bars: int = 10,
        max_divergence_bars: int = 50,
        rsi_threshold: float = 40.0,
        profit_target_pct: float = 0.015,
        stop_loss_pct: float = 0.01,
        trailing_stop_pct: float = 0.10,
        max_holding_bars: int = 96,
        long_only: bool = True,
        use_trailing_stop: bool = True,
    ):
        """
        Initialize strategy.

        Args:
            rsi_period: RSI calculation period
            pivot_bars: Bars to look left/right for pivot detection
            min_divergence_bars: Minimum bars between pivots for divergence
            max_divergence_bars: Maximum bars between pivots for divergence
            rsi_threshold: RSI must be below this for bullish divergence
            profit_target_pct: Fixed profit target (if not using trailing stop)
            stop_loss_pct: Stop loss percentage
            trailing_stop_pct: Trailing stop percentage from peak
            max_holding_bars: Maximum bars to hold position
            long_only: Only take long trades
            use_trailing_stop: Use trailing stop instead of fixed target
        """
        self.rsi_period = rsi_period
        self.pivot_bars = pivot_bars
        self.min_divergence_bars = min_divergence_bars
        self.max_divergence_bars = max_divergence_bars
        self.rsi_threshold = rsi_threshold
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_holding_bars = max_holding_bars
        self.long_only = long_only
        self.use_trailing_stop = use_trailing_stop

        # State for position management
        self._entry_price: Optional[float] = None
        self._entry_bar: Optional[int] = None
        self._best_price: Optional[float] = None
        self._position_side: Optional[str] = None

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def detect_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect pivot highs and lows."""
        df = df.copy()
        df['pivot_high'] = False
        df['pivot_low'] = False

        for i in range(self.pivot_bars, len(df) - self.pivot_bars):
            # Check for pivot high
            high_val = df.iloc[i]['high']
            is_pivot_high = all(
                high_val >= df.iloc[i-j]['high'] for j in range(1, self.pivot_bars+1)
            ) and all(
                high_val >= df.iloc[i+j]['high'] for j in range(1, self.pivot_bars+1)
            )
            if is_pivot_high:
                df.iloc[i, df.columns.get_loc('pivot_high')] = True

            # Check for pivot low
            low_val = df.iloc[i]['low']
            is_pivot_low = all(
                low_val <= df.iloc[i-j]['low'] for j in range(1, self.pivot_bars+1)
            ) and all(
                low_val <= df.iloc[i+j]['low'] for j in range(1, self.pivot_bars+1)
            )
            if is_pivot_low:
                df.iloc[i, df.columns.get_loc('pivot_low')] = True

        return df

    def check_entry_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """
        Check for entry signals based on RSI divergence.

        Args:
            df: DataFrame with OHLCV data and 'timestamp' column
            symbol: Trading symbol

        Returns:
            Signal object
        """
        if len(df) < self.rsi_period + self.max_divergence_bars:
            return Signal(
                signal_type='no_signal',
                symbol=symbol,
                price=df.iloc[-1]['close'],
                rsi=0.0,
                timestamp=df.iloc[-1]['timestamp'],
            )

        # Calculate RSI
        df = df.copy()
        df['rsi'] = self.calculate_rsi(df['close'])

        # Detect pivots
        df = self.detect_pivots(df)

        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        current_time = df.iloc[-1]['timestamp']

        if pd.isna(current_rsi):
            return Signal(
                signal_type='no_signal',
                symbol=symbol,
                price=current_price,
                rsi=0.0,
                timestamp=current_time,
            )

        # Collect recent pivots
        recent_pivot_lows: List[Tuple[int, float, float]] = []  # (idx, price, rsi)
        recent_pivot_highs: List[Tuple[int, float, float]] = []

        for i in range(len(df) - self.max_divergence_bars, len(df)):
            if i < 0:
                continue
            if df.iloc[i]['pivot_low']:
                rsi_val = df.iloc[i]['rsi']
                if not pd.isna(rsi_val):
                    recent_pivot_lows.append((i, df.iloc[i]['low'], rsi_val))
            if df.iloc[i]['pivot_high']:
                rsi_val = df.iloc[i]['rsi']
                if not pd.isna(rsi_val):
                    recent_pivot_highs.append((i, df.iloc[i]['high'], rsi_val))

        # Check for bullish divergence (price lower low, RSI higher low)
        if len(recent_pivot_lows) >= 2:
            for j in range(len(recent_pivot_lows) - 1):
                prev_idx, prev_price, prev_rsi = recent_pivot_lows[j]
                curr_idx, curr_price, curr_rsi = recent_pivot_lows[-1]

                bars_between = curr_idx - prev_idx
                if self.min_divergence_bars <= bars_between <= self.max_divergence_bars:
                    # Bullish divergence: price made lower low, RSI made higher low
                    if curr_price < prev_price and curr_rsi > prev_rsi and curr_rsi < self.rsi_threshold:
                        confidence = min(1.0, (prev_rsi - curr_rsi) / 10 + 0.5)
                        logger.info(
                            f"Bullish divergence detected: price {prev_price:.2f} -> {curr_price:.2f}, "
                            f"RSI {prev_rsi:.2f} -> {curr_rsi:.2f}"
                        )
                        return Signal(
                            signal_type='bullish_divergence',
                            symbol=symbol,
                            price=current_price,
                            rsi=current_rsi,
                            timestamp=current_time,
                            confidence=confidence,
                            pivot_price=curr_price,
                            pivot_rsi=curr_rsi,
                        )

        # Check for bearish divergence (price higher high, RSI lower high)
        if not self.long_only and len(recent_pivot_highs) >= 2:
            for j in range(len(recent_pivot_highs) - 1):
                prev_idx, prev_price, prev_rsi = recent_pivot_highs[j]
                curr_idx, curr_price, curr_rsi = recent_pivot_highs[-1]

                bars_between = curr_idx - prev_idx
                if self.min_divergence_bars <= bars_between <= self.max_divergence_bars:
                    # Bearish divergence: price made higher high, RSI made lower high
                    if curr_price > prev_price and curr_rsi < prev_rsi and curr_rsi > (100 - self.rsi_threshold):
                        confidence = min(1.0, (curr_rsi - prev_rsi) / 10 + 0.5)
                        logger.info(
                            f"Bearish divergence detected: price {prev_price:.2f} -> {curr_price:.2f}, "
                            f"RSI {prev_rsi:.2f} -> {curr_rsi:.2f}"
                        )
                        return Signal(
                            signal_type='bearish_divergence',
                            symbol=symbol,
                            price=current_price,
                            rsi=current_rsi,
                            timestamp=current_time,
                            confidence=confidence,
                            pivot_price=curr_price,
                            pivot_rsi=curr_rsi,
                        )

        return Signal(
            signal_type='no_signal',
            symbol=symbol,
            price=current_price,
            rsi=current_rsi,
            timestamp=current_time,
        )

    def enter_position(self, price: float, side: str, bar_index: int) -> None:
        """Record position entry for exit management."""
        self._entry_price = price
        self._entry_bar = bar_index
        self._best_price = price
        self._position_side = side
        logger.info(f"Position entered: {side} at {price:.2f}")

    def check_exit_signal(
        self,
        current_price: float,
        current_bar: int,
    ) -> ExitSignal:
        """
        Check if position should be exited.

        Args:
            current_price: Current market price
            current_bar: Current bar index

        Returns:
            ExitSignal object
        """
        if self._entry_price is None or self._entry_bar is None:
            return ExitSignal(
                should_exit=False,
                exit_reason='no_position',
                current_price=current_price,
                pnl_pct=0.0,
                bars_held=0,
            )

        bars_held = current_bar - self._entry_bar

        # Calculate PnL based on position side
        if self._position_side == 'long':
            pnl_pct = (current_price - self._entry_price) / self._entry_price
            # Update best price for trailing stop
            if current_price > self._best_price:
                self._best_price = current_price
        else:  # short
            pnl_pct = (self._entry_price - current_price) / self._entry_price
            # Update best price for trailing stop (lowest for shorts)
            if current_price < self._best_price:
                self._best_price = current_price

        exit_reason = None

        if self.use_trailing_stop:
            # Trailing stop logic
            if self._position_side == 'long':
                trailing_stop_price = self._best_price * (1 - self.trailing_stop_pct)
                if current_price <= trailing_stop_price and pnl_pct > 0:
                    exit_reason = 'trailing_stop'
            else:  # short
                trailing_stop_price = self._best_price * (1 + self.trailing_stop_pct)
                if current_price >= trailing_stop_price and pnl_pct > 0:
                    exit_reason = 'trailing_stop'

            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                exit_reason = 'stop_loss'

            # Max holding time
            if bars_held >= self.max_holding_bars:
                exit_reason = 'max_time'
        else:
            # Fixed target logic
            if pnl_pct >= self.profit_target_pct:
                exit_reason = 'profit_target'
            elif pnl_pct <= -self.stop_loss_pct:
                exit_reason = 'stop_loss'
            elif bars_held >= self.max_holding_bars:
                exit_reason = 'max_time'

        return ExitSignal(
            should_exit=exit_reason is not None,
            exit_reason=exit_reason or 'none',
            current_price=current_price,
            pnl_pct=pnl_pct,
            bars_held=bars_held,
        )

    def exit_position(self) -> None:
        """Clear position state after exit."""
        self._entry_price = None
        self._entry_bar = None
        self._best_price = None
        self._position_side = None
        logger.info("Position exited")

    @property
    def has_position(self) -> bool:
        """Check if strategy is tracking a position."""
        return self._entry_price is not None

    @property
    def position_info(self) -> dict:
        """Get current position info."""
        return {
            "entry_price": self._entry_price,
            "entry_bar": self._entry_bar,
            "best_price": self._best_price,
            "side": self._position_side,
        }
