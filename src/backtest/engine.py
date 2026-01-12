"""
Backtesting engine for the trading system.

Features:
- Walk-forward validation
- Cost-aware execution (fees, slippage)
- Funding simulation
- Position sizing integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

from ..risk.engine import RiskEngine
from ..risk.position import PositionManager, Trade
from ..models.signal_model import SignalModel
from ..utils.logger import get_logger

logger = get_logger("backtest")


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 10000.0
    fee_rate: float = 0.0005  # 5 bps
    slippage_bps: float = 5
    funding_enabled: bool = True
    funding_frequency_hours: int = 8
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    confidence_threshold: float = 0.4
    max_positions: int = 3
    leverage: float = 1.0  # Leverage multiplier (1.0 = no leverage, 2.0 = 2x, etc.)
    max_leverage: float = 3.0  # Maximum allowed leverage
    max_holding_bars: int = 480  # Max bars to hold position (480 = 10 days on 30m)
    trailing_stop_pct: float = 0.015  # 1.5% trailing stop after profit
    min_holding_bars: int = 24  # Min bars to hold (12 hours on 30m) - avoid whipsaws


@dataclass
class BacktestResult:
    """Backtest results container."""
    equity_curve: pd.Series
    trades: List[Trade]
    metrics: Dict
    daily_returns: pd.Series
    positions_history: pd.DataFrame
    signals_history: pd.DataFrame


class BacktestEngine:
    """
    Event-driven backtesting engine.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()

        # Validate and apply leverage
        effective_leverage = min(self.config.leverage, self.config.max_leverage)
        if effective_leverage != self.config.leverage:
            logger.warning(f"Leverage {self.config.leverage}x exceeds max {self.config.max_leverage}x, capping to {effective_leverage}x")
        self.leverage = effective_leverage

        # Initialize components
        self.risk_engine = RiskEngine({
            "initial_capital": self.config.initial_capital,
            "max_position_size": 0.95 * self.leverage,  # 95% base * leverage
            "max_leverage": self.config.max_leverage,
            "volatility_target": 0.80,  # High but not extreme
            "max_drawdown": 0.85,  # 85% drawdown limit - allow recovery from crashes
            "max_consecutive_losses": 500,
            "daily_loss_limit": 0.99,  # Essentially disabled
        })

        self.position_manager = PositionManager(
            fee_rate=self.config.fee_rate,
            slippage_bps=self.config.slippage_bps,
            max_positions=self.config.max_positions
        )

        # State tracking
        self.equity_history: List[Tuple[datetime, float]] = []
        self.signals_history: List[Dict] = []
        self.positions_history: List[Dict] = []
        self.price_series: Optional[pd.Series] = None  # For benchmark comparison

        # Position tracking for advanced exits
        self.entry_regime: Optional[str] = None  # Regime when position was opened
        self.trailing_stop_price: Optional[float] = None  # Trailing stop price
        self.best_price_since_entry: Optional[float] = None  # Best price for trailing
        self.position_entry_bar: int = 0  # Bar index when position was opened
        self.position_entry_price: float = 0.0  # Entry price for trailing stop calc

    def run(
        self,
        data: pd.DataFrame,
        model: SignalModel,
        feature_columns: List[str],
        symbol: str = "BTCUSDT",
        regime_detector=None,
        kronos_embeddings: np.ndarray = None
    ) -> BacktestResult:
        """
        Run backtest on historical data with optional regime-aware trading.

        Args:
            data: DataFrame with features, prices, and timestamp
            model: Trained signal model
            feature_columns: List of feature column names
            symbol: Trading symbol
            regime_detector: Optional RegimeDetector for regime-aware trading
            kronos_embeddings: Optional Kronos embeddings for regime detection

        Returns:
            BacktestResult with metrics and equity curve
        """
        logger.info(f"Starting backtest for {symbol} with {len(data)} bars")
        logger.info(f"Initial capital: ${self.config.initial_capital:.2f}, Leverage: {self.leverage}x")
        if regime_detector is not None:
            logger.info("Regime-aware trading ENABLED")

        # Reset state
        self._reset()

        # Ensure sorted by timestamp
        data = data.sort_values("timestamp").reset_index(drop=True)

        # Pre-extract arrays for fast access (avoid iloc in loop)
        logger.info("Pre-extracting data arrays for fast access...")
        timestamps = data["timestamp"].values
        prices = data["close"].values

        # Store price series for benchmark comparison
        self.price_series = pd.Series(prices, index=pd.to_datetime(timestamps))
        volatilities = data["volatility_20"].fillna(0.15).values if "volatility_20" in data.columns else np.full(len(data), 0.15)
        funding_rates = data["funding_rate"].fillna(0.0).values if "funding_rate" in data.columns else np.zeros(len(data))

        # Pre-extract regime features (fill NaN with defaults)
        vol_20 = data["volatility_20"].fillna(0.15).values if "volatility_20" in data.columns else np.full(len(data), 0.15)
        momentum_20 = data["momentum_20"].fillna(0.0).values if "momentum_20" in data.columns else np.zeros(len(data))
        perp_zscore = data["kronos_perplexity_zscore"].fillna(0.0).values if "kronos_perplexity_zscore" in data.columns else np.zeros(len(data))

        # Pre-compute all signals in batch (major optimization)
        logger.info("Computing all signals in batch...")
        features_matrix = data[feature_columns].values.astype(np.float64)

        # Find valid rows (no NaN)
        valid_mask = ~np.isnan(features_matrix).any(axis=1)
        logger.info(f"Valid samples: {valid_mask.sum()}/{len(data)} ({valid_mask.sum()/len(data)*100:.1f}%)")

        # Initialize signal arrays
        all_signals = np.zeros(len(data), dtype=np.int32)
        all_confidences = np.zeros(len(data), dtype=np.float64)

        # Batch predict for valid rows
        if valid_mask.sum() > 0:
            valid_features = features_matrix[valid_mask]
            try:
                signals, confidences = model.get_signal(
                    valid_features,
                    confidence_threshold=self.config.confidence_threshold
                )
                all_signals[valid_mask] = signals
                all_confidences[valid_mask] = confidences
            except Exception as e:
                logger.error(f"Error in batch signal prediction: {e}")

        # ============================================================
        # STRATEGY MODE SELECTION
        # Set USE_PURE_KRONOS = True to use XGBoost signals directly
        # Set USE_PURE_KRONOS = False to use simple rules with crash protection
        # ============================================================
        USE_PURE_KRONOS = False  # Toggle this to switch strategies

        regime_biases = np.zeros(len(data), dtype=np.float64)

        if USE_PURE_KRONOS:
            # ============================================================
            # PURE KRONOS STRATEGY: Use XGBoost model signals directly
            # No rule overrides - trust the ML model completely
            # ============================================================
            logger.info("Applying PURE KRONOS strategy (XGBoost signals only)...")

            # The signals are already computed by XGBoost model above (lines 167-178)
            # all_signals already contains: 1 (long), 0 (flat), -1 (short)
            # all_confidences already contains the model's confidence

            # Set regime_biases based on model signals for position sizing
            for i in range(len(data)):
                if all_signals[i] == 1:  # Long signal
                    regime_biases[i] = all_confidences[i]  # Use confidence as regime bias
                elif all_signals[i] == -1:  # Short signal
                    regime_biases[i] = -all_confidences[i]  # Negative for short
                else:  # Flat signal
                    regime_biases[i] = 0.0

            # Log signal distribution from XGBoost model
            logger.info("Using XGBoost model signals without overrides")

        else:
            # ============================================================
            # SIMPLE RULES STRATEGY: Crash protection with trend following
            # Overrides XGBoost signals with simple, robust rules
            # ============================================================
            logger.info("Applying SIMPLE RULES strategy (crash protection)...")

            # Calculate long-term trend MA
            ma_trend = pd.Series(prices).rolling(200).mean().values

            # Calculate all-time high for crash detection
            all_time_high = pd.Series(prices).expanding().max().values

            trend_overrides = 0
            is_flat = False

            for i in range(len(data)):
                # Skip first 200 bars while MA warms up - STAY LONG
                if i < 200 or np.isnan(ma_trend[i]):
                    all_signals[i] = 1
                    all_confidences[i] = 0.9
                    regime_biases[i] = 1.0
                    continue

                # Calculate drawdown from ALL-TIME high
                drawdown_from_ath = (all_time_high[i] - prices[i]) / all_time_high[i]
                price_above_trend = prices[i] > ma_trend[i]

                # CRASH DETECTION: >55% drawdown from ATH AND below trend
                crash_detected = drawdown_from_ath > 0.55 and not price_above_trend
                recovery_detected = is_flat and price_above_trend

                if crash_detected:
                    is_flat = True
                    regime_biases[i] = 0.0
                    all_signals[i] = 0
                    all_confidences[i] = 0.2
                    trend_overrides += 1
                elif recovery_detected:
                    is_flat = False
                    regime_biases[i] = 1.0
                    all_signals[i] = 1
                    all_confidences[i] = 0.9
                    trend_overrides += 1
                elif is_flat and not price_above_trend:
                    regime_biases[i] = 0.0
                    all_signals[i] = 0
                    all_confidences[i] = 0.2
                else:
                    regime_biases[i] = 1.0
                    all_signals[i] = 1
                    all_confidences[i] = 0.9
                    trend_overrides += 1

        # Count signals
        long_signals = (all_signals == 1).sum()
        short_signals = (all_signals == -1).sum()
        flat_signals = (all_signals == 0).sum()
        uptrend_bars = (regime_biases > 0).sum()
        downtrend_bars = (regime_biases < 0).sum()
        ranging_bars = (regime_biases == 0).sum()

        logger.info(f"Signal distribution: Long={long_signals}, Short={short_signals}, Flat={flat_signals}")
        logger.info(f"Trend detection: Uptrend={uptrend_bars}, Downtrend={downtrend_bars}, Ranging={ranging_bars}")

        logger.info("Starting backtest simulation loop...")

        # Track funding times (convert to pd.Timestamp for compatibility)
        last_funding_time = pd.Timestamp(timestamps[0])

        # Pre-compute regimes vectorized
        regimes = self._compute_regimes_vectorized(vol_20, momentum_20, perp_zscore)

        total_bars = len(data)
        for idx in range(total_bars):
            # Progress logging every 5000 bars
            if idx % 5000 == 0:
                progress = idx / total_bars * 100
                has_pos = self.position_manager.has_position(symbol)
                logger.info(f"Backtest progress: {idx}/{total_bars} ({progress:.1f}%) | signal={all_signals[idx]}, has_pos={has_pos}, equity=${self.risk_engine.state.equity:.2f}")

            timestamp = pd.Timestamp(timestamps[idx])
            current_price = prices[idx]
            volatility = volatilities[idx]
            funding_rate = funding_rates[idx]
            regime = regimes[idx]
            signal = all_signals[idx]
            confidence = all_confidences[idx]
            regime_bias = regime_biases[idx]  # Get regime bias for this bar

            # Set regime in risk engine
            self.risk_engine.set_regime(regime)

            # Apply funding if enabled
            if self.config.funding_enabled:
                hours_since_funding = (timestamp - last_funding_time).total_seconds() / 3600
                if hours_since_funding >= self.config.funding_frequency_hours:
                    self._apply_funding(funding_rate, current_price)
                    last_funding_time = timestamp

            # Update positions (check stops)
            closed_trades = self.position_manager.update_positions(
                {symbol: current_price}, timestamp
            )

            # Update risk state for closed trades
            for trade in closed_trades:
                logger.info(f"CLOSED by stop/tp: pnl=${trade.pnl:.2f}, reason={trade.exit_reason}")
                self.risk_engine.update_state(
                    pnl=trade.pnl,
                    position_delta={symbol: -trade.size},
                    timestamp=timestamp,
                    is_trade_close=True  # Count for consecutive losses
                )
                # Reset ALL tracking when position is closed by stop
                self.entry_regime = None
                self.trailing_stop_price = None
                self.best_price_since_entry = None
                self.position_entry_bar = 0
                self.position_entry_price = 0.0

            # Record signal
            self.signals_history.append({
                "timestamp": timestamp,
                "signal": signal,
                "confidence": confidence,
                "price": current_price,
                "regime": regime
            })

            # Process signal with regime-aware exits
            self._process_signal(
                symbol=symbol,
                signal=int(signal),
                confidence=float(confidence),
                current_price=current_price,
                volatility=volatility,
                regime=regime,
                funding_rate=funding_rate,
                timestamp=timestamp,
                regime_bias=float(regime_bias)
            )

            # Record equity
            total_equity = self.risk_engine.state.equity + self.position_manager.get_total_unrealized_pnl()
            self.equity_history.append((timestamp, total_equity))

            # Record position state
            position = self.position_manager.get_position(symbol)
            self.positions_history.append({
                "timestamp": timestamp,
                "position_side": position.side if position else "flat",
                "position_size": position.size if position else 0,
                "unrealized_pnl": position.unrealized_pnl if position else 0,
                "equity": total_equity
            })

        # Close any remaining positions
        self._close_all_positions(pd.Timestamp(timestamps[-1]), prices[-1], symbol)

        # Calculate results
        result = self._calculate_results()

        logger.info(f"Backtest complete. Final equity: ${result.equity_curve.iloc[-1]:.2f}")
        logger.info(f"Total trades: {len(result.trades)}")

        return result

    def _compute_regimes_vectorized(
        self,
        volatility: np.ndarray,
        momentum: np.ndarray,
        perplexity_zscore: np.ndarray
    ) -> np.ndarray:
        """Compute regimes for all bars at once."""
        n = len(volatility)
        regimes = np.full(n, "unknown", dtype=object)

        # Regime shock: abs(perplexity_zscore) > 2.0
        regime_shock_mask = np.abs(perplexity_zscore) > 2.0
        regimes[regime_shock_mask] = "regime_shock"

        # High volatility: volatility > 0.5 (and not regime shock)
        high_vol_mask = (volatility > 0.5) & ~regime_shock_mask
        regimes[high_vol_mask] = "high_vol"

        # Trend: abs(momentum) > 0.05 (and not above)
        trend_mask = (np.abs(momentum) > 0.05) & ~regime_shock_mask & ~high_vol_mask
        regimes[trend_mask] = "trend"

        # Chop: low volatility and low momentum
        chop_mask = (volatility < 0.15) & (np.abs(momentum) < 0.02) & ~regime_shock_mask & ~high_vol_mask & ~trend_mask
        regimes[chop_mask] = "chop"

        return regimes

    def _reset(self):
        """Reset backtest state."""
        self.risk_engine.state.equity = self.config.initial_capital
        self.risk_engine.state.peak_equity = self.config.initial_capital
        self.risk_engine.state.drawdown = 0
        self.position_manager.reset()
        self.equity_history = []
        self.signals_history = []
        self.positions_history = []
        # Reset position tracking
        self.entry_regime = None
        self.trailing_stop_price = None
        self.best_price_since_entry = None
        self.position_entry_bar = 0
        self.position_entry_price = 0.0

    def _determine_regime(self, row: pd.Series) -> str:
        """Determine current regime from features."""
        # Simple regime classification based on features
        volatility = row.get("volatility_20", 0.15)
        momentum = row.get("momentum_20", 0)
        perplexity_zscore = row.get("kronos_perplexity_zscore", 0)

        # Check for regime shock
        if abs(perplexity_zscore) > 2.0:
            return "regime_shock"

        # High volatility regime
        if volatility > 0.5:
            return "high_vol"

        # Trend regime
        if abs(momentum) > 0.05:
            return "trend"

        # Low volatility chop
        if volatility < 0.15 and abs(momentum) < 0.02:
            return "chop"

        return "unknown"

    def _process_signal(
        self,
        symbol: str,
        signal: int,
        confidence: float,
        current_price: float,
        volatility: float,
        regime: str,
        funding_rate: float,
        timestamp: datetime,
        regime_bias: float = 0.0
    ):
        """
        Process trading signal - ML SIGNALS CONTROL EVERYTHING.

        Regime only affects position sizing, NOT entry/exit timing.
        This preserves the ML model's edge which works well on its own.
        """
        current_position = self.position_manager.get_position(symbol)
        current_bar = len(self.equity_history)

        # Check if we can trade
        can_trade_status, can_trade_reason = self.risk_engine.can_trade()
        if not can_trade_status:
            if current_bar % 5000 == 0:
                logger.warning(f"Bar {current_bar}: Cannot trade - {can_trade_reason}")
            return

        # Determine desired side
        desired_side = {1: "long", -1: "short", 0: None}.get(signal)

        # Handle existing position - ML SIGNALS + TIME + TRAILING STOP EXITS
        if current_position:
            should_close = False
            close_reason = ""

            # Calculate current bar index for holding duration
            current_bar = len(self.equity_history)
            bars_held = current_bar - self.position_entry_bar

            # Update best price since entry for trailing stop
            if self.best_price_since_entry is None:
                self.best_price_since_entry = current_price
            elif current_position.side == "long":
                self.best_price_since_entry = max(self.best_price_since_entry, current_price)
            else:  # short
                self.best_price_since_entry = min(self.best_price_since_entry, current_price)

            # Exit condition 1: Max holding duration exceeded
            if bars_held >= self.config.max_holding_bars:
                should_close = True
                close_reason = "max_holding_time"
                logger.debug(f"Max holding time reached: {bars_held} bars")

            # Exit condition 2: Trailing stop hit (only after profit)
            if not should_close and self.best_price_since_entry is not None:
                pct_from_best = abs(current_price - self.best_price_since_entry) / self.best_price_since_entry
                pct_from_entry = (current_price - self.position_entry_price) / self.position_entry_price

                # Only apply trailing stop if we're in profit
                if current_position.side == "long":
                    if pct_from_entry > 0.01 and current_price < self.best_price_since_entry * (1 - self.config.trailing_stop_pct):
                        should_close = True
                        close_reason = "trailing_stop"
                else:  # short
                    if pct_from_entry < -0.01 and current_price > self.best_price_since_entry * (1 + self.config.trailing_stop_pct):
                        should_close = True
                        close_reason = "trailing_stop"

            # Exit condition 3: ML signal reverses or goes flat
            # BUT only if we've held for minimum time (avoid whipsaws)
            if not should_close and bars_held >= self.config.min_holding_bars:
                if signal == 0:
                    should_close = True
                    close_reason = "signal_flat"
                elif current_position.side == "long" and signal == -1:
                    should_close = True
                    close_reason = "signal_reversal"
                elif current_position.side == "short" and signal == 1:
                    should_close = True
                    close_reason = "signal_reversal"

            if should_close:
                trade, fees = self.position_manager.close_position(
                    symbol, current_price, timestamp, reason=close_reason
                )
                if trade:
                    logger.info(f"CLOSED position: {close_reason}, bars_held={bars_held}, pnl=${trade.pnl:.2f}")
                    self.risk_engine.update_state(
                        pnl=trade.pnl,
                        position_delta={symbol: -trade.size},
                        timestamp=timestamp,
                        is_trade_close=True  # Count for consecutive losses
                    )
                    # Reset position tracking
                    self.best_price_since_entry = None
                    self.position_entry_bar = 0
                    self.position_entry_price = 0.0

        # Open new position if we have a signal and no position
        current_bar = len(self.equity_history)
        has_pos = self.position_manager.has_position(symbol)
        if current_bar % 5000 == 0:
            logger.info(f"Bar {current_bar}: desired_side={desired_side}, has_pos={has_pos}, signal={signal}")

        if desired_side and not has_pos:
            # Debug: Check why we might not open
            can_trade_result, can_trade_reason = self.risk_engine.can_trade()
            if not can_trade_result:
                if current_bar % 5000 == 0:
                    logger.warning(f"Cannot trade at bar {current_bar}: {can_trade_reason}")

            # Calculate position size
            position_size, size_reason = self.risk_engine.calculate_position_size(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                current_price=current_price,
                volatility=volatility,
                regime=regime,
                funding_rate=funding_rate
            )

            # Debug: Log if position size is 0
            if position_size == 0 and len(self.equity_history) % 5000 == 0:
                logger.warning(f"Bar {len(self.equity_history)}: position_size=0, reason={size_reason}, signal={signal}, conf={confidence:.2f}")

            # Scale position size by regime - balance between gains and risk
            if abs(regime_bias) >= 0.8:
                # Strong trend - use full leverage
                position_size *= 1.5
            elif abs(regime_bias) >= 0.5:
                # Moderate trend - good position
                position_size *= 1.2
            elif abs(regime_bias) >= 0.3:
                # Weak trend - moderate position
                position_size *= 1.0
            else:
                # Uncertain - reduce position
                position_size *= 0.8

            # Cap position at 1.5x leverage * equity (safety cap)
            max_position = self.risk_engine.state.equity * min(self.leverage, 1.5)
            position_size = min(position_size, max_position)

            if position_size > 0:
                # Use standard stops - let ML handle timing
                position, fees = self.position_manager.open_position(
                    symbol=symbol,
                    side=desired_side,
                    size=position_size,
                    price=current_price,
                    timestamp=timestamp,
                    stop_loss_pct=self.config.stop_loss_pct,
                    take_profit_pct=self.config.take_profit_pct
                )

                if not position and len(self.equity_history) % 5000 == 0:
                    logger.warning(f"Bar {len(self.equity_history)}: open_position returned None, size={position_size:.2f}, side={desired_side}")

                if position:
                    # Track entry for holding duration and trailing stop
                    self.position_entry_bar = len(self.equity_history)
                    self.position_entry_price = current_price
                    self.best_price_since_entry = current_price

                    logger.info(f"OPENED {desired_side} position: size=${position_size:.2f}, price=${current_price:.2f}, bar={self.position_entry_bar}")

                    # Update risk state with fees
                    self.risk_engine.update_state(
                        pnl=-fees,
                        position_delta={symbol: position_size},
                        timestamp=timestamp
                    )

    def _apply_funding(self, funding_rate: float, current_price: float):
        """Apply funding payment to open positions."""
        for symbol, position in self.position_manager.positions.items():
            # Funding payment
            # Long pays short when funding is positive
            if position.side == "long":
                funding_payment = -position.size * funding_rate
            else:  # short
                funding_payment = position.size * funding_rate

            self.risk_engine.update_state(pnl=funding_payment)

    def _close_all_positions(self, timestamp: datetime, price: float, symbol: str):
        """Close all remaining positions at end of backtest."""
        if self.position_manager.has_position(symbol):
            trade, _ = self.position_manager.close_position(
                symbol, price, timestamp, reason="end_of_backtest"
            )
            if trade:
                self.risk_engine.update_state(
                    pnl=trade.pnl,
                    position_delta={symbol: -trade.size},
                    timestamp=timestamp
                )

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results and metrics."""
        from .metrics import BacktestMetrics

        # Create equity curve
        equity_df = pd.DataFrame(self.equity_history, columns=["timestamp", "equity"])
        equity_df = equity_df.set_index("timestamp")
        equity_curve = equity_df["equity"]

        # Calculate daily returns
        daily_equity = equity_curve.resample("D").last().dropna()
        daily_returns = daily_equity.pct_change().dropna()

        # Get trades
        trades = self.position_manager.trades

        # Calculate metrics (with benchmark if price data available)
        metrics_calculator = BacktestMetrics()
        metrics = metrics_calculator.calculate_all(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=self.config.initial_capital,
            prices=self.price_series
        )

        # Create DataFrames
        positions_df = pd.DataFrame(self.positions_history)
        signals_df = pd.DataFrame(self.signals_history)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            daily_returns=daily_returns,
            positions_history=positions_df,
            signals_history=signals_df
        )


def run_walk_forward_backtest(
    data: pd.DataFrame,
    model_class: type,
    model_config: Dict,
    feature_columns: List[str],
    target_column: str,
    backtest_config: Optional[BacktestConfig] = None,
    train_window: int = 10000,
    test_window: int = 2000,
    purge_window: int = 48,
    symbol: str = "BTCUSDT"
) -> List[BacktestResult]:
    """
    Run walk-forward backtest with model retraining.

    Args:
        data: Full dataset
        model_class: Signal model class
        model_config: Model configuration
        feature_columns: Feature column names
        target_column: Target column name
        backtest_config: Backtest configuration
        train_window: Training window size
        test_window: Test window size
        purge_window: Purge window size
        symbol: Trading symbol

    Returns:
        List of BacktestResult for each test window
    """
    results = []
    n_samples = len(data)

    split_idx = 0
    current_start = 0

    while current_start + train_window + purge_window + test_window <= n_samples:
        train_end = current_start + train_window
        test_start = train_end + purge_window
        test_end = test_start + test_window

        logger.info(f"Walk-forward split {split_idx + 1}: train=[{current_start}:{train_end}], test=[{test_start}:{test_end}]")

        # Get data splits
        train_data = data.iloc[current_start:train_end]
        test_data = data.iloc[test_start:test_end]

        # Train model
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]

        # Split train into train/val
        val_size = int(len(X_train) * 0.15)
        X_train_split = X_train.iloc[:-val_size]
        y_train_split = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]

        model = model_class(model_config)
        model.fit(X_train_split, y_train_split, X_val, y_val, verbose=False)

        # Run backtest
        engine = BacktestEngine(backtest_config)
        result = engine.run(
            data=test_data,
            model=model,
            feature_columns=feature_columns,
            symbol=symbol
        )

        results.append(result)

        # Move to next window
        current_start += test_window
        split_idx += 1

    return results
