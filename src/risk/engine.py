"""
Risk Engine - The component that keeps you alive.

Controls:
- Volatility targeting
- Position sizing
- Leverage
- Drawdown limits
- Max exposure
- Kill switches

Rules:
- Larger positions in stable trending regimes
- Smaller positions in chop or high volatility
- No trading during Kronos regime shocks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from ..utils.logger import get_logger

logger = get_logger("risk_engine")


@dataclass
class RiskState:
    """Current risk state tracking."""
    equity: float = 10000.0
    peak_equity: float = 10000.0
    drawdown: float = 0.0
    daily_pnl: float = 0.0
    daily_start_equity: float = 10000.0
    consecutive_losses: int = 0
    last_trade_time: Optional[datetime] = None
    positions: Dict[str, float] = field(default_factory=dict)
    total_exposure: float = 0.0
    regime: str = "unknown"
    is_active: bool = True


class RiskEngine:
    """
    Risk management engine for the trading system.

    Implements volatility targeting, position sizing, and kill switches.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize risk engine.

        Args:
            config: Risk configuration dictionary
        """
        self.config = config or {}

        # Capital and sizing
        self.initial_capital = self.config.get("initial_capital", 10000.0)
        self.max_position_size = self.config.get("max_position_size", 0.25)
        self.max_leverage = self.config.get("max_leverage", 3.0)

        # Volatility targeting
        self.volatility_target = self.config.get("volatility_target", 0.15)
        self.volatility_lookback = self.config.get("volatility_lookback", 20)

        # Drawdown limits
        self.max_drawdown = self.config.get("max_drawdown", 0.15)
        self.daily_loss_limit = self.config.get("daily_loss_limit", 0.03)

        # Regime-based adjustments - more aggressive for trends
        self.regime_sizing = self.config.get("regime_sizing", {
            "trend": 1.5,        # 1.5x in trending markets - capture momentum
            "trend_up": 1.5,     # Explicit uptrend
            "trend_down": 1.5,   # Explicit downtrend
            "chop": 0.4,         # Reduced but still trading in chop
            "high_vol": 0.7,     # Higher in volatility - vol = opportunity
            "regime_shock": 0.0,
            "unknown": 0.6,      # Default slightly positive
        })

        # Kill switches
        self.max_consecutive_losses = self.config.get("max_consecutive_losses", 5)
        self.funding_spike_threshold = self.config.get("funding_spike", 0.01)

        # State
        self.state = RiskState(
            equity=self.initial_capital,
            peak_equity=self.initial_capital,
            daily_start_equity=self.initial_capital,
        )

        # Trade history for risk metrics
        self.trade_history: List[Dict] = []

    def calculate_position_size(
        self,
        symbol: str,
        signal: int,
        confidence: float,
        current_price: float,
        volatility: float,
        regime: str,
        funding_rate: float = 0.0
    ) -> Tuple[float, str]:
        """
        Calculate position size based on risk parameters.

        Args:
            symbol: Trading symbol
            signal: 1 (long), 0 (flat), -1 (short)
            confidence: Signal confidence [0, 1]
            current_price: Current asset price
            volatility: Current volatility estimate
            regime: Current regime label
            funding_rate: Current funding rate

        Returns:
            Tuple of (position_size_usd, reason)
        """
        # Check kill switches first
        kill_check = self._check_kill_switches(funding_rate)
        if kill_check is not None:
            return 0.0, kill_check

        # No position for flat signal
        if signal == 0:
            return 0.0, "flat_signal"

        # Check drawdown
        if self.state.drawdown >= self.max_drawdown:
            return 0.0, "max_drawdown_exceeded"

        # Check daily loss
        daily_loss = (self.state.equity - self.state.daily_start_equity) / self.state.daily_start_equity
        if daily_loss <= -self.daily_loss_limit:
            return 0.0, "daily_loss_limit_exceeded"

        # Base position size (volatility targeting)
        base_size = self._volatility_target_size(volatility)

        # Regime adjustment
        regime_multiplier = self.regime_sizing.get(regime, 0.5)
        adjusted_size = base_size * regime_multiplier

        # Confidence adjustment
        confidence_multiplier = 0.5 + 0.5 * confidence  # Scale from 0.5 to 1.0
        adjusted_size *= confidence_multiplier

        # Cap at max position size
        max_size = self.state.equity * self.max_position_size
        position_size = min(adjusted_size, max_size)

        # Check total exposure
        new_total_exposure = self.state.total_exposure + position_size
        max_exposure = self.state.equity * self.max_leverage

        if new_total_exposure > max_exposure:
            position_size = max(0, max_exposure - self.state.total_exposure)
            if position_size == 0:
                return 0.0, "max_exposure_exceeded"

        # Round to reasonable precision
        position_size = round(position_size, 2)

        logger.debug(
            f"Position size for {symbol}: ${position_size:.2f} "
            f"(base=${base_size:.2f}, regime={regime}[{regime_multiplier}], conf={confidence:.2f})"
        )

        return position_size, "ok"

    def _volatility_target_size(self, current_volatility: float) -> float:
        """
        Calculate position size based on volatility targeting.

        Target a specific portfolio volatility by adjusting position size
        inversely to asset volatility.

        Args:
            current_volatility: Current annualized volatility

        Returns:
            Position size in USD
        """
        if current_volatility <= 0:
            current_volatility = self.volatility_target  # Default to target

        # Target volatility / Asset volatility
        vol_scalar = self.volatility_target / current_volatility

        # Position size = equity * vol_scalar (capped at max leverage)
        position_size = self.state.equity * min(vol_scalar, self.max_leverage)

        return position_size

    def _check_kill_switches(self, funding_rate: float) -> Optional[str]:
        """
        Check kill switch conditions.

        Args:
            funding_rate: Current funding rate

        Returns:
            Kill reason if triggered, None otherwise
        """
        # Check if trading is deactivated
        if not self.state.is_active:
            return "trading_deactivated"

        # Check consecutive losses
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            return "consecutive_losses_exceeded"

        # Check funding rate spike
        if abs(funding_rate) >= self.funding_spike_threshold:
            return "funding_spike"

        return None

    def update_state(
        self,
        pnl: float,
        position_delta: Dict[str, float] = None,
        timestamp: Optional[datetime] = None,
        is_trade_close: bool = False  # Only count for consecutive losses if True
    ):
        """
        Update risk state after a trade or mark-to-market.

        Args:
            pnl: Profit/loss in USD
            position_delta: Change in positions {symbol: delta_usd}
            timestamp: Current timestamp
            is_trade_close: True if this is a trade close (counts for consecutive losses)
        """
        # Update equity
        self.state.equity += pnl

        # Update peak and drawdown
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

        self.state.drawdown = (self.state.peak_equity - self.state.equity) / self.state.peak_equity

        # Update daily PnL
        self.state.daily_pnl += pnl

        # Update consecutive losses - ONLY for actual trade closes, not fees
        if is_trade_close:
            if pnl < 0:
                self.state.consecutive_losses += 1
            elif pnl > 0:
                self.state.consecutive_losses = 0

        # Update positions
        if position_delta:
            for symbol, delta in position_delta.items():
                current = self.state.positions.get(symbol, 0.0)
                self.state.positions[symbol] = current + delta

            # Recalculate total exposure
            self.state.total_exposure = sum(abs(p) for p in self.state.positions.values())

        # Update timestamp
        if timestamp:
            self.state.last_trade_time = timestamp

            # Check for new trading day
            if self.state.daily_start_equity != self.state.equity:
                # Simple day check - reset if more than 24h since last update
                pass  # Would implement proper day boundary logic here

        logger.debug(
            f"Risk state: equity=${self.state.equity:.2f}, "
            f"drawdown={self.state.drawdown:.2%}, "
            f"exposure=${self.state.total_exposure:.2f}"
        )

    def set_regime(self, regime: str):
        """Update current regime label."""
        self.state.regime = regime

    def reset_daily(self):
        """Reset daily tracking (call at day boundary)."""
        self.state.daily_start_equity = self.state.equity
        self.state.daily_pnl = 0.0
        logger.info(f"Daily reset: equity=${self.state.equity:.2f}")

    def activate(self):
        """Activate trading."""
        self.state.is_active = True
        logger.info("Trading activated")

    def deactivate(self, reason: str = "manual"):
        """Deactivate trading."""
        self.state.is_active = False
        logger.warning(f"Trading deactivated: {reason}")

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics."""
        return {
            "equity": self.state.equity,
            "peak_equity": self.state.peak_equity,
            "drawdown": self.state.drawdown,
            "drawdown_pct": f"{self.state.drawdown:.2%}",
            "daily_pnl": self.state.daily_pnl,
            "daily_return": (self.state.equity / self.state.daily_start_equity - 1),
            "total_exposure": self.state.total_exposure,
            "leverage": self.state.total_exposure / self.state.equity if self.state.equity > 0 else 0,
            "consecutive_losses": self.state.consecutive_losses,
            "regime": self.state.regime,
            "is_active": self.state.is_active,
            "positions": self.state.positions.copy(),
        }

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (can_trade, reason)
        """
        if not self.state.is_active:
            return False, "trading_deactivated"

        if self.state.drawdown >= self.max_drawdown:
            return False, "max_drawdown_exceeded"

        daily_return = (self.state.equity / self.state.daily_start_equity - 1)
        if daily_return <= -self.daily_loss_limit:
            return False, "daily_loss_limit"

        if self.state.consecutive_losses >= self.max_consecutive_losses:
            return False, "consecutive_losses"

        return True, "ok"


class RegimeRiskAdjuster:
    """
    Adjust risk parameters based on detected regime.
    """

    # Regime risk profiles - optimized for trend capture
    REGIME_PROFILES = {
        "trend_up": {
            "size_mult": 1.5,       # Larger positions in uptrends
            "stop_mult": 1.5,       # Wider stops to ride trends
            "take_profit_mult": 3.0, # Let winners run
            "max_leverage": 3.0,
        },
        "trend_down": {
            "size_mult": 1.5,       # Larger positions in downtrends
            "stop_mult": 1.5,       # Wider stops
            "take_profit_mult": 3.0, # Let winners run
            "max_leverage": 3.0,
        },
        "trend": {
            "size_mult": 1.5,       # Generic trend
            "stop_mult": 1.5,
            "take_profit_mult": 3.0,
            "max_leverage": 3.0,
        },
        "chop": {
            "size_mult": 0.5,       # Still trade, just smaller
            "stop_mult": 0.8,
            "take_profit_mult": 1.0,
            "max_leverage": 2.0,
        },
        "high_vol": {
            "size_mult": 0.8,       # Good opportunity in volatility
            "stop_mult": 2.0,       # Wider stops for volatility
            "take_profit_mult": 2.5,
            "max_leverage": 2.5,
        },
        "mean_reversion": {
            "size_mult": 0.8,
            "stop_mult": 1.0,
            "take_profit_mult": 1.2,
            "max_leverage": 2.0,
        },
        "panic": {
            "size_mult": 0.4,       # Reduced but still trading
            "stop_mult": 2.0,
            "take_profit_mult": 3.0,
            "max_leverage": 1.5,
        },
        "regime_shock": {
            "size_mult": 0.2,       # Small positions during regime shocks
            "stop_mult": 2.0,
            "take_profit_mult": 3.0,
            "max_leverage": 1.0,
        },
    }

    def __init__(self):
        self.current_regime = "unknown"
        self.regime_confidence = 0.0

    def update_regime(self, regime: str, confidence: float = 1.0):
        """Update current regime."""
        self.current_regime = regime
        self.regime_confidence = confidence

    def get_adjustments(self) -> Dict:
        """Get risk adjustments for current regime."""
        profile = self.REGIME_PROFILES.get(self.current_regime, {
            "size_mult": 0.5,
            "stop_mult": 1.0,
            "take_profit_mult": 1.0,
            "max_leverage": 2.0,
        })

        # Scale by confidence
        adjustments = {}
        for key, value in profile.items():
            if key == "size_mult":
                # Size scales with confidence
                adjustments[key] = value * (0.5 + 0.5 * self.regime_confidence)
            else:
                adjustments[key] = value

        return adjustments

    def should_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed in current regime."""
        profile = self.REGIME_PROFILES.get(self.current_regime, {})

        if profile.get("size_mult", 0) == 0:
            return False, f"no_trading_in_{self.current_regime}"

        return True, "ok"
