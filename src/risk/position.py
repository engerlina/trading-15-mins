"""
Position management for the trading system.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..utils.logger import get_logger

logger = get_logger("position_manager")


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: str  # "long" or "short"
    size: float  # Position size in USD
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    unrealized_pnl: float = 0.0

    def update_pnl(self, current_price: float):
        """Update unrealized PnL."""
        if self.side == "long":
            self.unrealized_pnl = self.size * (current_price / self.entry_price - 1)
        else:  # short
            self.unrealized_pnl = self.size * (1 - current_price / self.entry_price)

    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is hit."""
        if self.stop_loss is None:
            return False

        if self.side == "long":
            return current_price <= self.stop_loss
        else:  # short
            return current_price >= self.stop_loss

    def check_take_profit(self, current_price: float) -> bool:
        """Check if take profit is hit."""
        if self.take_profit is None:
            return False

        if self.side == "long":
            return current_price >= self.take_profit
        else:  # short
            return current_price <= self.take_profit


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    exit_reason: str  # "signal", "stop_loss", "take_profit", "manual"
    fees: float = 0.0
    funding: float = 0.0


class PositionManager:
    """
    Manage positions and trade execution.
    """

    def __init__(
        self,
        fee_rate: float = 0.0005,  # 5 bps taker fee
        slippage_bps: float = 5,
        max_positions: int = 5
    ):
        """
        Initialize position manager.

        Args:
            fee_rate: Trading fee rate
            slippage_bps: Slippage in basis points
            max_positions: Maximum concurrent positions
        """
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps / 10000
        self.max_positions = max_positions

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        timestamp: datetime,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None
    ) -> Tuple[Optional[Position], float]:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            size: Position size in USD
            price: Entry price
            timestamp: Entry time
            stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%)
            take_profit_pct: Take profit percentage

        Returns:
            Tuple of (Position or None, fees)
        """
        # Check max positions
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Max positions reached ({self.max_positions})")
            return None, 0.0

        # Check if position already exists
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return None, 0.0

        # Apply slippage to entry price
        if side == "long":
            entry_price = price * (1 + self.slippage_bps)
        else:
            entry_price = price * (1 - self.slippage_bps)

        # Calculate stop loss and take profit
        stop_loss = None
        take_profit = None

        if stop_loss_pct:
            if side == "long":
                stop_loss = entry_price * (1 - stop_loss_pct)
            else:
                stop_loss = entry_price * (1 + stop_loss_pct)

        if take_profit_pct:
            if side == "long":
                take_profit = entry_price * (1 + take_profit_pct)
            else:
                take_profit = entry_price * (1 - take_profit_pct)

        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.positions[symbol] = position

        # Calculate fees
        fees = size * self.fee_rate

        sl_str = f"${stop_loss:.2f}" if stop_loss else "None"
        tp_str = f"${take_profit:.2f}" if take_profit else "None"
        logger.debug(
            f"Opened {side} position: {symbol} @ ${entry_price:.2f}, "
            f"size=${size:.2f}, SL={sl_str}, TP={tp_str}"
        )

        return position, fees

    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        reason: str = "signal"
    ) -> Tuple[Optional[Trade], float]:
        """
        Close an existing position.

        Args:
            symbol: Trading symbol
            price: Exit price
            timestamp: Exit time
            reason: Exit reason

        Returns:
            Tuple of (Trade or None, fees)
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None, 0.0

        position = self.positions[symbol]

        # Apply slippage to exit price
        if position.side == "long":
            exit_price = price * (1 - self.slippage_bps)
        else:
            exit_price = price * (1 + self.slippage_bps)

        # Calculate PnL
        if position.side == "long":
            pnl_pct = exit_price / position.entry_price - 1
        else:
            pnl_pct = 1 - exit_price / position.entry_price

        pnl = position.size * pnl_pct

        # Calculate fees
        fees = position.size * self.fee_rate

        # Net PnL after fees
        net_pnl = pnl - fees

        # Create trade record
        trade = Trade(
            symbol=symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=timestamp,
            pnl=net_pnl,
            pnl_pct=pnl_pct - self.fee_rate,  # Net return
            exit_reason=reason,
            fees=fees
        )

        self.trades.append(trade)
        del self.positions[symbol]

        logger.debug(
            f"Closed {position.side} position: {symbol} @ ${exit_price:.2f}, "
            f"PnL=${net_pnl:.2f} ({pnl_pct:.2%}), reason={reason}"
        )

        return trade, fees

    def update_positions(
        self,
        prices: Dict[str, float],
        timestamp: datetime
    ) -> List[Trade]:
        """
        Update all positions and check stops.

        Args:
            prices: Current prices {symbol: price}
            timestamp: Current timestamp

        Returns:
            List of closed trades (from stops)
        """
        closed_trades = []

        for symbol in list(self.positions.keys()):
            if symbol not in prices:
                continue

            position = self.positions[symbol]
            current_price = prices[symbol]

            # Update unrealized PnL
            position.update_pnl(current_price)

            # Check stop loss
            if position.check_stop_loss(current_price):
                trade, _ = self.close_position(
                    symbol, current_price, timestamp, reason="stop_loss"
                )
                if trade:
                    closed_trades.append(trade)
                continue

            # Check take profit
            if position.check_take_profit(current_price):
                trade, _ = self.close_position(
                    symbol, current_price, timestamp, reason="take_profit"
                )
                if trade:
                    closed_trades.append(trade)

        return closed_trades

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if position exists."""
        return symbol in self.positions

    def get_total_exposure(self) -> float:
        """Get total position exposure in USD."""
        return sum(p.size for p in self.positions.values())

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized PnL."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_trade_statistics(self) -> Dict:
        """Calculate trade statistics."""
        if not self.trades:
            return {}

        pnls = [t.pnl for t in self.trades]
        pnl_pcts = [t.pnl_pct for t in self.trades]

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        stats = {
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades) if self.trades else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "avg_pnl_pct": np.mean(pnl_pcts),
            "max_win": max(pnls) if pnls else 0,
            "max_loss": min(pnls) if pnls else 0,
            "profit_factor": abs(sum(p for p in pnls if p > 0) / sum(p for p in pnls if p < 0)) if losing_trades else float("inf"),
            "avg_win": np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            "avg_loss": np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            "total_fees": sum(t.fees for t in self.trades),
        }

        # Exit reason breakdown
        exit_reasons = {}
        for trade in self.trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "pnl": 0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["pnl"] += trade.pnl

        stats["exit_reasons"] = exit_reasons

        return stats

    def reset(self):
        """Reset all positions and trades."""
        self.positions.clear()
        self.trades.clear()
