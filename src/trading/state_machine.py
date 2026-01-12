"""Trading state machine for managing trading lifecycle."""

from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List
import logging

logger = logging.getLogger(__name__)


class TradingState(Enum):
    """Trading engine states."""

    IDLE = auto()           # Not trading, waiting for start command
    SCANNING = auto()       # Actively looking for entry signals
    IN_POSITION = auto()    # Has open position, monitoring for exits
    COOLDOWN = auto()       # Post-trade cooldown period
    ERROR = auto()          # Error state, requires intervention
    STOPPING = auto()       # Graceful shutdown in progress


# Valid state transitions
VALID_TRANSITIONS: Dict[TradingState, List[TradingState]] = {
    TradingState.IDLE: [TradingState.SCANNING, TradingState.ERROR],
    TradingState.SCANNING: [TradingState.IN_POSITION, TradingState.IDLE, TradingState.STOPPING, TradingState.ERROR],
    TradingState.IN_POSITION: [TradingState.COOLDOWN, TradingState.SCANNING, TradingState.STOPPING, TradingState.ERROR],
    TradingState.COOLDOWN: [TradingState.SCANNING, TradingState.IDLE, TradingState.STOPPING, TradingState.ERROR],
    TradingState.ERROR: [TradingState.IDLE, TradingState.SCANNING],
    TradingState.STOPPING: [TradingState.IDLE],
}


@dataclass
class StateTransition:
    """Record of a state transition."""

    from_state: TradingState
    to_state: TradingState
    timestamp: datetime
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "from_state": self.from_state.name,
            "to_state": self.to_state.name,
            "timestamp": self.timestamp.isoformat(),
            "reason": reason,
            "metadata": self.metadata,
        }


@dataclass
class TradingContext:
    """Context information for the trading state machine."""

    current_state: TradingState = TradingState.IDLE
    last_state_change: datetime = field(default_factory=datetime.utcnow)
    state_history: List[StateTransition] = field(default_factory=list)

    # Position info
    position_symbol: Optional[str] = None
    position_side: Optional[str] = None
    position_entry_price: Optional[float] = None
    position_entry_time: Optional[datetime] = None
    position_size: Optional[float] = None
    position_trade_id: Optional[int] = None

    # Cooldown info
    cooldown_until: Optional[datetime] = None

    # Error info
    last_error: Optional[str] = None
    error_count: int = 0

    # Stats
    trades_today: int = 0
    daily_pnl: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "current_state": self.current_state.name,
            "last_state_change": self.last_state_change.isoformat(),
            "position": {
                "symbol": self.position_symbol,
                "side": self.position_side,
                "entry_price": self.position_entry_price,
                "entry_time": self.position_entry_time.isoformat() if self.position_entry_time else None,
                "size": self.position_size,
                "trade_id": self.position_trade_id,
            } if self.position_symbol else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "trades_today": self.trades_today,
            "daily_pnl": self.daily_pnl,
        }


class StateMachine:
    """State machine for managing trading lifecycle."""

    def __init__(self):
        self.context = TradingContext()
        self._callbacks: Dict[TradingState, List[Callable]] = {state: [] for state in TradingState}

    @property
    def state(self) -> TradingState:
        """Get current state."""
        return self.context.current_state

    @property
    def is_trading(self) -> bool:
        """Check if actively trading."""
        return self.context.current_state in [
            TradingState.SCANNING,
            TradingState.IN_POSITION,
            TradingState.COOLDOWN,
        ]

    @property
    def has_position(self) -> bool:
        """Check if currently in a position."""
        return self.context.current_state == TradingState.IN_POSITION

    def can_transition(self, to_state: TradingState) -> bool:
        """Check if transition to given state is valid."""
        return to_state in VALID_TRANSITIONS.get(self.context.current_state, [])

    def transition(
        self,
        to_state: TradingState,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Transition to a new state.

        Args:
            to_state: Target state
            reason: Reason for transition
            metadata: Additional metadata

        Returns:
            True if transition successful, False otherwise
        """
        if not self.can_transition(to_state):
            logger.warning(
                f"Invalid state transition: {self.context.current_state.name} -> {to_state.name}"
            )
            return False

        from_state = self.context.current_state

        # Record transition
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=datetime.utcnow(),
            reason=reason,
            metadata=metadata or {},
        )
        self.context.state_history.append(transition)

        # Keep only last 100 transitions
        if len(self.context.state_history) > 100:
            self.context.state_history = self.context.state_history[-100:]

        # Update state
        self.context.current_state = to_state
        self.context.last_state_change = datetime.utcnow()

        logger.info(f"State transition: {from_state.name} -> {to_state.name} ({reason})")

        # Execute callbacks
        for callback in self._callbacks[to_state]:
            try:
                callback(transition)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")

        return True

    def on_state(self, state: TradingState, callback: Callable) -> None:
        """Register a callback for when entering a state."""
        self._callbacks[state].append(callback)

    def enter_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        trade_id: int,
    ) -> bool:
        """Record entering a position."""
        if self.transition(TradingState.IN_POSITION, f"Entered {side} position"):
            self.context.position_symbol = symbol
            self.context.position_side = side
            self.context.position_entry_price = entry_price
            self.context.position_entry_time = datetime.utcnow()
            self.context.position_size = size
            self.context.position_trade_id = trade_id
            return True
        return False

    def exit_position(self, pnl: float, cooldown_minutes: int = 30) -> bool:
        """Record exiting a position."""
        self.context.daily_pnl += pnl
        self.context.trades_today += 1

        # Clear position info
        self.context.position_symbol = None
        self.context.position_side = None
        self.context.position_entry_price = None
        self.context.position_entry_time = None
        self.context.position_size = None
        self.context.position_trade_id = None

        # Set cooldown
        from datetime import timedelta
        self.context.cooldown_until = datetime.utcnow() + timedelta(minutes=cooldown_minutes)

        return self.transition(TradingState.COOLDOWN, f"Exited position, PnL: ${pnl:.2f}")

    def check_cooldown_complete(self) -> bool:
        """Check if cooldown period is complete."""
        if self.context.current_state != TradingState.COOLDOWN:
            return False

        if self.context.cooldown_until and datetime.utcnow() >= self.context.cooldown_until:
            self.context.cooldown_until = None
            return self.transition(TradingState.SCANNING, "Cooldown complete")

        return False

    def set_error(self, error: str) -> bool:
        """Transition to error state."""
        self.context.last_error = error
        self.context.error_count += 1
        return self.transition(TradingState.ERROR, error)

    def reset_error(self) -> bool:
        """Reset from error state."""
        if self.context.current_state == TradingState.ERROR:
            self.context.last_error = None
            return self.transition(TradingState.IDLE, "Error cleared")
        return False

    def start_trading(self) -> bool:
        """Start the trading loop."""
        if self.context.current_state == TradingState.IDLE:
            return self.transition(TradingState.SCANNING, "Trading started")
        return False

    def stop_trading(self) -> bool:
        """Stop the trading loop."""
        if self.context.current_state in [TradingState.SCANNING, TradingState.COOLDOWN]:
            return self.transition(TradingState.STOPPING, "Stop requested")
        elif self.context.current_state == TradingState.IN_POSITION:
            # Will need to close position first
            return self.transition(TradingState.STOPPING, "Stop requested - closing position")
        return False

    def complete_stop(self) -> bool:
        """Complete the stop process."""
        if self.context.current_state == TradingState.STOPPING:
            return self.transition(TradingState.IDLE, "Trading stopped")
        return False

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of new day)."""
        self.context.trades_today = 0
        self.context.daily_pnl = 0.0
