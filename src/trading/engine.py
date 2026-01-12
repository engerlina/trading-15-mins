"""Main trading engine for Kronos live trading system."""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Callable, List, Any

from .state_machine import StateMachine, TradingState, TradingContext
from .strategy import RSIDivergenceStrategy, Signal, ExitSignal
from .data_feed import DataFeed
from ..execution.hyperliquid import HyperliquidExecutor
from ..database.connection import get_db_session
from ..database.repository import TradeRepository, SignalRepository, EquityRepository

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main trading engine that coordinates:
    - Data feed
    - Strategy signals
    - Order execution
    - Position management
    - State tracking
    """

    def __init__(
        self,
        symbol: str = "BTC",
        timeframe: str = "15m",
        risk_per_trade: float = 0.02,
        max_leverage: float = 2.0,
        cooldown_minutes: int = 30,
        equity_snapshot_interval: int = 15,
        network: str = "mainnet",
    ):
        """
        Initialize trading engine.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH", "SOL")
            timeframe: Strategy timeframe
            risk_per_trade: Percentage of balance to risk per trade
            max_leverage: Maximum leverage to use
            cooldown_minutes: Minutes to wait after closing a trade
            equity_snapshot_interval: Minutes between equity snapshots
            network: Hyperliquid network ("mainnet" or "testnet")
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        self.cooldown_minutes = cooldown_minutes
        self.equity_snapshot_interval = equity_snapshot_interval
        self.network = network

        # Components
        self.state_machine = StateMachine()
        self.data_feed = DataFeed(
            symbol=symbol,
            timeframe=timeframe,
            buffer_size=200,
            network=network,
        )
        self.strategy = RSIDivergenceStrategy(
            long_only=True,
            use_trailing_stop=True,
            trailing_stop_pct=0.10,
        )
        self.executor = HyperliquidExecutor(network=network)

        # State
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._bar_counter = 0
        self._last_equity_snapshot: Optional[datetime] = None
        self._current_trade_id: Optional[int] = None

        # Callbacks for WebSocket broadcasting
        self._on_state_change: List[Callable] = []
        self._on_trade: List[Callable] = []
        self._on_signal: List[Callable] = []

    async def start(self) -> bool:
        """Start the trading engine."""
        if self._running:
            logger.warning("Trading engine already running")
            return False

        try:
            # Start data feed
            await self.data_feed.start()

            # Wait for data feed to be ready
            for _ in range(30):
                if self.data_feed.is_ready:
                    break
                await asyncio.sleep(1)

            if not self.data_feed.is_ready:
                logger.error("Data feed failed to initialize")
                await self.data_feed.stop()
                return False

            # Transition to scanning state
            if not self.state_machine.start_trading():
                logger.error("Failed to start trading state machine")
                return False

            # Start the main loop
            self._running = True
            self._loop_task = asyncio.create_task(self._trading_loop())

            logger.info(f"Trading engine started: {self.symbol} {self.timeframe}")
            await self._notify_state_change()
            return True

        except Exception as e:
            logger.error(f"Error starting trading engine: {e}")
            return False

    async def stop(self, close_positions: bool = True) -> bool:
        """
        Stop the trading engine.

        Args:
            close_positions: If True, close any open positions before stopping
        """
        if not self._running:
            return True

        logger.info("Stopping trading engine...")
        self._running = False

        # Signal stop to state machine
        self.state_machine.stop_trading()

        # Close positions if requested
        if close_positions and self.state_machine.has_position:
            await self._close_current_position("stop_requested")

        # Cancel loop task
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        # Stop data feed
        await self.data_feed.stop()

        # Close executor
        await self.executor.close()

        # Complete stop
        self.state_machine.complete_stop()

        logger.info("Trading engine stopped")
        await self._notify_state_change()
        return True

    async def _trading_loop(self) -> None:
        """Main trading loop."""
        logger.info("Trading loop started")

        # Calculate loop interval based on timeframe
        timeframe_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
        }
        interval = timeframe_seconds.get(self.timeframe, 900)
        check_interval = min(interval / 6, 60)  # Check more frequently than candle period

        while self._running:
            try:
                # Update data feed
                new_candle = await self.data_feed.update()
                if new_candle:
                    self._bar_counter += 1
                    logger.debug(f"New candle: {new_candle.close}")

                # Handle state-specific logic
                state = self.state_machine.state

                if state == TradingState.SCANNING:
                    await self._handle_scanning()

                elif state == TradingState.IN_POSITION:
                    await self._handle_in_position()

                elif state == TradingState.COOLDOWN:
                    self.state_machine.check_cooldown_complete()

                elif state == TradingState.STOPPING:
                    if not self.state_machine.has_position:
                        break

                # Take equity snapshot periodically
                await self._maybe_take_equity_snapshot()

                # Wait before next iteration
                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self.state_machine.set_error(str(e))
                await asyncio.sleep(60)  # Wait before retrying

        logger.info("Trading loop ended")

    async def _handle_scanning(self) -> None:
        """Handle SCANNING state - look for entry signals."""
        df = self.data_feed.get_dataframe()
        if df.empty:
            return

        # Check for entry signal
        signal = self.strategy.check_entry_signal(df, self.symbol)

        if signal.is_entry:
            # Record signal
            await self._record_signal(signal)
            await self._notify_signal(signal)

            # Calculate position size
            position_size = await self._calculate_position_size()
            if position_size <= 0:
                logger.warning("Insufficient balance for trade")
                return

            # Enter position
            success = await self._enter_position(signal, position_size)
            if success:
                await self._notify_state_change()

    async def _handle_in_position(self) -> None:
        """Handle IN_POSITION state - monitor for exits."""
        current_price = self.data_feed.get_latest_price()
        if current_price is None:
            return

        # Check exit signal
        exit_signal = self.strategy.check_exit_signal(current_price, self._bar_counter)

        if exit_signal.should_exit:
            await self._close_current_position(exit_signal.exit_reason)
            await self._notify_state_change()

    async def _calculate_position_size(self) -> float:
        """
        Calculate position size based on account balance and risk parameters.

        Returns:
            Position size in USD
        """
        try:
            balance = await self.executor.get_balance()
            if balance <= 0:
                return 0.0

            # Risk-based position sizing
            # Position size = (balance * risk%) / stop_loss%
            risk_amount = balance * self.risk_per_trade
            stop_distance = self.strategy.stop_loss_pct

            position_size = risk_amount / stop_distance

            # Apply leverage cap
            max_position = balance * self.max_leverage
            position_size = min(position_size, max_position)

            # Minimum position size
            if position_size < 10:
                return 0.0

            logger.info(
                f"Calculated position size: ${position_size:.2f} "
                f"(balance: ${balance:.2f}, risk: {self.risk_per_trade*100}%)"
            )
            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    async def _enter_position(self, signal: Signal, position_size: float) -> bool:
        """
        Enter a trading position.

        Args:
            signal: Entry signal
            position_size: Position size in USD

        Returns:
            True if successful
        """
        try:
            side = signal.side
            if not side:
                return False

            # Calculate order size (contracts)
            current_price = signal.price
            order_size = position_size / current_price

            # Place market order
            order_result = await self.executor.place_order(
                symbol=self.symbol,
                side="buy" if side == "long" else "sell",
                size=order_size,
                order_type="market",
            )

            if not order_result:
                logger.error("Failed to place entry order")
                return False

            # Record trade in database
            async with get_db_session() as session:
                trade_repo = TradeRepository(session)
                trade = await trade_repo.create(
                    symbol=self.symbol,
                    side=side,
                    entry_price=Decimal(str(current_price)),
                    size=Decimal(str(position_size)),
                    entry_time=datetime.utcnow(),
                    divergence_type=signal.signal_type,
                    rsi_at_entry=Decimal(str(signal.rsi)),
                )
                self._current_trade_id = trade.id

            # Update strategy state
            self.strategy.enter_position(current_price, side, self._bar_counter)

            # Update state machine
            self.state_machine.enter_position(
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                size=position_size,
                trade_id=self._current_trade_id,
            )

            logger.info(f"Entered {side} position: {self.symbol} @ {current_price:.2f}, size: ${position_size:.2f}")
            await self._notify_trade("entry", signal.price, position_size, side)
            return True

        except Exception as e:
            logger.error(f"Error entering position: {e}")
            return False

    async def _close_current_position(self, exit_reason: str) -> bool:
        """
        Close the current position.

        Args:
            exit_reason: Reason for closing

        Returns:
            True if successful
        """
        try:
            context = self.state_machine.context
            if not context.position_symbol:
                return False

            # Get current price
            current_price = self.data_feed.get_latest_price()
            if current_price is None:
                return False

            # Get position from exchange
            positions = await self.executor.get_positions()
            position = next((p for p in positions if p["symbol"] == self.symbol), None)

            if position:
                # Close position on exchange
                close_side = "sell" if context.position_side == "long" else "buy"
                order_result = await self.executor.place_order(
                    symbol=self.symbol,
                    side=close_side,
                    size=abs(position["size"]),
                    order_type="market",
                    reduce_only=True,
                )

                if not order_result:
                    logger.error("Failed to close position on exchange")

            # Calculate PnL
            entry_price = context.position_entry_price
            if context.position_side == "long":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            pnl = context.position_size * pnl_pct

            # Update trade in database
            if self._current_trade_id:
                async with get_db_session() as session:
                    trade_repo = TradeRepository(session)
                    await trade_repo.close_trade(
                        trade_id=self._current_trade_id,
                        exit_price=Decimal(str(current_price)),
                        exit_time=datetime.utcnow(),
                        pnl=Decimal(str(pnl)),
                        pnl_pct=Decimal(str(pnl_pct * 100)),
                        fees=Decimal("0"),  # TODO: Calculate actual fees
                        exit_reason=exit_reason,
                    )

            # Update strategy state
            self.strategy.exit_position()

            # Update state machine
            self.state_machine.exit_position(pnl, self.cooldown_minutes)

            self._current_trade_id = None

            logger.info(
                f"Closed position: {self.symbol} @ {current_price:.2f}, "
                f"PnL: ${pnl:.2f} ({pnl_pct*100:.2f}%), reason: {exit_reason}"
            )
            await self._notify_trade("exit", current_price, context.position_size, context.position_side, pnl)
            return True

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    async def _record_signal(self, signal: Signal) -> None:
        """Record a signal to the database."""
        try:
            async with get_db_session() as session:
                signal_repo = SignalRepository(session)
                await signal_repo.create(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type,
                    rsi_value=Decimal(str(signal.rsi)),
                    price=Decimal(str(signal.price)),
                    timestamp=signal.timestamp,
                    confidence=Decimal(str(signal.confidence)) if signal.confidence else None,
                )
        except Exception as e:
            logger.error(f"Error recording signal: {e}")

    async def _maybe_take_equity_snapshot(self) -> None:
        """Take equity snapshot if interval has passed."""
        now = datetime.utcnow()

        if self._last_equity_snapshot:
            elapsed = (now - self._last_equity_snapshot).total_seconds() / 60
            if elapsed < self.equity_snapshot_interval:
                return

        try:
            balance = await self.executor.get_balance()
            if balance <= 0:
                return

            # Calculate drawdown (simplified)
            context = self.state_machine.context
            peak_equity = balance  # TODO: Track actual peak
            drawdown = 0.0

            async with get_db_session() as session:
                equity_repo = EquityRepository(session)
                await equity_repo.create(
                    timestamp=now,
                    equity=Decimal(str(balance)),
                    drawdown=Decimal(str(drawdown)),
                    daily_pnl=Decimal(str(context.daily_pnl)),
                )

            self._last_equity_snapshot = now

        except Exception as e:
            logger.error(f"Error taking equity snapshot: {e}")

    # Event registration
    def on_state_change(self, callback: Callable) -> None:
        """Register callback for state changes."""
        self._on_state_change.append(callback)

    def on_trade(self, callback: Callable) -> None:
        """Register callback for trade events."""
        self._on_trade.append(callback)

    def on_signal(self, callback: Callable) -> None:
        """Register callback for signal events."""
        self._on_signal.append(callback)

    async def _notify_state_change(self) -> None:
        """Notify listeners of state change."""
        data = self.get_status()
        for callback in self._on_state_change:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

    async def _notify_trade(
        self,
        event_type: str,
        price: float,
        size: float,
        side: str,
        pnl: float = 0.0,
    ) -> None:
        """Notify listeners of trade event."""
        data = {
            "type": event_type,
            "symbol": self.symbol,
            "price": price,
            "size": size,
            "side": side,
            "pnl": pnl,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for callback in self._on_trade:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")

    async def _notify_signal(self, signal: Signal) -> None:
        """Notify listeners of signal event."""
        data = {
            "type": signal.signal_type,
            "symbol": signal.symbol,
            "price": signal.price,
            "rsi": signal.rsi,
            "confidence": signal.confidence,
            "timestamp": signal.timestamp.isoformat(),
        }
        for callback in self._on_signal:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")

    def get_status(self) -> dict:
        """Get current engine status."""
        return {
            "running": self._running,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "state": self.state_machine.state.name,
            "context": self.state_machine.context.to_dict(),
            "data_feed": {
                "ready": self.data_feed.is_ready,
                "candle_count": self.data_feed.candle_count,
                "latest_price": self.data_feed.get_latest_price(),
            },
            "strategy": {
                "has_position": self.strategy.has_position,
                "position_info": self.strategy.position_info,
            },
            "config": {
                "risk_per_trade": self.risk_per_trade,
                "max_leverage": self.max_leverage,
                "cooldown_minutes": self.cooldown_minutes,
                "network": self.network,
            },
        }

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    @property
    def state(self) -> TradingState:
        """Get current trading state."""
        return self.state_machine.state
