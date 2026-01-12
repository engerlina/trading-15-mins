"""Database repository classes for CRUD operations."""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from sqlalchemy import select, update, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Trade, Signal, EquitySnapshot


class TradeRepository:
    """Repository for Trade operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        size: Decimal,
        entry_time: datetime,
        divergence_type: Optional[str] = None,
        rsi_at_entry: Optional[Decimal] = None,
    ) -> Trade:
        """Create a new trade."""
        trade = Trade(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            entry_time=entry_time,
            divergence_type=divergence_type,
            rsi_at_entry=rsi_at_entry,
            status="open",
        )
        self.session.add(trade)
        await self.session.flush()
        await self.session.refresh(trade)
        return trade

    async def close_trade(
        self,
        trade_id: int,
        exit_price: Decimal,
        exit_time: datetime,
        pnl: Decimal,
        pnl_pct: Decimal,
        fees: Decimal,
        exit_reason: str,
    ) -> Optional[Trade]:
        """Close an open trade."""
        stmt = (
            update(Trade)
            .where(Trade.id == trade_id)
            .values(
                exit_price=exit_price,
                exit_time=exit_time,
                pnl=pnl,
                pnl_pct=pnl_pct,
                fees=fees,
                exit_reason=exit_reason,
                status="closed",
            )
            .returning(Trade)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_open_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        """Get all open trades, optionally filtered by symbol."""
        stmt = select(Trade).where(Trade.status == "open")
        if symbol:
            stmt = stmt.where(Trade.symbol == symbol)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_trade_by_id(self, trade_id: int) -> Optional[Trade]:
        """Get trade by ID."""
        stmt = select(Trade).where(Trade.id == trade_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_recent_trades(
        self,
        limit: int = 50,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Trade]:
        """Get recent trades with optional filters."""
        stmt = select(Trade).order_by(desc(Trade.entry_time)).limit(limit)
        if symbol:
            stmt = stmt.where(Trade.symbol == symbol)
        if status:
            stmt = stmt.where(Trade.status == status)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_trade_stats(self, symbol: Optional[str] = None) -> dict:
        """Get trade statistics."""
        stmt = select(Trade).where(Trade.status == "closed")
        if symbol:
            stmt = stmt.where(Trade.symbol == symbol)
        result = await self.session.execute(stmt)
        trades = list(result.scalars().all())

        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
            }

        winners = [t for t in trades if t.pnl and t.pnl > 0]
        losers = [t for t in trades if t.pnl and t.pnl < 0]

        total_pnl = sum(float(t.pnl) for t in trades if t.pnl)
        total_wins = sum(float(t.pnl) for t in winners)
        total_losses = abs(sum(float(t.pnl) for t in losers))

        return {
            "total_trades": len(trades),
            "win_rate": len(winners) / len(trades) * 100 if trades else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(trades) if trades else 0.0,
            "avg_win": total_wins / len(winners) if winners else 0.0,
            "avg_loss": -total_losses / len(losers) if losers else 0.0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
        }


class SignalRepository:
    """Repository for Signal operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        symbol: str,
        signal_type: str,
        rsi_value: Decimal,
        price: Decimal,
        timestamp: datetime,
        confidence: Optional[Decimal] = None,
    ) -> Signal:
        """Create a new signal."""
        signal = Signal(
            symbol=symbol,
            signal_type=signal_type,
            rsi_value=rsi_value,
            price=price,
            timestamp=timestamp,
            confidence=confidence,
            acted_upon=False,
        )
        self.session.add(signal)
        await self.session.flush()
        await self.session.refresh(signal)
        return signal

    async def mark_acted_upon(self, signal_id: int) -> Optional[Signal]:
        """Mark a signal as acted upon."""
        stmt = (
            update(Signal)
            .where(Signal.id == signal_id)
            .values(acted_upon=True)
            .returning(Signal)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_recent_signals(
        self,
        limit: int = 50,
        symbol: Optional[str] = None,
    ) -> List[Signal]:
        """Get recent signals."""
        stmt = select(Signal).order_by(desc(Signal.timestamp)).limit(limit)
        if symbol:
            stmt = stmt.where(Signal.symbol == symbol)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_unacted_signals(self, symbol: Optional[str] = None) -> List[Signal]:
        """Get signals that haven't been acted upon."""
        stmt = select(Signal).where(Signal.acted_upon == False).order_by(desc(Signal.timestamp))
        if symbol:
            stmt = stmt.where(Signal.symbol == symbol)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class EquityRepository:
    """Repository for EquitySnapshot operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        timestamp: datetime,
        equity: Decimal,
        drawdown: Decimal,
        daily_pnl: Optional[Decimal] = None,
        total_exposure: Optional[Decimal] = None,
    ) -> EquitySnapshot:
        """Create a new equity snapshot."""
        snapshot = EquitySnapshot(
            timestamp=timestamp,
            equity=equity,
            drawdown=drawdown,
            daily_pnl=daily_pnl,
            total_exposure=total_exposure,
        )
        self.session.add(snapshot)
        await self.session.flush()
        await self.session.refresh(snapshot)
        return snapshot

    async def get_recent_snapshots(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[EquitySnapshot]:
        """Get recent equity snapshots."""
        stmt = select(EquitySnapshot).order_by(desc(EquitySnapshot.timestamp)).limit(limit)
        if since:
            stmt = stmt.where(EquitySnapshot.timestamp >= since)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest(self) -> Optional[EquitySnapshot]:
        """Get the latest equity snapshot."""
        stmt = select(EquitySnapshot).order_by(desc(EquitySnapshot.timestamp)).limit(1)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_equity_curve(
        self,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[dict]:
        """Get equity curve data for charting."""
        stmt = select(EquitySnapshot).order_by(EquitySnapshot.timestamp).limit(limit)
        if since:
            stmt = stmt.where(EquitySnapshot.timestamp >= since)
        result = await self.session.execute(stmt)
        snapshots = list(result.scalars().all())

        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "equity": float(s.equity),
                "drawdown": float(s.drawdown),
            }
            for s in snapshots
        ]
