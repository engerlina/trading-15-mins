"""SQLAlchemy models for Kronos trading system."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Numeric,
    DateTime,
    Boolean,
    Index,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Trade(Base):
    """Trade record - both open and closed trades."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # 'long' or 'short'

    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    size: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)  # Position size in USD

    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    pnl_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    fees: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))

    exit_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    divergence_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    rsi_at_entry: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)

    status: Mapped[str] = mapped_column(String(20), default="open")  # 'open', 'closed'

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_trades_symbol", "symbol"),
        Index("idx_trades_status", "status"),
        Index("idx_trades_entry_time", "entry_time"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": float(self.entry_price) if self.entry_price else None,
            "exit_price": float(self.exit_price) if self.exit_price else None,
            "size": float(self.size) if self.size else None,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": float(self.pnl) if self.pnl else None,
            "pnl_pct": float(self.pnl_pct) if self.pnl_pct else None,
            "fees": float(self.fees) if self.fees else 0,
            "exit_reason": self.exit_reason,
            "divergence_type": self.divergence_type,
            "rsi_at_entry": float(self.rsi_at_entry) if self.rsi_at_entry else None,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Signal(Base):
    """Detected trading signals."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    signal_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'bullish_divergence', etc.

    rsi_value: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    acted_upon: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_signals_timestamp", "timestamp"),
        Index("idx_signals_symbol", "symbol"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "rsi_value": float(self.rsi_value) if self.rsi_value else None,
            "price": float(self.price) if self.price else None,
            "confidence": float(self.confidence) if self.confidence else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "acted_upon": self.acted_upon,
        }


class EquitySnapshot(Base):
    """Periodic equity tracking for performance monitoring."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    equity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    drawdown: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    daily_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)
    total_exposure: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_equity_timestamp", "timestamp"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "equity": float(self.equity) if self.equity else None,
            "drawdown": float(self.drawdown) if self.drawdown else None,
            "daily_pnl": float(self.daily_pnl) if self.daily_pnl else None,
            "total_exposure": float(self.total_exposure) if self.total_exposure else None,
        }
