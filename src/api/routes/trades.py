"""Trades endpoint for trade history."""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from ..main import verify_credentials

router = APIRouter(prefix="/trades", tags=["trades"], dependencies=[Depends(verify_credentials)])


class TradeResponse(BaseModel):
    """Trade response model."""

    id: int
    symbol: str
    side: str
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    size: Optional[float] = None
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fees: float = 0.0
    exit_reason: Optional[str] = None
    divergence_type: Optional[str] = None
    rsi_at_entry: Optional[float] = None
    status: str
    created_at: Optional[str] = None


class TradeStatsResponse(BaseModel):
    """Trade statistics response."""

    total_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float


class SignalResponse(BaseModel):
    """Signal response model."""

    id: int
    symbol: str
    signal_type: str
    rsi_value: Optional[float] = None
    price: Optional[float] = None
    confidence: Optional[float] = None
    timestamp: Optional[str] = None
    acted_upon: bool


@router.get("", response_model=List[TradeResponse])
async def get_trades(
    limit: int = Query(default=50, le=200),
    symbol: Optional[str] = None,
    status: Optional[str] = Query(default=None, regex="^(open|closed)$"),
) -> List[TradeResponse]:
    """
    Get trade history.

    Args:
        limit: Maximum number of trades to return
        symbol: Filter by symbol (e.g., "BTC")
        status: Filter by status ("open" or "closed")

    Returns:
        List of trades, most recent first
    """
    from ...database.connection import get_db_session
    from ...database.repository import TradeRepository

    try:
        async with get_db_session() as session:
            trade_repo = TradeRepository(session)
            trades = await trade_repo.get_recent_trades(
                limit=limit,
                symbol=symbol,
                status=status,
            )
            return [TradeResponse(**trade.to_dict()) for trade in trades]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trades: {e}")


@router.get("/open", response_model=List[TradeResponse])
async def get_open_trades(
    symbol: Optional[str] = None,
) -> List[TradeResponse]:
    """
    Get all open trades.

    Args:
        symbol: Filter by symbol (optional)

    Returns:
        List of open trades
    """
    from ...database.connection import get_db_session
    from ...database.repository import TradeRepository

    try:
        async with get_db_session() as session:
            trade_repo = TradeRepository(session)
            trades = await trade_repo.get_open_trades(symbol=symbol)
            return [TradeResponse(**trade.to_dict()) for trade in trades]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching open trades: {e}")


@router.get("/stats", response_model=TradeStatsResponse)
async def get_trade_stats(
    symbol: Optional[str] = None,
) -> TradeStatsResponse:
    """
    Get trade statistics.

    Args:
        symbol: Filter by symbol (optional)

    Returns:
        Trade statistics including win rate, PnL, profit factor
    """
    from ...database.connection import get_db_session
    from ...database.repository import TradeRepository

    try:
        async with get_db_session() as session:
            trade_repo = TradeRepository(session)
            stats = await trade_repo.get_trade_stats(symbol=symbol)
            return TradeStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trade stats: {e}")


@router.get("/{trade_id}", response_model=TradeResponse)
async def get_trade(trade_id: int) -> TradeResponse:
    """
    Get a specific trade by ID.

    Args:
        trade_id: Trade ID

    Returns:
        Trade details
    """
    from ...database.connection import get_db_session
    from ...database.repository import TradeRepository

    try:
        async with get_db_session() as session:
            trade_repo = TradeRepository(session)
            trade = await trade_repo.get_trade_by_id(trade_id)
            if trade is None:
                raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
            return TradeResponse(**trade.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trade: {e}")


@router.get("/signals/recent", response_model=List[SignalResponse])
async def get_recent_signals(
    limit: int = Query(default=50, le=200),
    symbol: Optional[str] = None,
) -> List[SignalResponse]:
    """
    Get recent signals.

    Args:
        limit: Maximum number of signals to return
        symbol: Filter by symbol (optional)

    Returns:
        List of signals, most recent first
    """
    from ...database.connection import get_db_session
    from ...database.repository import SignalRepository

    try:
        async with get_db_session() as session:
            signal_repo = SignalRepository(session)
            signals = await signal_repo.get_recent_signals(
                limit=limit,
                symbol=symbol,
            )
            return [SignalResponse(**signal.to_dict()) for signal in signals]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching signals: {e}")
