"""Status endpoint for trading engine."""

from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/status", tags=["status"])


class PositionInfo(BaseModel):
    """Position information."""

    symbol: Optional[str] = None
    side: Optional[str] = None
    entry_price: Optional[float] = None
    entry_time: Optional[str] = None
    size: Optional[float] = None
    trade_id: Optional[int] = None


class TradingContext(BaseModel):
    """Trading context information."""

    current_state: str
    last_state_change: str
    position: Optional[PositionInfo] = None
    cooldown_until: Optional[str] = None
    last_error: Optional[str] = None
    error_count: int = 0
    trades_today: int = 0
    daily_pnl: float = 0.0


class DataFeedStatus(BaseModel):
    """Data feed status."""

    ready: bool
    candle_count: int
    latest_price: Optional[float] = None


class StrategyStatus(BaseModel):
    """Strategy status."""

    has_position: bool
    position_info: dict


class ConfigInfo(BaseModel):
    """Configuration information."""

    risk_per_trade: float
    max_leverage: float
    cooldown_minutes: int
    network: str


class StatusResponse(BaseModel):
    """Full status response."""

    running: bool
    symbol: str
    timeframe: str
    state: str
    context: TradingContext
    data_feed: DataFeedStatus
    strategy: StrategyStatus
    config: ConfigInfo
    timestamp: str


class EquityPoint(BaseModel):
    """Single equity curve data point."""

    timestamp: str
    equity: float
    drawdown: float


@router.get("", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """
    Get current trading engine status.

    Returns comprehensive status including:
    - Trading state (IDLE, SCANNING, IN_POSITION, etc.)
    - Current position info
    - Data feed status
    - Configuration
    """
    # Import here to avoid circular imports
    from ..main import get_trading_engine

    engine = get_trading_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    status = engine.get_status()
    status["timestamp"] = datetime.utcnow().isoformat()

    return StatusResponse(**status)


@router.get("/equity", response_model=List[EquityPoint])
async def get_equity_curve(
    hours: int = 24,
    limit: int = 100,
) -> List[EquityPoint]:
    """
    Get equity curve data for charting.

    Args:
        hours: Number of hours of history to return
        limit: Maximum number of data points

    Returns:
        List of equity points with timestamp, equity, and drawdown
    """
    from ..main import get_trading_engine
    from ...database.connection import get_db_session
    from ...database.repository import EquityRepository

    since = datetime.utcnow() - timedelta(hours=hours)

    try:
        async with get_db_session() as session:
            equity_repo = EquityRepository(session)
            data = await equity_repo.get_equity_curve(since=since, limit=limit)
            return [EquityPoint(**point) for point in data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching equity curve: {e}")


@router.get("/balance")
async def get_balance() -> dict:
    """Get current account balance from exchange."""
    from ..main import get_trading_engine

    engine = get_trading_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    try:
        balance = await engine.executor.get_balance()
        return {
            "balance": balance,
            "currency": "USD",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching balance: {e}")


def truncate_address(address: str) -> str:
    """Truncate wallet address to show first 4 and last 4 characters."""
    if not address or len(address) < 10:
        return address or "Not configured"
    return f"{address[:6]}...{address[-4:]}"


@router.get("/wallet")
async def get_wallet_info() -> dict:
    """Get truncated wallet addresses for display."""
    from ..main import get_trading_engine

    engine = get_trading_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    executor = engine.executor
    return {
        "main_wallet": truncate_address(getattr(executor, 'wallet_address', None)),
        "api_wallet": truncate_address(getattr(executor, 'api_wallet_address', None)),
        "network": getattr(executor, 'network', 'unknown'),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/positions")
async def get_positions() -> dict:
    """Get current positions from exchange."""
    from ..main import get_trading_engine

    engine = get_trading_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    try:
        positions = await engine.executor.get_positions()
        return {
            "positions": positions,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching positions: {e}")
