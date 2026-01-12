"""Control endpoints for managing the trading engine."""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..main import verify_credentials

router = APIRouter(prefix="/control", tags=["control"], dependencies=[Depends(verify_credentials)])


class StartRequest(BaseModel):
    """Request to start trading."""

    symbol: Optional[str] = None
    risk_per_trade: Optional[float] = None


class StopRequest(BaseModel):
    """Request to stop trading."""

    close_positions: bool = True


class ControlResponse(BaseModel):
    """Control operation response."""

    success: bool
    message: str
    state: str
    timestamp: str


@router.post("/start", response_model=ControlResponse)
async def start_trading(request: StartRequest = None) -> ControlResponse:
    """
    Start the trading engine.

    Transitions from IDLE to SCANNING state and begins looking for signals.
    """
    from ..main import get_trading_engine

    engine = get_trading_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    if engine.is_running:
        return ControlResponse(
            success=False,
            message="Trading engine is already running",
            state=engine.state.name,
            timestamp=datetime.utcnow().isoformat(),
        )

    try:
        success = await engine.start()
        if success:
            return ControlResponse(
                success=True,
                message="Trading engine started successfully",
                state=engine.state.name,
                timestamp=datetime.utcnow().isoformat(),
            )
        else:
            return ControlResponse(
                success=False,
                message="Failed to start trading engine",
                state=engine.state.name,
                timestamp=datetime.utcnow().isoformat(),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting trading: {e}")


@router.post("/stop", response_model=ControlResponse)
async def stop_trading(request: StopRequest = None) -> ControlResponse:
    """
    Stop the trading engine.

    Optionally closes any open positions before stopping.
    """
    from ..main import get_trading_engine

    engine = get_trading_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    if not engine.is_running:
        return ControlResponse(
            success=False,
            message="Trading engine is not running",
            state=engine.state.name,
            timestamp=datetime.utcnow().isoformat(),
        )

    close_positions = request.close_positions if request else True

    try:
        success = await engine.stop(close_positions=close_positions)
        if success:
            return ControlResponse(
                success=True,
                message="Trading engine stopped successfully",
                state=engine.state.name,
                timestamp=datetime.utcnow().isoformat(),
            )
        else:
            return ControlResponse(
                success=False,
                message="Failed to stop trading engine",
                state=engine.state.name,
                timestamp=datetime.utcnow().isoformat(),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping trading: {e}")


@router.post("/restart", response_model=ControlResponse)
async def restart_trading() -> ControlResponse:
    """
    Restart the trading engine.

    Stops (with position close) and then starts again.
    """
    from ..main import get_trading_engine

    engine = get_trading_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    try:
        # Stop if running
        if engine.is_running:
            await engine.stop(close_positions=True)

        # Start
        success = await engine.start()
        if success:
            return ControlResponse(
                success=True,
                message="Trading engine restarted successfully",
                state=engine.state.name,
                timestamp=datetime.utcnow().isoformat(),
            )
        else:
            return ControlResponse(
                success=False,
                message="Failed to restart trading engine",
                state=engine.state.name,
                timestamp=datetime.utcnow().isoformat(),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restarting trading: {e}")


@router.post("/close-position", response_model=ControlResponse)
async def close_position() -> ControlResponse:
    """
    Manually close the current position.

    Forces position close regardless of exit signals.
    """
    from ..main import get_trading_engine

    engine = get_trading_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    if not engine.state_machine.has_position:
        return ControlResponse(
            success=False,
            message="No open position to close",
            state=engine.state.name,
            timestamp=datetime.utcnow().isoformat(),
        )

    try:
        success = await engine._close_current_position("manual_close")
        if success:
            return ControlResponse(
                success=True,
                message="Position closed successfully",
                state=engine.state.name,
                timestamp=datetime.utcnow().isoformat(),
            )
        else:
            return ControlResponse(
                success=False,
                message="Failed to close position",
                state=engine.state.name,
                timestamp=datetime.utcnow().isoformat(),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error closing position: {e}")


@router.post("/reset-error", response_model=ControlResponse)
async def reset_error() -> ControlResponse:
    """
    Reset from error state.

    Clears the error and transitions back to IDLE.
    """
    from ..main import get_trading_engine

    engine = get_trading_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    success = engine.state_machine.reset_error()

    return ControlResponse(
        success=success,
        message="Error state reset" if success else "Not in error state",
        state=engine.state.name,
        timestamp=datetime.utcnow().isoformat(),
    )
