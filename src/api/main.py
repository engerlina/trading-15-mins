"""FastAPI application for Kronos trading system."""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routes import health_router, status_router, trades_router, control_router
from .websocket import ws_manager
from ..trading.engine import TradingEngine
from ..database.connection import init_db, close_db

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global trading engine instance
_trading_engine: Optional[TradingEngine] = None


def get_trading_engine() -> Optional[TradingEngine]:
    """Get the trading engine instance."""
    return _trading_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global _trading_engine

    logger.info("Starting Kronos Trading System...")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Continue anyway for demo purposes
        logger.warning("Continuing without database...")

    # Initialize trading engine
    _trading_engine = TradingEngine(
        symbol=os.getenv("TRADING_SYMBOL", "BTC"),
        timeframe=os.getenv("TRADING_TIMEFRAME", "15m"),
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.02")),
        max_leverage=float(os.getenv("MAX_LEVERAGE", "2.0")),
        cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "30")),
        network=os.getenv("HYPERLIQUID_NETWORK", "mainnet"),
    )

    # Register callbacks for WebSocket broadcasting
    _trading_engine.on_state_change(lambda data: ws_manager.broadcast_state_update(data))
    _trading_engine.on_trade(lambda data: ws_manager.broadcast_trade(data))
    _trading_engine.on_signal(lambda data: ws_manager.broadcast_signal(data))

    logger.info(f"Trading engine initialized: {_trading_engine.symbol} {_trading_engine.timeframe}")

    # Auto-start if configured
    if os.getenv("AUTO_START_TRADING", "false").lower() == "true":
        logger.info("Auto-starting trading engine...")
        await _trading_engine.start()

    yield

    # Shutdown
    logger.info("Shutting down Kronos Trading System...")

    if _trading_engine and _trading_engine.is_running:
        await _trading_engine.stop(close_positions=True)

    await close_db()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Kronos Trading System",
    description="RSI Divergence trading bot with real-time monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(status_router)
app.include_router(trades_router)
app.include_router(control_router)

# Mount static files for dashboard
static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def root():
    """Serve the dashboard."""
    index_path = os.path.join(static_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Kronos Trading System API",
        "docs": "/docs",
        "health": "/health",
        "status": "/status",
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Broadcasts:
    - state_update: Trading state changes
    - trade: Trade entry/exit events
    - signal: Detected signals
    - price: Price updates (if enabled)
    """
    await ws_manager.connect(websocket)
    try:
        # Send current status on connect
        if _trading_engine:
            await websocket.send_json({
                "type": "initial_state",
                "data": _trading_engine.get_status(),
            })

        while True:
            # Keep connection alive and handle client messages
            try:
                data = await websocket.receive_text()
                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    finally:
        await ws_manager.disconnect(websocket)


# API info endpoint
@app.get("/api/info")
async def api_info():
    """Get API information."""
    return {
        "name": "Kronos Trading System",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "trades": "/trades",
            "control": {
                "start": "POST /control/start",
                "stop": "POST /control/stop",
                "restart": "POST /control/restart",
                "close_position": "POST /control/close-position",
            },
            "websocket": "/ws",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )
