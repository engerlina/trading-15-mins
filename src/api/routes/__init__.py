"""API routes for Kronos trading system."""

from .health import router as health_router
from .status import router as status_router
from .trades import router as trades_router
from .control import router as control_router

__all__ = [
    "health_router",
    "status_router",
    "trades_router",
    "control_router",
]
