"""WebSocket connection manager for real-time updates."""

import asyncio
import json
import logging
from typing import List, Dict, Any, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time broadcasting.

    Supports:
    - Multiple concurrent connections
    - Message broadcasting to all clients
    - Automatic connection cleanup
    - Message queuing for reliability
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
        """
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

        # Send initial state
        await self.send_personal_message(
            {
                "type": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Connected to Kronos trading system",
            },
            websocket,
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket to remove
        """
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket) -> bool:
        """
        Send a message to a specific client.

        Args:
            message: Message to send (will be JSON encoded)
            websocket: Target WebSocket

        Returns:
            True if sent successfully
        """
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            return False

    async def broadcast(self, message: Dict[str, Any]) -> int:
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast (will be JSON encoded)

        Returns:
            Number of clients successfully sent to
        """
        if not self.active_connections:
            return 0

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        sent_count = 0
        disconnected = []

        async with self._lock:
            connections = list(self.active_connections)

        for websocket in connections:
            try:
                await websocket.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            await self.disconnect(ws)

        return sent_count

    async def broadcast_state_update(self, state: Dict[str, Any]) -> int:
        """Broadcast a state update to all clients."""
        return await self.broadcast({
            "type": "state_update",
            "data": state,
        })

    async def broadcast_trade(self, trade_data: Dict[str, Any]) -> int:
        """Broadcast a trade event to all clients."""
        return await self.broadcast({
            "type": "trade",
            "data": trade_data,
        })

    async def broadcast_signal(self, signal_data: Dict[str, Any]) -> int:
        """Broadcast a signal event to all clients."""
        return await self.broadcast({
            "type": "signal",
            "data": signal_data,
        })

    async def broadcast_price(self, symbol: str, price: float) -> int:
        """Broadcast a price update to all clients."""
        return await self.broadcast({
            "type": "price",
            "data": {
                "symbol": symbol,
                "price": price,
            },
        })

    async def broadcast_error(self, error: str) -> int:
        """Broadcast an error to all clients."""
        return await self.broadcast({
            "type": "error",
            "data": {
                "message": error,
            },
        })

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)


# Global connection manager instance
ws_manager = ConnectionManager()
