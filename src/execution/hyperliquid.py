"""
Hyperliquid execution module.

For live trading execution on Hyperliquid exchange.
"""

import asyncio
import httpx
import hmac
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from ..utils.logger import get_logger

logger = get_logger("hyperliquid_executor")


class HyperliquidExecutor:
    """
    Hyperliquid exchange execution handler.

    Handles:
    - Order placement (limit, market)
    - Position management
    - Account info retrieval

    Uses Ethereum wallet-based authentication (not traditional API key/secret).
    """

    def __init__(
        self,
        wallet_address: Optional[str] = None,
        api_wallet_address: Optional[str] = None,
        private_key: Optional[str] = None,
        network: str = "mainnet",
        base_url: Optional[str] = None
    ):
        """
        Initialize Hyperliquid executor.

        Args:
            wallet_address: Main wallet address (with funds)
            api_wallet_address: API wallet address (vault)
            private_key: Private key for signing transactions
            network: 'mainnet' or 'testnet'
            base_url: API base URL (auto-set based on network)
        """
        # Load from environment if not provided
        self.wallet_address = wallet_address or os.getenv("HYPERLIQUID_WALLET_ADDRESS")
        self.api_wallet_address = api_wallet_address or os.getenv("HYPERLIQUID_API_WALLET_ADDRESS")
        self.private_key = private_key or os.getenv("HYPERLIQUID_PRIVATE_KEY")
        self.network = network or os.getenv("HYPERLIQUID_NETWORK", "mainnet")

        # Set base URL based on network
        if base_url:
            self.base_url = base_url
        elif self.network == "testnet":
            self.base_url = "https://api.hyperliquid-testnet.xyz"
        else:
            self.base_url = "https://api.hyperliquid.xyz"

        self._client: Optional[httpx.AsyncClient] = None
        self.account = None
        self._eth_account_available = False

        # Initialize signing account if private key provided (lazy import)
        if self.private_key:
            try:
                from eth_account import Account
                pk = self.private_key if self.private_key.startswith("0x") else f"0x{self.private_key}"
                self.account = Account.from_key(pk)
                self._eth_account_available = True
                logger.info(f"Initialized Hyperliquid executor for wallet {self.wallet_address[:10]}...")
            except ImportError:
                logger.warning("eth-account not installed - order signing will not work")
            except Exception as e:
                logger.warning(f"Could not initialize signing account: {e}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_account_info(self) -> Optional[Dict]:
        """Get account information."""
        client = await self._get_client()

        user_address = self.api_wallet_address or self.wallet_address
        logger.debug(f"Fetching account info for wallet: {user_address}")

        payload = {
            "type": "clearinghouseState",
            "user": user_address
        }

        try:
            response = await client.post(f"{self.base_url}/info", json=payload)
            if response.status_code != 200:
                logger.error(f"Failed to get account info: {response.text}")
                return None

            data = response.json()
            logger.debug(f"Account info response: {data}")
            return data

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    async def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        account_info = await self.get_account_info()

        if not account_info:
            return []

        positions = []
        for position in account_info.get("assetPositions", []):
            pos_info = position.get("position", {})
            if float(pos_info.get("szi", 0)) != 0:
                positions.append({
                    "symbol": pos_info.get("coin"),
                    "size": float(pos_info.get("szi", 0)),
                    "entry_price": float(pos_info.get("entryPx", 0)),
                    "unrealized_pnl": float(pos_info.get("unrealizedPnl", 0)),
                    "margin": float(pos_info.get("positionValue", 0)),
                })

        return positions

    async def get_balance(self) -> float:
        """Get account balance."""
        logger.info(f"Fetching balance from Hyperliquid ({self.network})...")
        account_info = await self.get_account_info()

        if not account_info:
            logger.warning("No account info returned - balance is 0")
            return 0.0

        margin_summary = account_info.get("marginSummary", {})
        balance = float(margin_summary.get("accountValue", 0))
        logger.info(f"Balance retrieved: ${balance:.2f} (withdrawable: ${float(margin_summary.get('withdrawable', 0)):.2f})")
        return balance

    async def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        order_type: str = "limit",
        reduce_only: bool = False,
        time_in_force: str = "GTC"
    ) -> Optional[Dict]:
        """
        Place an order on Hyperliquid.

        Args:
            symbol: Asset symbol (e.g., 'BTC')
            side: 'buy' or 'sell'
            size: Order size in contracts
            price: Limit price (None for market)
            order_type: 'limit' or 'market'
            reduce_only: If True, only reduce position
            time_in_force: GTC, IOC, etc.

        Returns:
            Order response or None on failure
        """
        if not self.private_key:
            logger.error("Private key not configured")
            return None

        client = await self._get_client()

        # Build order payload
        is_buy = side.lower() == "buy"

        order = {
            "a": self._get_asset_index(symbol),  # Asset index
            "b": is_buy,  # Is buy
            "p": str(price) if price else "0",  # Price
            "s": str(abs(size)),  # Size
            "r": reduce_only,  # Reduce only
            "t": {"limit": {"tif": time_in_force}} if order_type == "limit" else {"trigger": {"isMarket": True}}
        }

        payload = {
            "action": {
                "type": "order",
                "orders": [order],
                "grouping": "na"
            },
            "nonce": int(time.time() * 1000),
            "signature": None  # Would be computed with actual signing
        }

        # Sign request (simplified - actual implementation needs proper signing)
        payload["signature"] = self._sign_request(payload)

        try:
            response = await client.post(f"{self.base_url}/exchange", json=payload)
            if response.status_code != 200:
                logger.error(f"Order failed: {response.text}")
                return None

            data = response.json()

            if data.get("status") == "ok":
                logger.info(f"Order placed: {side} {size} {symbol} @ {price or 'market'}")
                return data

            logger.error(f"Order rejected: {data}")
            return None

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        client = await self._get_client()

        payload = {
            "action": {
                "type": "cancel",
                "cancels": [{
                    "a": self._get_asset_index(symbol),
                    "o": int(order_id)
                }]
            },
            "nonce": int(time.time() * 1000),
            "signature": None
        }

        payload["signature"] = self._sign_request(payload)

        try:
            response = await client.post(f"{self.base_url}/exchange", json=payload)
            if response.status_code == 200:
                logger.info(f"Order cancelled: {order_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """Cancel all open orders."""
        client = await self._get_client()

        payload = {
            "action": {
                "type": "cancelByCloid",
                "cancels": []
            },
            "nonce": int(time.time() * 1000),
            "signature": None
        }

        if symbol:
            payload["action"]["cancels"].append({
                "asset": self._get_asset_index(symbol)
            })

        payload["signature"] = self._sign_request(payload)

        try:
            response = await client.post(f"{self.base_url}/exchange", json=payload)
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return False

    async def close_position(
        self,
        symbol: str,
        price: Optional[float] = None
    ) -> Optional[Dict]:
        """Close a position."""
        positions = await self.get_positions()

        for pos in positions:
            if pos["symbol"] == symbol:
                size = pos["size"]
                side = "sell" if size > 0 else "buy"

                return await self.place_order(
                    symbol=symbol,
                    side=side,
                    size=abs(size),
                    price=price,
                    order_type="market" if price is None else "limit",
                    reduce_only=True
                )

        logger.warning(f"No position to close for {symbol}")
        return None

    def _get_asset_index(self, symbol: str) -> int:
        """Get asset index for a symbol."""
        # Simplified mapping - actual implementation would query exchange
        asset_map = {
            "BTC": 0,
            "ETH": 1,
            "SOL": 2,
        }
        return asset_map.get(symbol.upper().replace("USDT", "").replace("USD", ""), 0)

    def _sign_request(self, payload: Dict) -> str:
        """Sign a request (simplified)."""
        if not self.api_secret:
            return ""

        message = json.dumps(payload, separators=(",", ":"))
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature


class PaperTradingExecutor:
    """
    Paper trading executor for testing without real money.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.orders: List[Dict] = []
        self.trades: List[Dict] = []

    async def get_balance(self) -> float:
        """Get paper trading balance."""
        return self.balance

    async def get_positions(self) -> List[Dict]:
        """Get paper positions."""
        return list(self.positions.values())

    async def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        **kwargs
    ) -> Dict:
        """Place a paper order."""
        order_id = f"paper_{len(self.orders) + 1}"

        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "status": "filled",
            "timestamp": datetime.utcnow().isoformat()
        }

        self.orders.append(order)

        # Update position
        if symbol in self.positions:
            pos = self.positions[symbol]
            if side == "buy":
                pos["size"] += size
            else:
                pos["size"] -= size

            if pos["size"] == 0:
                del self.positions[symbol]
        else:
            self.positions[symbol] = {
                "symbol": symbol,
                "size": size if side == "buy" else -size,
                "entry_price": price,
                "unrealized_pnl": 0
            }

        # Update balance (simplified)
        cost = size * price * 0.0005  # Fees only
        self.balance -= cost

        logger.info(f"Paper order filled: {side} {size} {symbol} @ {price}")
        return order

    async def close_position(self, symbol: str, price: float) -> Optional[Dict]:
        """Close a paper position."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        side = "sell" if pos["size"] > 0 else "buy"

        return await self.place_order(
            symbol=symbol,
            side=side,
            size=abs(pos["size"]),
            price=price
        )
