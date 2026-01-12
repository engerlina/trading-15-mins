"""
Kronos Trading System
=====================
A regime-aware AI trading system for crypto perpetuals.

Architecture:
- Layer 1: Market Data (OHLCV, funding, volatility)
- Layer 2: Kronos (regime detection + distribution modeling)
- Layer 3: Signal Model (XGBoost on Kronos embeddings)
- Layer 4: Risk Engine (position sizing, drawdown limits)
"""

__version__ = "0.1.0"
__author__ = "Kronos Trading"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
