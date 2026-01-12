"""
Configuration management for Kronos Trading System.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override with environment variables
    config = _override_with_env(config)

    return config


def _override_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """Override config values with environment variables."""
    # Binance (optional - public endpoints work without auth)
    if os.getenv("BINANCE_API_KEY"):
        config["data"]["sources"]["binance"]["api_key"] = os.getenv("BINANCE_API_KEY")

    # Hyperliquid (wallet-based auth)
    if os.getenv("HYPERLIQUID_WALLET_ADDRESS"):
        config["data"]["sources"]["hyperliquid"]["wallet_address"] = os.getenv("HYPERLIQUID_WALLET_ADDRESS")
    if os.getenv("HYPERLIQUID_API_WALLET_ADDRESS"):
        config["data"]["sources"]["hyperliquid"]["api_wallet_address"] = os.getenv("HYPERLIQUID_API_WALLET_ADDRESS")
    if os.getenv("HYPERLIQUID_PRIVATE_KEY"):
        config["data"]["sources"]["hyperliquid"]["private_key"] = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    if os.getenv("HYPERLIQUID_NETWORK"):
        config["data"]["sources"]["hyperliquid"]["network"] = os.getenv("HYPERLIQUID_NETWORK")

    return config


@dataclass
class DataConfig:
    """Data configuration."""
    symbols: list = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    timeframes: Dict[str, str] = field(default_factory=lambda: {
        "regime": "4h",
        "signal": "30m",
        "execution": "5m"
    })
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/cache"


@dataclass
class KronosConfig:
    """Kronos model configuration."""
    model_path: str = "Kronos/model/base/model.safetensors"
    context_length: int = 512
    device: str = "cuda"
    frozen: bool = True
    batch_size: int = 32


@dataclass
class RiskConfig:
    """Risk management configuration."""
    initial_capital: float = 10000.0
    max_position_size: float = 0.25
    max_leverage: float = 3.0
    volatility_target: float = 0.15
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.03


@dataclass
class TradingConfig:
    """Complete trading configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    kronos: KronosConfig = field(default_factory=KronosConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TradingConfig":
        """Create TradingConfig from dictionary."""
        data_config = DataConfig(
            symbols=config_dict.get("data", {}).get("symbols", []),
            timeframes=config_dict.get("data", {}).get("timeframes", {}),
            raw_dir=config_dict.get("data", {}).get("storage", {}).get("raw_dir", "data/raw"),
            processed_dir=config_dict.get("data", {}).get("storage", {}).get("processed_dir", "data/processed"),
            cache_dir=config_dict.get("data", {}).get("storage", {}).get("cache_dir", "data/cache"),
        )

        kronos_config = KronosConfig(
            model_path=config_dict.get("kronos", {}).get("model_path", ""),
            context_length=config_dict.get("kronos", {}).get("context_length", 512),
            device=config_dict.get("kronos", {}).get("device", "cuda"),
            frozen=config_dict.get("kronos", {}).get("frozen", True),
            batch_size=config_dict.get("kronos", {}).get("batch_size", 32),
        )

        risk_config = RiskConfig(
            initial_capital=config_dict.get("risk", {}).get("initial_capital", 10000.0),
            max_position_size=config_dict.get("risk", {}).get("max_position_size", 0.25),
            max_leverage=config_dict.get("risk", {}).get("max_leverage", 3.0),
            volatility_target=config_dict.get("risk", {}).get("volatility_target", 0.15),
            max_drawdown=config_dict.get("risk", {}).get("max_drawdown", 0.15),
            daily_loss_limit=config_dict.get("risk", {}).get("daily_loss_limit", 0.03),
        )

        return cls(data=data_config, kronos=kronos_config, risk=risk_config)

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "TradingConfig":
        """Load TradingConfig from YAML file."""
        config_dict = load_config(config_path)
        return cls.from_dict(config_dict)
