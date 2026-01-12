"""
Logging utilities for Kronos Trading System.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime


_loggers = {}


def setup_logger(
    name: str = "kronos",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console: Whether to log to console
        file: Whether to log to file

    Returns:
        Configured logger
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_path / f"{name}_{timestamp}.log"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "kronos") -> logging.Logger:
    """Get an existing logger or create a new one."""
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class TradeLogger:
    """Specialized logger for trade events."""

    def __init__(self, log_dir: str = "logs"):
        self.logger = setup_logger(
            name="trades",
            level="INFO",
            log_dir=log_dir,
            console=False,
            file=True
        )

    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        reason: str = ""
    ):
        """Log a trade event."""
        msg = f"TRADE | {symbol} | {side} | qty={quantity:.4f} | price={price:.2f}"
        if pnl is not None:
            msg += f" | pnl={pnl:.2f}"
        if reason:
            msg += f" | reason={reason}"
        self.logger.info(msg)

    def log_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        regime: str
    ):
        """Log a signal event."""
        self.logger.info(
            f"SIGNAL | {symbol} | {signal} | conf={confidence:.3f} | regime={regime}"
        )

    def log_risk_event(self, event_type: str, details: str):
        """Log a risk management event."""
        self.logger.warning(f"RISK | {event_type} | {details}")
