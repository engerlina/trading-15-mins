"""
Data storage utilities for Kronos Trading System.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
import pickle
import logging

from ..utils.logger import get_logger

logger = get_logger("data_storage")


class DataStorage:
    """
    Manage data storage and retrieval for the trading system.

    Supports:
    - Parquet for OHLCV data
    - JSON for metadata
    - Pickle for model artifacts
    """

    def __init__(
        self,
        base_dir: str = "data",
        raw_subdir: str = "raw",
        processed_subdir: str = "processed",
        cache_subdir: str = "cache"
    ):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / raw_subdir
        self.processed_dir = self.base_dir / processed_subdir
        self.cache_dir = self.base_dir / cache_subdir

        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        data_type: str = "raw"
    ) -> Path:
        """
        Save OHLCV data to parquet.

        Args:
            df: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Data timeframe
            data_type: 'raw' or 'processed'

        Returns:
            Path to saved file
        """
        dir_path = self.raw_dir if data_type == "raw" else self.processed_dir
        file_path = dir_path / f"{symbol}_{timeframe}.parquet"

        df.to_parquet(file_path, index=False, compression="snappy")
        logger.info(f"Saved {len(df)} rows to {file_path}")

        return file_path

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        data_type: str = "raw",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from parquet.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            data_type: 'raw' or 'processed'
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            OHLCV DataFrame
        """
        dir_path = self.raw_dir if data_type == "raw" else self.processed_dir
        file_path = dir_path / f"{symbol}_{timeframe}.parquet"

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()

        df = pd.read_parquet(file_path)

        # Apply date filters
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df["timestamp"] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df["timestamp"] <= end_dt]

        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df

    def save_processed(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Path:
        """Save fully processed data."""
        file_path = self.processed_dir / f"{symbol}_processed.parquet"
        df.to_parquet(file_path, index=False, compression="snappy")
        logger.info(f"Saved processed data to {file_path}")
        return file_path

    def load_processed(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load processed data."""
        file_path = self.processed_dir / f"{symbol}_processed.parquet"

        if not file_path.exists():
            logger.warning(f"Processed file not found: {file_path}")
            return pd.DataFrame()

        df = pd.read_parquet(file_path)

        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]

        return df

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        symbol: str,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save Kronos embeddings to cache.

        Args:
            embeddings: NumPy array of embeddings
            symbol: Trading symbol
            metadata: Optional metadata dict

        Returns:
            Path to saved file
        """
        file_path = self.cache_dir / f"{symbol}_embeddings.npz"

        np.savez_compressed(
            file_path,
            embeddings=embeddings,
            **({} if metadata is None else {"metadata": np.array([json.dumps(metadata)])})
        )

        logger.info(f"Saved embeddings shape {embeddings.shape} to {file_path}")
        return file_path

    def load_embeddings(self, symbol: str) -> tuple:
        """
        Load Kronos embeddings from cache.

        Returns:
            Tuple of (embeddings, metadata)
        """
        file_path = self.cache_dir / f"{symbol}_embeddings.npz"

        if not file_path.exists():
            logger.warning(f"Embeddings not found: {file_path}")
            return None, None

        data = np.load(file_path, allow_pickle=True)
        embeddings = data["embeddings"]

        metadata = None
        if "metadata" in data:
            metadata = json.loads(str(data["metadata"][0]))

        return embeddings, metadata

    def save_model(
        self,
        model: object,
        name: str,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save a trained model.

        Args:
            model: Model object (XGBoost, etc.)
            name: Model name
            metadata: Optional metadata

        Returns:
            Path to saved model
        """
        models_dir = self.base_dir.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / f"{name}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "metadata": metadata}, f)

        logger.info(f"Saved model to {model_path}")
        return model_path

    def load_model(self, name: str) -> tuple:
        """
        Load a trained model.

        Returns:
            Tuple of (model, metadata)
        """
        models_dir = self.base_dir.parent / "models"
        model_path = models_dir / f"{name}.pkl"

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None, None

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        return data["model"], data.get("metadata")

    def save_backtest_results(
        self,
        results: Dict,
        name: str
    ) -> Path:
        """Save backtest results."""
        results_dir = self.base_dir.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = results_dir / f"{name}_{timestamp}.json"

        # Convert numpy/pandas types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()  # Convert Series to dict
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()  # Convert DataFrame to dict
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return str(obj)  # Fallback to string for other types

        results_clean = json.loads(
            json.dumps(results, default=convert_numpy)
        )

        with open(file_path, "w") as f:
            json.dump(results_clean, f, indent=2)

        logger.info(f"Saved backtest results to {file_path}")
        return file_path

    def list_available_data(self) -> Dict[str, List[str]]:
        """List all available data files."""
        result = {
            "raw": [],
            "processed": [],
            "cache": []
        }

        for f in self.raw_dir.glob("*.parquet"):
            result["raw"].append(f.stem)

        for f in self.processed_dir.glob("*.parquet"):
            result["processed"].append(f.stem)

        for f in self.cache_dir.glob("*.npz"):
            result["cache"].append(f.stem)

        return result

    def get_data_info(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get information about stored data."""
        file_path = self.raw_dir / f"{symbol}_{timeframe}.parquet"

        if not file_path.exists():
            return None

        df = pd.read_parquet(file_path)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": len(df),
            "start_date": df["timestamp"].min().isoformat() if not df.empty else None,
            "end_date": df["timestamp"].max().isoformat() if not df.empty else None,
            "columns": list(df.columns),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024)
        }

    def clear_cache(self):
        """Clear the cache directory."""
        for f in self.cache_dir.glob("*"):
            f.unlink()
        logger.info("Cleared cache directory")
