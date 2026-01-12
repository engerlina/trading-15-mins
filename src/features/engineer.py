"""
Feature engineering for the signal model.

Combines Kronos embeddings with traditional features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from sklearn.decomposition import PCA

from ..utils.logger import get_logger

logger = get_logger("feature_engineer")


class FeatureEngineer:
    """
    Generate features for the signal model.

    Features include:
    - Kronos embeddings (regime representation)
    - Kronos shift alarms (perplexity, reconstruction error)
    - Forward return quantiles
    - Funding rate
    - Volatility measures
    - Momentum indicators
    - Multi-timeframe features (4h regime filter)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Feature configuration
        self.momentum_periods = self.config.get("momentum", [5, 10, 20])
        self.volatility_window = self.config.get("volatility_window", 20)
        self.return_quantiles = self.config.get("return_quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])

    def generate_features(
        self,
        ohlcv: pd.DataFrame,
        kronos_embeddings: Optional[np.ndarray] = None,
        kronos_metrics: Optional[Dict] = None,
        funding_rates: Optional[pd.DataFrame] = None,
        higher_tf_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate all features for the signal model.

        Args:
            ohlcv: Base OHLCV DataFrame (30m timeframe)
            kronos_embeddings: Kronos embedding vectors
            kronos_metrics: Dict with perplexity, reconstruction error
            funding_rates: Funding rate data
            higher_tf_data: Higher timeframe data (4h)

        Returns:
            DataFrame with all features
        """
        features = ohlcv.copy()

        # Price-based features
        features = self._add_price_features(features)

        # Momentum features
        features = self._add_momentum_features(features)

        # Volatility features
        features = self._add_volatility_features(features)

        # Volume features
        features = self._add_volume_features(features)

        # Kronos embeddings
        if kronos_embeddings is not None:
            features = self._add_kronos_embeddings(features, kronos_embeddings)

        # Kronos shift alarms
        if kronos_metrics is not None:
            features = self._add_kronos_metrics(features, kronos_metrics)

        # Funding rate
        if funding_rates is not None and not funding_rates.empty:
            features = self._add_funding_features(features, funding_rates)

        # Higher timeframe features (4h regime filter)
        if higher_tf_data is not None and not higher_tf_data.empty:
            features = self._add_higher_tf_features(features, higher_tf_data)

        # Forward return quantiles (for distribution modeling)
        features = self._add_return_quantiles(features)

        # Drop rows with NaN in critical features
        features = features.dropna(subset=["close", "volatility_20"])

        logger.info(f"Generated {len(features.columns)} features for {len(features)} rows")
        return features

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        df = df.copy()

        # Returns at various horizons
        for period in [1, 3, 6, 12, 24]:
            df[f"return_{period}"] = df["close"].pct_change(period)

        # Log returns
        df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

        # Price relative to moving averages
        for period in [10, 20, 50]:
            ma = df["close"].rolling(period).mean()
            df[f"price_ma_ratio_{period}"] = df["close"] / ma - 1

        # Candle body ratio
        df["body_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)

        # Upper/lower shadow
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-8)
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-8)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        df = df.copy()

        for period in self.momentum_periods:
            # Price momentum
            df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1

            # RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

            # Rate of change
            df[f"roc_{period}"] = df["close"].pct_change(period) * 100

        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        df = df.copy()

        # Returns-based volatility
        returns = df["close"].pct_change()

        for window in [10, 20, 50]:
            df[f"volatility_{window}"] = returns.rolling(window).std() * np.sqrt(252 * 48)  # Annualized for 30m bars

        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        for window in [14, 20]:
            df[f"atr_{window}"] = true_range.rolling(window).mean()
            df[f"atr_pct_{window}"] = df[f"atr_{window}"] / df["close"]

        # Bollinger Bands
        window = 20
        ma = df["close"].rolling(window).mean()
        std = df["close"].rolling(window).std()
        df["bb_upper"] = ma + 2 * std
        df["bb_lower"] = ma - 2 * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / ma
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)

        # Volatility regime (rolling percentile of squared returns)
        df["vol_percentile"] = returns.rolling(100).apply(
            lambda x: (x**2).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = df.copy()

        # Volume moving averages
        for period in [10, 20]:
            df[f"volume_ma_{period}"] = df["volume"].rolling(period).mean()
            df[f"volume_ratio_{period}"] = df["volume"] / df[f"volume_ma_{period}"]

        # Volume trend
        df["volume_trend"] = df["volume"].rolling(10).mean() / df["volume"].rolling(50).mean()

        # OBV (On Balance Volume)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        df["obv_ma"] = df["obv"].rolling(20).mean()

        # Volume-price trend
        df["vpt"] = ((df["close"] - df["close"].shift(1)) / df["close"].shift(1) * df["volume"]).cumsum()

        return df

    def _add_kronos_embeddings(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        n_components: int = 50
    ) -> pd.DataFrame:
        """Add Kronos embedding features with PCA dimensionality reduction.

        Using PCA reduces 832 dimensions to 50, which:
        1. Captures 95%+ of variance
        2. Reduces overfitting
        3. Creates features that change more frequently (less stable)
        """
        df = df.copy()

        # embeddings shape: (n_samples, embedding_dim)
        n_samples, embedding_dim = embeddings.shape

        # Align embeddings with DataFrame (embeddings are for last n_samples rows)
        offset = len(df) - n_samples

        if offset < 0:
            logger.warning("More embeddings than data rows, truncating embeddings")
            embeddings = embeddings[-len(df):]
            offset = 0

        # Apply PCA to reduce dimensionality
        n_components = min(n_components, embedding_dim, n_samples)
        pca = PCA(n_components=n_components)
        embeddings_pca = pca.fit_transform(embeddings)

        variance_explained = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA: {embedding_dim} dims -> {n_components} components ({variance_explained:.1%} variance)")

        # Create PCA embedding columns
        for i in range(n_components):
            col_name = f"kronos_pca_{i}"
            df[col_name] = np.nan
            df.loc[df.index[offset:], col_name] = embeddings_pca[:, i]

        # Add rolling differences to capture regime changes
        for i in range(min(10, n_components)):
            col_name = f"kronos_pca_{i}"
            df[f"{col_name}_diff"] = df[col_name].diff()
            df[f"{col_name}_diff5"] = df[col_name].diff(5)

        # Summary features
        df["kronos_pca_sum"] = df[[f"kronos_pca_{i}" for i in range(min(10, n_components))]].sum(axis=1)
        df["kronos_pca_std"] = df[[f"kronos_pca_{i}" for i in range(min(10, n_components))]].std(axis=1)

        logger.info(f"Added {n_components} PCA + {min(10, n_components)*2 + 2} derived Kronos features")
        return df

    def _add_kronos_metrics(
        self,
        df: pd.DataFrame,
        metrics: Dict
    ) -> pd.DataFrame:
        """Add Kronos regime shift alarm features."""
        df = df.copy()

        # Perplexity (uncertainty measure)
        if "perplexity" in metrics:
            perplexity = metrics["perplexity"]
            offset = len(df) - len(perplexity)

            df["kronos_perplexity"] = np.nan
            if offset >= 0:
                df.loc[df.index[offset:], "kronos_perplexity"] = perplexity

            # Rolling stats
            df["kronos_perplexity_ma"] = df["kronos_perplexity"].rolling(20).mean()
            df["kronos_perplexity_std"] = df["kronos_perplexity"].rolling(20).std()

            # Perplexity spike (regime shift alarm)
            df["kronos_perplexity_zscore"] = (
                df["kronos_perplexity"] - df["kronos_perplexity_ma"]
            ) / df["kronos_perplexity_std"]

        # Reconstruction error
        if "reconstruction_error" in metrics:
            recon_error = metrics["reconstruction_error"]
            offset = len(df) - len(recon_error)

            df["kronos_recon_error"] = np.nan
            if offset >= 0:
                df.loc[df.index[offset:], "kronos_recon_error"] = recon_error

            df["kronos_recon_error_ma"] = df["kronos_recon_error"].rolling(20).mean()

        # Predicted volatility
        if "predicted_volatility" in metrics:
            pred_vol = metrics["predicted_volatility"]
            offset = len(df) - len(pred_vol)

            df["kronos_pred_vol"] = np.nan
            if offset >= 0:
                df.loc[df.index[offset:], "kronos_pred_vol"] = pred_vol

        return df

    def _add_funding_features(
        self,
        df: pd.DataFrame,
        funding: pd.DataFrame
    ) -> pd.DataFrame:
        """Add funding rate features."""
        df = df.copy()

        # Merge funding rates
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            funding.sort_values("timestamp")[["timestamp", "funding_rate"]],
            on="timestamp",
            direction="backward"
        )

        df["funding_rate"] = df["funding_rate"].ffill().fillna(0)

        # Funding rate features
        df["funding_ma"] = df["funding_rate"].rolling(8).mean()  # 8 funding periods = 1 day
        df["funding_std"] = df["funding_rate"].rolling(24).std()
        df["funding_extreme"] = (df["funding_rate"].abs() > 0.01).astype(int)

        return df

    def _add_higher_tf_features(
        self,
        df: pd.DataFrame,
        higher_tf: pd.DataFrame
    ) -> pd.DataFrame:
        """Add 4h timeframe features (regime filter)."""
        df = df.copy()

        # Calculate 4h features
        higher_tf = higher_tf.copy()

        # 4h trend
        higher_tf["4h_trend"] = np.sign(higher_tf["close"] - higher_tf["close"].shift(1))
        higher_tf["4h_trend_strength"] = (
            higher_tf["close"] / higher_tf["close"].rolling(6).mean() - 1
        )

        # 4h volatility
        higher_tf["4h_volatility"] = higher_tf["close"].pct_change().rolling(6).std()

        # 4h momentum
        higher_tf["4h_momentum"] = higher_tf["close"] / higher_tf["close"].shift(6) - 1

        # 4h RSI
        delta = higher_tf["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        higher_tf["4h_rsi"] = 100 - (100 / (1 + rs))

        # Select columns to merge
        merge_cols = ["timestamp", "4h_trend", "4h_trend_strength", "4h_volatility", "4h_momentum", "4h_rsi"]
        higher_tf_merge = higher_tf[merge_cols].copy()

        # Merge as-of
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            higher_tf_merge.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        # Forward fill
        for col in ["4h_trend", "4h_trend_strength", "4h_volatility", "4h_momentum", "4h_rsi"]:
            df[col] = df[col].ffill()

        return df

    def _add_return_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling return distribution quantiles."""
        df = df.copy()

        returns = df["close"].pct_change()
        window = 100

        for q in self.return_quantiles:
            df[f"return_q{int(q*100)}"] = returns.rolling(window).quantile(q)

        # Distribution shape
        df["return_skew"] = returns.rolling(window).skew()
        df["return_kurt"] = returns.rolling(window).kurt()

        return df

    def create_target(
        self,
        df: pd.DataFrame,
        horizon: int = 6,
        threshold: float = 0.001,
        target_type: str = "classification"
    ) -> pd.DataFrame:
        """
        Create target variable for the signal model.

        Args:
            df: Feature DataFrame
            horizon: Number of bars to look forward
            threshold: Minimum return to consider profitable
            target_type: 'classification' or 'regression'

        Returns:
            DataFrame with target column added
        """
        df = df.copy()

        # Forward return
        df["fwd_return"] = df["close"].shift(-horizon) / df["close"] - 1

        if target_type == "classification":
            # 3-class: long (0), flat (1), short (2)
            df["target"] = 1  # Default flat

            df.loc[df["fwd_return"] > threshold, "target"] = 0  # Long
            df.loc[df["fwd_return"] < -threshold, "target"] = 2  # Short

        else:
            # Regression target
            df["target"] = df["fwd_return"]

        # Remove rows without valid target
        df = df.dropna(subset=["target"])

        logger.info(f"Created target with horizon={horizon}, threshold={threshold}")

        if target_type == "classification":
            class_counts = df["target"].value_counts()
            logger.info(f"Class distribution: Long={class_counts.get(0, 0)}, Flat={class_counts.get(1, 0)}, Short={class_counts.get(2, 0)}")

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names (excluding target and metadata)."""
        exclude = ["timestamp", "target", "fwd_return", "open", "high", "low", "close", "volume"]
        return [col for col in self._all_features if col not in exclude]
