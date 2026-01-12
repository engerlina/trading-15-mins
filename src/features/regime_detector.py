"""
Regime Detection using Kronos Embeddings

Clusters market states into distinct regimes and assigns
optimal trading strategies to each regime.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("regime_detector")


class RegimeType(Enum):
    """Market regime types with associated trading strategies."""
    STRONG_UPTREND = "strong_uptrend"      # Strong bullish momentum
    WEAK_UPTREND = "weak_uptrend"          # Mild bullish bias
    RANGING = "ranging"                     # Sideways, mean-reverting
    WEAK_DOWNTREND = "weak_downtrend"      # Mild bearish bias
    STRONG_DOWNTREND = "strong_downtrend"  # Strong bearish momentum
    HIGH_VOLATILITY = "high_volatility"    # Uncertain, high risk


@dataclass
class RegimeConfig:
    """Configuration for each regime's trading behavior."""
    regime_type: RegimeType
    position_bias: float          # -1 (short) to +1 (long)
    position_scale: float         # 0 to 1, scales position size
    use_trend_following: bool     # Follow momentum or mean-revert
    confidence_threshold: float   # Min confidence to trade
    stop_loss_multiplier: float   # Multiplier for stop loss
    take_profit_multiplier: float # Multiplier for take profit


# Default regime configurations - AGGRESSIVE TREND FOLLOWING
# Strong trends: Maximum position, ride the trend
# Weak trends: Good position, follow momentum
# Ranging: Moderate position for swing trading
# High vol: Opportunity in volatility
DEFAULT_REGIME_CONFIGS = {
    RegimeType.STRONG_UPTREND: RegimeConfig(
        regime_type=RegimeType.STRONG_UPTREND,
        position_bias=1.0,           # Always long in strong uptrend
        position_scale=1.5,          # INCREASED: 1.5x position in strong trends
        use_trend_following=True,
        confidence_threshold=0.0,    # No threshold - just follow trend
        stop_loss_multiplier=2.5,    # Wider stops
        take_profit_multiplier=5.0   # INCREASED: Let winners run
    ),
    RegimeType.WEAK_UPTREND: RegimeConfig(
        regime_type=RegimeType.WEAK_UPTREND,
        position_bias=0.6,           # Slight bias increase
        position_scale=1.0,          # INCREASED: Full position
        use_trend_following=True,
        confidence_threshold=0.1,    # Lower threshold
        stop_loss_multiplier=2.0,
        take_profit_multiplier=3.0
    ),
    RegimeType.RANGING: RegimeConfig(
        regime_type=RegimeType.RANGING,
        position_bias=0.0,
        position_scale=0.7,          # INCREASED: Still trade ranging
        use_trend_following=False,   # Mean revert - use ML signals
        confidence_threshold=0.15,   # Lower threshold
        stop_loss_multiplier=1.2,
        take_profit_multiplier=2.0
    ),
    RegimeType.WEAK_DOWNTREND: RegimeConfig(
        regime_type=RegimeType.WEAK_DOWNTREND,
        position_bias=-0.6,          # Slight bias increase
        position_scale=1.0,          # INCREASED: Full position
        use_trend_following=True,
        confidence_threshold=0.1,    # Lower threshold
        stop_loss_multiplier=2.0,
        take_profit_multiplier=3.0
    ),
    RegimeType.STRONG_DOWNTREND: RegimeConfig(
        regime_type=RegimeType.STRONG_DOWNTREND,
        position_bias=-1.0,          # Always short in strong downtrend
        position_scale=1.5,          # INCREASED: 1.5x position in strong trends
        use_trend_following=True,
        confidence_threshold=0.0,    # No threshold - just follow trend
        stop_loss_multiplier=2.5,    # Wider stops
        take_profit_multiplier=5.0   # INCREASED: Let winners run
    ),
    RegimeType.HIGH_VOLATILITY: RegimeConfig(
        regime_type=RegimeType.HIGH_VOLATILITY,
        position_bias=0.0,
        position_scale=0.8,          # INCREASED: Volatility = opportunity
        use_trend_following=True,
        confidence_threshold=0.15,   # Lower threshold
        stop_loss_multiplier=3.0,    # Very wide stops for vol
        take_profit_multiplier=4.0   # Big targets in vol
    ),
}


class RegimeDetector:
    """
    Detects market regimes using Kronos embeddings.

    The detector clusters embeddings and labels each cluster based on
    forward returns and volatility characteristics.
    """

    def __init__(
        self,
        n_regimes: int = 6,
        pca_components: int = 50,
        lookback_returns: int = 24,  # Bars for calculating regime characteristics
        forward_horizon: int = 12,   # Bars for labeling returns
        random_state: int = 42
    ):
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of regime clusters
            pca_components: PCA dimensions for embedding reduction
            lookback_returns: Bars to look back for regime characteristics
            forward_horizon: Bars to look forward for labeling
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.pca_components = pca_components
        self.lookback_returns = lookback_returns
        self.forward_horizon = forward_horizon
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)

        self.regime_labels: Dict[int, RegimeType] = {}
        self.regime_configs: Dict[int, RegimeConfig] = {}
        self.cluster_centers_: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(
        self,
        embeddings: np.ndarray,
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> 'RegimeDetector':
        """
        Fit the regime detector on historical data.

        Args:
            embeddings: (n_samples, embedding_dim) Kronos embeddings
            prices: (n_samples,) Price series for labeling regimes
            timestamps: Optional timestamps for logging

        Returns:
            self
        """
        logger.info(f"Fitting regime detector on {len(embeddings)} samples")

        # Reduce dimensionality
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        embeddings_pca = self.pca.fit_transform(embeddings_scaled)

        logger.info(f"PCA explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")

        # Cluster embeddings
        cluster_labels = self.kmeans.fit_predict(embeddings_pca)
        self.cluster_centers_ = self.kmeans.cluster_centers_

        # Label each cluster based on market characteristics
        self._label_clusters(cluster_labels, prices)

        self.is_fitted = True
        logger.info(f"Regime detector fitted with {self.n_regimes} regimes")

        return self

    def _label_clusters(
        self,
        cluster_labels: np.ndarray,
        prices: np.ndarray
    ) -> None:
        """
        Label each cluster with a regime type based on market characteristics.

        Args:
            cluster_labels: Cluster assignment for each sample
            prices: Price series
        """
        # Calculate forward returns and volatility for each sample
        returns = np.diff(prices) / prices[:-1]

        # Pad returns to match length
        returns = np.concatenate([[0], returns])

        # Forward returns (what happens after this regime)
        forward_returns = np.zeros(len(prices))
        for i in range(len(prices) - self.forward_horizon):
            forward_returns[i] = (prices[i + self.forward_horizon] - prices[i]) / prices[i]

        # Rolling volatility
        volatility = pd.Series(returns).rolling(self.lookback_returns).std().fillna(0).values

        # Analyze each cluster
        cluster_stats = {}
        for cluster_id in range(self.n_regimes):
            mask = cluster_labels == cluster_id
            if mask.sum() == 0:
                continue

            cluster_stats[cluster_id] = {
                'mean_forward_return': np.mean(forward_returns[mask]),
                'std_forward_return': np.std(forward_returns[mask]),
                'mean_volatility': np.mean(volatility[mask]),
                'count': mask.sum(),
                'pct': mask.sum() / len(mask) * 100
            }

        # Sort clusters by forward return
        sorted_clusters = sorted(
            cluster_stats.keys(),
            key=lambda x: cluster_stats[x]['mean_forward_return'],
            reverse=True
        )

        # Calculate volatility threshold (top 20% is "high volatility")
        all_vols = [cluster_stats[c]['mean_volatility'] for c in sorted_clusters]
        vol_threshold = np.percentile(all_vols, 80)

        # Assign regime types
        n = len(sorted_clusters)
        for i, cluster_id in enumerate(sorted_clusters):
            stats = cluster_stats[cluster_id]

            # Check if high volatility regime
            if stats['mean_volatility'] > vol_threshold:
                regime_type = RegimeType.HIGH_VOLATILITY
            else:
                # Assign based on position in sorted order
                position = i / (n - 1) if n > 1 else 0.5

                if position < 0.2:
                    regime_type = RegimeType.STRONG_UPTREND
                elif position < 0.4:
                    regime_type = RegimeType.WEAK_UPTREND
                elif position < 0.6:
                    regime_type = RegimeType.RANGING
                elif position < 0.8:
                    regime_type = RegimeType.WEAK_DOWNTREND
                else:
                    regime_type = RegimeType.STRONG_DOWNTREND

            self.regime_labels[cluster_id] = regime_type
            self.regime_configs[cluster_id] = DEFAULT_REGIME_CONFIGS[regime_type]

            logger.info(
                f"Cluster {cluster_id} -> {regime_type.value}: "
                f"fwd_ret={stats['mean_forward_return']:.4f}, "
                f"vol={stats['mean_volatility']:.4f}, "
                f"pct={stats['pct']:.1f}%"
            )

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict regime for new embeddings.

        Args:
            embeddings: (n_samples, embedding_dim) embeddings

        Returns:
            (n_samples,) cluster IDs
        """
        if not self.is_fitted:
            raise RuntimeError("Regime detector not fitted. Call fit() first.")

        embeddings_scaled = self.scaler.transform(embeddings)
        embeddings_pca = self.pca.transform(embeddings_scaled)
        return self.kmeans.predict(embeddings_pca)

    def get_regime_type(self, cluster_id: int) -> RegimeType:
        """Get regime type for a cluster ID."""
        return self.regime_labels.get(cluster_id, RegimeType.RANGING)

    def get_regime_config(self, cluster_id: int) -> RegimeConfig:
        """Get trading configuration for a cluster ID."""
        return self.regime_configs.get(
            cluster_id,
            DEFAULT_REGIME_CONFIGS[RegimeType.RANGING]
        )

    def get_regime_probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get soft regime probabilities based on distance to cluster centers.

        Args:
            embeddings: (n_samples, embedding_dim) embeddings

        Returns:
            (n_samples, n_regimes) probability for each regime
        """
        if not self.is_fitted:
            raise RuntimeError("Regime detector not fitted. Call fit() first.")

        embeddings_scaled = self.scaler.transform(embeddings)
        embeddings_pca = self.pca.transform(embeddings_scaled)

        # Calculate distances to each cluster center
        distances = np.zeros((len(embeddings_pca), self.n_regimes))
        for i, center in enumerate(self.cluster_centers_):
            distances[:, i] = np.linalg.norm(embeddings_pca - center, axis=1)

        # Convert to probabilities (softmax of negative distances)
        neg_distances = -distances
        exp_distances = np.exp(neg_distances - neg_distances.max(axis=1, keepdims=True))
        probabilities = exp_distances / exp_distances.sum(axis=1, keepdims=True)

        return probabilities

    def get_trading_signals(
        self,
        embeddings: np.ndarray,
        base_signals: np.ndarray,
        base_confidences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust trading signals based on detected regime.

        Args:
            embeddings: (n_samples, embedding_dim) embeddings
            base_signals: (n_samples,) base signals (-1, 0, 1)
            base_confidences: (n_samples,) base confidence scores

        Returns:
            adjusted_signals: (n_samples,) regime-adjusted signals
            adjusted_confidences: (n_samples,) regime-adjusted confidences
        """
        if not self.is_fitted:
            return base_signals, base_confidences

        cluster_ids = self.predict(embeddings)
        adjusted_signals = base_signals.copy()
        adjusted_confidences = base_confidences.copy()

        for i, cluster_id in enumerate(cluster_ids):
            config = self.get_regime_config(cluster_id)

            # Apply confidence threshold
            if base_confidences[i] < config.confidence_threshold:
                adjusted_signals[i] = 0
                adjusted_confidences[i] = 0
                continue

            # Apply position bias
            if config.use_trend_following:
                # Trend following: bias towards regime direction
                if config.position_bias > 0.3 and base_signals[i] == -1:
                    # Strong uptrend but signal is short - reduce confidence
                    adjusted_confidences[i] *= 0.5
                elif config.position_bias < -0.3 and base_signals[i] == 1:
                    # Strong downtrend but signal is long - reduce confidence
                    adjusted_confidences[i] *= 0.5
            else:
                # Mean reversion: trade against extremes
                pass  # Keep base signal for ranging markets

            # Scale confidence by regime's position scale
            adjusted_confidences[i] *= config.position_scale

        return adjusted_signals, adjusted_confidences

    def get_position_size_multiplier(self, cluster_id: int) -> float:
        """Get position size multiplier for a regime."""
        config = self.get_regime_config(cluster_id)
        return config.position_scale

    def summary(self) -> pd.DataFrame:
        """Get summary of detected regimes."""
        if not self.is_fitted:
            return pd.DataFrame()

        rows = []
        for cluster_id, regime_type in self.regime_labels.items():
            config = self.regime_configs[cluster_id]
            rows.append({
                'cluster_id': cluster_id,
                'regime_type': regime_type.value,
                'position_bias': config.position_bias,
                'position_scale': config.position_scale,
                'trend_following': config.use_trend_following,
                'confidence_threshold': config.confidence_threshold
            })

        return pd.DataFrame(rows)
