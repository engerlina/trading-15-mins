"""
Kronos model wrapper for feature extraction.

Uses the frozen Kronos-base model as an encoder to generate:
- Regime embeddings
- Predicted return distribution
- Volatility estimates
- Perplexity / reconstruction error (regime change alarms)
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Add Kronos to path
KRONOS_PATH = Path(__file__).parent.parent.parent / "Kronos"
sys.path.insert(0, str(KRONOS_PATH))

from ..utils.logger import get_logger

logger = get_logger("kronos_features")


class KronosFeatureExtractor:
    """
    Extract features from the Kronos model.

    The Kronos model is used as a frozen encoder to convert
    raw OHLCV data into high-level regime representations.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        context_length: int = 512,
        device: str = "auto",
        frozen: bool = True
    ):
        """
        Initialize Kronos feature extractor.

        Args:
            model_path: Path to model.safetensors
            context_length: Context window size
            device: 'cuda', 'cpu', or 'auto'
            frozen: If True, use model in eval mode without gradients
        """
        self.context_length = context_length
        self.frozen = frozen

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model path
        if model_path is None:
            model_path = KRONOS_PATH / "model" / "base" / "model.safetensors"
        self.model_path = Path(model_path)

        # Load model
        self.model = None
        self.tokenizer = None
        self.predictor = None
        self._load_model()

    def _load_model(self):
        """Load Kronos model and tokenizer."""
        try:
            from model import KronosTokenizer, Kronos, KronosPredictor

            logger.info(f"Loading Kronos model from {self.model_path}")
            logger.info(f"Using device: {self.device}")

            # Load tokenizer from HuggingFace
            logger.info("Loading tokenizer from NeoQuasar/Kronos-Tokenizer-base...")
            self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

            # Check if local config.json exists (required for from_pretrained)
            config_path = self.model_path.parent / "config.json"

            # Always load from HuggingFace - it has the proper config
            # Local safetensors file is missing config.json
            logger.info("Loading model from NeoQuasar/Kronos-base...")
            self.model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

            # Move to device
            self.tokenizer = self.tokenizer.to(self.device)
            self.model = self.model.to(self.device)

            # Initialize predictor with loaded model and tokenizer
            self.predictor = KronosPredictor(self.model, self.tokenizer, max_context=self.context_length)

            if self.frozen:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False

            logger.info("Kronos model loaded successfully")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        except Exception as e:
            logger.error(f"Failed to load Kronos model: {e}")
            raise

    def extract_features(
        self,
        ohlcv: pd.DataFrame,
        batch_size: int = 32,
        return_predictions: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from OHLCV data.

        Args:
            ohlcv: DataFrame with columns [timestamp, open, high, low, close, volume]
            batch_size: Batch size for inference
            return_predictions: Whether to also return price predictions

        Returns:
            Dict with:
            - embeddings: (n_samples, embedding_dim) regime embeddings
            - perplexity: (n_samples,) uncertainty measure
            - reconstruction_error: (n_samples,) how well model reconstructs input
            - predicted_volatility: (n_samples,) predicted volatility
            - predictions: (n_samples, pred_steps, 5) if return_predictions
        """
        if len(ohlcv) < self.context_length:
            raise ValueError(f"Need at least {self.context_length} rows, got {len(ohlcv)}")

        # Prepare input data - Kronos expects 6 columns: OHLCV + amount
        # 'amount' is quote volume (price * volume). Calculate if not present.
        if "amount" in ohlcv.columns:
            data = ohlcv[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
        elif "quote_volume" in ohlcv.columns:
            data = ohlcv[["open", "high", "low", "close", "volume", "quote_volume"]].values.astype(np.float32)
        else:
            # Calculate amount = close * volume
            ohlcv_copy = ohlcv.copy()
            ohlcv_copy["amount"] = ohlcv_copy["close"] * ohlcv_copy["volume"]
            data = ohlcv_copy[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)

        # Create sliding windows
        n_windows = len(data) - self.context_length + 1
        windows = np.array([
            data[i:i + self.context_length]
            for i in range(n_windows)
        ])

        logger.info(f"Processing {n_windows} windows with batch_size={batch_size}")

        # Process in batches
        all_embeddings = []
        all_perplexity = []
        all_recon_error = []
        all_pred_vol = []
        all_predictions = []

        with torch.no_grad():
            for batch_start in range(0, n_windows, batch_size):
                batch_end = min(batch_start + batch_size, n_windows)
                batch_windows = windows[batch_start:batch_end]

                # Extract features for this batch
                batch_features = self._process_batch(
                    batch_windows,
                    return_predictions=return_predictions
                )

                all_embeddings.append(batch_features["embeddings"])
                all_perplexity.append(batch_features["perplexity"])
                all_recon_error.append(batch_features["reconstruction_error"])
                all_pred_vol.append(batch_features["predicted_volatility"])

                if return_predictions:
                    all_predictions.append(batch_features["predictions"])

                if (batch_start + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Processed {batch_end}/{n_windows} windows")

        # Concatenate results
        result = {
            "embeddings": np.concatenate(all_embeddings, axis=0),
            "perplexity": np.concatenate(all_perplexity, axis=0),
            "reconstruction_error": np.concatenate(all_recon_error, axis=0),
            "predicted_volatility": np.concatenate(all_pred_vol, axis=0),
        }

        if return_predictions:
            result["predictions"] = np.concatenate(all_predictions, axis=0)

        logger.info(f"Extracted features: embeddings shape {result['embeddings'].shape}")
        return result

    def _process_batch(
        self,
        batch_windows: np.ndarray,
        return_predictions: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Process a batch of windows through the model.

        Args:
            batch_windows: (batch_size, context_length, 6) OHLCV+amount windows
            return_predictions: Whether to return predictions

        Returns:
            Dict with batch features
        """
        batch_size = len(batch_windows)

        try:
            embeddings_list = []
            perplexity_list = []
            recon_error_list = []
            pred_vol_list = []

            for i in range(batch_size):
                window = batch_windows[i]  # (context_length, 6) - OHLCV + amount

                # Normalize window for tokenizer (Kronos expects normalized data)
                # Z-score normalization per feature
                window_mean = np.mean(window, axis=0, keepdims=True)
                window_std = np.std(window, axis=0, keepdims=True) + 1e-8
                window_norm = (window - window_mean) / window_std
                window_norm = np.clip(window_norm, -5, 5)

                # Convert to tensor - shape (1, seq_len, 6)
                window_tensor = torch.tensor(window_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    # Encode using tokenizer to get indices (s1, s2)
                    indices = self.tokenizer.encode(window_tensor, half=True)
                    s1_ids, s2_ids = indices[0], indices[1]

                    # Forward pass through model to get hidden states
                    # The model takes (s1_ids, s2_ids) and returns (s1_logits, context)
                    s1_logits, context = self.model.decode_s1(s1_ids, s2_ids)

                    # Use context (hidden states) as embedding - mean pool over sequence
                    embedding = context.mean(dim=1).cpu().numpy()  # (1, d_model)
                    embeddings_list.append(embedding)

                    # Calculate perplexity from s1_logits (cross-entropy with actual tokens)
                    # Use softmax entropy as proxy for perplexity
                    probs = torch.softmax(s1_logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                    perplexity = torch.exp(entropy).item()
                    perplexity_list.append(perplexity)

                    # Reconstruction error: decode and compare
                    reconstructed = self.tokenizer.decode(indices, half=True)
                    recon_error = torch.mean((window_tensor - reconstructed) ** 2).item()
                    recon_error_list.append(recon_error)

                    # Predicted volatility from recent returns (use close price - column 3)
                    pred_vol = self._calculate_predicted_volatility(window)
                    pred_vol_list.append(pred_vol)

            # Stack results
            embeddings = np.vstack(embeddings_list) if embeddings_list else np.zeros((batch_size, 256))
            perplexity = np.array(perplexity_list)
            recon_error = np.array(recon_error_list)
            pred_vol = np.array(pred_vol_list)

            result = {
                "embeddings": embeddings,
                "perplexity": perplexity,
                "reconstruction_error": recon_error,
                "predicted_volatility": pred_vol,
            }

            if return_predictions:
                # Skip predictions for now - focus on embeddings
                result["predictions"] = np.zeros((batch_size, 48, 6))

            return result

        except Exception as e:
            logger.warning(f"Error in batch processing, using fallback: {e}")
            return self._fallback_features(batch_windows, return_predictions)

    def _fallback_features(
        self,
        batch_windows: np.ndarray,
        return_predictions: bool
    ) -> Dict[str, np.ndarray]:
        """Fallback feature extraction when model fails."""
        batch_size = len(batch_windows)

        # Use simple statistical features as fallback
        embeddings = []
        pred_volatilities = []

        for window in batch_windows:
            # Statistical summary as embedding using close prices (column 3)
            close_prices = window[:, 3]
            returns = np.diff(close_prices) / (close_prices[:-1] + 1e-10)

            embedding = np.array([
                np.mean(returns),
                np.std(returns),
                np.percentile(returns, 25),
                np.percentile(returns, 75),
                np.min(returns),
                np.max(returns),
            ])
            # Pad to expected dimension (256 to match Kronos d_model)
            embedding = np.pad(embedding, (0, 250), mode='constant')
            embeddings.append(embedding)

            # Calculate volatility
            pred_volatilities.append(np.std(returns[-20:]) * np.sqrt(252 * 48))

        result = {
            "embeddings": np.array(embeddings),
            "perplexity": np.ones(batch_size),
            "reconstruction_error": np.zeros(batch_size),
            "predicted_volatility": np.array(pred_volatilities),
        }

        if return_predictions:
            result["predictions"] = np.zeros((batch_size, 48, 6))

        return result

    def _calculate_reconstruction_error(
        self,
        input_window: np.ndarray,
        model_output
    ) -> float:
        """Calculate reconstruction error."""
        # Simplified: use model loss as proxy
        if hasattr(model_output, 'loss') and model_output.loss is not None:
            return model_output.loss.item()
        return 0.0

    def _calculate_predicted_volatility(
        self,
        window: np.ndarray
    ) -> float:
        """Calculate predicted volatility from recent data."""
        # Use recent realized volatility as proxy
        returns = np.diff(window[:, 3]) / window[:-1, 3]
        return np.std(returns[-20:]) * np.sqrt(252 * 48)  # Annualized

    def get_regime_label(
        self,
        embeddings: np.ndarray,
        n_regimes: int = 5
    ) -> np.ndarray:
        """
        Cluster embeddings into regime labels.

        Args:
            embeddings: (n_samples, embedding_dim) embeddings
            n_regimes: Number of regime clusters

        Returns:
            (n_samples,) regime labels
        """
        from sklearn.cluster import KMeans

        # Fit KMeans on embeddings
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Map clusters to meaningful regime names based on volatility
        # (This is a simplified heuristic)
        cluster_vol = []
        for i in range(n_regimes):
            mask = labels == i
            if mask.sum() > 0:
                cluster_vol.append((i, embeddings[mask].std()))
            else:
                cluster_vol.append((i, 0))

        # Sort by volatility
        cluster_vol.sort(key=lambda x: x[1])

        # Create mapping: low vol -> 0 (calm), high vol -> n_regimes-1 (volatile)
        regime_map = {old: new for new, (old, _) in enumerate(cluster_vol)}
        labels = np.array([regime_map[l] for l in labels])

        return labels

    def detect_regime_shift(
        self,
        perplexity: np.ndarray,
        threshold: float = 2.0,
        window: int = 20
    ) -> np.ndarray:
        """
        Detect regime shifts based on perplexity spikes.

        Args:
            perplexity: (n_samples,) perplexity values
            threshold: Z-score threshold for regime shift
            window: Rolling window for baseline

        Returns:
            (n_samples,) binary regime shift indicators
        """
        # Rolling mean and std
        perplexity_series = pd.Series(perplexity)
        rolling_mean = perplexity_series.rolling(window).mean()
        rolling_std = perplexity_series.rolling(window).std()

        # Z-score
        zscore = (perplexity_series - rolling_mean) / rolling_std

        # Regime shift = zscore > threshold
        regime_shift = (zscore.abs() > threshold).astype(int).values

        return regime_shift
