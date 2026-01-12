"""
Signal Model for trading decisions.

Uses XGBoost to learn when Kronos regime signals are trustworthy.

Target: Probability that the next N hours of trading will be profitable after costs

Output:
- P(long)
- P(short)
- P(flat)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle
import json
import logging

from ..utils.logger import get_logger

logger = get_logger("signal_model")


class SignalModel:
    """
    XGBoost-based signal model.

    Learns to predict profitable trading opportunities based on:
    - Kronos embeddings
    - Kronos regime shift alarms
    - Forward return quantiles
    - Funding rate
    - Volatility
    - Momentum
    - 4h regime filter
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize signal model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}

        # XGBoost parameters
        self.params = {
            "n_estimators": self.config.get("n_estimators", 500),
            "max_depth": self.config.get("max_depth", 6),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "min_child_weight": self.config.get("min_child_weight", 5),
            "reg_alpha": self.config.get("reg_alpha", 0.1),
            "reg_lambda": self.config.get("reg_lambda", 1.0),
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",  # GPU-compatible
            "device": "cuda" if self.config.get("use_gpu", True) else "cpu",
            "random_state": 42,
        }

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self.training_history: List[Dict] = []

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> "SignalModel":
        """
        Train the signal model.

        Args:
            X_train: Training features
            y_train: Training targets (0=long, 1=flat, 2=short)
            X_val: Validation features
            y_val: Validation targets
            early_stopping_rounds: Stop if no improvement
            verbose: Print training progress

        Returns:
            self
        """
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train = X_train.values

        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values

        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        logger.info(f"Training signal model with {len(X_train)} samples, {X_train.shape[1]} features")

        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)

        # Prepare eval set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        # Store feature importance
        self._calculate_feature_importance()

        # Log results
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = (val_pred == y_val).mean()
            logger.info(f"Validation accuracy: {val_acc:.4f}")

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features

        Returns:
            Predicted labels (0=long, 1=flat, 2=short)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            (n_samples, 3) array of [P(long), P(flat), P(short)]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict_proba(X)

    def get_signal(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        confidence_threshold: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trading signals with confidence filtering.

        Args:
            X: Features
            confidence_threshold: Minimum probability to generate signal

        Returns:
            Tuple of:
            - signals: 1 (long), 0 (flat), -1 (short)
            - confidences: Probability of the signal
        """
        proba = self.predict_proba(X)

        # Get max probability and its class
        max_proba = proba.max(axis=1)
        max_class = proba.argmax(axis=1)

        # Convert to signal: 0=long->1, 1=flat->0, 2=short->-1
        signal_map = {0: 1, 1: 0, 2: -1}
        signals = np.array([signal_map[c] for c in max_class])
        confidences = max_proba

        # Apply confidence threshold (set to flat if below threshold)
        signals[max_proba < confidence_threshold] = 0

        return signals, confidences

    def _calculate_feature_importance(self):
        """Calculate and store feature importance."""
        if self.model is None or not self.feature_names:
            return

        importance = self.model.feature_importances_

        self.feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        # Log top features
        top_10 = self.feature_importance.head(10)
        logger.info("Top 10 features:")
        for _, row in top_10.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance DataFrame."""
        if self.feature_importance is None:
            raise ValueError("No feature importance available. Train model first.")
        return self.feature_importance

    def save(self, path: str) -> Path:
        """
        Save model to disk.

        Args:
            path: Save path

        Returns:
            Path to saved model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model_data = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history,
            "config": self.config,
        }

        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {save_path}")
        return save_path

    @classmethod
    def load(cls, path: str) -> "SignalModel":
        """
        Load model from disk.

        Args:
            path: Model path

        Returns:
            Loaded SignalModel
        """
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        with open(load_path, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(config=model_data.get("config", {}))
        instance.model = model_data["model"]
        instance.params = model_data["params"]
        instance.feature_names = model_data["feature_names"]
        instance.feature_importance = model_data.get("feature_importance")
        instance.training_history = model_data.get("training_history", [])

        logger.info(f"Model loaded from {load_path}")
        return instance


class EnsembleSignalModel:
    """
    Ensemble of signal models for improved robustness.
    """

    def __init__(self, n_models: int = 5, config: Optional[Dict] = None):
        """
        Initialize ensemble.

        Args:
            n_models: Number of models in ensemble
            config: Model configuration
        """
        self.n_models = n_models
        self.config = config or {}
        self.models: List[SignalModel] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        val_ratio: float = 0.15
    ) -> "EnsembleSignalModel":
        """
        Train ensemble with bootstrap sampling.

        Args:
            X: Features
            y: Targets
            val_ratio: Validation set ratio

        Returns:
            self
        """
        n_samples = len(X)
        val_size = int(n_samples * val_ratio)

        for i in range(self.n_models):
            logger.info(f"Training ensemble model {i+1}/{self.n_models}")

            # Bootstrap sample
            indices = np.random.choice(n_samples - val_size, size=n_samples - val_size, replace=True)

            X_train = X.iloc[indices]
            y_train = y.iloc[indices]

            X_val = X.iloc[n_samples - val_size:]
            y_val = y.iloc[n_samples - val_size:]

            # Train model
            model = SignalModel(self.config)
            model.fit(X_train, y_train, X_val, y_val, verbose=False)
            self.models.append(model)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with ensemble averaging."""
        probas = np.array([model.predict_proba(X) for model in self.models])
        return probas.mean(axis=0)

    def get_signal(
        self,
        X: pd.DataFrame,
        confidence_threshold: float = 0.4,
        agreement_threshold: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get signals with ensemble agreement.

        Args:
            X: Features
            confidence_threshold: Minimum probability
            agreement_threshold: Fraction of models that must agree

        Returns:
            Tuple of (signals, confidences)
        """
        # Get predictions from all models
        all_signals = []
        all_confidences = []

        for model in self.models:
            signals, confidences = model.get_signal(X, confidence_threshold)
            all_signals.append(signals)
            all_confidences.append(confidences)

        all_signals = np.array(all_signals)
        all_confidences = np.array(all_confidences)

        # Majority vote
        final_signals = []
        final_confidences = []

        for i in range(len(X)):
            sample_signals = all_signals[:, i]
            sample_confidences = all_confidences[:, i]

            # Count votes
            unique, counts = np.unique(sample_signals, return_counts=True)
            vote_dict = dict(zip(unique, counts))

            # Get majority
            majority_signal = unique[np.argmax(counts)]
            agreement = counts.max() / self.n_models

            # Check agreement threshold
            if agreement >= agreement_threshold:
                final_signals.append(majority_signal)
                # Confidence is average confidence of agreeing models
                mask = sample_signals == majority_signal
                final_confidences.append(sample_confidences[mask].mean())
            else:
                final_signals.append(0)  # Flat if no agreement
                final_confidences.append(0.0)

        return np.array(final_signals), np.array(final_confidences)

    def save(self, path: str) -> Path:
        """Save ensemble to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        for i, model in enumerate(self.models):
            model.save(save_path / f"model_{i}.pkl")

        # Save metadata
        metadata = {
            "n_models": self.n_models,
            "config": self.config
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return save_path

    @classmethod
    def load(cls, path: str) -> "EnsembleSignalModel":
        """Load ensemble from disk."""
        load_path = Path(path)

        with open(load_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        instance = cls(n_models=metadata["n_models"], config=metadata.get("config", {}))

        for i in range(instance.n_models):
            model_path = load_path / f"model_{i}.pkl"
            instance.models.append(SignalModel.load(str(model_path)))

        return instance
