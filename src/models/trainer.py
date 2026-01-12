"""
Model training utilities with walk-forward validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from .signal_model import SignalModel
from ..utils.logger import get_logger

logger = get_logger("model_trainer")


class ModelTrainer:
    """
    Train signal models with proper validation methodology.

    Implements:
    - Walk-forward validation
    - Purged cross-validation
    - Hyperparameter tuning
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or {}

        # Walk-forward parameters
        self.train_window = self.config.get("train_window", 10000)
        self.test_window = self.config.get("test_window", 500)
        self.purge_window = self.config.get("purge_window", 48)

    def walk_forward_train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_config: Optional[Dict] = None,
        n_splits: Optional[int] = None
    ) -> Tuple[List[SignalModel], List[Dict]]:
        """
        Train models using walk-forward validation.

        Walk-forward validation:
        1. Train on historical data
        2. Test on out-of-sample future data
        3. Roll forward and repeat

        Args:
            X: Feature DataFrame with datetime index or 'timestamp' column
            y: Target Series
            model_config: Signal model configuration
            n_splits: Number of walk-forward splits (auto-calculated if None)

        Returns:
            Tuple of:
            - List of trained models (one per split)
            - List of performance metrics per split
        """
        # Ensure aligned
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")

        n_samples = len(X)
        min_required = self.train_window + self.purge_window + self.test_window

        # Check if we have enough data for walk-forward
        if n_samples < min_required:
            logger.warning(f"Not enough data for walk-forward ({n_samples} < {min_required}). Using simple train/test split.")
            return self._simple_train_test_split(X, y, model_config)

        # Calculate number of splits
        if n_splits is None:
            available = n_samples - self.train_window
            n_splits = available // self.test_window
            n_splits = max(1, min(n_splits, 20))  # Cap at 20 splits

        logger.info(f"Walk-forward training with {n_splits} splits")
        logger.info(f"Train window: {self.train_window}, Test window: {self.test_window}, Purge: {self.purge_window}")

        models = []
        metrics_list = []

        for split in range(n_splits):
            # Calculate indices
            train_end = self.train_window + split * self.test_window
            test_start = train_end + self.purge_window
            test_end = test_start + self.test_window

            if test_end > n_samples:
                logger.info(f"Stopping at split {split}, not enough data")
                break

            # Split data
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]

            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            # Further split train into train/val
            val_size = int(len(X_train) * 0.15)
            X_train_split = X_train.iloc[:-val_size]
            y_train_split = y_train.iloc[:-val_size]
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]

            logger.info(f"Split {split + 1}/{n_splits}: train={len(X_train_split)}, val={len(X_val)}, test={len(X_test)}")

            # Train model
            model = SignalModel(model_config)
            model.fit(X_train_split, y_train_split, X_val, y_val, verbose=False)

            # Evaluate on test set
            metrics = self._evaluate_model(model, X_test, y_test)
            metrics["split"] = split
            metrics["train_end_idx"] = train_end
            metrics["test_start_idx"] = test_start
            metrics["test_end_idx"] = test_end

            models.append(model)
            metrics_list.append(metrics)

            logger.info(f"Split {split + 1} metrics: accuracy={metrics['accuracy']:.4f}, "
                       f"long_precision={metrics.get('precision_long', 0):.4f}, "
                       f"short_precision={metrics.get('precision_short', 0):.4f}")

        # Log overall results
        self._log_walk_forward_results(metrics_list)

        return models, metrics_list

    def _evaluate_model(
        self,
        model: SignalModel,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """Evaluate model on test set."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Basic metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "n_samples": len(y_test),
        }

        # Per-class metrics
        for i, label in enumerate(["long", "flat", "short"]):
            mask = y_test == i
            if mask.sum() > 0:
                pred_mask = y_pred == i
                metrics[f"precision_{label}"] = precision_score(y_test, y_pred, labels=[i], average="macro", zero_division=0)
                metrics[f"recall_{label}"] = recall_score(y_test, y_pred, labels=[i], average="macro", zero_division=0)
                metrics[f"n_{label}"] = mask.sum()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        metrics["confusion_matrix"] = cm.tolist()

        # Signal quality (only count non-flat predictions)
        non_flat = y_pred != 1
        if non_flat.sum() > 0:
            metrics["signal_rate"] = non_flat.mean()
            metrics["signal_accuracy"] = (y_pred[non_flat] == y_test.values[non_flat]).mean()

        return metrics

    def _simple_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_config: Optional[Dict] = None,
        test_ratio: float = 0.2
    ) -> Tuple[List[SignalModel], List[Dict]]:
        """
        Fallback simple train/test split for small datasets.

        Args:
            X: Features
            y: Targets
            model_config: Model configuration
            test_ratio: Ratio of data for testing

        Returns:
            Tuple of (models, metrics)
        """
        n_samples = len(X)
        test_size = int(n_samples * test_ratio)
        train_size = n_samples - test_size - self.purge_window

        logger.info(f"Using simple train/test split: train={train_size}, purge={self.purge_window}, test={test_size}")

        # Split data with purge gap
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]

        X_test = X.iloc[train_size + self.purge_window:]
        y_test = y.iloc[train_size + self.purge_window:]

        # Further split train into train/val
        val_size = int(len(X_train) * 0.15)
        X_train_split = X_train.iloc[:-val_size]
        y_train_split = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]

        logger.info(f"Training split: train={len(X_train_split)}, val={len(X_val)}, test={len(X_test)}")

        # Train model
        model = SignalModel(model_config)
        model.fit(X_train_split, y_train_split, X_val, y_val, verbose=True)

        # Evaluate
        metrics = self._evaluate_model(model, X_test, y_test)
        metrics["split"] = 0
        metrics["type"] = "simple_split"

        logger.info(f"Test metrics: accuracy={metrics['accuracy']:.4f}, "
                   f"long_precision={metrics.get('precision_long', 0):.4f}, "
                   f"short_precision={metrics.get('precision_short', 0):.4f}")

        return [model], [metrics]

    def _log_walk_forward_results(self, metrics_list: List[Dict]):
        """Log summary of walk-forward results."""
        if not metrics_list:
            return

        # Aggregate metrics
        accuracies = [m["accuracy"] for m in metrics_list]
        signal_accuracies = [m.get("signal_accuracy", 0) for m in metrics_list if "signal_accuracy" in m]

        logger.info("=" * 50)
        logger.info("Walk-Forward Validation Results")
        logger.info("=" * 50)
        logger.info(f"Number of splits: {len(metrics_list)}")
        logger.info(f"Mean accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")

        if signal_accuracies:
            logger.info(f"Mean signal accuracy: {np.mean(signal_accuracies):.4f} (+/- {np.std(signal_accuracies):.4f})")

        # Per-class summary
        for label in ["long", "flat", "short"]:
            precisions = [m.get(f"precision_{label}", 0) for m in metrics_list]
            if precisions:
                logger.info(f"Mean {label} precision: {np.mean(precisions):.4f}")

    def purged_kfold_train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        model_config: Optional[Dict] = None
    ) -> Tuple[List[SignalModel], List[Dict]]:
        """
        Train with purged K-fold cross-validation.

        Purging removes samples too close to the test set boundary
        to prevent look-ahead bias.

        Args:
            X: Features
            y: Targets
            n_folds: Number of folds
            model_config: Model configuration

        Returns:
            Tuple of (models, metrics)
        """
        n_samples = len(X)
        fold_size = n_samples // n_folds

        models = []
        metrics_list = []

        for fold in range(n_folds):
            # Test indices
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples

            # Purge boundaries
            purge_start = max(0, test_start - self.purge_window)
            purge_end = min(n_samples, test_end + self.purge_window)

            # Train indices (everything not in test or purge zone)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:purge_end] = False

            X_train = X.iloc[train_mask]
            y_train = y.iloc[train_mask]

            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            logger.info(f"Fold {fold + 1}/{n_folds}: train={len(X_train)}, test={len(X_test)}")

            # Train
            model = SignalModel(model_config)

            # Split train into train/val
            val_size = int(len(X_train) * 0.15)
            model.fit(
                X_train.iloc[:-val_size],
                y_train.iloc[:-val_size],
                X_train.iloc[-val_size:],
                y_train.iloc[-val_size:],
                verbose=False
            )

            # Evaluate
            metrics = self._evaluate_model(model, X_test, y_test)
            metrics["fold"] = fold

            models.append(model)
            metrics_list.append(metrics)

        self._log_walk_forward_results(metrics_list)
        return models, metrics_list

    def hyperparameter_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict,
        n_trials: int = 20
    ) -> Tuple[Dict, List[Dict]]:
        """
        Random search for hyperparameters.

        Args:
            X: Features
            y: Targets
            param_grid: Dict of parameter ranges
            n_trials: Number of random trials

        Returns:
            Tuple of (best_params, all_results)
        """
        from sklearn.model_selection import ParameterSampler

        # Generate parameter combinations
        param_list = list(ParameterSampler(param_grid, n_iter=n_trials, random_state=42))

        results = []
        best_score = -np.inf
        best_params = None

        for i, params in enumerate(param_list):
            logger.info(f"Trial {i + 1}/{n_trials}: {params}")

            # Train with walk-forward
            models, metrics = self.walk_forward_train(X, y, model_config=params, n_splits=3)

            # Calculate mean score
            mean_accuracy = np.mean([m["accuracy"] for m in metrics])
            mean_signal_acc = np.mean([m.get("signal_accuracy", 0) for m in metrics])

            # Combined score
            score = 0.3 * mean_accuracy + 0.7 * mean_signal_acc

            results.append({
                "params": params,
                "mean_accuracy": mean_accuracy,
                "mean_signal_accuracy": mean_signal_acc,
                "score": score
            })

            if score > best_score:
                best_score = score
                best_params = params

            logger.info(f"Trial {i + 1} score: {score:.4f}")

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")

        return best_params, results
