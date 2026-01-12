#!/usr/bin/env python3
"""
Full Pipeline Script

Runs the complete pipeline:
1. Load and process data
2. Extract Kronos features
3. Train signal model
4. Run backtest

Usage:
    python scripts/run_pipeline.py --symbol BTCUSDT --start 2023-01-01 --end 2024-12-31
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.storage import DataStorage
from src.data.processor import DataProcessor, process_symbol_data
from src.features.engineer import FeatureEngineer
from src.features.kronos_features import KronosFeatureExtractor
from src.features.regime_detector import RegimeDetector, RegimeType
from src.models.signal_model import SignalModel
from src.models.trainer import ModelTrainer
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.metrics import BacktestMetrics
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("pipeline", log_dir="logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Run full trading pipeline")

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Start date"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--skip-kronos",
        action="store_true",
        help="Skip Kronos feature extraction (use cached or fallback)"
    )
    parser.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip backtest (train only)"
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=2.0,
        help="Leverage multiplier (1.0 = no leverage, 2.0 = 2x, max 3.0)"
    )

    return parser.parse_args()


def load_and_process_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load and process data for a symbol."""
    logger.info(f"Loading data for {symbol}...")

    storage = DataStorage()
    processor = DataProcessor()

    # Try to load processed data first
    processed_file = Path(f"data/processed/{symbol}_processed.parquet")
    if processed_file.exists():
        logger.info("Loading cached processed data...")
        data = pd.read_parquet(processed_file)
    else:
        # Load raw data
        data_30m = storage.load_ohlcv(symbol, "30m", start_date=start, end_date=end)
        data_4h = storage.load_ohlcv(symbol, "4h", start_date=start, end_date=end)
        funding = storage.load_ohlcv(symbol, "funding", start_date=start, end_date=end)

        if data_30m.empty:
            raise ValueError(f"No data found for {symbol}. Run collect_data.py first.")

        # Process data
        data_30m = processor.fill_gaps(data_30m, "30m")
        data_30m = processor.calculate_returns(data_30m)
        data_30m = processor.calculate_volatility(data_30m)
        data_30m = processor.calculate_momentum(data_30m)

        if not funding.empty:
            data_30m = processor.merge_funding_rates(data_30m, funding)

        if not data_4h.empty:
            data_4h = processor.fill_gaps(data_4h, "4h")
            data = processor.align_timeframes(
                {"30m": data_30m, "4h": data_4h},
                base_timeframe="30m"
            )
        else:
            data = data_30m

        # Save processed data
        data.to_parquet(processed_file, index=False)
        logger.info(f"Saved processed data to {processed_file}")

    # Filter by date
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data[(data["timestamp"] >= start) & (data["timestamp"] <= end)]

    logger.info(f"Loaded {len(data)} rows")
    return data


def extract_kronos_features(data: pd.DataFrame, skip: bool = False) -> tuple:
    """Extract features from Kronos model and fit regime detector."""
    if skip:
        logger.info("Skipping Kronos features (using fallback)...")
        return None, None, None

    logger.info("Extracting Kronos features...")

    try:
        extractor = KronosFeatureExtractor(
            context_length=512,
            device="cuda"
        )

        features = extractor.extract_features(
            data[["timestamp", "open", "high", "low", "close", "volume"]].copy(),
            batch_size=32,
            return_predictions=False
        )

        logger.info(f"Extracted embeddings shape: {features['embeddings'].shape}")

        # Fit regime detector on embeddings
        logger.info("Fitting regime detector...")
        regime_detector = RegimeDetector(
            n_regimes=6,
            pca_components=50,
            lookback_returns=24,
            forward_horizon=12
        )

        # Align embeddings with prices (embeddings start after context_length)
        context_length = 512
        prices = data["close"].values[context_length - 1:]
        if len(prices) >= len(features["embeddings"]):
            prices = prices[:len(features["embeddings"])]

        regime_detector.fit(
            features["embeddings"],
            prices
        )

        # Predict regimes for all samples
        regime_labels = regime_detector.predict(features["embeddings"])
        logger.info(f"Regime distribution: {np.bincount(regime_labels)}")

        # Add regime info to metrics
        metrics = {
            "perplexity": features["perplexity"],
            "reconstruction_error": features["reconstruction_error"],
            "predicted_volatility": features["predicted_volatility"],
            "regime_labels": regime_labels
        }

        return features["embeddings"], metrics, regime_detector

    except Exception as e:
        logger.warning(f"Kronos extraction failed: {e}. Using fallback features.")
        import traceback
        traceback.print_exc()
        return None, None, None


def generate_features(
    data: pd.DataFrame,
    kronos_embeddings: np.ndarray = None,
    kronos_metrics: dict = None
) -> pd.DataFrame:
    """Generate all features for the signal model."""
    logger.info("Generating features...")

    engineer = FeatureEngineer()

    features = engineer.generate_features(
        ohlcv=data,
        kronos_embeddings=kronos_embeddings,
        kronos_metrics=kronos_metrics
    )

    # Create target
    features = engineer.create_target(
        features,
        horizon=6,  # 6 x 30m = 3 hours
        threshold=0.001,
        target_type="classification"
    )

    logger.info(f"Generated {len(features.columns)} features")
    return features


def train_model(
    features: pd.DataFrame,
    model_config: dict = None
) -> tuple:
    """Train signal model with walk-forward validation."""
    logger.info("Training signal model...")

    # Identify feature columns (exclude metadata, target, and ANY forward-looking features)
    exclude_cols = ["timestamp", "target", "open", "high", "low", "close", "volume"]
    # CRITICAL: Exclude ALL forward-looking columns (fwd_return, fwd_return_1, etc.)
    # ALSO: Exclude Kronos embedding features to prevent over-stable predictions
    # Kronos embeddings cause the model to lock onto a direction for too long
    # Regime detection still uses embeddings for position sizing
    feature_cols = [
        c for c in features.columns
        if c not in exclude_cols
        and not c.startswith("fwd_")  # Exclude ALL forward-looking features
        and not c.startswith("kronos_pca_")  # Exclude Kronos PCA embeddings
        and not c.startswith("kronos_emb_")  # Exclude raw Kronos embeddings
        and not features[c].isna().all()
    ]

    X = features[feature_cols].copy()
    y = features["target"].copy()

    # Drop rows with NaN
    valid_mask = ~X.isna().any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"Training on {len(X)} samples with {len(feature_cols)} features")

    # Diagnostic: Check target distribution
    target_dist = y.value_counts(normalize=True)
    logger.info(f"Target distribution: Long={target_dist.get(0, 0):.1%}, Flat={target_dist.get(1, 0):.1%}, Short={target_dist.get(2, 0):.1%}")

    # Diagnostic: Check for potential leakage columns
    suspicious_cols = [c for c in feature_cols if 'fwd' in c.lower() or 'future' in c.lower() or 'target' in c.lower()]
    if suspicious_cols:
        logger.warning(f"POTENTIAL LEAKAGE: Suspicious columns found: {suspicious_cols}")

    # Diagnostic: Print first 20 feature names
    logger.info(f"Feature columns (first 20): {feature_cols[:20]}")

    # Dynamic training window based on data size
    n_samples = len(X)
    train_window = min(10000, int(n_samples * 0.6))  # Use 60% of data or 10k, whichever is smaller
    test_window = min(500, int(n_samples * 0.1))  # Use 10% of data or 500
    logger.info(f"Dynamic window sizes: train={train_window}, test={test_window}")

    # Train with walk-forward validation
    trainer = ModelTrainer({
        "train_window": train_window,
        "test_window": test_window,
        "purge_window": 48
    })

    models, metrics_list = trainer.walk_forward_train(
        X, y,
        model_config=model_config,
        n_splits=5
    )

    # Diagnostic: Show top feature importances
    final_model = models[-1]
    if hasattr(final_model.model, 'feature_importances_'):
        importance = final_model.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        logger.info("Top 10 feature importances:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Return the last model (trained on most recent data)
    return models[-1], metrics_list, feature_cols


def run_backtest(
    data: pd.DataFrame,
    model: SignalModel,
    feature_cols: list,
    regime_detector: RegimeDetector = None,
    kronos_embeddings: np.ndarray = None,
    config: BacktestConfig = None,
    leverage: float = 1.0
) -> dict:
    """Run backtest with trained model and regime-aware trading."""
    logger.info("Running backtest...")

    if config is None:
        config = BacktestConfig(
            initial_capital=10000.0,
            fee_rate=0.0005,
            slippage_bps=5,
            stop_loss_pct=0.99,         # 99% stop - effectively disabled
            take_profit_pct=5.00,       # 500% take profit - let winners run
            confidence_threshold=0.10,  # Low threshold for more signals
            leverage=leverage,
            max_leverage=3.0,
            max_holding_bars=99999,     # Essentially infinite - hold until signal changes
            trailing_stop_pct=0.22,     # 22% trailing stop - best combined result
            min_holding_bars=96         # Min 2 days - reduce whipsaws
        )

    engine = BacktestEngine(config)

    # Filter to features that exist in data
    valid_features = [c for c in feature_cols if c in data.columns]

    result = engine.run(
        data=data,
        model=model,
        feature_columns=valid_features,
        symbol="BTCUSDT",
        regime_detector=regime_detector,
        kronos_embeddings=kronos_embeddings
    )

    # Print report
    metrics_calc = BacktestMetrics()
    report = metrics_calc.format_report(result.metrics)
    print(report)

    return result


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Kronos Trading Pipeline")
    logger.info("=" * 60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Leverage: {args.leverage}x")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)

    # Step 1: Load and process data
    data = load_and_process_data(args.symbol, args.start, args.end)

    # Step 2: Extract Kronos features and fit regime detector
    kronos_embeddings, kronos_metrics, regime_detector = extract_kronos_features(
        data, skip=args.skip_kronos
    )

    # Print regime summary if available
    if regime_detector is not None:
        logger.info("Regime Summary:")
        regime_summary = regime_detector.summary()
        for _, row in regime_summary.iterrows():
            logger.info(f"  Cluster {row['cluster_id']}: {row['regime_type']} "
                       f"(bias={row['position_bias']:.2f}, scale={row['position_scale']:.2f})")

    # Step 3: Generate features
    features = generate_features(data, kronos_embeddings, kronos_metrics)

    # Step 4: Train model
    model_config = config.get("signal_model", {}).get("xgboost", {})
    model, train_metrics, feature_cols = train_model(features, model_config)

    # Save model
    model_path = Path("models") / f"{args.symbol}_signal_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Step 5: Run backtest with regime-aware trading
    if not args.no_backtest:
        result = run_backtest(
            features, model, feature_cols,
            regime_detector=regime_detector,
            kronos_embeddings=kronos_embeddings,
            leverage=args.leverage
        )

        # Save results
        storage = DataStorage()
        storage.save_backtest_results(
            result.metrics,
            f"{args.symbol}_backtest"
        )

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
