# Kronos Trading System

A regime-aware AI trading system for crypto perpetuals, designed to turn $10k into a high-Sharpe swing trading system.

## Architecture

```
Market Data
   ↓
Kronos (Regime + Distribution Model)
   ↓
Signal Model (XGBoost)
   ↓
Risk & Execution Engine
```

### Four Layers

1. **Market Data Layer**: OHLCV at 5m, 30m, 4h timeframes + funding rates
2. **Kronos Layer**: Neural regime sensor - detects market state changes
3. **Signal Model Layer**: XGBoost learns when Kronos signals are profitable
4. **Risk Engine Layer**: Position sizing, volatility targeting, kill switches

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### 2. Collect Historical Data

```bash
python scripts/collect_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --start 2024-01-01 --end 2025-12-31
```

### 3. Run Full Pipeline

```bash
python scripts/run_pipeline.py --symbol BTCUSDT --start 2024-01-01 --end 2025-12-31
```

### 4. Run Backtest

```bash
python scripts/run_backtest.py --model models/BTCUSDT_signal_model.pkl --data data/processed/BTCUSDT_processed.parquet --monte-carlo
```

## Project Structure

```
Kronos02/
├── configs/
│   └── config.yaml          # Main configuration
├── data/
│   ├── raw/                  # Raw OHLCV data
│   ├── processed/            # Processed features
│   └── cache/                # Cached embeddings
├── Kronos/                   # Kronos model (frozen encoder)
│   └── model/base/model.safetensors
├── models/                   # Trained signal models
├── logs/                     # Log files
├── results/                  # Backtest results
├── notebooks/                # Jupyter notebooks
├── scripts/
│   ├── collect_data.py       # Data collection
│   ├── run_pipeline.py       # Full pipeline
│   └── run_backtest.py       # Standalone backtest
├── src/
│   ├── data/                 # Data modules
│   │   ├── collector.py      # Binance/Hyperliquid collectors
│   │   ├── processor.py      # Data processing
│   │   └── storage.py        # Storage utilities
│   ├── features/             # Feature engineering
│   │   ├── engineer.py       # Feature generation
│   │   └── kronos_features.py # Kronos wrapper
│   ├── models/               # Signal models
│   │   ├── signal_model.py   # XGBoost model
│   │   └── trainer.py        # Training utilities
│   ├── risk/                 # Risk management
│   │   ├── engine.py         # Risk engine
│   │   └── position.py       # Position manager
│   ├── backtest/             # Backtesting
│   │   ├── engine.py         # Backtest engine
│   │   └── metrics.py        # Performance metrics
│   ├── execution/            # Live execution
│   │   └── hyperliquid.py    # Hyperliquid connector
│   └── utils/                # Utilities
│       ├── config.py         # Configuration
│       └── logger.py         # Logging
├── tests/                    # Unit tests
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Data sources
data:
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
  timeframes:
    regime: "4h"      # Regime detection
    signal: "30m"     # Signal generation
    execution: "5m"   # Execution/stops

# Kronos model
kronos:
  model_path: "Kronos/model/base/model.safetensors"
  context_length: 512
  frozen: true

# Risk management
risk:
  initial_capital: 10000
  max_position_size: 0.25
  max_leverage: 3.0
  volatility_target: 0.15
  max_drawdown: 0.15
```

## Timeframes

| Layer      | Timeframe | Purpose                    |
|------------|-----------|----------------------------|
| Regime     | 4h        | Regime detection           |
| Signal     | 30m       | Trade signal generation    |
| Execution  | 5m        | Stop management, entries   |

## Risk Management

### Regime-Based Position Sizing

| Regime       | Size Multiplier |
|--------------|-----------------|
| Trend        | 1.0x            |
| Chop         | 0.3x            |
| High Vol     | 0.5x            |
| Regime Shock | 0.0x (no trade) |

### Kill Switches

- Max drawdown: 15%
- Daily loss limit: 3%
- Max consecutive losses: 5
- Funding spike: > 1%

## Expected Performance

| Outcome      | Monthly   | Annual     |
|--------------|-----------|------------|
| Conservative | 2-4%      | 25-60%     |
| Strong       | 4-7%      | 60-120%    |
| Exceptional  | 7-12%     | 150%+      |

*With drawdowns controlled*

## Environment Variables

Set environment variables for API access:

```bash
# Binance (optional - public endpoints work without auth)
export BINANCE_API_KEY="your_key"

# Hyperliquid (wallet-based auth for live trading)
export HYPERLIQUID_WALLET_ADDRESS="0x..."
export HYPERLIQUID_API_WALLET_ADDRESS="0x..."
export HYPERLIQUID_PRIVATE_KEY="0x..."
export HYPERLIQUID_NETWORK="mainnet"
```

## Backtesting

The backtest engine implements:

- **Walk-forward validation**: Train on past, test on future
- **Purged windows**: Prevent data leakage
- **Cost-aware execution**: Fees, slippage, funding
- **Position sizing**: Integrated with risk engine

### Metrics Calculated

- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Calmar Ratio
- Profit Factor
- Win Rate
- Exposure

## Development Roadmap

### Phase 1: Local Development (Current)
- [x] Data collection (Binance)
- [x] Feature engineering
- [x] Kronos integration
- [x] Signal model (XGBoost)
- [x] Risk engine
- [x] Backtesting framework

### Phase 2: Validation
- [ ] Walk-forward testing
- [ ] Out-of-sample validation
- [ ] Monte Carlo simulation
- [ ] Parameter sensitivity analysis

### Phase 3: Paper Trading
- [ ] Hyperliquid paper trading
- [ ] Real-time signal generation
- [ ] Performance monitoring

### Phase 4: Live Trading (AWS)
- [ ] AWS deployment
- [ ] Live execution
- [ ] Monitoring dashboard
- [ ] Alert system

## Core Principles

1. **Regime awareness**: Markets switch between states. Adapt accordingly.
2. **Risk first**: The risk engine keeps you alive long enough to profit.
3. **Cost awareness**: Account for fees, slippage, funding in all decisions.
4. **Simplicity**: XGBoost on good features beats complex models on bad features.
5. **Validation**: Walk-forward testing prevents overfitting.

## License

MIT

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results.
