1. The mission

Turn $10k of retail capital into a high-Sharpe, regime-aware swing trading system that:

Trades crypto perps

Survives drawdowns

Compounds steadily

Runs on consumer hardware + AWS

This is not HFT.
This is not day trading.
This is AI-driven regime trading on 30m to 4h timeframes.

2. The core idea

Markets are not random.
They switch between latent regimes:

Trend

Chop

Volatility expansion

Mean reversion

Panic

Funding squeezes

Most strategies fail because they assume one regime.

Kronos gives you a neural regime sensor.

Your system uses Kronos to decide:

Is this a market worth trading right now, and in what style?

3. The three-layer architecture
Market Data
   ↓
Kronos (Regime + Distribution Model)
   ↓
Signal Model (LightGBM / XGBoost)
   ↓
Risk & Execution Engine


Each layer does exactly one job.

Layer 1: Market data

You collect:

OHLCV (5m, 30m, 4h)

Funding rates

Volatility

Optional open interest

Source:

Binance for deep history

Hyperliquid for live + validation

Stored as:

Parquet on disk or S3

Normalized, resampled, gap-free

This is the raw reality the system sees.

Layer 2: Kronos (your AI edge)

You run Kronos-base as a frozen encoder.

For every rolling window of bars (context=512), Kronos outputs:

Regime embedding vector

Predicted return distribution

Volatility estimate

Perplexity / reconstruction error (regime change alarm)

This converts noisy candles into high-level market state.

You do not trade from Kronos directly.

You turn its output into features.

Layer 3: Signal model

A simple, powerful learner:

XGBoost

Inputs:

Kronos embeddings

Kronos shift alarms

Forward return quantiles

Funding rate

Volatility

Momentum

4h regime filter

Target:

Probability that the next 3 hours of trading will be profitable after costs

Output:

P(long)

P(short)

P(flat)

This layer learns when Kronos is trustworthy.

This is where real alpha comes from.

Layer 4: Risk engine

This is what keeps you alive.

It controls:

Volatility targeting

Position sizing

Leverage

Drawdown limits

Max exposure

Kill switches

Rules:

Larger positions in stable trending regimes

Smaller positions in chop or high volatility

No trading during Kronos regime shocks

This is why $10k survives long enough to grow.

4. Timeframes

The system trades where retail has an edge:

Layer	Timeframe
Regime	4h
Signal	30m
Execution + stops	5m

You never trade against HFTs.
You never wait months.

You trade regime-filtered swing moves.

5. What you trade

Crypto perps:

BTC

ETH

SOL

Later:

Add top-20 alts

Execution:

Hyperliquid (low fees, good API)

Backtests:

Binance candles + funding data

6. How you backtest

You use:

Walk-forward splits

Purged windows

Cost-aware execution

Funding simulation

Slippage

Position sizing

You measure:

CAGR

Max drawdown

Sharpe

Profit factor

Exposure

If it fails here, it dies.

7. Hardware and infrastructure

Local RTX 4090:

Kronos inference

Feature extraction

Model iteration

Modal:

Parallel runs

Optional fine-tuning (LoRA)

No colo.
No latency games.

Just intelligence.

8. Retraining strategy

You do not retrain Kronos first.

You:

Use Kronos-base as frozen

Train signal model on top

If needed, LoRA-fine-tune Kronos on BTC/ETH/SOL

That is the correct order.

9. Expected performance

If built correctly:

Outcome	Monthly	Annual
Conservative	2 to 4%	25 to 60%
Strong	4 to 7%	60 to 120%
Exceptional	7 to 12%	150%+

With drawdowns controlled.

That is how 10k becomes 100k.

Final truth

You are not building a trading bot.

You are building a regime-aware financial machine that:

Knows when markets change

Knows when not to trade

Sizes risk dynamically


The source code and instructions for kronos is here \Kronos

Base model is here: \Kronos\model\base\model.safetensors