# Kronos Strategy Research Log

## Session: January 11, 2026

### Overview
Systematic testing of different strategy approaches to beat buy-and-hold benchmark for BTC, ETH, SOL.

---

## Approach 1: Simple Rules (Crash Protection)

**Description:** Long-only strategy with crash protection
- Stay LONG unless >55% drawdown from ATH AND price below 200-bar MA
- 22% trailing stop
- 96-bar minimum holding period
- 2x leverage (capped at 1.5x actual position)

**Results (2023-2024):**
| Symbol | Strategy | Benchmark | Alpha | Status |
|--------|----------|-----------|-------|--------|
| BTCUSDT | 619.13% | 459.60% | +159.53% | BEATS |
| ETHUSDT | 46.23% | 45.74% | +0.49% | BEATS |
| SOLUSDT | 77.34% | 83.34% | -6.00% | UNDER |

**Pros:**
- Consistent across BTC and ETH
- Minimal trades (2-3/year) = low fees
- Simple to understand and implement

**Cons:**
- Doesn't beat SOL benchmark
- Can't profit from downturns (long-only)
- Starting at market peak (2022) causes major losses

---

## Approach 2: Pure Kronos (XGBoost Signals)

**Description:** Use trained XGBoost model signals directly
- Model predicts: P(long), P(flat), P(short)
- Signals based on Kronos embeddings, perplexity, momentum, etc.
- No rule overrides

**Results (2023-2024):**
| Symbol | Strategy | Benchmark | Alpha | Status |
|--------|----------|-----------|-------|--------|
| BTCUSDT | -27.74% | 459.60% | -487.33% | LOSS |
| ETHUSDT | -21.53% | 45.74% | -67.27% | LOSS |
| SOLUSDT | 766.52% | 83.34% | +683.18% | BEATS |

**Pros:**
- Amazing performance on SOL (+683% alpha)
- ML-based adaptation

**Cons:**
- Catastrophic losses on BTC and ETH
- 170-348 trades = $1,000-5,600 in fees
- Shorts in a bull market (loses money)
- Inconsistent across assets

**Why it failed for BTC/ETH:**
- Model trained to predict "profitable trades" not "beat benchmark"
- Too many signals causing whipsaws
- Shorting during secular bull market

---

## Approach 3: Hybrid (Kronos + Simple Rules)

**Description:** Simple rules with Kronos uncertainty adjustment
- Base: Same as Simple Rules
- Enhancement: Reduce position size when Kronos perplexity z-score > 2
- Enhancement: Reduce position size when reconstruction error high

**Results (2023-2024):**
| Symbol | Strategy | Benchmark | Alpha | Status |
|--------|----------|-----------|-------|--------|
| BTCUSDT | 513.58% | 459.60% | +53.98% | BEATS |
| ETHUSDT | 46.45% | 45.74% | +0.71% | BEATS |
| SOLUSDT | 74.69% | 83.34% | -8.65% | UNDER |

**Pros:**
- Still beats benchmark on BTC/ETH
- Uses ML features without full dependency

**Cons:**
- Worse than pure simple rules for BTC (513% vs 619%)
- Kronos penalties reduced position during profitable periods
- Added complexity without benefit

---

## Key Learnings

1. **Simple beats complex** - Rule-based approach outperformed ML for BTC/ETH
2. **Long-only in bull market** - Shorting crypto in 2023-2024 was a losing strategy
3. **Fees matter** - 348 trades × fees destroys returns vs 2 trades
4. **ML inconsistency** - Kronos worked great for SOL but failed for BTC/ETH
5. **Benchmark is hard** - Buy-and-hold during bull market is tough to beat

---

---

## Research: Short Timeframe Strategies (5m/15m)

### Current Infrastructure Status
- **5m data available:** ~80 days (need 6+ months for ML)
- **15m data:** Not collected yet (supported by collector)
- **Current base timeframe:** 30m
- **5m currently used for:** Stop management only, not primary signals

### Strategy 1: RSI Mean Reversion (5m/15m)

**Concept:** Fade overbought/oversold extremes when RSI hits 30/70 levels

**Recommended Settings:**
- Scalping (5m): RSI period 5-7, levels 80/20
- Day trading (15m): RSI period 9-10, levels 75/25
- High volatility: Widen to 65/35

**Implementation:**
```python
# Pseudo-code
if rsi_5m < 30 and rsi_1h < 40:  # Multi-TF confirmation
    signal = LONG
elif rsi_5m > 70 and rsi_1h > 60:
    signal = SHORT
```

**Expected Performance:**
- Win rate: 55-65% with multi-TF confirmation
- Avg trade: 0.5-1% profit target
- Stop: 0.3-0.5% (tight)
- Trades per day: 5-15

**Pros:** High frequency, consistent small gains
**Cons:** High fees, needs tight execution, choppy in trends

**Source:** [Best RSI Settings for 5-Minute Charts](https://eplanetbrokers.com/en-US/training/best-rsi-settings-for-5-minute-charts)

---

### Strategy 2: RSI Divergence (15m)

**Concept:** Trade when price makes new high/low but RSI doesn't confirm

**Types:**
- Bullish divergence: Price lower low + RSI higher low → BUY
- Bearish divergence: Price higher high + RSI lower high → SELL

**Backtesting Results:**
- One study: 87.5% win rate (7/8 trades) with candlestick confirmation
- Caveat: Small sample size (16 setups), need 100+ for validity

**Implementation:**
- Use pivot detection (5 bars left/right)
- Require candlestick confirmation (engulfing, pin bar)
- Stop: Below/above divergence swing point

**Pros:** High probability when confirmed
**Cons:** Rare setups (2-5 per week), need fast execution on 5m

**Source:** [RSI Trading Strategy Backtest](https://www.quantifiedstrategies.com/rsi-trading-strategy/)

---

### Strategy 3: Bollinger Band Scalping (5m)

**Concept:** Trade bounces off Bollinger Band extremes

**Entry Rules:**
- LONG: Price touches lower band + RSI < 30
- SHORT: Price touches upper band + RSI > 70
- Confirmation: Wait for close back inside bands

**Exit Rules:**
- Target: Middle band (20 SMA)
- Stop: Beyond the band that was touched

**Expected Performance:**
- Win rate: 60-70% in ranging markets
- Fails badly in trending markets (need regime filter)

**Implementation Notes:**
- Use 4h regime filter (only trade when NOT in strong trend)
- Reduce size when Bollinger Band width expanding (volatility spike)

**Source:** [Crypto Scalping Strategies](https://fxopen.com/blog/en/5-best-crypto-scalp-trading-strategies/)

---

### Strategy 4: Momentum Breakout (15m/1h)

**Concept:** Enter when price breaks key levels with volume confirmation

**Entry Rules:**
- Price breaks above resistance (or below support)
- Volume > 1.5x average
- RSI confirms direction (>50 for longs, <50 for shorts)

**Exit Rules:**
- Trail stop at prior support/resistance
- Take profit at 1.5-2x risk

**Implementation:**
- Identify S/R from 4h chart
- Execute on 15m breakout with volume
- Use ATR for stop distance

**Pros:** Catches big moves, good R:R
**Cons:** Many false breakouts, needs good S/R identification

---

### Strategy 5: Funding Rate Arbitrage (Market Neutral)

**Concept:** Earn funding without directional exposure

**Mechanics:**
- Go LONG spot (buy BTC)
- Go SHORT perpetual futures (same size)
- Collect funding rate payments (every 8 hours)

**2025 Performance:**
- Average funding rate: 0.015% per 8 hours
- Annualized return: ~19% (up from 14% in 2024)
- Research showed up to 115.9% returns over 6 months
- Max drawdown: ~2% (from basis risk)

**Implementation Requirements:**
- Need spot AND futures trading capability
- Need to monitor basis (perp vs spot price)
- Need capital on two exchanges or use delta-neutral perp

**Pros:** Market neutral, consistent returns, low drawdown
**Cons:** Capital intensive, execution complexity, basis risk

**Source:** [Ultimate Guide to Funding Rate Arbitrage](https://blog.amberdata.io/the-ultimate-guide-to-funding-rate-arbitrage-amberdata)

---

### Strategy 6: Multi-Timeframe RSI Confluence (5m + 1h)

**Concept:** Only trade when multiple timeframes agree

**Rules:**
- 1h RSI sets direction (>50 = bullish bias, <50 = bearish bias)
- 5m RSI provides entry (oversold in uptrend, overbought in downtrend)
- Both must be extreme (5m < 30 AND 1h < 50 for longs)

**Backtesting Claims:**
- Multi-TF confirmation raises accuracy to 90%+ in robust backtests

**Implementation:**
```python
# Buy when 5m oversold in 1h uptrend
if rsi_1h > 50 and rsi_5m < 30:
    signal = LONG
# Sell when 5m overbought in 1h downtrend
if rsi_1h < 50 and rsi_5m > 70:
    signal = SHORT
```

**Source:** [Time Interval Analysis in Crypto](https://www.youhodler.com/education/time-interval-analysis-1m-5m-15m-1h-4h-1d-1w)

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 weeks)
1. **Collect more 5m data** - Need 6+ months minimum
2. **Implement RSI mean reversion** - Simplest to code and test
3. **Add Bollinger Bands to feature engineer** - Already have infrastructure

### Phase 2: Advanced (2-4 weeks)
4. **Multi-TF RSI system** - Combine 5m signals with 1h/4h filter
5. **Divergence detection** - Requires pivot/swing point detection
6. **Breakout system** - Requires S/R level identification

### Phase 3: Market Neutral (4-6 weeks)
7. **Funding rate arbitrage** - Requires spot trading capability
8. **Basis trading** - Complex execution logic

---

## BACKTEST RESULTS: Short Timeframe Strategies (January 12, 2026)

### Data Used
- **Timeframe**: 5m resampled to 15m for RSI Divergence
- **Period**: 2024-01-01 to 2024-12-31 (full year)
- **Leverage**: 2x

### Strategy 1: RSI Mean Reversion - FAILED

**Parameters tested:**
- RSI period: 7
- Oversold: 25, Overbought: 75
- Profit target: 2%, Stop loss: 1%
- Long-only mode

**Results (BTC 2024):**
- Return: -51.79% vs Benchmark +118.57%
- Win rate: 45.49%
- Problem: Exits too early (RSI neutral or max_time), misses trends

**Conclusion:** Mean reversion doesn't work in trending crypto markets.

---

### Strategy 2: RSI Divergence with Trailing Stops - SUCCESS!

**Parameters:**
- RSI period: 14
- Pivot detection: 5 bars
- Divergence window: 10-50 bars
- Entry: Bullish divergence when RSI < 40
- Exit: 10% trailing stop OR 1% stop loss OR 96 bars max
- Long-only mode, 2x leverage

**Results (2024):**

| Symbol | Strategy | Benchmark | Alpha | Win Rate | Trades |
|--------|----------|-----------|-------|----------|--------|
| BTCUSDT | 224.51% | 118.08% | +106.42% | 45.12% | 215 |
| ETHUSDT | 220.47% | 46.00% | +174.47% | 40.57% | 212 |
| SOLUSDT | 391.86% | 86.88% | +304.98% | 30.91% | 275 |

**Key Stats (BTC):**
- Sharpe Ratio: 2.75
- Max Drawdown: 43.28%
- Avg Win: $1131, Avg Loss: $695 (1.63:1 ratio)
- Profit Factor: 1.34

**Why it works:**
1. **Bullish divergence** catches bottoms in uptrends
2. **Trailing stop** lets winners run (no fixed profit cap)
3. **Long-only** aligned with crypto bull market
4. **15m timeframe** reduces noise vs 5m

**Script:** `scripts/backtest_short_tf.py`
```bash
python scripts/backtest_short_tf.py --symbol BTCUSDT --strategy rsi_divergence \
  --start 2024-01-01 --end 2024-12-31 --long-only --leverage 2.0 \
  --trailing-stop --trailing-pct 0.10
```

---

### Timeframe Comparison (January 12, 2026)

**Important finding:** The 15m timeframe is crucial for this strategy to beat benchmark!

| Timeframe | Period | BTC Return | Benchmark | Alpha | Trades | Sharpe |
|-----------|--------|------------|-----------|-------|--------|--------|
| 5m→15m | 2024 | 224.51% | 118.08% | **+106.42%** | 215 | 2.75 |
| 30m | 2024 | 103.17% | 117.97% | -14.79% | 107 | 2.71 |
| 30m | 2023-2024 (2yr) | 316.61% | 459.55% | -142.94% | 225 | 2.86 |

**Why 15m beats 30m:**
1. More signals (215 vs 107 trades) = more opportunities
2. Better divergence detection granularity
3. Tighter stops work better on shorter timeframe

**Data Limitation:** 5m data only starts Dec 2023, so 2-year backtest only possible on 30m (which underperforms).

---

### Full Metrics: RSI Divergence (15m, 2024)

| Symbol | Return | Benchmark | Alpha | Trades | Win Rate | Sharpe | Max DD | Profit Factor |
|--------|--------|-----------|-------|--------|----------|--------|--------|---------------|
| BTCUSDT | 224.51% | 118.08% | +106.42% | 215 | 45.12% | 2.75 | 43.28% | 1.34 |
| ETHUSDT | 220.47% | 45.44% | +175.03% | 212 | 40.57% | 2.63 | 34.45% | 1.38 |
| SOLUSDT | 391.86% | 84.50% | +307.35% | 275 | 30.91% | 2.61 | 40.89% | 1.38 |

---

### Comparison: Short TF vs Original 30m Strategy

| Metric | RSI Divergence (15m) | Simple Rules (30m) |
|--------|---------------------|-------------------|
| BTC Return | 224.51% | 619.13% |
| BTC Alpha | +106.42% | +159.53% |
| Trades/Year | 215 | 2 |
| Complexity | Medium | Low |
| Drawdown | 43.28% | 33.22% |

**Verdict:** The 30m Simple Rules strategy still outperforms on raw returns, but the 15m RSI Divergence:
- Provides more trading opportunities
- Works well across all three symbols (Simple Rules fails on SOL)
- Could be combined with the 30m strategy

---

## TODO: Research Areas

### Shorter Timeframe Strategies
- [x] Mean reversion on 5m/15m timeframes (researched above)
- [x] Scalping with tight stops (RSI + Bollinger)
- [x] Momentum breakouts on hourly (researched above)
- [x] Funding rate arbitrage (researched above)

### Alternative Approaches
- [ ] Pairs trading (BTC/ETH spread)
- [ ] Volatility targeting with dynamic leverage
- [ ] Grid trading in ranging markets
- [ ] VWAP strategies for intraday

### Kronos Improvements
- [ ] Retrain model with different labels (beat benchmark vs profit)
- [ ] Use Kronos only for SOL (where it works)
- [ ] Use Kronos perplexity as volatility filter only
- [ ] Ensemble: Kronos for SOL, Simple Rules for BTC/ETH

---

## Data Notes

- **Available data:** Dec 2021 - present
- **Timeframes:** 5m, 30m, 4h
- **Symbols:** BTCUSDT, ETHUSDT, SOLUSDT
- **ETH/SOL limitation:** Only ~1 year of clean data (2024)

---

*Last updated: 2026-01-12*
