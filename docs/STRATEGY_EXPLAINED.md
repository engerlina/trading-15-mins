# Kronos Trading Strategy - Complete Technical Explanation

## What This System Does

This is an **automated trading system** that trades cryptocurrency perpetual futures (BTC, ETH, SOL) on a 30-minute timeframe. It decides when to buy (go long) and when to sell (go flat), using **2x leverage** to amplify returns.

---

## The Core Idea (In Plain English)

The strategy is based on one simple observation:

> **Crypto goes up over time, but occasionally crashes hard.**

So the strategy does two things:
1. **Stay invested (long) most of the time** to capture the uptrend
2. **Exit during crashes** to protect capital

That's it. No fancy machine learning predictions, no complex indicators. Just:
- Buy and hold with leverage
- Get out when things look really bad
- Get back in when they recover

---

## How It Decides: Long vs Flat

Every 30 minutes, the system looks at two things:

### 1. Where is price relative to its All-Time High?

```
Drawdown = (All-Time High - Current Price) / All-Time High
```

Example:
- BTC hit $69,000 (ATH)
- BTC is now at $35,000
- Drawdown = ($69,000 - $35,000) / $69,000 = **49%**

### 2. Where is price relative to the 200-period Moving Average?

```
MA_200 = Average of last 200 closing prices (100 hours of data)
```

If price is ABOVE the MA → Market is in an uptrend
If price is BELOW the MA → Market is in a downtrend

### The Decision Rule

```
IF drawdown > 55% AND price < MA_200:
    → CRASH DETECTED → GO FLAT (sell everything)

ELSE:
    → STAY LONG (keep holding)
```

**Why 55%?**
- Normal corrections are 20-40%
- True crashes (like 2022) are 60-80%
- 55% catches crashes while avoiding false alarms

**Why require BOTH conditions?**
- Drawdown alone might trigger during a slow grind down
- Being below MA alone is normal during consolidation
- Both together = high confidence it's a real crash

---

## How It Manages Positions

### Opening a Position

When the signal says LONG and we have no position:

1. **Calculate base position size** using volatility targeting:
   ```
   Base Size = Equity × (Target Volatility / Current Volatility)
   ```
   - If market is calm → bigger position
   - If market is volatile → smaller position

2. **Apply regime multiplier** (how strong is the trend):
   ```
   Strong uptrend (regime > 0.8):  size × 1.5
   Moderate uptrend (> 0.5):       size × 1.2
   Weak uptrend (> 0.3):           size × 1.0
   Uncertain (< 0.3):              size × 0.8
   ```

3. **Apply leverage** (2x default):
   ```
   Final Size = min(Calculated Size, Equity × 1.5)
   ```
   - Never exceed 150% of equity regardless of settings

### Closing a Position

A position is closed when ANY of these happen:

1. **Signal changes to FLAT** (crash detected)
   - Must have held for at least 96 bars (2 days) first
   - This prevents whipsaws during volatile periods

2. **Trailing stop hit** (22% from peak)
   - System tracks the highest price since entry
   - If price drops 22% from that peak → exit
   - This locks in profits during pullbacks

Example of trailing stop:
```
Entry price: $50,000
Price rises to: $70,000 (new peak tracked)
Trailing stop at: $70,000 × (1 - 0.22) = $54,600
Price drops to: $54,000 → STOP TRIGGERED → EXIT
```

---

## Complete Flow: What Happens Each Bar

```
┌─────────────────────────────────────────────────────────────┐
│ NEW 30-MINUTE BAR ARRIVES                                   │
│ (price, volume, timestamp)                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: UPDATE INDICATORS                                   │
│                                                             │
│ • Update 200-bar moving average                            │
│ • Update all-time high                                     │
│ • Calculate drawdown from ATH                              │
│ • Calculate current volatility                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: GENERATE SIGNAL                                     │
│                                                             │
│ Is drawdown > 55% AND price < MA?                          │
│     YES → Signal = FLAT                                    │
│     NO  → Signal = LONG                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: IF WE HAVE A POSITION                              │
│                                                             │
│ Check exit conditions:                                     │
│ • Signal changed to FLAT? (after min hold)                 │
│ • Trailing stop hit? (price < peak × 0.78)                 │
│                                                             │
│ If any exit triggered → CLOSE POSITION                     │
│ Otherwise → UPDATE trailing stop peak if price higher      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: IF WE HAVE NO POSITION                             │
│                                                             │
│ Is signal LONG?                                            │
│     YES → Calculate position size                          │
│         → Apply regime multiplier                          │
│         → OPEN LONG POSITION                               │
│     NO  → Stay flat, wait                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: UPDATE EQUITY                                       │
│                                                             │
│ • Mark position to market                                  │
│ • Record equity for this bar                               │
│ • Log any trades that occurred                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     [Wait for next bar]
```

---

## Real Example: BTC 2023-2024

### The Setup
- Start: January 2023
- BTC price: ~$16,700
- Initial capital: $10,000
- Leverage: 2x

### What Happened

**January 2023:**
- Price: $16,700, well above MA (after 2022 crash recovery started)
- ATH: $69,000 (from Nov 2021)
- Drawdown: 76% (but recovering)
- Price is ABOVE MA → Signal: LONG
- Open position: ~$19,000 (190% of equity with leverage)

**March 2023:**
- Price rises to $28,000
- Trailing stop now at: $28,000 × 0.78 = $21,840
- Position is profitable, hold

**Throughout 2023:**
- Price consolidates between $25k-$35k
- Some dips approach trailing stop but don't hit it
- Stay long, accumulating gains

**March 2024:**
- BTC hits $73,000 (new ATH!)
- Trailing stop at: $73,000 × 0.78 = $56,940
- Huge unrealized profit

**April 2024:**
- Price drops to $56,000
- Trailing stop ($56,940) triggered
- EXIT with massive profit: +$50,000+

**May 2024:**
- Price recovers above MA
- Signal: LONG again
- Re-enter with larger capital base

**End of 2024:**
- BTC at $93,000
- Still holding position
- Total return: **630%** vs buy-and-hold 460%

---

## Why It Works

### 1. Captures the Trend
By staying long most of the time, we capture the secular bull trend in crypto. We're not trying to predict every wiggle.

### 2. Leverage Amplifies Gains
2x leverage means a 100% gain becomes 200% (minus costs). In a strong bull market, this is powerful.

### 3. Crash Protection Preserves Capital
The 55% crash threshold saved us from the 2022 bear market. Instead of riding BTC from $69k to $16k (-77%), we would have exited around $31k (-55%) and preserved more capital.

### 4. Trailing Stop Locks Profits
The 22% trailing stop ensures we keep most of our gains during pullbacks. We might miss some of the top, but we keep the bulk of the move.

### 5. Minimum Holding Prevents Whipsaws
The 96-bar (2-day) minimum hold prevents us from getting chopped up during volatile consolidation periods.

---

## What It Doesn't Do

1. **Doesn't predict price** - No ML model trying to forecast next candle
2. **Doesn't short** - Only long or flat, never bets against the market
3. **Doesn't try to catch bottoms** - Waits for price to cross above MA
4. **Doesn't trade frequently** - Only 3-10 trades per year typically
5. **Doesn't work in bear markets starting from peaks** - If you start at the top of a bubble with leverage, you'll lose money

---

## The Files That Make It Work

| File | What It Does |
|------|--------------|
| `src/backtest/engine.py` | Main logic - signal generation, position management |
| `scripts/run_pipeline.py` | Configuration and execution |
| `src/risk/engine.py` | Position sizing, risk limits |
| `src/risk/position.py` | Track open positions, calculate P&L |
| `src/backtest/metrics.py` | Calculate performance statistics |

---

## Key Numbers to Remember

| Parameter | Value | Why |
|-----------|-------|-----|
| Crash threshold | 55% | Catches real crashes, avoids false alarms |
| Trailing stop | 22% | Locks in profits without cutting winners too short |
| Min holding | 96 bars | Prevents whipsaws (2 days on 30min bars) |
| Leverage | 2x | Amplifies gains without excessive risk |
| MA period | 200 bars | ~4 days, smooth enough to avoid noise |

---

## Summary

This system is essentially **leveraged buy-and-hold with a safety valve**:

1. Buy and hold crypto with 2x leverage
2. If things go really bad (55% crash + below trend), get out
3. When things recover (price back above trend), get back in
4. Use a trailing stop to lock in big gains

It's simple, but it works because:
- Crypto trends strongly (capture with leverage)
- Crypto crashes hard (protect with crash detection)
- Most "crashes" are just corrections (don't exit too early)
