# Multi-Sport Specialized Model Results

## Date: December 23, 2025

## Summary
Tested specialized (pace-focused) models vs V6 (behavioral proxy) for ALL sports and bet types.

## 🏆 MAJOR FINDINGS

### 1. NHL Moneyline: MASSIVE Improvement (+20.8pp!)
```
V6:          51.2%
Specialized: 72.0%   ← +20.8pp improvement!
```
The NHL moneyline was severely underperforming with V6. Simple win/loss features work much better.

### 2. TOTALS Improved Across ALL Sports
Pace-focused features (combined_pace, defense_allowing) beat behavioral proxy for predicting over/under.

| Sport | V6 Total | Specialized | Improvement |
|-------|----------|-------------|-------------|
| NBA | 55.0% | 62.2% | **+7.2pp** |
| Soccer | 55.0% | 61.5% | **+6.5pp** |
| MLB | 53.0% | 58.4% | **+5.4pp** |
| NHL | 56.0% | 60.1% | **+4.1pp** |
| NFL | 53.0% | 56.3% | **+3.3pp** |

### 3. Moneyline/Spread: V6 Still Best (mostly)
For predicting winners and margins, V6's behavioral features remain optimal.

## Production Configuration

| Sport | Moneyline | Spread | Total |
|-------|-----------|--------|-------|
| NBA | V6 (65.4%) | V6 (73.4%) | **Specialized (62.2%)** |
| NFL | V6 (65.1%) | V6 (65.2%) | **Specialized (56.3%)** |
| NHL | **Specialized (72.0%)** | V6 (59.1%) | **Specialized (60.1%)** |
| MLB | **Specialized (53.4%)** | V6 (55.6%) | **Specialized (58.4%)** |
| Soccer | V6 (64.3%) | V6 (75.3%) | **Specialized (61.5%)** |
| Tennis | V6 (62.8%) | - | - |

## Key Insight

**DIFFERENT PREDICTION PROBLEMS NEED DIFFERENT FEATURES:**

1. **Moneyline/Spread** = "Who is better?"
   - Best features: Win%, net rating, behavioral proxies
   - V6 approach works well

2. **Total/O-U** = "How much will they score combined?"
   - Best features: PACE, combined scoring, defense allowing
   - Specialized approach works better

## Files
- Testing script: `scripts/test_all_sports_specialized.py`
- Results JSON: `models/multi_sport_comparison.json`
