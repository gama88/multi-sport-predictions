# V14 Best Model Selection

## Summary

After extensive testing, here are the **BEST** models for each bet type:

| Bet Type | Best Model | Accuracy | Key Features |
|----------|------------|----------|--------------|
| Moneyline | **V6** | **65.4%** | Behavioral proxy + outcomes |
| Spread | **V6** | **73.4%** | Behavioral proxy + outcomes |
| Total/O-U | **V13** | **61.9%** | Pace + combined scoring |
| Contracts | **V6** | **65.4%** | Same as moneyline |

## Why Different Models Work for Different Bet Types

### Moneyline (V6 wins)
- **Predicts: WHO wins**
- Best features: Win%, net rating, behavioral proxies (steals, blocks, turnovers)
- These capture team QUALITY which determines winners

### Spread (V6 wins)
- **Predicts: BY HOW MUCH**
- Best features: Same as moneyline + consistency metrics
- V6's behavioral features capture the underlying margin dynamics

### Total/O-U (V13 wins!)
- **Predicts: COMBINED SCORE**
- Best features: PACE, combined scoring, defense allowing
- V6's approach focused on winner/margin, NOT combined output
- V13's pace features (combined_pace, home_pace, away_pace) are KEY

## Key Insight
Different bet types are fundamentally different prediction problems:
- Moneyline/Spread = comparative (who is better)
- Total = additive (how much will both teams score combined)

## Production Recommendation
Use a **hybrid approach**:
```python
def get_prediction(bet_type):
    if bet_type == 'total':
        return v13_total_model.predict(X)
    else:  # moneyline, spread, contracts
        return v6_model.predict(X)
```

## Combined Accuracy
| Bet Type | Best Accuracy |
|----------|---------------|
| Moneyline | 65.4% |
| Spread | 73.4% |
| Total | 61.9% |
| **Average** | **66.9%** |

This is an improvement from V6's average of 64.6%!
