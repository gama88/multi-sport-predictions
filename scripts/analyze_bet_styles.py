"""
Analyze V6 model performance across bet styles.
Determine if we need specialized models for each bet type.
"""
import json
from pathlib import Path

# Load V6 metrics
with open('models/v6_nba_metrics.json') as f:
    metrics = json.load(f)

print("="*60)
print("V6 BEHAVIORAL PROXY MODEL - BET STYLE ANALYSIS")
print("="*60)

print("\n📊 Current Results (Same Features, Different Targets):")
print("-"*50)
for bet_type, m in metrics.items():
    acc = m['accuracy'] * 100
    auc = m['auc']
    status = "✅ Great" if acc >= 65 else "⚠️ Needs work" if acc < 58 else "👍 Good"
    print(f"  {bet_type.upper():12} | Accuracy: {acc:5.1f}% | AUC: {auc:.4f} | {status}")

print("\n" + "="*60)
print("🔍 ANALYSIS")
print("="*60)

print("""
SPREAD (73.4%) >>> BEST PERFORMER! ✨
  - Behavioral features DIRECTLY predict margin of victory
  - Fatigue → tired teams lose by MORE points
  - Discipline → undisciplined teams give up more points
  - Chemistry → well-gelled teams dominate

MONEYLINE (65.4%) - Good Performance 👍
  - Core use case for behavioral proxies
  - Answers: "Who is the better team TODAY?"
  - Behavioral factors capture "form" better than just win rate

CONTRACTS (65.4%) - Good Performance 👍
  - Same underlying model as moneyline
  - Isotonic calibration makes probabilities market-ready
  - Good for prediction market pricing

OVER/UNDER (55.0%) - NEEDS SPECIALIZED FEATURES ⚠️
  - Current features don't capture PACE/TEMPO
  - Behavioral proxies answer WHO wins, not HOW MANY points
  - Near-random performance indicates wrong feature set
""")

print("="*60)
print("💡 RECOMMENDATIONS")
print("="*60)

print("""
OPTION 1: Keep Single Model (Current) 
  ✅ Moneyline: 65.4% - Use as-is
  ✅ Spread: 73.4% - Use as-is (best performer!)
  ✅ Contracts: 65.4% - Use as-is
  ⛔ O/U: 55.0% - DON'T USE (upgrade to specialized model)

OPTION 2: Create Specialized O/U Model
  Need different features for O/U:
  - Combined pace indicator (possessions/game for both teams)
  - Combined offensive efficiency (both teams)
  - Combined defensive efficiency (both teams) 
  - Historical head-to-head totals
  - Recent over/under trends (last 5 games for each team)
  - Vegas line as feature (if available)
  
  This would likely push O/U to 60-65%

OPTION 3: Ensemble Approach (Best)
  - Use V6 behavioral for ML/Spread/Contracts
  - Create V6-OU specialized for Over/Under
  - Each model optimized for its task
""")

print("\n" + "="*60)
print("🎯 VERDICT")  
print("="*60)
print("""
The V6 behavioral model works GREAT for:
  ✅ Spread (73.4% - best!)
  ✅ Moneyline (65.4%)
  ✅ Contracts (65.4%)

RECOMMENDED: Create a specialized O/U model with pace/tempo features
""")
