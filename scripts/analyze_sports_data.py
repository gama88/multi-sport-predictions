"""
Analyze available data for all sports to determine
what behavioral proxy features we can create.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

print("="*60)
print("SPORTS DATA ANALYSIS FOR BEHAVIORAL PROXIES")
print("="*60)

# ============ NHL ============
print("\n" + "="*60)
print("1. NHL - EXCELLENT DATA AVAILABLE ✅")
print("="*60)
try:
    nhl = pd.read_csv(DATA_DIR / "nhl" / "game_teams_stats.csv")
    print(f"   Rows: {len(nhl):,}")
    print(f"   Columns: {list(nhl.columns)}")
    print("\n   BEHAVIORAL STATS AVAILABLE:")
    print("   ✅ shots - shot attempts")
    print("   ✅ hits - physical play")
    print("   ✅ pim - penalty minutes (discipline)")
    print("   ✅ giveaways - turnovers")
    print("   ✅ takeaways - steals")
    print("   ✅ blocked - blocked shots")
    print("   ✅ faceOffWinPercentage - possession")
    print("   ✅ powerPlayGoals/Opportunities - special teams")
    print("\n   READY FOR V6 BEHAVIORAL TRAINING!")
except Exception as e:
    print(f"   Error: {e}")

# ============ NFL ============
print("\n" + "="*60)
print("2. NFL - LIMITED DATA ⚠️")
print("="*60)
try:
    nfl = pd.read_csv(DATA_DIR / "nfl" / "spreadspoke_scores.csv")
    print(f"   Rows: {len(nfl):,}")
    print(f"   Columns: {list(nfl.columns)}")
    print("\n   HAS:")
    print("   ✅ scores, spread, over/under")
    print("   ✅ weather data (temperature, wind, humidity)")
    print("\n   MISSING FOR BEHAVIORAL:")
    print("   ❌ Passing yards, rushing yards")
    print("   ❌ Turnovers (fumbles, interceptions)")
    print("   ❌ Sacks, penalties")
    print("   ❌ Time of possession")
    print("\n   NEED: Fetch from NFL stats API or pro-football-reference")
except Exception as e:
    print(f"   Error: {e}")

# ============ MLB ============
print("\n" + "="*60)
print("3. MLB - NEED TO COMBINE FILES ⚠️")
print("="*60)
try:
    games = pd.read_csv(DATA_DIR / "mlb" / "games.csv")
    print(f"   games.csv Rows: {len(games):,}")
    print(f"   games.csv Columns: {list(games.columns)}")
    
    # Check batting stats
    batting = pd.read_csv(DATA_DIR / "mlb" / "Batting.csv", nrows=5)
    print(f"\n   Batting.csv Columns: {list(batting.columns)}")
    
    pitching = pd.read_csv(DATA_DIR / "mlb" / "Pitching.csv", nrows=5)
    print(f"   Pitching.csv Columns: {list(pitching.columns)}")
    
    print("\n   BATTING HAS:")
    print("   ✅ AB, R, H, 2B, 3B, HR - hitting stats")
    print("   ✅ RBI, BB, SO - plate discipline")
    print("   ✅ SB, CS - base running")
    
    print("\n   PITCHING HAS:")
    print("   ✅ W, L, ERA, IP - pitcher performance")
    print("   ✅ SO, BB, H - pitcher control")
    
    print("\n   CHALLENGE: Need to aggregate per-game team stats")
except Exception as e:
    print(f"   Error: {e}")

# ============ SUMMARY ============
print("\n" + "="*60)
print("SUMMARY - ACTION PLAN")
print("="*60)
print("""
PRIORITY ORDER:

1. NHL - READY NOW ✅
   - game_teams_stats.csv has all behavioral stats
   - Can directly apply V6 approach
   
2. NFL - NEED EXTERNAL DATA ⚠️
   - Current data is scores only
   - Need to fetch team stats from:
     - NFL.com API
     - ESPN API
     - Pro Football Reference scraping
   
3. MLB - NEEDS PROCESSING ⚠️
   - Has batting/pitching stats but not per-game
   - Need to aggregate to team-level per-game
   - Or fetch from MLB Stats API
""")
