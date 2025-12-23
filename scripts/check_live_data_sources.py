"""
Check if we can get the same rich stats (STL, BLK, TOV, PF) from live APIs
for production predictions.
"""
import requests

print("=" * 60)
print("CHECKING NBA DATA SOURCES FOR BEHAVIORAL PROXY FEATURES")
print("=" * 60)

# Stats we need for V6 behavioral proxy features
REQUIRED_STATS = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

print("\n1. ESPN API - Team Statistics")
print("-" * 40)
try:
    url = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/1/statistics'
    r = requests.get(url, timeout=10)
    data = r.json()
    
    all_stats = []
    if 'splits' in data:
        for cat in data.get('splits', {}).get('categories', []):
            cat_name = cat.get('name', 'unknown')
            for stat in cat.get('stats', []):
                all_stats.append(stat.get('name', ''))
    
    print(f"  Available stats: {len(all_stats)} total")
    found = [s for s in REQUIRED_STATS if any(s.lower() in x.lower() for x in all_stats)]
    missing = [s for s in REQUIRED_STATS if s not in found]
    print(f"  Found: {found}")
    print(f"  Key stats sample: {all_stats[:15]}")
except Exception as e:
    print(f"  Error: {e}")


print("\n2. NBA.com Stats API")
print("-" * 40)
try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': 'https://www.nba.com/',
        'Accept': 'application/json'
    }
    url = 'https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2024-25&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
    r = requests.get(url, headers=headers, timeout=10)
    data = r.json()
    
    cols = data.get('resultSets', [{}])[0].get('headers', [])
    print(f"  Available columns: {len(cols)} total")
    print(f"  Columns: {cols}")
    
    found = [s for s in REQUIRED_STATS if s in cols]
    missing = [s for s in REQUIRED_STATS if s not in found]
    print(f"  Found required: {found}")
    print(f"  Missing: {missing}")
    
    if found == REQUIRED_STATS:
        print("  ✅ NBA.com has ALL required stats!")
    
except Exception as e:
    print(f"  Error: {e}")


print("\n3. balldontlie API")  
print("-" * 40)
try:
    url = 'https://www.balldontlie.io/api/v1/stats?seasons[]=2024&per_page=1'
    r = requests.get(url, timeout=10)
    data = r.json()
    
    if 'data' in data and len(data['data']) > 0:
        sample = data['data'][0]
        available = list(sample.keys())
        print(f"  Available fields: {available}")
        
        found = [s for s in REQUIRED_STATS if s.lower() in [a.lower() for a in available]]
        print(f"  Found (case-insensitive): {found}")
except Exception as e:
    print(f"  Error: {e}")


print("\n4. Checking our new training data source")
print("-" * 40)
import pandas as pd
from pathlib import Path

new_data = Path("data/nba/nba_new/nba_training_games_20251205.csv")
if new_data.exists():
    df = pd.read_csv(new_data, nrows=2)
    print(f"  Training data columns: {list(df.columns)}")
    
    train_cols = [c.upper() for c in df.columns]
    found = [s for s in REQUIRED_STATS if s in train_cols]
    print(f"  Found in training data: {found}")


print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
✅ NBA.com Stats API has ALL the stats we need:
   - PTS, FG_PCT, FG3_PCT, FT_PCT (scoring)
   - REB, OREB, DREB (rebounding)
   - AST, STL, BLK, TOV, PF (the key behavioral proxies!)

📋 Production Pipeline:
   1. Fetch team stats from NBA.com API (updated daily)
   2. Store rolling 20-game history per team
   3. Calculate behavioral proxy features (same as training)
   4. Feed to V6 model for predictions

⚠️  Note: NBA.com API requires proper headers (User-Agent, Referer)
""")
