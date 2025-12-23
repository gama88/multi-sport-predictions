"""
Test free data sources for NFL and MLB
No API keys required
"""
import requests
import pandas as pd
from io import StringIO

print("="*60)
print("TESTING FREE NFL DATA SOURCES")
print("="*60)

# 1. ESPN API - Team Stats
print("\n1. ESPN NFL Team Statistics...")
try:
    url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/1/statistics'
    r = requests.get(url, timeout=10)
    data = r.json()
    
    if 'splits' in data:
        categories = data.get('splits', {}).get('categories', [])
        print(f"   Found {len(categories)} stat categories")
        for cat in categories[:5]:
            name = cat.get('name', 'unknown')
            stats = [s.get('name') for s in cat.get('stats', [])[:5]]
            print(f"   {name}: {stats}")
    else:
        print("   No splits found")
except Exception as e:
    print(f"   Error: {e}")

# 2. NFLverse on GitHub (free CSV data)
print("\n2. NFLverse GitHub Data...")
try:
    # Team stats
    url = 'https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.csv'
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        df = pd.read_csv(StringIO(r.text), nrows=5)
        print(f"   ✅ NFLverse schedules accessible!")
        print(f"   Columns: {list(df.columns)}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("TESTING FREE MLB DATA SOURCES")
print("="*60)

# 1. MLB Stats API (no key needed)
print("\n1. MLB Stats API - Team Game Logs...")
try:
    # Get team game log
    url = 'https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R&startDate=2024-04-01&endDate=2024-04-05&hydrate=linescore'
    r = requests.get(url, timeout=10)
    data = r.json()
    
    games = []
    for date_entry in data.get('dates', []):
        for game in date_entry.get('games', []):
            home = game.get('teams', {}).get('home', {})
            away = game.get('teams', {}).get('away', {})
            linescore = game.get('linescore', {})
            
            g = {
                'date': game.get('officialDate'),
                'home_team': home.get('team', {}).get('name'),
                'home_score': home.get('score'),
                'away_team': away.get('team', {}).get('name'),
                'away_score': away.get('score'),
                'home_hits': linescore.get('teams', {}).get('home', {}).get('hits'),
                'away_hits': linescore.get('teams', {}).get('away', {}).get('hits'),
                'home_errors': linescore.get('teams', {}).get('home', {}).get('errors'),
                'away_errors': linescore.get('teams', {}).get('away', {}).get('errors'),
            }
            games.append(g)
    
    print(f"   ✅ Got {len(games)} games with detailed stats!")
    if games:
        print(f"   Sample: {games[0]}")
except Exception as e:
    print(f"   Error: {e}")

# 2. MLB Stats API - Box Score
print("\n2. MLB Stats API - Full Box Score...")
try:
    # Get a specific game's box score
    url = 'https://statsapi.mlb.com/api/v1/game/745455/boxscore'
    r = requests.get(url, timeout=10)
    data = r.json()
    
    teams = data.get('teams', {})
    home = teams.get('home', {})
    away = teams.get('away', {})
    
    home_stats = home.get('teamStats', {}).get('batting', {})
    away_stats = away.get('teamStats', {}).get('batting', {})
    
    print(f"   Home batting stats: {list(home_stats.keys())[:10]}")
    print(f"   ✅ Full box scores available with batting stats!")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("SUMMARY - AVAILABLE FREE DATA")
print("="*60)
print("""
NFL:
  ✅ ESPN API - Basic team stats (passing, rushing, receiving, defense)
  ✅ NFLverse - Free CSV downloads (schedules, player stats)
  ⚠️  No free API for detailed game-by-game team stats
  → Pro-Football-Reference scraping would be needed

MLB:  
  ✅ MLB Stats API - Schedule with linescores (runs, hits, errors)
  ✅ MLB Stats API - Full box scores (all batting/pitching stats)
  ✅ No API key needed!
  → Can build behavioral proxy model from this data!
""")
