"""
Fetch MLB Historical Data from MLB Stats API
=============================================
No API key required! Fetches detailed game-by-game stats.
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "mlb"


def fetch_season_games(season=2024):
    """Fetch all games for a season with detailed stats."""
    print(f"\nFetching {season} MLB games...")
    
    all_games = []
    
    # Fetch by month chunks
    start = datetime(season, 3, 28)  # Opening day around March 28
    end = datetime(season, 10, 1)    # Regular season ends ~Oct 1
    
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=7), end)
        
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R"
        url += f"&startDate={current.strftime('%Y-%m-%d')}"
        url += f"&endDate={chunk_end.strftime('%Y-%m-%d')}"
        url += "&hydrate=linescore,decisions"
        
        try:
            r = requests.get(url, timeout=30)
            data = r.json()
            
            for date_entry in data.get('dates', []):
                for game in date_entry.get('games', []):
                    status = game.get('status', {}).get('abstractGameState', '')
                    if status != 'Final':
                        continue
                    
                    home = game.get('teams', {}).get('home', {})
                    away = game.get('teams', {}).get('away', {})
                    linescore = game.get('linescore', {})
                    
                    g = {
                        'game_id': game.get('gamePk'),
                        'date': game.get('officialDate'),
                        'season': season,
                        'home_team_id': home.get('team', {}).get('id'),
                        'home_team': home.get('team', {}).get('name'),
                        'away_team_id': away.get('team', {}).get('id'),
                        'away_team': away.get('team', {}).get('name'),
                        'home_score': home.get('score', 0),
                        'away_score': away.get('score', 0),
                        'home_hits': linescore.get('teams', {}).get('home', {}).get('hits', 0),
                        'away_hits': linescore.get('teams', {}).get('away', {}).get('hits', 0),
                        'home_errors': linescore.get('teams', {}).get('home', {}).get('errors', 0),
                        'away_errors': linescore.get('teams', {}).get('away', {}).get('errors', 0),
                        'home_lob': linescore.get('teams', {}).get('home', {}).get('leftOnBase', 0),
                        'away_lob': linescore.get('teams', {}).get('away', {}).get('leftOnBase', 0),
                        'innings': len(linescore.get('innings', [])),
                    }
                    all_games.append(g)
            
            print(f"  {current.strftime('%b %d')} - {chunk_end.strftime('%b %d')}: {len(all_games)} games total")
            
        except Exception as e:
            print(f"  Error for {current}: {e}")
        
        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)  # Rate limit
    
    return pd.DataFrame(all_games)


def fetch_multiple_seasons(seasons=[2022, 2023, 2024]):
    """Fetch multiple seasons."""
    all_data = []
    
    for season in seasons:
        df = fetch_season_games(season)
        if len(df) > 0:
            all_data.append(df)
            print(f"  Season {season}: {len(df)} games")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    return None


def main():
    """Fetch and save MLB data."""
    print("="*60)
    print("FETCHING MLB HISTORICAL DATA FROM MLB STATS API")
    print("="*60)
    print("No API key required!")
    
    # Fetch 3 seasons
    df = fetch_multiple_seasons([2022, 2023, 2024])
    
    if df is not None and len(df) > 0:
        path = DATA_DIR / "mlb_games_enhanced.csv"
        df.to_csv(path, index=False)
        print(f"\n✅ Saved {len(df)} games to {path}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample:\n{df.head()}")
    else:
        print("No data fetched")


if __name__ == "__main__":
    main()
