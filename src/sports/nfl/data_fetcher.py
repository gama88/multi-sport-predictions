"""
NFL Live Data Fetcher
=====================
Fetches NFL team stats from ESPN API.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "nfl"


class NFLDataFetcher:
    """Fetch NFL team stats from ESPN API."""
    
    ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    
    def __init__(self, season=2024):
        self.season = season
    
    def get_team_stats(self):
        """Get team statistics."""
        print("  Fetching NFL team stats from ESPN...")
        
        try:
            # Get teams
            url = f"{self.ESPN_BASE}/teams"
            r = requests.get(url, timeout=15)
            data = r.json()
            
            teams = []
            for team in data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', []):
                t = team.get('team', {})
                teams.append({
                    'id': t.get('id'),
                    'name': t.get('displayName'),
                    'abbreviation': t.get('abbreviation'),
                })
            
            print(f"  Found {len(teams)} teams")
            return teams
        except Exception as e:
            print(f"  Error: {e}")
            return []
    
    def get_scoreboard(self, week=None):
        """Get games for a specific week."""
        url = f"{self.ESPN_BASE}/scoreboard"
        if week:
            url += f"?week={week}"
        
        try:
            r = requests.get(url, timeout=15)
            data = r.json()
            
            games = []
            for event in data.get('events', []):
                game = {
                    'id': event.get('id'),
                    'date': event.get('date'),
                    'name': event.get('name'),
                    'status': event.get('status', {}).get('type', {}).get('name'),
                }
                
                comp = event.get('competitions', [{}])[0]
                for c in comp.get('competitors', []):
                    prefix = 'home_' if c.get('homeAway') == 'home' else 'away_'
                    game[f'{prefix}team'] = c.get('team', {}).get('abbreviation')
                    game[f'{prefix}score'] = c.get('score')
                
                games.append(game)
            
            return games
        except Exception as e:
            print(f"  Error: {e}")
            return []
    
    def fetch_historical_stats(self):
        """
        NFL doesn't have easy API access for detailed team stats.
        Would need to scrape Pro-Football-Reference.
        
        Alternative: Use the spreadspoke_scores.csv for spreads data.
        """
        print("  Note: NFL detailed stats require scraping")
        print("  Using spreadspoke_scores.csv for historical data")
        
        path = DATA_DIR / "spreadspoke_scores.csv"
        if path.exists():
            df = pd.read_csv(path)
            print(f"  Loaded {len(df)} games from spreadspoke")
            return df
        return None


def test_nfl_fetcher():
    """Test the NFL fetcher."""
    fetcher = NFLDataFetcher()
    
    print("\n1. Testing Team List...")
    teams = fetcher.get_team_stats()
    if teams:
        print(f"   ✅ Got {len(teams)} teams")
        print(f"   Sample: {teams[:3]}")
    
    print("\n2. Testing Scoreboard...")
    games = fetcher.get_scoreboard()
    print(f"   ✅ Got {len(games)} games")
    if games:
        print(f"   Sample: {games[0]}")
    
    print("\n3. Checking Historical Data...")
    hist = fetcher.fetch_historical_stats()
    if hist is not None:
        print(f"   ✅ Loaded {len(hist)} historical games")


if __name__ == "__main__":
    test_nfl_fetcher()
