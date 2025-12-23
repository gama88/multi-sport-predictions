"""
MLB Live Data Fetcher
======================
Fetches MLB team stats from MLB Stats API.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "mlb"


class MLBDataFetcher:
    """Fetch MLB stats from official MLB Stats API."""
    
    MLB_API = "https://statsapi.mlb.com/api/v1"
    
    def __init__(self, season=2024):
        self.season = season
    
    def get_teams(self):
        """Get all MLB teams."""
        print("  Fetching MLB teams...")
        
        try:
            url = f"{self.MLB_API}/teams?sportId=1"
            r = requests.get(url, timeout=15)
            data = r.json()
            
            teams = []
            for t in data.get('teams', []):
                teams.append({
                    'id': t.get('id'),
                    'name': t.get('name'),
                    'abbreviation': t.get('abbreviation'),
                    'division': t.get('division', {}).get('name'),
                    'league': t.get('league', {}).get('name'),
                })
            
            print(f"  Found {len(teams)} teams")
            return teams
        except Exception as e:
            print(f"  Error: {e}")
            return []
    
    def get_team_stats(self, team_id):
        """Get team stats for a season."""
        try:
            url = f"{self.MLB_API}/teams/{team_id}/stats?stats=season&season={self.season}"
            r = requests.get(url, timeout=15)
            data = r.json()
            
            stats = {}
            for stat_group in data.get('stats', []):
                if stat_group.get('type', {}).get('displayName') == 'season':
                    splits = stat_group.get('splits', [])
                    if splits:
                        stats = splits[0].get('stat', {})
            
            return stats
        except Exception as e:
            print(f"  Error fetching team {team_id}: {e}")
            return {}
    
    def get_schedule(self, start_date, end_date):
        """Get games in date range."""
        try:
            url = f"{self.MLB_API}/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
            r = requests.get(url, timeout=15)
            data = r.json()
            
            games = []
            for date_data in data.get('dates', []):
                for game in date_data.get('games', []):
                    games.append({
                        'id': game.get('gamePk'),
                        'date': game.get('officialDate'),
                        'status': game.get('status', {}).get('abstractGameState'),
                        'home_team': game.get('teams', {}).get('home', {}).get('team', {}).get('name'),
                        'home_score': game.get('teams', {}).get('home', {}).get('score'),
                        'away_team': game.get('teams', {}).get('away', {}).get('team', {}).get('name'),
                        'away_score': game.get('teams', {}).get('away', {}).get('score'),
                    })
            
            return games
        except Exception as e:
            print(f"  Error: {e}")
            return []
    
    def get_all_team_stats(self):
        """Get stats for all teams."""
        print("  Fetching all team stats...")
        
        teams = self.get_teams()
        team_stats = []
        
        for team in teams:
            time.sleep(0.3)  # Rate limit
            stats = self.get_team_stats(team['id'])
            stats['team_id'] = team['id']
            stats['team_name'] = team['name']
            team_stats.append(stats)
            if stats:
                print(f"    Got stats for {team['name']}")
        
        return pd.DataFrame(team_stats)


def test_mlb_fetcher():
    """Test the MLB fetcher."""
    fetcher = MLBDataFetcher()
    
    print("\n1. Testing Team List...")
    teams = fetcher.get_teams()
    if teams:
        print(f"   ✅ Got {len(teams)} teams")
        print(f"   Sample: {teams[:3]}")
    
    print("\n2. Testing Schedule...")
    games = fetcher.get_schedule("2024-04-01", "2024-04-07")
    print(f"   ✅ Got {len(games)} games")
    
    print("\n3. Testing Team Stats...")
    if teams:
        stats = fetcher.get_team_stats(teams[0]['id'])
        print(f"   ✅ Got stats: {list(stats.keys())[:10]}...")


if __name__ == "__main__":
    test_mlb_fetcher()
