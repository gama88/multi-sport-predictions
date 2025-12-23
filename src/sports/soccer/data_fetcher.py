"""
Soccer Live Data Fetcher
=========================
Uses ESPN API (free, no key needed) for Premier League and other leagues.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "soccer"


class SoccerDataFetcher:
    """Fetch soccer data from ESPN API."""
    
    ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
    
    LEAGUES = {
        'eng.1': 'Premier League',
        'esp.1': 'La Liga',
        'ger.1': 'Bundesliga', 
        'ita.1': 'Serie A',
        'fra.1': 'Ligue 1',
        'usa.1': 'MLS',
        'uefa.champions': 'Champions League',
    }
    
    def __init__(self, league='eng.1'):
        self.league = league
    
    def get_scoreboard(self):
        """Get current/recent games."""
        url = f"{self.ESPN_BASE}/{self.league}/scoreboard"
        
        try:
            r = requests.get(url, timeout=15)
            data = r.json()
            
            games = []
            for event in data.get('events', []):
                comp = event.get('competitions', [{}])[0]
                
                home = None
                away = None
                for c in comp.get('competitors', []):
                    if c.get('homeAway') == 'home':
                        home = c
                    else:
                        away = c
                
                if home and away:
                    games.append({
                        'id': event.get('id'),
                        'date': event.get('date'),
                        'status': event.get('status', {}).get('type', {}).get('name'),
                        'home_team': home.get('team', {}).get('displayName'),
                        'home_score': home.get('score'),
                        'away_team': away.get('team', {}).get('displayName'),
                        'away_score': away.get('score'),
                    })
            
            return games
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def get_standings(self):
        """Get league standings."""
        url = f"{self.ESPN_BASE}/{self.league}/standings"
        
        try:
            r = requests.get(url, timeout=15)
            data = r.json()
            
            standings = []
            for standing in data.get('children', [{}])[0].get('standings', {}).get('entries', []):
                team = standing.get('team', {})
                stats = {s.get('name'): s.get('value') for s in standing.get('stats', [])}
                standings.append({
                    'team': team.get('displayName'),
                    'team_id': team.get('id'),
                    'rank': standing.get('stats', [{}])[0].get('value') if standing.get('stats') else None,
                    'played': stats.get('gamesPlayed'),
                    'wins': stats.get('wins'),
                    'draws': stats.get('ties'),
                    'losses': stats.get('losses'),
                    'goals_for': stats.get('pointsFor'),
                    'goals_against': stats.get('pointsAgainst'),
                    'goal_diff': stats.get('pointDifferential'),
                    'points': stats.get('points'),
                })
            
            return standings
        except Exception as e:
            print(f"Error: {e}")
            return []


def test_soccer_fetcher():
    """Test the soccer data fetcher."""
    print("="*60)
    print("TESTING SOCCER DATA FETCHER")
    print("="*60)
    
    for league_id, league_name in SoccerDataFetcher.LEAGUES.items():
        print(f"\n{league_name} ({league_id}):")
        fetcher = SoccerDataFetcher(league_id)
        
        games = fetcher.get_scoreboard()
        print(f"  Games: {len(games)}")
        if games:
            print(f"  Sample: {games[0]}")
        
        time.sleep(0.5)
    
    # Get standings for Premier League
    print("\n" + "-"*40)
    print("Premier League Standings:")
    fetcher = SoccerDataFetcher('eng.1')
    standings = fetcher.get_standings()
    for s in standings[:5]:
        print(f"  {s.get('rank', '?'):>2}. {s.get('team', ''):30} {s.get('points', 0):>3} pts")


if __name__ == "__main__":
    test_soccer_fetcher()
