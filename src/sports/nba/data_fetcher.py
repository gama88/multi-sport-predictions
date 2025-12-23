"""
NBA Live Data Fetcher
=====================
Fetches real-time NBA stats from NBA.com API for V6 model predictions.
Gets all stats needed for behavioral proxy features: STL, BLK, TOV, PF, etc.
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

BASE_DIR = Path(__file__).parent.parent.parent.parent  # src/sports/nba -> src/sports -> src -> repo root
DATA_DIR = BASE_DIR / "data" / "nba"


class NBADataFetcher:
    """
    Fetches NBA team/game data from NBA.com Stats API.
    Uses proper headers to avoid 403 errors.
    """
    
    BASE_URL = "https://stats.nba.com/stats"
    
    # Required headers for NBA.com API
    HEADERS = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'Connection': 'keep-alive',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
    }
    
    def __init__(self, season="2024-25"):
        self.season = season
        self.cache_dir = DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def _make_request(self, endpoint, params=None):
        """Make request with proper headers and rate limiting."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            time.sleep(0.5)  # Rate limit
            response = requests.get(url, headers=self.HEADERS, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {endpoint}: {e}")
            return None
    
    def get_team_list(self):
        """Get list of all NBA teams."""
        print("  Fetching team list...")
        
        data = self._make_request("leaguedashteamstats", {
            'Conference': '',
            'DateFrom': '',
            'DateTo': '',
            'Division': '',
            'GameScope': '',
            'GameSegment': '',
            'LastNGames': '0',
            'LeagueID': '00',
            'Location': '',
            'MeasureType': 'Base',
            'Month': '0',
            'OpponentTeamID': '0',
            'Outcome': '',
            'PORound': '0',
            'PaceAdjust': 'N',
            'PerMode': 'PerGame',
            'Period': '0',
            'PlayerExperience': '',
            'PlayerPosition': '',
            'PlusMinus': 'N',
            'Rank': 'N',
            'Season': self.season,
            'SeasonSegment': '',
            'SeasonType': 'Regular Season',
            'ShotClockRange': '',
            'StarterBench': '',
            'TeamID': '0',
            'TwoWay': '0',
            'VsConference': '',
            'VsDivision': '',
        })
        
        if not data:
            return None
        
        headers = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']
        
        df = pd.DataFrame(rows, columns=headers)
        return df
    
    def get_team_game_log(self, team_id):
        """Get game log for a specific team (last 30 games)."""
        print(f"  Fetching game log for team {team_id}...")
        
        data = self._make_request("teamgamelog", {
            'TeamID': str(team_id),
            'Season': self.season,
            'SeasonType': 'Regular Season',
        })
        
        if not data:
            return None
        
        headers = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']
        
        df = pd.DataFrame(rows, columns=headers)
        return df
    
    def get_all_team_game_logs(self):
        """Get game logs for all teams."""
        print("\nFetching all team game logs...")
        
        teams = self.get_team_list()
        if teams is None:
            print("  Error: Could not fetch team list")
            return None
        
        print(f"  Found {len(teams)} teams")
        
        all_games = []
        for idx, row in teams.iterrows():
            team_id = row['TEAM_ID']
            team_name = row['TEAM_NAME']
            
            game_log = self.get_team_game_log(team_id)
            if game_log is not None and len(game_log) > 0:
                game_log['TEAM_NAME'] = team_name
                all_games.append(game_log)
                print(f"    {team_name}: {len(game_log)} games")
        
        if all_games:
            combined = pd.concat(all_games, ignore_index=True)
            return combined
        
        return None
    
    def get_todays_games(self):
        """Get today's scheduled games."""
        print("  Fetching today's games...")
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        data = self._make_request("scoreboardv2", {
            'GameDate': today,
            'LeagueID': '00',
            'DayOffset': '0',
        })
        
        if not data:
            return None
        
        # Find the GameHeader result set
        for rs in data.get('resultSets', []):
            if rs['name'] == 'GameHeader':
                headers = rs['headers']
                rows = rs['rowSet']
                return pd.DataFrame(rows, columns=headers)
        
        return None
    
    def fetch_and_save_current_data(self):
        """Fetch all current data and save for model predictions."""
        print("\n" + "="*60)
        print("FETCHING CURRENT NBA DATA FOR V6 MODEL")
        print("="*60)
        
        # Get all team game logs
        game_logs = self.get_all_team_game_logs()
        
        if game_logs is not None:
            output_path = DATA_DIR / "cache" / f"team_game_logs_{datetime.now().strftime('%Y%m%d')}.csv"
            game_logs.to_csv(output_path, index=False)
            print(f"\n  Saved {len(game_logs)} game records to {output_path}")
            
            # Print available columns
            print(f"\n  Available stats: {list(game_logs.columns)}")
            
            # Check for required stats
            required = ['PTS', 'FG_PCT', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'REB']
            found = [c for c in required if c in game_logs.columns]
            print(f"  Required stats found: {found}")
            
            return game_logs
        
        return None
    
    def build_team_histories(self, game_logs):
        """
        Convert game logs to team histories format for V6 features.
        Returns dict: {team_id: [list of game dicts]}
        """
        if game_logs is None:
            return {}
        
        print("\n  Building team histories from game logs...")
        
        # Parse dates
        game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'])
        game_logs = game_logs.sort_values('GAME_DATE')
        
        team_histories = {}
        
        for team_id in game_logs['Team_ID'].unique():
            team_games = game_logs[game_logs['Team_ID'] == team_id].copy()
            
            history = []
            for _, row in team_games.iterrows():
                is_home = ' vs. ' in str(row.get('MATCHUP', ''))
                won = row.get('WL', 'L') == 'W'
                
                game = {
                    'date': row['GAME_DATE'],
                    'is_home': is_home,
                    'won': won,
                    'pts': row.get('PTS', 0) or 0,
                    'opp_pts': row.get('PTS', 0) or 0,  # Would need opponent lookup
                    'fg_pct': row.get('FG_PCT', 0.45) or 0.45,
                    'fg3_pct': row.get('FG3_PCT', 0.35) or 0.35,
                    'ft_pct': row.get('FT_PCT', 0.75) or 0.75,
                    'reb': row.get('REB', 40) or 40,
                    'ast': row.get('AST', 24) or 24,
                    'stl': row.get('STL', 7) or 7,
                    'blk': row.get('BLK', 5) or 5,
                    'tov': row.get('TOV', 14) or 14,
                    'pf': row.get('PF', 20) or 20,
                }
                history.append(game)
            
            team_histories[team_id] = history
        
        print(f"  Built histories for {len(team_histories)} teams")
        return team_histories


def test_fetcher():
    """Test the NBA data fetcher."""
    fetcher = NBADataFetcher(season="2024-25")
    
    # Test team list
    print("\n1. Testing Team List...")
    teams = fetcher.get_team_list()
    if teams is not None:
        print(f"   ✅ Got {len(teams)} teams")
        print(f"   Columns: {list(teams.columns)[:10]}...")
    else:
        print("   ❌ Failed to get teams")
        return
    
    # Test single team game log
    print("\n2. Testing Team Game Log...")
    team_id = teams.iloc[0]['TEAM_ID']
    team_name = teams.iloc[0]['TEAM_NAME']
    game_log = fetcher.get_team_game_log(team_id)
    if game_log is not None:
        print(f"   ✅ Got {len(game_log)} games for {team_name}")
        print(f"   Columns: {list(game_log.columns)}")
    else:
        print("   ❌ Failed to get game log")
    
    # Test today's games
    print("\n3. Testing Today's Games...")
    today = fetcher.get_todays_games()
    if today is not None:
        print(f"   ✅ Got {len(today)} games scheduled today")
    else:
        print("   ⚠️ No games today or error")


if __name__ == "__main__":
    test_fetcher()
