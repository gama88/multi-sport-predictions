"""
Data Gathering Script - Uses only FREE data sources.
Sources:
- Kaggle (free, requires account)
- ESPN API (free, public)
- balldontlie API (free tier)
"""
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import requests

# Try to import kagglehub
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    print("Installing kagglehub...")
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])
    import kagglehub
    KAGGLE_AVAILABLE = True

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Free Kaggle datasets with betting data
KAGGLE_DATASETS = {
    'nba': [
        'nathanlauga/nba-games',  # Already have
        'wyattowalsh/basketball',  # Comprehensive NBA data
    ],
    'nfl': [
        'tobycrabtree/nfl-scores-and-betting-data',  # Already have
        'maxhorowitz/nflplaybyplay2009to2016',  # Play by play
    ],
    'nhl': [
        'martinellis/nhl-game-data',  # Already have
    ],
    'mlb': [
        'pschale/mlb-pitch-data-20152018',  # Pitch data
    ],
    'ncaa_basketball': [
        'andrewsundberg/college-basketball-dataset',  # Already have
    ],
    'soccer': [
        'davidcariboo/player-scores',  # Additional soccer
    ],
}


def download_kaggle_datasets():
    """Download datasets from Kaggle."""
    print("\n" + "="*60)
    print("📦 DOWNLOADING KAGGLE DATASETS (FREE)")
    print("="*60)
    
    for sport, datasets in KAGGLE_DATASETS.items():
        sport_dir = DATA_DIR / sport
        sport_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset in datasets:
            print(f"\n📥 Downloading {dataset}...")
            try:
                path = kagglehub.dataset_download(dataset)
                print(f"   ✅ Downloaded to: {path}")
                
                # Copy CSV files to our data directory
                import shutil
                source_path = Path(path)
                for csv_file in source_path.glob("**/*.csv"):
                    dest = sport_dir / csv_file.name
                    if not dest.exists():
                        shutil.copy(csv_file, dest)
                        print(f"   📊 Copied: {csv_file.name}")
                        
            except Exception as e:
                print(f"   ⚠️ Error: {str(e)[:50]}")


def fetch_espn_historical_data():
    """Fetch historical data from ESPN API (free)."""
    print("\n" + "="*60)
    print("🌐 FETCHING ESPN HISTORICAL DATA (FREE)")
    print("="*60)
    
    ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
    
    endpoints = {
        'nba': '/basketball/nba/scoreboard',
        'nfl': '/football/nfl/scoreboard',
        'nhl': '/hockey/nhl/scoreboard',
        'mlb': '/baseball/mlb/scoreboard',
    }
    
    # Fetch recent dates
    today = datetime.now()
    all_games = {}
    
    for sport, endpoint in endpoints.items():
        sport_dir = DATA_DIR / sport
        sport_dir.mkdir(parents=True, exist_ok=True)
        games = []
        
        print(f"\n📊 Fetching {sport.upper()} games...")
        
        # Fetch last 30 days of games
        for days_ago in range(30):
            date = today - timedelta(days=days_ago)
            date_str = date.strftime('%Y%m%d')
            
            url = f"{ESPN_BASE}{endpoint}?dates={date_str}"
            
            try:
                response = requests.get(url, timeout=10)
                data = response.json()
                
                events = data.get('events', [])
                for event in events:
                    comp = event.get('competitions', [{}])[0]
                    competitors = comp.get('competitors', [])
                    
                    if len(competitors) >= 2:
                        home = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
                        away = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])
                        
                        game_data = {
                            'game_id': event.get('id'),
                            'date': event.get('date'),
                            'home_team': home.get('team', {}).get('displayName'),
                            'away_team': away.get('team', {}).get('displayName'),
                            'home_score': home.get('score'),
                            'away_score': away.get('score'),
                            'home_record': home.get('records', [{}])[0].get('summary', ''),
                            'away_record': away.get('records', [{}])[0].get('summary', ''),
                            'status': event.get('status', {}).get('type', {}).get('name'),
                        }
                        games.append(game_data)
                        
            except Exception as e:
                pass  # Skip failed requests
        
        if games:
            df = pd.DataFrame(games)
            output_path = sport_dir / f"espn_recent_games.csv"
            df.to_csv(output_path, index=False)
            print(f"   ✅ Saved {len(games)} games to {output_path.name}")
            all_games[sport] = len(games)
    
    return all_games


def fetch_balldontlie_data():
    """Fetch NBA data from balldontlie API (free tier)."""
    print("\n" + "="*60)
    print("🏀 FETCHING BALLDONTLIE NBA DATA (FREE)")
    print("="*60)
    
    nba_dir = DATA_DIR / "nba"
    nba_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch teams
    print("\n📊 Fetching NBA teams...")
    try:
        url = "https://api.balldontlie.io/v1/teams"
        headers = {"Authorization": ""}  # Free tier doesn't need auth
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            teams_data = response.json().get('data', [])
            if teams_data:
                df = pd.DataFrame(teams_data)
                df.to_csv(nba_dir / "balldontlie_teams.csv", index=False)
                print(f"   ✅ Saved {len(teams_data)} teams")
        else:
            print(f"   ⚠️ API returned status {response.status_code}")
            
    except Exception as e:
        print(f"   ⚠️ Error: {str(e)[:50]}")
    
    # Fetch recent games
    print("\n📊 Fetching recent games...")
    try:
        today = datetime.now()
        games = []
        
        for page in range(1, 6):  # Get 5 pages
            url = f"https://api.balldontlie.io/v1/games?per_page=100&page={page}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('data', [])
                games.extend(data)
            else:
                break
        
        if games:
            # Flatten the structure
            flat_games = []
            for g in games:
                flat_games.append({
                    'game_id': g.get('id'),
                    'date': g.get('date'),
                    'season': g.get('season'),
                    'home_team': g.get('home_team', {}).get('full_name'),
                    'away_team': g.get('visitor_team', {}).get('full_name'),
                    'home_score': g.get('home_team_score'),
                    'away_score': g.get('visitor_team_score'),
                    'status': g.get('status'),
                })
            
            df = pd.DataFrame(flat_games)
            df.to_csv(nba_dir / "balldontlie_games.csv", index=False)
            print(f"   ✅ Saved {len(flat_games)} games")
            
    except Exception as e:
        print(f"   ⚠️ Error: {str(e)[:50]}")


def fetch_nhl_api_data():
    """Fetch NHL data from NHL API (free)."""
    print("\n" + "="*60)
    print("🏒 FETCHING NHL API DATA (FREE)")
    print("="*60)
    
    nhl_dir = DATA_DIR / "nhl"
    nhl_dir.mkdir(parents=True, exist_ok=True)
    
    # NHL API endpoints
    base_url = "https://statsapi.web.nhl.com/api/v1"
    
    # Fetch teams
    print("\n📊 Fetching NHL teams...")
    try:
        url = f"{base_url}/teams"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            teams = response.json().get('teams', [])
            if teams:
                df = pd.DataFrame(teams)
                df.to_csv(nhl_dir / "nhl_api_teams.csv", index=False)
                print(f"   ✅ Saved {len(teams)} teams")
    except Exception as e:
        print(f"   ⚠️ Error: {str(e)[:50]}")
    
    # Fetch standings
    print("\n📊 Fetching NHL standings...")
    try:
        url = f"{base_url}/standings"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            records = response.json().get('records', [])
            standings = []
            for div in records:
                for team in div.get('teamRecords', []):
                    standings.append({
                        'team': team.get('team', {}).get('name'),
                        'wins': team.get('leagueRecord', {}).get('wins'),
                        'losses': team.get('leagueRecord', {}).get('losses'),
                        'ot': team.get('leagueRecord', {}).get('ot'),
                        'points': team.get('points'),
                        'games_played': team.get('gamesPlayed'),
                    })
            
            if standings:
                df = pd.DataFrame(standings)
                df.to_csv(nhl_dir / "nhl_standings.csv", index=False)
                print(f"   ✅ Saved standings for {len(standings)} teams")
                
    except Exception as e:
        print(f"   ⚠️ Error: {str(e)[:50]}")


def summarize_data():
    """Summarize all available data."""
    print("\n" + "="*60)
    print("📊 DATA SUMMARY")
    print("="*60)
    
    total_rows = 0
    
    for sport_dir in DATA_DIR.iterdir():
        if sport_dir.is_dir() and sport_dir.name not in ['__pycache__']:
            csv_files = list(sport_dir.glob("*.csv"))
            sport_rows = 0
            
            print(f"\n{sport_dir.name.upper()}:")
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    rows = len(df)
                    sport_rows += rows
                    print(f"  • {csv_file.name}: {rows:,} rows, {len(df.columns)} cols")
                except:
                    pass
            
            total_rows += sport_rows
            print(f"  Total: {sport_rows:,} rows")
    
    print(f"\n{'='*60}")
    print(f"📈 TOTAL DATA: {total_rows:,} rows across all sports")
    print("="*60)


def main():
    """Run all data gathering."""
    print("\n" + "🎰 " * 20)
    print("   FREE DATA GATHERING FOR SPORTS PREDICTIONS")
    print("🎰 " * 20)
    
    # 1. Download Kaggle datasets
    download_kaggle_datasets()
    
    # 2. Fetch ESPN data
    fetch_espn_historical_data()
    
    # 3. Fetch balldontlie NBA data
    fetch_balldontlie_data()
    
    # 4. Fetch NHL API data
    fetch_nhl_api_data()
    
    # 5. Summarize
    summarize_data()
    
    print("\n✅ Data gathering complete!")
    print("Run 'python scripts/train_advanced_models.py' to retrain models")


if __name__ == "__main__":
    main()
