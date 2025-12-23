"""
Fetch NFL Data from NFLverse (Free GitHub Data)
================================================
No API key required! Uses publicly available CSV data.
"""
import requests
import pandas as pd
from pathlib import Path
from io import StringIO
import time

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "nfl"


def fetch_nflverse_schedules():
    """Fetch NFL schedules with scores."""
    print("Fetching NFL schedules from NFLverse...")
    
    url = "https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.csv"
    
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text))
            print(f"  ✅ Got {len(df)} schedule entries")
            return df
    except Exception as e:
        print(f"  Error: {e}")
    
    return None


def fetch_nflverse_team_stats(season=2023):
    """Fetch team-level stats from NFLverse."""
    print(f"Fetching NFL team stats for {season}...")
    
    # Player stats that we'll aggregate to team level
    url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.csv"
    
    try:
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text))
            print(f"  ✅ Got {len(df)} player stat entries")
            return df
    except Exception as e:
        print(f"  Error: {e}")
    
    return None


def aggregate_team_stats(player_stats):
    """Aggregate player stats to team game level."""
    if player_stats is None:
        return None
    
    print("Aggregating player stats to team level...")
    
    # Group by team and week
    team_stats = player_stats.groupby(['recent_team', 'week']).agg({
        'passing_yards': 'sum',
        'rushing_yards': 'sum',
        'receiving_yards': 'sum',
        'passing_tds': 'sum',
        'rushing_tds': 'sum',
        'receiving_tds': 'sum',
        'interceptions': 'sum',
        'sacks': 'sum',
        'rushing_fumbles_lost': 'sum',
        'receiving_fumbles_lost': 'sum',
    }).reset_index()
    
    # Calculate total turnovers
    team_stats['turnovers'] = team_stats['interceptions'] + team_stats['rushing_fumbles_lost'] + team_stats['receiving_fumbles_lost']
    team_stats['total_yards'] = team_stats['passing_yards'] + team_stats['rushing_yards']
    team_stats['total_tds'] = team_stats['passing_tds'] + team_stats['rushing_tds'] + team_stats['receiving_tds']
    
    print(f"  Created {len(team_stats)} team-week entries")
    return team_stats


def fetch_and_process_nfl_data():
    """Fetch and process NFL data."""
    print("\n" + "="*60)
    print("FETCHING NFL DATA FROM NFLVERSE (FREE)")
    print("="*60)
    
    # 1. Fetch schedules
    schedules = fetch_nflverse_schedules()
    
    if schedules is not None:
        # Filter to recent seasons
        schedules = schedules[schedules['season'] >= 2020]
        print(f"  Filtered to {len(schedules)} games (2020+)")
        print(f"  Columns: {list(schedules.columns)}")
        
        # Save
        path = DATA_DIR / "nflverse_schedules.csv"
        schedules.to_csv(path, index=False)
        print(f"  Saved to: {path}")
    
    # 2. Fetch player stats for aggregation
    all_team_stats = []
    for season in [2022, 2023, 2024]:
        player_stats = fetch_nflverse_team_stats(season)
        if player_stats is not None:
            team_stats = aggregate_team_stats(player_stats)
            if team_stats is not None:
                team_stats['season'] = season
                all_team_stats.append(team_stats)
        time.sleep(1)
    
    if all_team_stats:
        combined = pd.concat(all_team_stats, ignore_index=True)
        path = DATA_DIR / "nflverse_team_stats.csv"
        combined.to_csv(path, index=False)
        print(f"\n  Saved team stats to: {path}")
        print(f"  Total entries: {len(combined)}")
        print(f"  Columns: {list(combined.columns)}")
    
    return schedules


def main():
    schedules = fetch_and_process_nfl_data()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if schedules is not None:
        print(f"  Schedules: {len(schedules)} games loaded")
        print(f"  Has scores: 'home_score' in columns = {'home_score' in schedules.columns}")
        print(f"  Has spread: 'spread_line' in columns = {'spread_line' in schedules.columns}")
    else:
        print("  No data fetched")


if __name__ == "__main__":
    main()
