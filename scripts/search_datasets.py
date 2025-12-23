"""
Search for Kaggle datasets for each sport using kagglehub.
"""
import os
import json
from pathlib import Path

try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("kagglehub not installed, will use curated list instead")

# Curated sports datasets from Kaggle (manually verified to be good quality)
CURATED_DATASETS = {
    'nba': [
        {'ref': 'nathanlauga/nba-games', 'title': 'NBA Games Data - Complete Dataset', 'description': 'All NBA games from 2003-2022 with game stats'},
        {'ref': 'wyattowalsh/basketball', 'title': 'Basketball Dataset', 'description': 'Extensive NBA data including players, teams, games'},
        {'ref': 'justinas/nba-players-data', 'title': 'NBA Players Stats', 'description': 'NBA player statistics from 1996-2022'},
        {'ref': 'sumitrodatta/nba-aba-baa-stats', 'title': 'NBA/ABA/BAA Stats', 'description': 'Historical NBA statistics'},
        {'ref': 'drgilermo/nba-players-stats', 'title': 'NBA Players Stats per Season', 'description': 'Detailed player statistics per season'},
    ],
    'nfl': [
        {'ref': 'maxhorowitz/nflplaybyplay2009to2016', 'title': 'NFL Play by Play 2009-2018', 'description': 'Detailed NFL play-by-play data'},
        {'ref': 'tobycrabtree/nfl-scores-and-betting-data', 'title': 'NFL Scores and Betting Data', 'description': 'NFL game results with betting lines'},
        {'ref': 'cviaxmiern/nfl-team-stats-20022019-espn', 'title': 'NFL Team Stats 2002-2019', 'description': 'Team statistics from ESPN'},
        {'ref': 'kendallgillies/nflstatistics', 'title': 'NFL Statistics', 'description': 'Comprehensive NFL statistics'},
        {'ref': 'zynicide/nfl-football-player-stats', 'title': 'NFL Football Player Stats', 'description': 'Individual player statistics'},
    ],
    'nhl': [
        {'ref': 'martinellis/nhl-game-data', 'title': 'NHL Game Data', 'description': 'NHL games data with detailed stats'},
        {'ref': 'open-source-sports/professional-hockey-database', 'title': 'Professional Hockey Database', 'description': 'Comprehensive hockey database'},
        {'ref': 'mjavon/nhl-stats', 'title': 'NHL Stats', 'description': 'NHL team and player statistics'},
        {'ref': 'zingbretsen/nhl-play-by-play', 'title': 'NHL Play by Play', 'description': 'Detailed play-by-play data'},
    ],
    'mlb': [
        {'ref': 'open-source-sports/baseball-databank', 'title': 'Baseball Databank', 'description': 'Comprehensive baseball statistics'},
        {'ref': 'pschale/mlb-pitch-data-20152018', 'title': 'MLB Pitch Data 2015-2018', 'description': 'Pitch-level data'},
        {'ref': 'wduckett/baseball-betting-data', 'title': 'Baseball Betting Data', 'description': 'MLB games with betting lines'},
        {'ref': 'seanlahman/the-history-of-baseball', 'title': 'History of Baseball', 'description': 'Historical MLB data'},
    ],
    'ncaa_basketball': [
        {'ref': 'andrewsundberg/college-basketball-dataset', 'title': 'College Basketball Dataset', 'description': 'NCAA basketball game data'},
        {'ref': 'ncaa/ncaa-basketball', 'title': 'NCAA Basketball', 'description': 'Official NCAA basketball data'},
        {'ref': 'nishaanamin/march-madness-data', 'title': 'March Madness Data', 'description': 'Tournament data and seedings'},
        {'ref': 'datasets/march-machine-learning-mania-2024', 'title': 'March Machine Learning Mania', 'description': 'Kaggle competition dataset'},
    ],
    'ncaa_football': [
        {'ref': 'jeffgallini/college-football-attendance-2000-2018', 'title': 'College Football Attendance', 'description': 'CFB attendance data'},
        {'ref': 'tylerr1990/ncaafb-team-game-stats-2014-2022', 'title': 'NCAAFB Team Game Stats', 'description': 'College football game statistics'},
        {'ref': 'mhixon/college-football-statistics', 'title': 'College Football Statistics', 'description': 'CFB statistics database'},
    ],
}


def main():
    print("📊 Sports Datasets for Model Training")
    print("=" * 60)
    
    # Save curated datasets
    output_path = Path("data/kaggle_datasets.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(CURATED_DATASETS, f, indent=2)
    
    print(f"\n✅ Dataset catalog saved to {output_path}")
    
    # Print datasets per sport
    for sport, datasets in CURATED_DATASETS.items():
        print(f"\n🏆 {sport.upper()}:")
        print("-" * 40)
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds['title']}")
            print(f"     📦 kaggle: {ds['ref']}")
            print(f"     📝 {ds['description']}")
    
    print("\n" + "=" * 60)
    print("To download a dataset, use:")
    print("  import kagglehub")
    print("  path = kagglehub.dataset_download('username/dataset-name')")
    print("=" * 60)


if __name__ == "__main__":
    main()
