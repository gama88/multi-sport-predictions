"""
Download datasets from Kaggle for all sports.
"""
import json
from pathlib import Path
import sys

try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    print("Installing kagglehub...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])
    import kagglehub
    KAGGLE_AVAILABLE = True

# Load curated datasets
DATASETS_PATH = Path(__file__).parent.parent / "data" / "kaggle_datasets.json"

# Define which datasets to download per sport (primary dataset for each)
PRIMARY_DATASETS = {
    'nba': 'nathanlauga/nba-games',
    'nfl': 'tobycrabtree/nfl-scores-and-betting-data',
    'nhl': 'martinellis/nhl-game-data',
    'mlb': 'open-source-sports/baseball-databank',
    'ncaa_basketball': 'andrewsundberg/college-basketball-dataset',
    'ncaa_football': 'jeffgallini/college-football-attendance-2000-2018',
    'tennis': 'guillemservera/tennis-atp-world-tour',
    'soccer': 'davidcariboo/player-scores',
}


def download_dataset(dataset_ref: str, sport: str) -> Path:
    """Download a dataset from Kaggle."""
    print(f"  📥 Downloading: {dataset_ref}")
    
    try:
        # Download using kagglehub
        path = kagglehub.dataset_download(dataset_ref)
        print(f"  ✅ Downloaded to: {path}")
        return Path(path)
    except Exception as e:
        print(f"  ❌ Error downloading {dataset_ref}: {e}")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download sports datasets from Kaggle")
    parser.add_argument(
        "--sport",
        choices=list(PRIMARY_DATASETS.keys()) + ['all'],
        default='all',
        help="Sport to download data for"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets without downloading"
    )
    args = parser.parse_args()

    print("📊 Kaggle Sports Dataset Downloader")
    print("=" * 60)

    if args.list:
        print("\nAvailable datasets per sport:\n")
        for sport, ref in PRIMARY_DATASETS.items():
            print(f"  {sport.upper():20} → {ref}")
        return

    # Determine which sports to download
    if args.sport == 'all':
        sports = list(PRIMARY_DATASETS.keys())
    else:
        sports = [args.sport]

    print(f"\nDownloading datasets for: {', '.join(sports)}\n")

    # Download each sport's dataset
    results = {}
    for sport in sports:
        print(f"\n🏆 {sport.upper()}")
        print("-" * 40)
        
        dataset_ref = PRIMARY_DATASETS.get(sport)
        if dataset_ref:
            path = download_dataset(dataset_ref, sport)
            results[sport] = {
                'ref': dataset_ref,
                'path': str(path) if path else None,
                'success': path is not None
            }

    # Summary
    print("\n" + "=" * 60)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 60)
    
    for sport, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {sport.upper():20} → {result['ref']}")
        if result['path']:
            print(f"     📁 {result['path']}")


if __name__ == "__main__":
    main()
