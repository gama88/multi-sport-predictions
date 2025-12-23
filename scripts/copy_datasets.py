"""
Copy downloaded Kaggle datasets to project data folder and explore contents.
"""
import shutil
from pathlib import Path
import os


# Kaggle cache location
KAGGLE_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets"

# Dataset mappings
DATASETS = {
    'nba': 'nathanlauga/nba-games',
    'nfl': 'tobycrabtree/nfl-scores-and-betting-data',
    'nhl': 'martinellis/nhl-game-data',
    'mlb': 'open-source-sports/baseball-databank',
    'ncaa_basketball': 'andrewsundberg/college-basketball-dataset',
    'ncaa_football': 'jeffgallini/college-football-attendance-2000-2018',
    'tennis': 'guillemservera/tennis-atp-world-tour',
    'soccer': 'davidcariboo/player-scores',
}

# Project data directory
PROJECT_DATA = Path(__file__).parent.parent / "data"


def find_dataset_path(dataset_ref: str) -> Path:
    """Find the downloaded dataset in Kaggle cache."""
    parts = dataset_ref.split('/')
    if len(parts) != 2:
        return None
    
    owner, name = parts
    dataset_path = KAGGLE_CACHE / owner / name
    
    if not dataset_path.exists():
        return None
    
    # Find latest version
    versions = list(dataset_path.glob("versions/*"))
    if versions:
        return max(versions, key=lambda p: int(p.name) if p.name.isdigit() else 0)
    
    return dataset_path


def list_files(path: Path, indent: int = 0) -> list:
    """List all files in a directory recursively."""
    files = []
    if path.is_dir():
        for item in sorted(path.iterdir()):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                files.append((item, size_mb))
            elif item.is_dir():
                files.extend(list_files(item, indent + 1))
    return files


def main():
    print("📦 Kaggle Dataset Explorer & Copier")
    print("=" * 60)
    
    PROJECT_DATA.mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    total_size = 0
    
    for sport, dataset_ref in DATASETS.items():
        print(f"\n🏆 {sport.upper()}")
        print("-" * 40)
        
        dataset_path = find_dataset_path(dataset_ref)
        
        if not dataset_path:
            print(f"  ❌ Not found: {dataset_ref}")
            continue
        
        print(f"  📁 Source: {dataset_path}")
        
        # List files
        files = list_files(dataset_path)
        print(f"  📊 Files found: {len(files)}")
        
        for file_path, size_mb in files[:10]:  # Show first 10
            rel_path = file_path.relative_to(dataset_path)
            print(f"    • {rel_path} ({size_mb:.2f} MB)")
        
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more files")
        
        # Copy to project data folder
        sport_data_dir = PROJECT_DATA / sport
        sport_data_dir.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        for file_path, size_mb in files:
            if file_path.suffix.lower() in ['.csv', '.json', '.parquet', '.xlsx']:
                dest = sport_data_dir / file_path.name
                if not dest.exists():
                    shutil.copy2(file_path, dest)
                    copied += 1
                    total_size += size_mb
        
        if copied > 0:
            print(f"  ✅ Copied {copied} data files to {sport_data_dir}")
        
        total_files += len(files)
    
    print("\n" + "=" * 60)
    print(f"📋 SUMMARY")
    print("=" * 60)
    print(f"  Total files found: {total_files}")
    print(f"  Total size: {total_size:.2f} MB")
    print(f"  Data directory: {PROJECT_DATA}")
    
    # List what's in the data directory now
    print("\n📁 Project Data Directory Contents:")
    for sport_dir in sorted(PROJECT_DATA.iterdir()):
        if sport_dir.is_dir():
            files = list(sport_dir.glob("*"))
            data_files = [f for f in files if f.suffix.lower() in ['.csv', '.json', '.parquet', '.xlsx']]
            print(f"  {sport_dir.name}/: {len(data_files)} data files")
            for f in data_files[:3]:
                print(f"    • {f.name}")
            if len(data_files) > 3:
                print(f"    ... and {len(data_files) - 3} more")


if __name__ == "__main__":
    main()
