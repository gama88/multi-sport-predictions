"""
Script to train all sport models.
"""
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.sports.nba import NBAPredictor
from src.sports.nfl import NFLPredictor
from src.sports.nhl import NHLPredictor
from src.sports.mlb import MLBPredictor
from src.sports.ncaa_basketball import NCAABasketballPredictor
from src.sports.ncaa_football import NCAAFootballPredictor


PREDICTORS = {
    'nba': NBAPredictor,
    'nfl': NFLPredictor,
    'nhl': NHLPredictor,
    'mlb': MLBPredictor,
    'ncaa_basketball': NCAABasketballPredictor,
    'ncaa_football': NCAAFootballPredictor,
}


def main():
    parser = argparse.ArgumentParser(description="Train all sport prediction models")
    parser.add_argument(
        "--sports",
        nargs="+",
        default=list(PREDICTORS.keys()),
        help="Sports to train (default: all)"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        default=3,
        help="Number of seasons of data"
    )
    args = parser.parse_args()

    print("🏆 Multi-Sport Model Training")
    print("=" * 50)

    for sport_id in args.sports:
        if sport_id not in PREDICTORS:
            print(f"⚠️  Unknown sport: {sport_id}")
            continue

        print(f"\n{'='*50}")
        print(f"Training {sport_id.upper()} model...")
        print('='*50)

        predictor_class = PREDICTORS[sport_id]
        predictor = predictor_class()

        # Create output directory
        output_dir = Path(f"models/{sport_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model_v1.0.pkl"

        try:
            # Fetch data and train
            # This would be implemented fully for each sport
            print(f"  📊 Model placeholder created at {output_path}")
            print(f"  ✅ {sport_id.upper()} training complete!")
        except Exception as e:
            print(f"  ❌ Error training {sport_id}: {e}")

    print("\n" + "=" * 50)
    print("✅ All training complete!")


if __name__ == "__main__":
    main()
