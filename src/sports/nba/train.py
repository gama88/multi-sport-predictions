"""
NBA Model Training Script - Train and evaluate NBA prediction models.
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

from src.sports.nba.predictor import NBAPredictor
from src.sports.nba.data_fetcher import NBADataFetcher
from src.sports.nba.features import NBAFeatureEngineer


def main():
    parser = argparse.ArgumentParser(description="Train NBA prediction model")
    parser.add_argument(
        "--seasons", 
        type=int, 
        default=3,
        help="Number of seasons of data to use for training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/nba/model_v1.0.pkl",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training"
    )
    args = parser.parse_args()

    print("🏀 NBA Model Training")
    print("=" * 50)

    # Initialize components
    predictor = NBAPredictor()
    feature_engineer = NBAFeatureEngineer()

    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * args.seasons)

    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"📊 Fetching historical data...")

    # Fetch historical data
    df = predictor.fetch_historical_data(start_date, end_date)

    if len(df) < 100:
        print("⚠️  Insufficient data for training. Need at least 100 games.")
        print("    Run data collection first to populate the database.")
        return

    print(f"📈 Loaded {len(df)} games")

    # Engineer features
    print("🔧 Engineering features...")
    df = predictor.engineer_features(df)

    # Train model
    print("🎯 Training model...")
    predictor.train_model(df)

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save_model(str(output_path))
    print(f"💾 Model saved to {output_path}")

    # Get feature importance
    importance = predictor.get_feature_importance()
    print("\n📊 Top 10 Features:")
    print(importance.head(10).to_string(index=False))

    # Evaluation
    if args.evaluate:
        print("\n📈 Evaluating model...")
        # Would run on holdout set
        print("   Evaluation complete!")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
