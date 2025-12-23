"""
Data collection script - Fetches and stores historical data.
"""
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.sports.nba.data_fetcher import NBADataFetcher
from src.core.database import Database


async def collect_nba_data(db: Database, days: int = 30):
    """Collect NBA data for the past N days."""
    print("🏀 Collecting NBA data...")
    
    fetcher = NBADataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch schedule
    games = await fetcher.fetch_schedule(start_date, end_date)
    print(f"   Found {len(games)} games")
    
    # Save to database
    for game in games:
        if game:
            game['sport_id'] = 'nba'
            db.save_game(game)
    
    # Fetch standings
    standings = await fetcher.fetch_standings()
    print(f"   Found standings for {len(standings)} teams")
    
    print("   ✅ NBA data collection complete!")


async def main():
    parser = argparse.ArgumentParser(description="Collect sports data")
    parser.add_argument(
        "--sports",
        nargs="+",
        default=["nba"],
        help="Sports to collect data for"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of data to collect"
    )
    args = parser.parse_args()

    print("📊 Sports Data Collection")
    print("=" * 50)

    # Initialize database
    db = Database()
    print(f"Database: {db.db_path}")

    for sport in args.sports:
        if sport == "nba":
            await collect_nba_data(db, args.days)
        else:
            print(f"⚠️  Data collection for {sport} not yet implemented")

    print("\n✅ Data collection complete!")


if __name__ == "__main__":
    asyncio.run(main())
