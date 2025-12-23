"""
Database management for sports predictions.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from contextlib import contextmanager


class Database:
    """SQLite database manager for sports data."""

    def __init__(self, db_path: str = "data/sports.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Teams table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    id TEXT PRIMARY KEY,
                    sport_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    abbreviation TEXT,
                    conference TEXT,
                    division TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Games table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id TEXT PRIMARY KEY,
                    sport_id TEXT NOT NULL,
                    home_team_id TEXT NOT NULL,
                    away_team_id TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    status TEXT DEFAULT 'scheduled',
                    home_score INTEGER DEFAULT 0,
                    away_score INTEGER DEFAULT 0,
                    period TEXT,
                    time_remaining TEXT,
                    venue TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (home_team_id) REFERENCES teams(id),
                    FOREIGN KEY (away_team_id) REFERENCES teams(id)
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    sport_id TEXT NOT NULL,
                    home_win_probability REAL NOT NULL,
                    predicted_home_score REAL,
                    predicted_away_score REAL,
                    predicted_total REAL,
                    predicted_spread REAL,
                    confidence REAL,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES games(id)
                )
            """)
            
            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sport_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    accuracy REAL DEFAULT 0,
                    precision_score REAL DEFAULT 0,
                    recall_score REAL DEFAULT 0,
                    roi REAL DEFAULT 0,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Team stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS team_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT NOT NULL,
                    sport_id TEXT NOT NULL,
                    season TEXT NOT NULL,
                    games_played INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    points_for REAL DEFAULT 0,
                    points_against REAL DEFAULT 0,
                    stats_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team_id) REFERENCES teams(id)
                )
            """)
            
            conn.commit()

    def save_game(self, game: Dict[str, Any]) -> None:
        """Save a game to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO games 
                (id, sport_id, home_team_id, away_team_id, start_time, 
                 status, home_score, away_score, period, time_remaining, venue, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game['id'],
                game.get('sport_id', 'unknown'),
                game['home_team_id'],
                game['away_team_id'],
                game['start_time'],
                game.get('status', 'scheduled'),
                game.get('home_score', 0),
                game.get('away_score', 0),
                game.get('period'),
                game.get('time_remaining'),
                game.get('venue'),
                datetime.now().isoformat(),
            ))
            conn.commit()

    def save_prediction(self, prediction: Dict[str, Any]) -> None:
        """Save a prediction to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions 
                (game_id, sport_id, home_win_probability, predicted_home_score,
                 predicted_away_score, predicted_total, predicted_spread, 
                 confidence, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction['game_id'],
                prediction.get('sport_id', 'unknown'),
                prediction['home_win_probability'],
                prediction.get('predicted_home_score'),
                prediction.get('predicted_away_score'),
                prediction.get('predicted_total'),
                prediction.get('predicted_spread'),
                prediction.get('confidence', 0.5),
                prediction.get('model_version', 'v1.0'),
            ))
            conn.commit()

    def get_games_for_date(
        self, sport_id: str, date: datetime
    ) -> List[Dict[str, Any]]:
        """Get all games for a specific date."""
        start = date.replace(hour=0, minute=0, second=0)
        end = date.replace(hour=23, minute=59, second=59)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM games 
                WHERE sport_id = ? 
                AND start_time BETWEEN ? AND ?
                ORDER BY start_time
            """, (sport_id, start.isoformat(), end.isoformat()))
            
            return [dict(row) for row in cursor.fetchall()]

    def get_live_games(self, sport_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all live games."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if sport_id:
                cursor.execute("""
                    SELECT * FROM games 
                    WHERE status = 'live' AND sport_id = ?
                    ORDER BY start_time
                """, (sport_id,))
            else:
                cursor.execute("""
                    SELECT * FROM games 
                    WHERE status = 'live'
                    ORDER BY start_time
                """)
            
            return [dict(row) for row in cursor.fetchall()]

    def get_model_stats(self, sport_id: str) -> Dict[str, Any]:
        """Get model performance statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM model_performance 
                WHERE sport_id = ?
                ORDER BY recorded_at DESC
                LIMIT 1
            """, (sport_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else {}

    def update_model_stats(self, stats: Dict[str, Any]) -> None:
        """Update model performance statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_performance 
                (sport_id, model_version, total_predictions, correct_predictions,
                 accuracy, precision_score, recall_score, roi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats['sport_id'],
                stats.get('model_version', 'v1.0'),
                stats.get('total_predictions', 0),
                stats.get('correct_predictions', 0),
                stats.get('accuracy', 0),
                stats.get('precision_score', 0),
                stats.get('recall_score', 0),
                stats.get('roi', 0),
            ))
            conn.commit()

    def get_historical_games(
        self, sport_id: str, limit: int = 1000
    ) -> pd.DataFrame:
        """Get historical games for training."""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM games 
                WHERE sport_id = ? AND status = 'final'
                ORDER BY start_time DESC
                LIMIT ?
            """
            return pd.read_sql_query(
                query, conn, params=(sport_id, limit)
            )
