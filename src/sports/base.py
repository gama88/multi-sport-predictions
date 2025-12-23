"""
Base Sport Module - Abstract base classes for sport implementations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class Team:
    """Represents a team."""
    id: str
    name: str
    abbreviation: str
    conference: Optional[str] = None
    division: Optional[str] = None
    record_wins: int = 0
    record_losses: int = 0
    record_ties: int = 0


@dataclass
class Game:
    """Represents a game."""
    id: str
    home_team: Team
    away_team: Team
    start_time: datetime
    status: str = "scheduled"  # scheduled, live, final
    home_score: int = 0
    away_score: int = 0
    period: Optional[str] = None
    time_remaining: Optional[str] = None
    venue: Optional[str] = None
    broadcast: Optional[str] = None
    odds_spread: Optional[float] = None
    odds_total: Optional[float] = None
    odds_moneyline_home: Optional[int] = None
    odds_moneyline_away: Optional[int] = None


@dataclass
class GamePrediction:
    """Prediction for a game."""
    game_id: str
    home_win_probability: float
    predicted_home_score: Optional[float] = None
    predicted_away_score: Optional[float] = None
    predicted_total: Optional[float] = None
    predicted_spread: Optional[float] = None
    confidence: float = 0.5
    model_version: str = "v1.0"
    created_at: datetime = field(default_factory=datetime.now)
    features_used: Dict[str, Any] = field(default_factory=dict)


class BaseSportPredictor(ABC):
    """Abstract base class for sport-specific predictors."""

    def __init__(self, sport_id: str, sport_name: str):
        self.sport_id = sport_id
        self.sport_name = sport_name
        self.model = None
        self.model_version = "v1.0"
        self.feature_columns: List[str] = []

    @abstractmethod
    def fetch_live_games(self) -> List[Game]:
        """Fetch currently live games."""
        pass

    @abstractmethod
    def fetch_upcoming_games(self, days: int = 7) -> List[Game]:
        """Fetch upcoming games."""
        pass

    @abstractmethod
    def fetch_historical_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical game data for training."""
        pass

    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw game data."""
        pass

    @abstractmethod
    def train_model(self, df: pd.DataFrame) -> None:
        """Train the prediction model."""
        pass

    @abstractmethod
    def predict(self, game: Game) -> GamePrediction:
        """Make a prediction for a single game."""
        pass

    def predict_batch(self, games: List[Game]) -> List[GamePrediction]:
        """Make predictions for multiple games."""
        return [self.predict(game) for game in games]

    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        import joblib
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'version': self.model_version,
                'features': self.feature_columns,
                'sport_id': self.sport_id,
            }, path)

    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.model_version = data['version']
        self.feature_columns = data['features']

    def evaluate_model(
        self, predictions: List[GamePrediction], actuals: List[Game]
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        correct = 0
        for pred, actual in zip(predictions, actuals):
            predicted_winner = pred.home_win_probability > 0.5
            actual_winner = actual.home_score > actual.away_score
            if predicted_winner == actual_winner:
                correct += 1

        accuracy = correct / len(predictions) if predictions else 0

        return {
            'accuracy': accuracy,
            'total_predictions': len(predictions),
            'correct_predictions': correct,
        }


class BaseDataFetcher(ABC):
    """Abstract base class for fetching sport data."""

    def __init__(self, sport_id: str):
        self.sport_id = sport_id
        self.base_url = ""
        self.api_key: Optional[str] = None

    @abstractmethod
    async def fetch_schedule(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch game schedule."""
        pass

    @abstractmethod
    async def fetch_live_scores(self) -> List[Dict[str, Any]]:
        """Fetch live game scores."""
        pass

    @abstractmethod
    async def fetch_team_stats(self, team_id: str) -> Dict[str, Any]:
        """Fetch team statistics."""
        pass

    @abstractmethod
    async def fetch_standings(self) -> List[Dict[str, Any]]:
        """Fetch league standings."""
        pass
