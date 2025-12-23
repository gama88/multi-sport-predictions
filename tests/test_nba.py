"""
Tests for NBA predictor.
"""
import pytest
from datetime import datetime
from src.sports.nba.predictor import NBAPredictor
from src.sports.base import Game, Team


class TestNBAPredictor:
    """Test suite for NBA predictor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = NBAPredictor()

    def test_predictor_initialization(self):
        """Test predictor initializes correctly."""
        assert self.predictor.sport_id == "nba"
        assert self.predictor.sport_name == "NBA Basketball"
        assert len(self.predictor.feature_columns) > 0

    def test_predict_without_model(self):
        """Test prediction returns default when no model trained."""
        game = Game(
            id="test_game_1",
            home_team=Team(id="lal", name="Lakers", abbreviation="LAL"),
            away_team=Team(id="bos", name="Celtics", abbreviation="BOS"),
            start_time=datetime.now(),
        )
        
        prediction = self.predictor.predict(game)
        
        assert prediction.game_id == "test_game_1"
        assert prediction.home_win_probability == 0.5
        assert prediction.confidence == 0.5

    def test_feature_columns_structure(self):
        """Test feature columns are properly defined."""
        assert "home_win_pct" in self.predictor.feature_columns
        assert "away_win_pct" in self.predictor.feature_columns
        assert "home_net_rating" in self.predictor.feature_columns

    def test_predict_batch(self):
        """Test batch prediction."""
        games = [
            Game(
                id=f"test_game_{i}",
                home_team=Team(id=f"home_{i}", name=f"Home {i}", abbreviation=f"H{i}"),
                away_team=Team(id=f"away_{i}", name=f"Away {i}", abbreviation=f"A{i}"),
                start_time=datetime.now(),
            )
            for i in range(3)
        ]
        
        predictions = self.predictor.predict_batch(games)
        
        assert len(predictions) == 3
        for pred in predictions:
            assert pred.home_win_probability == 0.5
