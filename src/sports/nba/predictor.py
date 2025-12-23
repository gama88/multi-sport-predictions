"""
NBA Predictor - Machine learning model for NBA game predictions.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ..base import BaseSportPredictor, Game, GamePrediction, Team


class NBAPredictor(BaseSportPredictor):
    """NBA-specific prediction model."""

    def __init__(self):
        super().__init__(sport_id="nba", sport_name="NBA Basketball")
        self.feature_columns = [
            # Team performance
            'home_win_pct', 'away_win_pct',
            'home_ppg', 'away_ppg',
            'home_opp_ppg', 'away_opp_ppg',
            'home_net_rating', 'away_net_rating',
            
            # Recent form
            'home_last10_wins', 'away_last10_wins',
            'home_streak', 'away_streak',
            
            # Rest and travel
            'home_rest_days', 'away_rest_days',
            'home_is_b2b', 'away_is_b2b',
            
            # Head-to-head
            'h2h_home_wins', 'h2h_total_games',
            
            # Advanced metrics
            'home_off_rating', 'away_off_rating',
            'home_def_rating', 'away_def_rating',
            'home_pace', 'away_pace',
            'home_efg_pct', 'away_efg_pct',
            'home_tov_pct', 'away_tov_pct',
            'home_reb_pct', 'away_reb_pct',
            'home_ft_rate', 'away_ft_rate',
        ]

    def fetch_live_games(self) -> List[Game]:
        """Fetch currently live NBA games."""
        # This will be implemented with actual API calls
        # For now, return empty list (sample data handled in TUI)
        return []

    def fetch_upcoming_games(self, days: int = 7) -> List[Game]:
        """Fetch upcoming NBA games."""
        # Will be implemented with ESPN/NBA API
        return []

    def fetch_historical_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical NBA game data."""
        # Will fetch from basketball-reference or NBA API
        # For now, generate sample data structure
        columns = [
            'game_id', 'date', 'home_team', 'away_team',
            'home_score', 'away_score', 'home_win'
        ] + self.feature_columns
        
        return pd.DataFrame(columns=columns)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create NBA-specific features."""
        # Calculate derived features
        if len(df) == 0:
            return df
            
        # Point differential
        df['home_point_diff'] = df['home_ppg'] - df['home_opp_ppg']
        df['away_point_diff'] = df['away_ppg'] - df['away_opp_ppg']
        
        # Win percentage difference
        df['win_pct_diff'] = df['home_win_pct'] - df['away_win_pct']
        
        # Rating differential
        df['rating_diff'] = df['home_net_rating'] - df['away_net_rating']
        
        # Rest advantage
        df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
        
        # Form differential
        df['form_diff'] = df['home_last10_wins'] - df['away_last10_wins']
        
        return df

    def train_model(self, df: pd.DataFrame) -> None:
        """Train the NBA prediction model."""
        if len(df) < 100:
            raise ValueError("Need at least 100 games for training")
        
        # Prepare features
        df = self.engineer_features(df)
        X = df[self.feature_columns]
        y = df['home_win'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train XGBoost model
        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"Model trained - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

    def predict(self, game: Game) -> GamePrediction:
        """Make a prediction for a single NBA game."""
        if self.model is None:
            # Return default prediction if no model trained
            return GamePrediction(
                game_id=game.id,
                home_win_probability=0.5,
                confidence=0.5,
                model_version=self.model_version,
            )
        
        # Extract features for the game
        features = self._extract_game_features(game)
        X = pd.DataFrame([features], columns=self.feature_columns)
        
        # Get prediction probability
        proba = self.model.predict_proba(X)[0]
        home_win_prob = proba[1]  # Probability of home win
        
        # Calculate confidence
        confidence = abs(home_win_prob - 0.5) * 2  # 0-1 scale
        
        return GamePrediction(
            game_id=game.id,
            home_win_probability=home_win_prob,
            confidence=confidence,
            model_version=self.model_version,
            features_used=features,
        )

    def _extract_game_features(self, game: Game) -> Dict[str, float]:
        """Extract feature values for a game."""
        # This would fetch real stats from database/API
        # For now, return placeholder values
        return {col: 0.0 for col in self.feature_columns}

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if self.model is None:
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)
