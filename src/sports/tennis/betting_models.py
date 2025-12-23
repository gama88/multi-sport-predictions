"""
Tennis Betting Models - Match winner, set betting, game props.
"""
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime


@dataclass
class BettingPrediction:
    game_id: str
    bet_type: str
    pick: str
    odds: int
    confidence: float
    expected_value: float
    model_version: str
    timestamp: datetime = None


class TennisMatchWinnerModel:
    """Predict match winners."""
    
    def __init__(self):
        self.model_version = "tennis_ml_v1.0"
    
    def predict(self, match_data: Dict) -> BettingPrediction:
        p1_rank = match_data.get('player1_rank', 50)
        p2_rank = match_data.get('player2_rank', 50)
        
        rank_factor = (p2_rank - p1_rank) * 0.005
        p1_win_prob = min(max(0.5 + rank_factor, 0.15), 0.90)
        
        pick = match_data.get('player1') if p1_win_prob > 0.5 else match_data.get('player2')
        confidence = p1_win_prob if p1_win_prob > 0.5 else 1 - p1_win_prob
        
        return BettingPrediction(
            game_id=match_data.get('match_id', ''),
            bet_type='match_winner',
            pick=pick,
            odds=match_data.get('odds', -110),
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class TennisBettingEngine:
    def __init__(self):
        self.match_winner = TennisMatchWinnerModel()
    
    def get_all_predictions(self, match_data: Dict) -> Dict[str, BettingPrediction]:
        return {'match_winner': self.match_winner.predict(match_data)}
