"""
Soccer Betting Models - Match result, both teams to score, correct score.
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


class SoccerMatchResultModel:
    """Predict 3-way result (Home/Draw/Away)."""
    
    def __init__(self):
        self.model_version = "soccer_1x2_v1.0"
    
    def predict(self, match_data: Dict) -> BettingPrediction:
        home_xg = match_data.get('home_xg', 1.4)
        away_xg = match_data.get('away_xg', 1.1)
        
        home_prob = 0.40 + (home_xg - away_xg) * 0.1
        draw_prob = 0.28
        away_prob = 1 - home_prob - draw_prob
        
        probs = {'1': home_prob, 'X': draw_prob, '2': away_prob}
        best = max(probs, key=probs.get)
        
        pick_map = {'1': match_data.get('home_team'), 'X': 'Draw', '2': match_data.get('away_team')}
        
        return BettingPrediction(
            game_id=match_data.get('game_id', ''),
            bet_type='match_result',
            pick=pick_map[best],
            odds=match_data.get('odds', +150 if best == 'X' else -110),
            confidence=probs[best],
            expected_value=(probs[best] - 0.33) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class SoccerBTTSModel:
    """Predict Both Teams To Score (Yes/No)."""
    
    def __init__(self):
        self.model_version = "soccer_btts_v1.0"
    
    def predict(self, match_data: Dict) -> BettingPrediction:
        home_goals_avg = match_data.get('home_goals_avg', 1.4)
        away_goals_avg = match_data.get('away_goals_avg', 1.1)
        
        btts_prob = min((home_goals_avg * away_goals_avg) * 0.5, 0.70)
        
        pick = "BTTS Yes" if btts_prob > 0.5 else "BTTS No"
        confidence = btts_prob if btts_prob > 0.5 else 1 - btts_prob
        
        return BettingPrediction(
            game_id=match_data.get('game_id', ''),
            bet_type='btts',
            pick=pick,
            odds=-110,
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class SoccerOverUnderModel:
    """Predict total goals."""
    
    def __init__(self):
        self.model_version = "soccer_totals_v1.0"
    
    def predict(self, match_data: Dict) -> BettingPrediction:
        total = match_data.get('total', 2.5)
        home_xg = match_data.get('home_xg', 1.4)
        away_xg = match_data.get('away_xg', 1.1)
        
        predicted = home_xg + away_xg
        margin = predicted - total
        
        pick = f"OVER {total}" if margin > 0 else f"UNDER {total}"
        confidence = min(0.5 + abs(margin) * 0.15, 0.68)
        
        return BettingPrediction(
            game_id=match_data.get('game_id', ''),
            bet_type='over_under',
            pick=pick,
            odds=-110,
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class SoccerBettingEngine:
    def __init__(self):
        self.match_result = SoccerMatchResultModel()
        self.btts = SoccerBTTSModel()
        self.over_under = SoccerOverUnderModel()
    
    def get_all_predictions(self, match_data: Dict) -> Dict[str, BettingPrediction]:
        return {
            'match_result': self.match_result.predict(match_data),
            'btts': self.btts.predict(match_data),
            'over_under': self.over_under.predict(match_data),
        }
