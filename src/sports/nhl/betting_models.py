"""
NHL Betting Models - Hockey-specific betting predictions.
"""
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime
import numpy as np


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


class NHLMoneylineModel:
    """Predict straight-up winners (3-way: Home/Away/OT)."""
    
    def __init__(self):
        self.model_version = "nhl_ml_v1.0"
        self.features = [
            'home_corsi', 'away_corsi',
            'home_fenwick', 'away_fenwick',
            'home_goals_for', 'away_goals_for',
            'home_goals_against', 'away_goals_against',
            'home_pp_pct', 'away_pp_pct',
            'home_pk_pct', 'away_pk_pct',
            'home_save_pct', 'away_save_pct',
        ]
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        home_win_prob = 0.54
        
        pick = game_data.get('home_team', 'Home') if home_win_prob > 0.5 else game_data.get('away_team', 'Away')
        confidence = home_win_prob if home_win_prob > 0.5 else 1 - home_win_prob
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='moneyline',
            pick=pick,
            odds=game_data.get('moneyline_odds', -120),
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class NHLPucklineModel:
    """Predict puckline (spread) - typically -1.5/+1.5."""
    
    def __init__(self):
        self.model_version = "nhl_puckline_v1.0"
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        puckline = game_data.get('puckline', -1.5)
        
        # Predict if favorite wins by 2+ goals
        predicted_margin = 2.1
        covers = predicted_margin > abs(puckline)
        
        confidence = 0.42 if covers else 0.58  # Puckline favorites often lose
        pick = f"{game_data.get('home_team')} {puckline}" if covers else f"{game_data.get('away_team')} +1.5"
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='puckline',
            pick=pick,
            odds=game_data.get('puckline_odds', +150) if covers else -180,
            confidence=confidence,
            expected_value=(confidence - 0.50) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class NHLOverUnderModel:
    """Predict total goals."""
    
    def __init__(self):
        self.model_version = "nhl_totals_v1.0"
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        total = game_data.get('total', 6.0)
        predicted_total = 5.8
        
        margin = predicted_total - total
        pick = f"OVER {total}" if margin > 0 else f"UNDER {total}"
        confidence = min(0.5 + abs(margin) * 0.08, 0.72)
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='over_under',
            pick=pick,
            odds=-110,
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class NHLPeriodPropsModel:
    """Predict period-specific props."""
    
    def __init__(self):
        self.model_version = "nhl_period_v1.0"
    
    def predict_first_period(self, game_data: Dict) -> BettingPrediction:
        """Predict first period result."""
        # First periods are often low-scoring and tied
        tie_prob = 0.45
        home_prob = 0.30
        away_prob = 0.25
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='first_period',
            pick="TIE (1st Period)",
            odds=+175,
            confidence=tie_prob,
            expected_value=(tie_prob - 0.36) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class NHLGoaliePropsModel:
    """Predict goalie save props."""
    
    def __init__(self):
        self.model_version = "nhl_goalie_v1.0"
    
    def predict(self, goalie_data: Dict, save_line: float) -> BettingPrediction:
        avg_saves = goalie_data.get('avg_saves', 28)
        opp_shots = goalie_data.get('opp_shots_per_game', 30)
        
        predicted_saves = avg_saves * 0.5 + opp_shots * 0.5
        margin = predicted_saves - save_line
        
        pick = f"{goalie_data.get('name')} OVER {save_line} saves" if margin > 0 else f"{goalie_data.get('name')} UNDER {save_line} saves"
        confidence = min(0.5 + abs(margin) * 0.04, 0.70)
        
        return BettingPrediction(
            game_id=goalie_data.get('game_id', ''),
            bet_type='goalie_saves',
            pick=pick,
            odds=-110,
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class NHLBettingEngine:
    """Main NHL betting engine."""
    
    def __init__(self):
        self.moneyline = NHLMoneylineModel()
        self.puckline = NHLPucklineModel()
        self.over_under = NHLOverUnderModel()
        self.period = NHLPeriodPropsModel()
        self.goalie = NHLGoaliePropsModel()
    
    def get_all_predictions(self, game_data: Dict) -> Dict[str, BettingPrediction]:
        return {
            'moneyline': self.moneyline.predict(game_data),
            'puckline': self.puckline.predict(game_data),
            'over_under': self.over_under.predict(game_data),
            'first_period': self.period.predict_first_period(game_data),
        }
    
    def get_best_bets(self, games: List[Dict], min_ev: float = 2.0) -> List[BettingPrediction]:
        all_preds = []
        for game in games:
            all_preds.extend(self.get_all_predictions(game).values())
        return sorted([p for p in all_preds if p.expected_value >= min_ev],
                     key=lambda x: x.expected_value, reverse=True)
