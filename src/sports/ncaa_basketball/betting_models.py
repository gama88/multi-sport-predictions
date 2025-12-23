"""
NCAA Basketball Betting Models - March Madness & regular season.
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


class NCAABMoneylineModel:
    """Predict straight-up winners."""
    
    def __init__(self):
        self.model_version = "ncaab_ml_v1.0"
        self.features = [
            'home_kenpom_eff', 'away_kenpom_eff',
            'home_adj_off', 'away_adj_off',
            'home_adj_def', 'away_adj_def',
            'home_tempo', 'away_tempo',
            'home_exp', 'away_exp',  # Experience
        ]
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        home_kenpom = game_data.get('home_kenpom', 0)
        away_kenpom = game_data.get('away_kenpom', 0)
        
        kenpom_diff = home_kenpom - away_kenpom
        home_win_prob = 0.52 + kenpom_diff * 0.02
        home_win_prob = min(max(home_win_prob, 0.20), 0.85)
        
        pick = game_data.get('home_team', 'Home') if home_win_prob > 0.5 else game_data.get('away_team', 'Away')
        confidence = home_win_prob if home_win_prob > 0.5 else 1 - home_win_prob
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='moneyline',
            pick=pick,
            odds=game_data.get('moneyline_odds', -110),
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class NCAABSpreadModel:
    """Predict against the spread."""
    
    def __init__(self):
        self.model_version = "ncaab_spread_v1.0"
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        spread = game_data.get('spread', -5.5)
        
        # Use efficiency metrics
        home_eff = game_data.get('home_kenpom', 0)
        away_eff = game_data.get('away_kenpom', 0)
        
        predicted_margin = (home_eff - away_eff) * 0.8 + 3.5  # Home court
        
        covers = predicted_margin > abs(spread) if spread < 0 else predicted_margin < spread
        margin = abs(predicted_margin - abs(spread))
        confidence = min(0.5 + margin * 0.03, 0.72)
        
        pick = f"{game_data.get('home_team')} {spread}" if covers else f"{game_data.get('away_team')} +{abs(spread)}"
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='spread',
            pick=pick,
            odds=-110,
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class NCAABOverUnderModel:
    """Predict game totals."""
    
    def __init__(self):
        self.model_version = "ncaab_totals_v1.0"
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        total = game_data.get('total', 140.5)
        
        home_tempo = game_data.get('home_tempo', 68)
        away_tempo = game_data.get('away_tempo', 68)
        
        expected_possessions = (home_tempo + away_tempo) / 2
        expected_efficiency = 1.05  # Points per possession
        predicted_total = expected_possessions * 2 * expected_efficiency
        
        margin = predicted_total - total
        pick = f"OVER {total}" if margin > 0 else f"UNDER {total}"
        confidence = min(0.5 + abs(margin) * 0.015, 0.70)
        
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


class MarchMadnessModel:
    """Special model for NCAA Tournament predictions."""
    
    def __init__(self):
        self.model_version = "march_madness_v1.0"
        self.upset_factors = {
            (12, 5): 0.35,  # 12 seed vs 5 seed upset rate
            (11, 6): 0.38,
            (13, 4): 0.22,
            (14, 3): 0.15,
            (15, 2): 0.06,
            (16, 1): 0.01,
        }
    
    def predict_upset_probability(self, higher_seed: int, lower_seed: int) -> float:
        """Calculate upset probability based on seeds."""
        key = (higher_seed, lower_seed)
        return self.upset_factors.get(key, 0.0)
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        home_seed = game_data.get('home_seed', 8)
        away_seed = game_data.get('away_seed', 9)
        
        # In tournament, lower seed is favorite
        if home_seed < away_seed:
            favorite = game_data.get('home_team')
            underdog = game_data.get('away_team')
            upset_prob = self.predict_upset_probability(away_seed, home_seed)
        else:
            favorite = game_data.get('away_team')
            underdog = game_data.get('home_team')
            upset_prob = self.predict_upset_probability(home_seed, away_seed)
        
        # Look for upset value
        if upset_prob > 0.25:
            pick = underdog
            confidence = upset_prob
            ev = 3.0  # Positive EV on underdogs
        else:
            pick = favorite
            confidence = 1 - upset_prob
            ev = (confidence - 0.524) * 100
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='tournament',
            pick=pick,
            odds=game_data.get('moneyline_odds', -110),
            confidence=confidence,
            expected_value=ev,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class NCAABBettingEngine:
    """Main NCAA basketball betting engine."""
    
    def __init__(self):
        self.moneyline = NCAABMoneylineModel()
        self.spread = NCAABSpreadModel()
        self.over_under = NCAABOverUnderModel()
        self.tournament = MarchMadnessModel()
    
    def get_all_predictions(self, game_data: Dict) -> Dict[str, BettingPrediction]:
        preds = {
            'moneyline': self.moneyline.predict(game_data),
            'spread': self.spread.predict(game_data),
            'over_under': self.over_under.predict(game_data),
        }
        
        # Add tournament prediction if it's March Madness
        if game_data.get('is_tournament', False):
            preds['tournament'] = self.tournament.predict(game_data)
        
        return preds
    
    def get_best_bets(self, games: List[Dict], min_ev: float = 2.0) -> List[BettingPrediction]:
        all_preds = []
        for game in games:
            all_preds.extend(self.get_all_predictions(game).values())
        return sorted([p for p in all_preds if p.expected_value >= min_ev],
                     key=lambda x: x.expected_value, reverse=True)
