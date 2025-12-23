"""
MLB Betting Models - Baseball-specific betting predictions.
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


class MLBMoneylineModel:
    """Predict straight-up winners."""
    
    def __init__(self):
        self.model_version = "mlb_ml_v1.0"
        self.features = [
            'home_era', 'away_era',
            'home_whip', 'away_whip',
            'home_ops', 'away_ops',
            'home_starter_era', 'away_starter_era',
            'home_bullpen_era', 'away_bullpen_era',
            'home_runs_per_game', 'away_runs_per_game',
        ]
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        # Starting pitcher is huge in MLB
        home_starter_era = game_data.get('home_starter_era', 4.0)
        away_starter_era = game_data.get('away_starter_era', 4.0)
        
        era_diff = away_starter_era - home_starter_era
        base_prob = 0.52 + era_diff * 0.02
        home_win_prob = min(max(base_prob, 0.35), 0.70)
        
        pick = game_data.get('home_team', 'Home') if home_win_prob > 0.5 else game_data.get('away_team', 'Away')
        confidence = home_win_prob if home_win_prob > 0.5 else 1 - home_win_prob
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='moneyline',
            pick=pick,
            odds=game_data.get('moneyline_odds', -130),
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class MLBRunlineModel:
    """Predict runline (spread) - typically -1.5/+1.5."""
    
    def __init__(self):
        self.model_version = "mlb_runline_v1.0"
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        runline = game_data.get('runline', -1.5)
        
        predicted_margin = 1.8
        covers = predicted_margin > abs(runline)
        
        # MLB favorites cover runline about 35-40% of time
        confidence = 0.38 if covers else 0.62
        pick = f"{game_data.get('home_team')} {runline}" if covers else f"{game_data.get('away_team')} +1.5"
        
        odds = +140 if covers else -160
        implied_prob = 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='runline',
            pick=pick,
            odds=odds,
            confidence=confidence,
            expected_value=(confidence - implied_prob) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class MLBOverUnderModel:
    """Predict total runs."""
    
    def __init__(self):
        self.model_version = "mlb_totals_v1.0"
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        total = game_data.get('total', 8.5)
        
        # Consider starters, weather, ballpark
        park_factor = game_data.get('park_factor', 1.0)
        predicted_total = 8.2 * park_factor
        
        margin = predicted_total - total
        pick = f"OVER {total}" if margin > 0 else f"UNDER {total}"
        confidence = min(0.5 + abs(margin) * 0.04, 0.68)
        
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


class MLBFirst5InningsModel:
    """Predict first 5 innings result (F5) - removes bullpen variance."""
    
    def __init__(self):
        self.model_version = "mlb_f5_v1.0"
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        home_starter_era = game_data.get('home_starter_era', 4.0)
        away_starter_era = game_data.get('away_starter_era', 4.0)
        
        # F5 is more predictable due to known starters
        era_diff = away_starter_era - home_starter_era
        base_prob = 0.52 + era_diff * 0.03
        home_win_prob = min(max(base_prob, 0.35), 0.72)
        
        pick = f"{game_data.get('home_team')} F5" if home_win_prob > 0.5 else f"{game_data.get('away_team')} F5"
        confidence = home_win_prob if home_win_prob > 0.5 else 1 - home_win_prob
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='first_5',
            pick=pick,
            odds=-110,
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class MLBPlayerPropsModel:
    """Predict player props - hits, strikeouts, HRs, etc."""
    
    PROP_TYPES = ['hits', 'total_bases', 'strikeouts', 'home_runs', 'rbis']
    
    def __init__(self):
        self.model_version = "mlb_props_v1.0"
    
    def predict_strikeouts(self, pitcher_data: Dict, line: float) -> BettingPrediction:
        """Predict pitcher strikeout total."""
        avg_ks = pitcher_data.get('k_per_game', line)
        opp_k_rate = pitcher_data.get('opp_k_rate', 0.22)
        
        predicted_ks = avg_ks * (opp_k_rate / 0.22)
        margin = predicted_ks - line
        
        pick = f"{pitcher_data.get('name')} OVER {line} Ks" if margin > 0 else f"{pitcher_data.get('name')} UNDER {line} Ks"
        confidence = min(0.5 + abs(margin) * 0.06, 0.70)
        
        return BettingPrediction(
            game_id=pitcher_data.get('game_id', ''),
            bet_type='prop_strikeouts',
            pick=pick,
            odds=-110,
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class MLBBettingEngine:
    """Main MLB betting engine."""
    
    def __init__(self):
        self.moneyline = MLBMoneylineModel()
        self.runline = MLBRunlineModel()
        self.over_under = MLBOverUnderModel()
        self.first5 = MLBFirst5InningsModel()
        self.props = MLBPlayerPropsModel()
    
    def get_all_predictions(self, game_data: Dict) -> Dict[str, BettingPrediction]:
        return {
            'moneyline': self.moneyline.predict(game_data),
            'runline': self.runline.predict(game_data),
            'over_under': self.over_under.predict(game_data),
            'first_5': self.first5.predict(game_data),
        }
    
    def get_best_bets(self, games: List[Dict], min_ev: float = 2.0) -> List[BettingPrediction]:
        all_preds = []
        for game in games:
            all_preds.extend(self.get_all_predictions(game).values())
        return sorted([p for p in all_preds if p.expected_value >= min_ev],
                     key=lambda x: x.expected_value, reverse=True)
