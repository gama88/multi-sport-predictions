"""
NCAA Football Betting Models - College football predictions.
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


class NCAAFMoneylineModel:
    """Predict straight-up winners."""
    
    def __init__(self):
        self.model_version = "ncaaf_ml_v1.0"
        self.features = [
            'home_sp_plus', 'away_sp_plus',
            'home_fpi', 'away_fpi',
            'home_recruiting_rank', 'away_recruiting_rank',
        ]
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        home_sp = game_data.get('home_sp_plus', 0)
        away_sp = game_data.get('away_sp_plus', 0)
        
        sp_diff = home_sp - away_sp
        home_win_prob = 0.55 + sp_diff * 0.01  # CFB has bigger home advantage
        home_win_prob = min(max(home_win_prob, 0.15), 0.90)
        
        pick = game_data.get('home_team') if home_win_prob > 0.5 else game_data.get('away_team')
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


class NCAAFSpreadModel:
    """Predict against the spread."""
    
    def __init__(self):
        self.model_version = "ncaaf_spread_v1.0"
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        spread = game_data.get('spread', -7.0)
        
        home_sp = game_data.get('home_sp_plus', 0)
        away_sp = game_data.get('away_sp_plus', 0)
        
        # SP+ is calibrated to expected margin
        predicted_margin = (home_sp - away_sp) + 2.5  # Home field in CFB
        
        covers = predicted_margin > abs(spread) if spread < 0 else predicted_margin < spread
        margin = abs(predicted_margin - abs(spread))
        confidence = min(0.5 + margin * 0.02, 0.70)
        
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


class NCAAFOverUnderModel:
    """Predict game totals."""
    
    def __init__(self):
        self.model_version = "ncaaf_totals_v1.0"
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        total = game_data.get('total', 52.5)
        
        home_ppg = game_data.get('home_ppg', 28)
        away_ppg = game_data.get('away_ppg', 28)
        home_def_ppg = game_data.get('home_def_ppg', 24)
        away_def_ppg = game_data.get('away_def_ppg', 24)
        
        predicted_total = (home_ppg + away_ppg + home_def_ppg + away_def_ppg) / 2
        
        margin = predicted_total - total
        pick = f"OVER {total}" if margin > 0 else f"UNDER {total}"
        confidence = min(0.5 + abs(margin) * 0.02, 0.68)
        
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


class CFBPlayoffModel:
    """Special model for College Football Playoff games."""
    
    def __init__(self):
        self.model_version = "cfb_playoff_v1.0"
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        # Playoff games are tighter - look for underdog value
        home_rank = game_data.get('home_rank', 5)
        away_rank = game_data.get('away_rank', 5)
        
        rank_diff = away_rank - home_rank
        base_prob = 0.52 + rank_diff * 0.02
        home_win_prob = min(max(base_prob, 0.35), 0.65)
        
        # In playoffs, underdogs often cover
        pick = game_data.get('home_team') if home_win_prob > 0.5 else game_data.get('away_team')
        confidence = home_win_prob if home_win_prob > 0.5 else 1 - home_win_prob
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='playoff',
            pick=pick,
            odds=game_data.get('moneyline_odds', -110),
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class NCAAFBettingEngine:
    """Main NCAA football betting engine."""
    
    def __init__(self):
        self.moneyline = NCAAFMoneylineModel()
        self.spread = NCAAFSpreadModel()
        self.over_under = NCAAFOverUnderModel()
        self.playoff = CFBPlayoffModel()
    
    def get_all_predictions(self, game_data: Dict) -> Dict[str, BettingPrediction]:
        preds = {
            'moneyline': self.moneyline.predict(game_data),
            'spread': self.spread.predict(game_data),
            'over_under': self.over_under.predict(game_data),
        }
        
        if game_data.get('is_playoff', False):
            preds['playoff'] = self.playoff.predict(game_data)
        
        return preds
    
    def get_best_bets(self, games: List[Dict], min_ev: float = 2.0) -> List[BettingPrediction]:
        all_preds = []
        for game in games:
            all_preds.extend(self.get_all_predictions(game).values())
        return sorted([p for p in all_preds if p.expected_value >= min_ev],
                     key=lambda x: x.expected_value, reverse=True)
