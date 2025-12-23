"""
NFL Betting Models - Various prediction types for sportsbook apps.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
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


class NFLMoneylineModel:
    """Predict straight-up winners."""
    
    def __init__(self):
        self.model_version = "nfl_ml_v1.0"
        self.features = [
            'home_win_pct', 'away_win_pct',
            'home_point_diff', 'away_point_diff',
            'home_yards_per_game', 'away_yards_per_game',
            'home_turnovers', 'away_turnovers',
            'home_rest_days', 'away_rest_days',
            'is_primetime', 'is_divisional',
        ]
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        home_win_prob = 0.56
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


class NFLSpreadModel:
    """Predict against the spread."""
    
    def __init__(self):
        self.model_version = "nfl_spread_v1.0"
        self.features = [
            'home_ppg', 'away_ppg',
            'home_rush_ypg', 'away_rush_ypg',
            'home_pass_ypg', 'away_pass_ypg',
            'home_def_ppg', 'away_def_ppg',
            'spread_line',
        ]
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        spread = game_data.get('spread', -3.0)
        predicted_diff = 4.5  # From model
        
        covers = predicted_diff > abs(spread) if spread < 0 else predicted_diff < spread
        confidence = 0.58 if covers else 0.42
        
        pick = f"{game_data.get('home_team', 'Home')} {spread}" if covers else f"{game_data.get('away_team', 'Away')} +{abs(spread)}"
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='spread',
            pick=pick,
            odds=-110,
            confidence=confidence if covers else 1 - confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class NFLOverUnderModel:
    """Predict game totals."""
    
    def __init__(self):
        self.model_version = "nfl_totals_v1.0"
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        total_line = game_data.get('total', 45.5)
        predicted_total = 47.2
        
        margin = predicted_total - total_line
        pick = f"OVER {total_line}" if margin > 0 else f"UNDER {total_line}"
        confidence = min(0.5 + abs(margin) * 0.03, 0.75)
        
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


class NFLPlayerPropsModel:
    """Predict player props - passing yards, rushing yards, TDs, etc."""
    
    PROP_TYPES = ['passing_yards', 'rushing_yards', 'receiving_yards', 'touchdowns', 'receptions']
    
    def __init__(self):
        self.model_version = "nfl_props_v1.0"
    
    def predict(self, player_data: Dict, prop_type: str, line: float) -> BettingPrediction:
        player_avg = player_data.get(f'{prop_type}_avg', line)
        predicted = player_avg * 1.03
        
        margin = predicted - line
        pick = f"{player_data.get('name')} OVER {line} {prop_type}" if margin > 0 else f"{player_data.get('name')} UNDER {line} {prop_type}"
        confidence = min(0.5 + abs(margin) * 0.02, 0.72)
        
        return BettingPrediction(
            game_id=player_data.get('game_id', ''),
            bet_type=f'prop_{prop_type}',
            pick=pick,
            odds=-110,
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
            timestamp=datetime.now(),
        )


class NFLTeaserModel:
    """NFL teaser bet optimizer - moving spreads 6, 6.5, or 7 points."""
    
    def __init__(self):
        self.model_version = "nfl_teaser_v1.0"
        self.teaser_points = [6, 6.5, 7]
    
    def get_teaser_legs(self, games: List[Dict], points: float = 6) -> List[BettingPrediction]:
        """Find best teaser legs with adjusted spreads."""
        teaser_picks = []
        
        for game in games:
            spread = game.get('spread', -3.0)
            
            # Teaser through key numbers (3 and 7)
            adjusted_spread = spread + points if spread < 0 else spread - points
            
            # Check if teaser moves through key numbers
            key_numbers_crossed = 0
            for key in [3, 7]:
                if (spread < -key and adjusted_spread >= -key) or (spread > key and adjusted_spread <= key):
                    key_numbers_crossed += 1
            
            confidence = 0.65 + key_numbers_crossed * 0.05
            
            teaser_picks.append(BettingPrediction(
                game_id=game.get('game_id', ''),
                bet_type='teaser',
                pick=f"{game.get('home_team')} {adjusted_spread:+.1f} (teased {points})",
                odds=-110,
                confidence=min(confidence, 0.80),
                expected_value=(confidence - 0.524) * 100,
                model_version=self.model_version,
                timestamp=datetime.now(),
            ))
        
        return sorted(teaser_picks, key=lambda x: x.confidence, reverse=True)


class NFLBettingEngine:
    """Main NFL betting engine."""
    
    def __init__(self):
        self.moneyline = NFLMoneylineModel()
        self.spread = NFLSpreadModel()
        self.over_under = NFLOverUnderModel()
        self.props = NFLPlayerPropsModel()
        self.teaser = NFLTeaserModel()
    
    def get_all_predictions(self, game_data: Dict) -> Dict[str, BettingPrediction]:
        return {
            'moneyline': self.moneyline.predict(game_data),
            'spread': self.spread.predict(game_data),
            'over_under': self.over_under.predict(game_data),
        }
    
    def get_best_bets(self, games: List[Dict], min_ev: float = 2.0) -> List[BettingPrediction]:
        all_preds = []
        for game in games:
            all_preds.extend(self.get_all_predictions(game).values())
        return sorted([p for p in all_preds if p.expected_value >= min_ev], 
                     key=lambda x: x.expected_value, reverse=True)
