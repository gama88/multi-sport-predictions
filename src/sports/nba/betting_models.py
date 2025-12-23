"""
NBA Betting Models - Various prediction types for sportsbook apps.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class BettingPrediction:
    """A single betting prediction."""
    game_id: str
    bet_type: str  # moneyline, spread, over_under, prop, parlay_leg
    pick: str
    odds: int  # American odds
    confidence: float  # 0-1
    expected_value: float  # Expected value in units
    model_version: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ParlayLeg:
    """A single leg of a parlay."""
    game_id: str
    bet_type: str
    pick: str
    odds: int
    confidence: float


@dataclass 
class ParlayRecommendation:
    """A parlay recommendation with multiple legs."""
    legs: List[ParlayLeg]
    combined_odds: int
    combined_confidence: float
    expected_value: float
    risk_level: str  # low, medium, high


class NBAMoneylineModel:
    """Predict straight-up winners (moneyline bets)."""
    
    def __init__(self):
        self.model = None
        self.model_version = "moneyline_v1.0"
        self.features = [
            'home_win_pct', 'away_win_pct',
            'home_net_rating', 'away_net_rating',
            'home_last10_wins', 'away_last10_wins',
            'home_rest_days', 'away_rest_days',
            'h2h_home_wins_pct',
        ]
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        """Predict moneyline winner."""
        # Feature extraction would go here
        # For now, return a sample prediction
        home_win_prob = 0.58  # Would be from model
        
        if home_win_prob > 0.5:
            pick = game_data.get('home_team', 'Home')
            confidence = home_win_prob
        else:
            pick = game_data.get('away_team', 'Away')
            confidence = 1 - home_win_prob
        
        # Calculate expected value
        odds = game_data.get('moneyline_odds', -110)
        implied_prob = self._odds_to_prob(odds)
        ev = (confidence - implied_prob) * 100  # EV in units
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='moneyline',
            pick=pick,
            odds=odds,
            confidence=confidence,
            expected_value=ev,
            model_version=self.model_version,
        )
    
    def _odds_to_prob(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)


class NBASpreadModel:
    """Predict against the spread (ATS)."""
    
    def __init__(self):
        self.model = None
        self.model_version = "spread_v1.0"
        self.features = [
            'home_point_diff', 'away_point_diff',
            'home_off_rating', 'away_off_rating',
            'home_def_rating', 'away_def_rating',
            'home_pace', 'away_pace',
            'spread_line',
        ]
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        """Predict spread cover."""
        spread = game_data.get('spread', -3.5)
        
        # Model would predict point differential
        predicted_diff = 4.2  # Would be from model
        
        # Determine if home covers
        covers_spread = predicted_diff > abs(spread) if spread < 0 else predicted_diff < spread
        
        margin = abs(predicted_diff - abs(spread))
        confidence = min(0.5 + margin * 0.05, 0.85)  # Higher margin = higher confidence
        
        if covers_spread:
            pick = f"{game_data.get('home_team', 'Home')} {spread}"
        else:
            pick = f"{game_data.get('away_team', 'Away')} +{abs(spread)}"
        
        ev = (confidence - 0.524) * 100  # 0.524 is break-even at -110
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='spread',
            pick=pick,
            odds=-110,
            confidence=confidence,
            expected_value=ev,
            model_version=self.model_version,
        )


class NBAOverUnderModel:
    """Predict game totals (over/under)."""
    
    def __init__(self):
        self.model = None
        self.model_version = "totals_v1.0"
        self.features = [
            'home_ppg', 'away_ppg',
            'home_opp_ppg', 'away_opp_ppg',
            'home_pace', 'away_pace',
            'total_line',
        ]
    
    def predict(self, game_data: Dict) -> BettingPrediction:
        """Predict over/under."""
        total_line = game_data.get('total', 220.5)
        
        # Model would predict total points
        predicted_total = 224.8  # Would be from model
        
        margin = predicted_total - total_line
        if margin > 0:
            pick = f"OVER {total_line}"
            confidence = min(0.5 + abs(margin) * 0.02, 0.80)
        else:
            pick = f"UNDER {total_line}"
            confidence = min(0.5 + abs(margin) * 0.02, 0.80)
        
        ev = (confidence - 0.524) * 100
        
        return BettingPrediction(
            game_id=game_data.get('game_id', ''),
            bet_type='over_under',
            pick=pick,
            odds=-110,
            confidence=confidence,
            expected_value=ev,
            model_version=self.model_version,
        )


class NBAPlayerPropsModel:
    """Predict player prop bets."""
    
    PROP_TYPES = ['points', 'rebounds', 'assists', 'threes', 'pts_rebs_asts']
    
    def __init__(self):
        self.model = None
        self.model_version = "props_v1.0"
    
    def predict(self, player_data: Dict, prop_type: str, line: float) -> BettingPrediction:
        """Predict player prop."""
        # Would use player's recent stats, matchup, etc.
        player_avg = player_data.get(f'{prop_type}_avg', line)
        
        predicted_value = player_avg * 1.02  # Simplified
        
        margin = predicted_value - line
        if margin > 0:
            pick = f"{player_data.get('name', 'Player')} OVER {line} {prop_type}"
            confidence = min(0.5 + abs(margin) * 0.03, 0.75)
        else:
            pick = f"{player_data.get('name', 'Player')} UNDER {line} {prop_type}"
            confidence = min(0.5 + abs(margin) * 0.03, 0.75)
        
        return BettingPrediction(
            game_id=player_data.get('game_id', ''),
            bet_type=f'prop_{prop_type}',
            pick=pick,
            odds=-110,
            confidence=confidence,
            expected_value=(confidence - 0.524) * 100,
            model_version=self.model_version,
        )


class NBAParlayOptimizer:
    """Build optimal parlays from multiple predictions."""
    
    def __init__(self):
        self.min_confidence = 0.55
        self.max_legs = 6
    
    def build_parlay(
        self, 
        predictions: List[BettingPrediction],
        num_legs: int = 3,
        risk_level: str = 'medium'
    ) -> ParlayRecommendation:
        """Build an optimal parlay from predictions."""
        
        # Filter by confidence threshold based on risk level
        thresholds = {'low': 0.62, 'medium': 0.58, 'high': 0.52}
        min_conf = thresholds.get(risk_level, 0.58)
        
        eligible = [p for p in predictions if p.confidence >= min_conf]
        
        # Sort by expected value
        eligible.sort(key=lambda x: x.expected_value, reverse=True)
        
        # Take top N legs
        selected = eligible[:min(num_legs, len(eligible))]
        
        if len(selected) < 2:
            return None
        
        # Build parlay legs
        legs = [
            ParlayLeg(
                game_id=p.game_id,
                bet_type=p.bet_type,
                pick=p.pick,
                odds=p.odds,
                confidence=p.confidence,
            )
            for p in selected
        ]
        
        # Calculate combined odds and confidence
        combined_conf = np.prod([leg.confidence for leg in legs])
        combined_odds = self._calculate_parlay_odds([leg.odds for leg in legs])
        
        # Expected value
        implied_prob = self._combined_odds_to_prob(combined_odds)
        ev = (combined_conf - implied_prob) * 100
        
        return ParlayRecommendation(
            legs=legs,
            combined_odds=combined_odds,
            combined_confidence=combined_conf,
            expected_value=ev,
            risk_level=risk_level,
        )
    
    def _calculate_parlay_odds(self, odds_list: List[int]) -> int:
        """Calculate combined parlay odds."""
        decimal_odds = []
        for odds in odds_list:
            if odds > 0:
                decimal_odds.append(1 + odds / 100)
            else:
                decimal_odds.append(1 + 100 / abs(odds))
        
        combined_decimal = np.prod(decimal_odds)
        
        # Convert back to American
        if combined_decimal >= 2:
            return int((combined_decimal - 1) * 100)
        else:
            return int(-100 / (combined_decimal - 1))
    
    def _combined_odds_to_prob(self, odds: int) -> float:
        """Convert American odds to probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def get_best_parlays(
        self,
        predictions: List[BettingPrediction],
        count: int = 5
    ) -> List[ParlayRecommendation]:
        """Get top parlay recommendations."""
        parlays = []
        
        # Generate parlays of different sizes and risk levels
        for num_legs in [2, 3, 4]:
            for risk in ['low', 'medium', 'high']:
                parlay = self.build_parlay(predictions, num_legs, risk)
                if parlay:
                    parlays.append(parlay)
        
        # Sort by expected value
        parlays.sort(key=lambda x: x.expected_value, reverse=True)
        
        return parlays[:count]


class NBABettingEngine:
    """Main engine combining all betting models."""
    
    def __init__(self):
        self.moneyline = NBAMoneylineModel()
        self.spread = NBASpreadModel()
        self.over_under = NBAOverUnderModel()
        self.props = NBAPlayerPropsModel()
        self.parlay_optimizer = NBAParlayOptimizer()
    
    def get_all_predictions(self, game_data: Dict) -> Dict[str, BettingPrediction]:
        """Get all prediction types for a game."""
        return {
            'moneyline': self.moneyline.predict(game_data),
            'spread': self.spread.predict(game_data),
            'over_under': self.over_under.predict(game_data),
        }
    
    def get_best_bets(
        self, 
        games: List[Dict],
        min_ev: float = 2.0
    ) -> List[BettingPrediction]:
        """Get best bets across all games."""
        all_predictions = []
        
        for game in games:
            preds = self.get_all_predictions(game)
            all_predictions.extend(preds.values())
        
        # Filter by minimum EV
        best = [p for p in all_predictions if p.expected_value >= min_ev]
        best.sort(key=lambda x: x.expected_value, reverse=True)
        
        return best
    
    def get_parlay_recommendations(
        self,
        games: List[Dict]
    ) -> List[ParlayRecommendation]:
        """Get parlay recommendations from all games."""
        all_predictions = []
        
        for game in games:
            preds = self.get_all_predictions(game)
            all_predictions.extend(preds.values())
        
        return self.parlay_optimizer.get_best_parlays(all_predictions)
