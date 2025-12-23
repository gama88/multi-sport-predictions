"""
Universal Parlay Builder - Build optimal parlays across all sports.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np


@dataclass
class ParlayLeg:
    sport: str
    game_id: str
    bet_type: str
    pick: str
    odds: int
    confidence: float


@dataclass
class Parlay:
    legs: List[ParlayLeg]
    combined_odds: int
    combined_confidence: float
    expected_value: float
    risk_level: str
    potential_payout: float = 0.0
    
    def __post_init__(self):
        # Calculate payout for $100 bet
        if self.combined_odds > 0:
            self.potential_payout = 100 + (100 * self.combined_odds / 100)
        else:
            self.potential_payout = 100 + (100 * 100 / abs(self.combined_odds))


class UniversalParlayBuilder:
    """Build optimal parlays across all sports."""
    
    def __init__(self):
        self.min_legs = 2
        self.max_legs = 8
        
        # Sport-specific confidence adjustments
        self.sport_reliability = {
            'nba': 1.0,
            'nfl': 0.98,
            'nhl': 0.95,
            'mlb': 0.92,
            'ncaa_basketball': 0.90,
            'ncaa_football': 0.88,
            'tennis': 0.93,
            'soccer': 0.85,
        }
    
    def calculate_parlay_odds(self, odds_list: List[int]) -> int:
        """Calculate combined parlay odds."""
        decimal_odds = []
        for odds in odds_list:
            if odds > 0:
                decimal_odds.append(1 + odds / 100)
            else:
                decimal_odds.append(1 + 100 / abs(odds))
        
        combined = np.prod(decimal_odds)
        
        if combined >= 2:
            return int((combined - 1) * 100)
        else:
            return int(-100 / (combined - 1))
    
    def build_parlay(
        self,
        predictions: List[Dict],
        num_legs: int = 3,
        risk_level: str = 'medium',
        prefer_correlation: bool = True
    ) -> Optional[Parlay]:
        """Build an optimal parlay from predictions."""
        
        # Confidence thresholds by risk level
        thresholds = {
            'conservative': 0.62,
            'medium': 0.56,
            'aggressive': 0.52,
        }
        min_conf = thresholds.get(risk_level, 0.56)
        
        # Filter eligible picks
        eligible = []
        for pred in predictions:
            adj_conf = pred['confidence'] * self.sport_reliability.get(pred['sport'], 0.9)
            if adj_conf >= min_conf:
                eligible.append({**pred, 'adj_confidence': adj_conf})
        
        if len(eligible) < num_legs:
            return None
        
        # Sort by adjusted expected value
        eligible.sort(key=lambda x: x.get('expected_value', 0), reverse=True)
        
        # Select legs (avoiding same game parlays for now)
        selected = []
        used_games = set()
        
        for pred in eligible:
            if pred['game_id'] not in used_games:
                selected.append(pred)
                used_games.add(pred['game_id'])
            if len(selected) >= num_legs:
                break
        
        if len(selected) < 2:
            return None
        
        # Build parlay
        legs = [
            ParlayLeg(
                sport=p['sport'],
                game_id=p['game_id'],
                bet_type=p['bet_type'],
                pick=p['pick'],
                odds=p['odds'],
                confidence=p['adj_confidence'],
            )
            for p in selected
        ]
        
        combined_conf = np.prod([leg.confidence for leg in legs])
        combined_odds = self.calculate_parlay_odds([leg.odds for leg in legs])
        
        # Calculate EV
        implied_prob = 100 / (combined_odds + 100) if combined_odds > 0 else abs(combined_odds) / (abs(combined_odds) + 100)
        ev = (combined_conf - implied_prob) * 100
        
        return Parlay(
            legs=legs,
            combined_odds=combined_odds,
            combined_confidence=combined_conf,
            expected_value=ev,
            risk_level=risk_level,
        )
    
    def get_best_parlays(
        self,
        predictions: List[Dict],
        count: int = 5
    ) -> List[Parlay]:
        """Get top parlay recommendations."""
        parlays = []
        
        for num_legs in [2, 3, 4, 5]:
            for risk in ['conservative', 'medium', 'aggressive']:
                parlay = self.build_parlay(predictions, num_legs, risk)
                if parlay and parlay.expected_value > 0:
                    parlays.append(parlay)
        
        # Sort by EV
        parlays.sort(key=lambda x: x.expected_value, reverse=True)
        
        return parlays[:count]
    
    def get_same_game_parlays(
        self,
        game_predictions: List[Dict],
        game_id: str
    ) -> List[Parlay]:
        """Build same-game parlays (SGP)."""
        game_preds = [p for p in game_predictions if p['game_id'] == game_id]
        
        if len(game_preds) < 2:
            return []
        
        # For SGP, combine different bet types
        sgp_legs = []
        used_types = set()
        
        for pred in sorted(game_preds, key=lambda x: x['confidence'], reverse=True):
            if pred['bet_type'] not in used_types:
                sgp_legs.append(pred)
                used_types.add(pred['bet_type'])
        
        if len(sgp_legs) < 2:
            return []
        
        # Build SGP with 2-4 legs
        sgps = []
        for num_legs in range(2, min(5, len(sgp_legs) + 1)):
            legs = [
                ParlayLeg(
                    sport=p['sport'],
                    game_id=p['game_id'],
                    bet_type=p['bet_type'],
                    pick=p['pick'],
                    odds=p['odds'],
                    confidence=p['confidence'] * 0.95,  # SGP correlation penalty
                )
                for p in sgp_legs[:num_legs]
            ]
            
            combined_conf = np.prod([leg.confidence for leg in legs])
            combined_odds = self.calculate_parlay_odds([leg.odds for leg in legs])
            
            implied_prob = 100 / (combined_odds + 100) if combined_odds > 0 else abs(combined_odds) / (abs(combined_odds) + 100)
            
            sgps.append(Parlay(
                legs=legs,
                combined_odds=combined_odds,
                combined_confidence=combined_conf,
                expected_value=(combined_conf - implied_prob) * 100,
                risk_level='sgp',
            ))
        
        return sgps


def main():
    """Demo the parlay builder."""
    print("🎰 Universal Parlay Builder")
    print("=" * 50)
    
    # Sample predictions
    predictions = [
        {'sport': 'nba', 'game_id': 'nba_1', 'bet_type': 'spread', 'pick': 'Lakers -4.5', 'odds': -110, 'confidence': 0.58, 'expected_value': 3.2},
        {'sport': 'nba', 'game_id': 'nba_2', 'bet_type': 'moneyline', 'pick': 'Celtics', 'odds': -150, 'confidence': 0.62, 'expected_value': 4.1},
        {'sport': 'nfl', 'game_id': 'nfl_1', 'bet_type': 'over_under', 'pick': 'OVER 45.5', 'odds': -110, 'confidence': 0.57, 'expected_value': 2.5},
        {'sport': 'nhl', 'game_id': 'nhl_1', 'bet_type': 'puckline', 'pick': 'Rangers +1.5', 'odds': -180, 'confidence': 0.65, 'expected_value': 5.0},
    ]
    
    builder = UniversalParlayBuilder()
    parlays = builder.get_best_parlays(predictions)
    
    for i, parlay in enumerate(parlays, 1):
        print(f"\n🎲 Parlay #{i} ({parlay.risk_level})")
        print(f"   Legs: {len(parlay.legs)}")
        print(f"   Combined Odds: {parlay.combined_odds:+d}")
        print(f"   Win Probability: {parlay.combined_confidence:.1%}")
        print(f"   Expected Value: {parlay.expected_value:.1f}")
        print(f"   Potential Payout ($100): ${parlay.potential_payout:.2f}")
        for leg in parlay.legs:
            print(f"      • {leg.sport.upper()}: {leg.pick} ({leg.odds:+d})")


if __name__ == "__main__":
    main()
