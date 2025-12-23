"""
V6 Soccer Behavioral Proxy Model
=================================
Uses comprehensive Transfermarkt data:
- Goals scored, goals against
- Win/draw/loss history
- Home/away performance
- League position context
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "soccer"
MODELS_DIR = BASE_DIR / "models"


class SoccerBehavioralEngine:
    """Feature engineering for soccer."""
    
    def load_data(self):
        """Load soccer data."""
        print("  Loading soccer data...")
        
        # Load games
        games = pd.read_csv(DATA_DIR / "games.csv")
        
        # Parse dates
        games['date'] = pd.to_datetime(games['date'], errors='coerce')
        
        # Filter to completed games with goals
        games = games.dropna(subset=['home_club_goals', 'away_club_goals', 'date'])
        
        # Filter to domestic leagues only (more predictable)
        games = games[games['competition_type'] == 'domestic_league']
        
        # Recent seasons
        games = games[games['season'] >= 2018]
        
        print(f"  Loaded {len(games)} games")
        print(f"  Seasons: {sorted(games['season'].unique())[-5:]}")
        print(f"  Competitions: {games['competition_id'].nunique()} leagues")
        
        return games.sort_values('date').reset_index(drop=True)
    
    def build_histories(self, df):
        """Build team game histories."""
        print("  Building team histories...")
        
        teams = set(df['home_club_id'].unique()) | set(df['away_club_id'].unique())
        histories = {t: [] for t in teams}
        
        for _, row in df.iterrows():
            home_id = row['home_club_id']
            away_id = row['away_club_id']
            date = row['date']
            
            home_goals = row['home_club_goals']
            away_goals = row['away_club_goals']
            
            # Determine result
            if home_goals > away_goals:
                home_result = 'win'
                away_result = 'loss'
            elif home_goals < away_goals:
                home_result = 'loss'
                away_result = 'win'
            else:
                home_result = 'draw'
                away_result = 'draw'
            
            home_game = {
                'date': date,
                'is_home': True,
                'result': home_result,
                'goals': home_goals,
                'goals_against': away_goals,
                'goal_diff': home_goals - away_goals,
                'points': 3 if home_result == 'win' else (1 if home_result == 'draw' else 0),
                'competition': row['competition_id'],
            }
            
            away_game = {
                'date': date,
                'is_home': False,
                'result': away_result,
                'goals': away_goals,
                'goals_against': home_goals,
                'goal_diff': away_goals - home_goals,
                'points': 3 if away_result == 'win' else (1 if away_result == 'draw' else 0),
                'competition': row['competition_id'],
            }
            
            histories[home_id].append(home_game)
            histories[away_id].append(away_game)
        
        return histories
    
    def get_team_stats(self, history, n=10):
        """Get stats from recent games."""
        if len(history) < 3:
            return None
        
        recent = history[-n:] if len(history) >= n else history
        
        def safe_mean(vals):
            valid = [v for v in vals if v is not None and not pd.isna(v)]
            return np.mean(valid) if valid else 0
        
        def safe_std(vals):
            valid = [v for v in vals if v is not None and not pd.isna(v)]
            return np.std(valid) if len(valid) > 1 else 0
        
        results = [g['result'] for g in recent]
        wins = sum(1 for r in results if r == 'win')
        draws = sum(1 for r in results if r == 'draw')
        losses = sum(1 for r in results if r == 'loss')
        
        return {
            'goals_mean': safe_mean([g['goals'] for g in recent]),
            'goals_std': safe_std([g['goals'] for g in recent]),
            'goals_against_mean': safe_mean([g['goals_against'] for g in recent]),
            'goal_diff_mean': safe_mean([g['goal_diff'] for g in recent]),
            'points_mean': safe_mean([g['points'] for g in recent]),
            'win_rate': wins / len(recent),
            'draw_rate': draws / len(recent),
            'loss_rate': losses / len(recent),
            'unbeaten_rate': (wins + draws) / len(recent),
            'last_3_points': sum(g['points'] for g in recent[-3:]) if len(recent) >= 3 else sum(g['points'] for g in recent),
            'clean_sheets': sum(1 for g in recent if g['goals_against'] == 0) / len(recent),
            'failed_to_score': sum(1 for g in recent if g['goals'] == 0) / len(recent),
            'home_games': [g for g in recent if g['is_home']],
            'away_games': [g for g in recent if not g['is_home']],
            'games': len(recent),
        }
    
    def calculate_features(self, home_stats, away_stats, home_hist, away_hist, date):
        """Calculate behavioral features for soccer."""
        features = {}
        
        # === FATIGUE/REST (2) ===
        def calc_rest(history, d):
            if len(history) < 1:
                return 7
            last = history[-1]['date']
            return (d - last).days if d > last else 7
        
        home_rest = calc_rest(home_hist, date)
        away_rest = calc_rest(away_hist, date)
        features['rest_diff'] = (home_rest - away_rest) / 7
        features['congestion'] = 1 if home_rest < 4 or away_rest < 4 else 0
        
        # === SCORING (4) ===
        features['goals_diff'] = (home_stats['goals_mean'] - away_stats['goals_mean']) / 3
        features['goals_against_diff'] = (away_stats['goals_against_mean'] - home_stats['goals_against_mean']) / 3
        features['goal_diff_diff'] = (home_stats['goal_diff_mean'] - away_stats['goal_diff_mean']) / 3
        features['scoring_variance'] = home_stats['goals_std'] / 2 - away_stats['goals_std'] / 2
        
        # === FORM (5) ===
        features['win_rate_diff'] = home_stats['win_rate'] - away_stats['win_rate']
        features['draw_rate_diff'] = home_stats['draw_rate'] - away_stats['draw_rate']
        features['unbeaten_diff'] = home_stats['unbeaten_rate'] - away_stats['unbeaten_rate']
        features['points_diff'] = (home_stats['points_mean'] - away_stats['points_mean']) / 3
        features['momentum_diff'] = (home_stats['last_3_points'] - away_stats['last_3_points']) / 9
        
        # === DEFENSIVE QUALITY (3) ===
        features['clean_sheet_diff'] = home_stats['clean_sheets'] - away_stats['clean_sheets']
        features['defensive_quality'] = 1 - (home_stats['goals_against_mean'] + away_stats['goals_against_mean']) / 4
        features['failed_to_score_diff'] = away_stats['failed_to_score'] - home_stats['failed_to_score']
        
        # === HOME/AWAY SPLITS (3) ===
        home_home_pts = np.mean([g['points'] for g in home_stats['home_games']]) if home_stats['home_games'] else 1.5
        away_away_pts = np.mean([g['points'] for g in away_stats['away_games']]) if away_stats['away_games'] else 1.5
        features['home_points_rate'] = home_home_pts / 3
        features['away_points_rate'] = away_away_pts / 3
        features['home_advantage'] = home_home_pts - away_away_pts
        
        return features
    
    def create_all_features(self, df, histories):
        """Create features for all games."""
        print("  Creating behavioral features...")
        
        features = []
        targets = {'moneyline': [], 'spread': [], 'total': [], 'draw': []}
        valid_idx = []
        
        for idx, row in df.iterrows():
            home_id = row['home_club_id']
            away_id = row['away_club_id']
            date = row['date']
            
            home_hist = [g for g in histories.get(home_id, []) if g['date'] < date]
            away_hist = [g for g in histories.get(away_id, []) if g['date'] < date]
            
            if len(home_hist) < 3 or len(away_hist) < 3:
                continue
            
            home_stats = self.get_team_stats(home_hist, 10)
            away_stats = self.get_team_stats(away_hist, 10)
            
            if home_stats is None or away_stats is None:
                continue
            
            f = self.calculate_features(home_stats, away_stats, home_hist, away_hist, date)
            features.append(f)
            valid_idx.append(idx)
            
            # Targets
            home_goals = row['home_club_goals']
            away_goals = row['away_club_goals']
            
            # Moneyline (home win = 1, draw or away = 0)
            targets['moneyline'].append(1 if home_goals > away_goals else 0)
            
            # Draw (for 3-way market)
            targets['draw'].append(1 if home_goals == away_goals else 0)
            
            # Spread (Asian Handicap style)
            margin = home_goals - away_goals
            pred_spread = -(home_stats['goal_diff_mean'] - away_stats['goal_diff_mean']) * 0.3
            targets['spread'].append(1 if margin > pred_spread else 0)
            
            # Total (Over/Under 2.5)
            total = home_goals + away_goals
            targets['total'].append(1 if total > 2.5 else 0)
            
            if len(features) % 10000 == 0:
                print(f"    Processed {len(features)} games...")
        
        print(f"  Created {len(features)} feature rows")
        return pd.DataFrame(features), targets, valid_idx


class V6SoccerModel:
    """V6 Soccer model."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.metrics = {}
        self.feature_names = None
    
    def train_bet_style(self, X, y, bet_type='moneyline'):
        """Train for a bet type."""
        print(f"\n  Training {bet_type.upper()}...")
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[bet_type] = scaler
        self.feature_names = list(X.columns)
        
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = np.array(y[:split]), np.array(y[split:])
        
        print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.01,
            reg_lambda=5.0, reg_alpha=1.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, n_jobs=-1, use_label_encoder=False
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.01,
            reg_lambda=5.0, reg_alpha=1.0, subsample=0.8, colsample_bytree=0.7,
            random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        
        # Ensemble
        xgb_p = xgb_model.predict_proba(X_test)[:, 1]
        lgb_p = lgb_model.predict_proba(X_test)[:, 1]
        ensemble_p = 0.5 * xgb_p + 0.5 * lgb_p
        
        pred = (ensemble_p >= 0.5).astype(int)
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, ensemble_p)
        
        self.models[bet_type] = {'xgb': xgb_model, 'lgb': lgb_model}
        self.calibrators[bet_type] = IsotonicRegression(out_of_bounds='clip').fit(ensemble_p, y_test)
        self.metrics[bet_type] = {'accuracy': acc, 'auc': auc, 'test_size': len(y_test)}
        
        print(f"    Accuracy: {acc:.1%}")
        print(f"    AUC: {auc:.4f}")
        
        return acc
    
    def train_all(self, X, targets):
        """Train all bet types."""
        self.train_bet_style(X, targets['moneyline'], 'moneyline')
        self.train_bet_style(X, targets['spread'], 'spread')
        self.train_bet_style(X, targets['total'], 'total')
        self.train_bet_style(X, targets['moneyline'], 'contracts')
    
    def save(self):
        """Save model."""
        path = MODELS_DIR / "v6_soccer_complete.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'calibrators': self.calibrators,
                'metrics': self.metrics,
                'feature_names': self.feature_names,
            }, f)
        
        metrics_path = MODELS_DIR / "v6_soccer_metrics.json"
        json_metrics = {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv 
                           for kk, vv in v.items()} 
                       for k, v in self.metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"\n  Saved to: {path}")


def train_v6_soccer():
    """Train V6 Soccer model."""
    print("\n" + "="*60)
    print("V6 SOCCER BEHAVIORAL PROXY MODEL")
    print("="*60)
    
    engine = SoccerBehavioralEngine()
    
    df = engine.load_data()
    if df is None or len(df) == 0:
        return None
    
    histories = engine.build_histories(df)
    X, targets, valid_idx = engine.create_all_features(df, histories)
    
    print(f"\n  Features: {X.shape}")
    print(f"  Columns: {list(X.columns)}")
    
    model = V6SoccerModel()
    model.train_all(X, targets)
    
    print("\n" + "="*60)
    print("V6 SOCCER RESULTS")
    print("="*60)
    for bt, m in model.metrics.items():
        print(f"  {bt.upper():12} | Accuracy: {m['accuracy']:.1%} | AUC: {m['auc']:.4f}")
    
    # Home win rate as baseline
    home_wins = sum(targets['moneyline']) / len(targets['moneyline'])
    print(f"\n  Baseline (home win rate): {home_wins:.1%}")
    
    model.save()
    return model


if __name__ == "__main__":
    model = train_v6_soccer()
