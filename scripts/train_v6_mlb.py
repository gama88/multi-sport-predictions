"""
V6 MLB Behavioral Proxy Model
==============================
Uses MLB's behavioral stats:
- Hits, Runs, Errors (offensive execution)
- LOB (left on base - clutch hitting)
- Innings (extra innings indicator)
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
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "mlb"
MODELS_DIR = BASE_DIR / "models"


class MLBBehavioralEngine:
    """Feature engineering for MLB behavioral proxy model."""
    
    def load_data(self):
        """Load MLB enhanced data."""
        print("  Loading MLB data...")
        
        path = DATA_DIR / "mlb_games_enhanced.csv"
        if not path.exists():
            print(f"  Error: {path} not found")
            return None
        
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"  Loaded {len(df)} games")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df.sort_values('date').reset_index(drop=True)
    
    def build_histories(self, df):
        """Build team game histories."""
        print("  Building team histories...")
        
        teams = set(df['home_team_id'].unique()) | set(df['away_team_id'].unique())
        histories = {t: [] for t in teams}
        
        for _, row in df.iterrows():
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            date = row['date']
            
            home_game = {
                'date': date,
                'is_home': True,
                'won': row['home_score'] > row['away_score'],
                'runs': row['home_score'],
                'runs_against': row['away_score'],
                'hits': row.get('home_hits', 8) or 8,
                'errors': row.get('home_errors', 0) or 0,
                'lob': row.get('home_lob', 6) or 6,
                'innings': row.get('innings', 9) or 9,
            }
            
            away_game = {
                'date': date,
                'is_home': False,
                'won': row['away_score'] > row['home_score'],
                'runs': row['away_score'],
                'runs_against': row['home_score'],
                'hits': row.get('away_hits', 8) or 8,
                'errors': row.get('away_errors', 0) or 0,
                'lob': row.get('away_lob', 6) or 6,
                'innings': row.get('innings', 9) or 9,
            }
            
            histories[home_id].append(home_game)
            histories[away_id].append(away_game)
        
        return histories
    
    def get_team_stats(self, history, n=20):
        """Get stats from last n games."""
        if len(history) < 3:
            return None
        
        recent = history[-n:] if len(history) >= n else history
        
        def safe_mean(vals):
            valid = [v for v in vals if v is not None and v >= 0]
            return np.mean(valid) if valid else 0
        
        def safe_std(vals):
            valid = [v for v in vals if v is not None and v >= 0]
            return np.std(valid) if len(valid) > 1 else 0
        
        wins = [1 if g['won'] else 0 for g in recent]
        
        return {
            # Scoring
            'runs_mean': safe_mean([g['runs'] for g in recent]),
            'runs_std': safe_std([g['runs'] for g in recent]),
            'runs_against_mean': safe_mean([g['runs_against'] for g in recent]),
            
            # Hits/Execution
            'hits_mean': safe_mean([g['hits'] for g in recent]),
            'hits_std': safe_std([g['hits'] for g in recent]),
            
            # Errors (discipline)
            'errors_mean': safe_mean([g['errors'] for g in recent]),
            
            # LOB (clutch hitting - lower is better)
            'lob_mean': safe_mean([g['lob'] for g in recent]),
            
            # Extra innings tendency
            'extra_innings': sum(1 for g in recent if g.get('innings', 9) > 9) / len(recent),
            
            # Win rate
            'win_rate': np.mean(wins),
            'last_5_wins': sum(wins[-5:]) if len(wins) >= 5 else sum(wins),
            'first_5_wins': sum(wins[:5]) if len(wins) >= 5 else sum(wins),
            
            # Home/Away splits
            'home_games': [g for g in recent if g['is_home']],
            'away_games': [g for g in recent if not g['is_home']],
            'games': len(recent),
        }
    
    def calculate_features(self, home_stats, away_stats, home_hist, away_hist, date):
        """Calculate behavioral features for MLB."""
        features = {}
        
        # === FATIGUE (3) ===
        def calc_fatigue(history, d):
            if len(history) < 2:
                return {'rest': 0.5, 'g7d': 0.5}
            last = history[-1]['date']
            rest = (d - last).days if d > last else 1
            week_ago = d - timedelta(days=7)
            g7d = sum(1 for g in history if g['date'] >= week_ago and g['date'] < d)
            return {
                'rest': min(rest / 7.0, 1.0),
                'g7d': min(g7d / 6.0, 1.0),
            }
        
        home_fat = calc_fatigue(home_hist, date)
        away_fat = calc_fatigue(away_hist, date)
        
        features['fatigue_rest_diff'] = home_fat['rest'] - away_fat['rest']
        features['fatigue_g7d_diff'] = away_fat['g7d'] - home_fat['g7d']
        
        # === OFFENSIVE EXECUTION (4) ===
        features['runs_diff'] = (home_stats['runs_mean'] - away_stats['runs_mean']) / 5
        features['hits_diff'] = (home_stats['hits_mean'] - away_stats['hits_mean']) / 10
        features['runs_variance'] = home_stats['runs_std'] / 4 - away_stats['runs_std'] / 4
        
        # Run differential (defense combined)
        home_run_diff = home_stats['runs_mean'] - home_stats['runs_against_mean']
        away_run_diff = away_stats['runs_mean'] - away_stats['runs_against_mean']
        features['run_differential_diff'] = (home_run_diff - away_run_diff) / 4
        
        # === DISCIPLINE (3) ===
        # Errors hurt
        features['errors_diff'] = (away_stats['errors_mean'] - home_stats['errors_mean']) / 2
        
        # LOB - leaving runners on = not clutch
        features['lob_diff'] = (away_stats['lob_mean'] - home_stats['lob_mean']) / 10
        
        # Hits per run efficiency
        home_hit_eff = home_stats['runs_mean'] / max(home_stats['hits_mean'], 1)
        away_hit_eff = away_stats['runs_mean'] / max(away_stats['hits_mean'], 1)
        features['hit_efficiency_diff'] = home_hit_eff - away_hit_eff
        
        # === WIN RATE (4) ===
        features['win_rate_diff'] = home_stats['win_rate'] - away_stats['win_rate']
        features['momentum_diff'] = (home_stats['last_5_wins'] - away_stats['last_5_wins']) / 5
        
        # Form trend
        home_form = (home_stats['last_5_wins'] - home_stats['first_5_wins']) / 5
        away_form = (away_stats['last_5_wins'] - away_stats['first_5_wins']) / 5
        features['form_trend_diff'] = home_form - away_form
        
        # Home/Away split
        home_home_rate = np.mean([1 if g['won'] else 0 for g in home_stats['home_games']]) if home_stats['home_games'] else 0.5
        away_away_rate = np.mean([1 if g['won'] else 0 for g in away_stats['away_games']]) if away_stats['away_games'] else 0.5
        features['home_win_rate'] = home_home_rate
        features['away_road_rate'] = away_away_rate
        
        # === EXTRA FEATURES (2) ===
        features['extra_innings_diff'] = home_stats['extra_innings'] - away_stats['extra_innings']
        features['experience_diff'] = (home_stats['games'] - away_stats['games']) / 20
        
        return features
    
    def create_all_features(self, df, histories):
        """Create features for all games."""
        print("  Creating behavioral features...")
        
        features = []
        targets = {'moneyline': [], 'spread': [], 'total': []}
        valid_idx = []
        
        for idx, row in df.iterrows():
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            date = row['date']
            
            home_hist = [g for g in histories.get(home_id, []) if g['date'] < date]
            away_hist = [g for g in histories.get(away_id, []) if g['date'] < date]
            
            if len(home_hist) < 5 or len(away_hist) < 5:
                continue
            
            home_stats = self.get_team_stats(home_hist, 20)
            away_stats = self.get_team_stats(away_hist, 20)
            
            if home_stats is None or away_stats is None:
                continue
            
            f = self.calculate_features(home_stats, away_stats, home_hist, away_hist, date)
            features.append(f)
            valid_idx.append(idx)
            
            # Targets
            home_runs = row['home_score']
            away_runs = row['away_score']
            
            targets['moneyline'].append(1 if home_runs > away_runs else 0)
            
            # Spread
            margin = home_runs - away_runs
            pred_spread = -(home_stats['runs_mean'] - away_stats['runs_mean']) * 0.5
            targets['spread'].append(1 if margin > pred_spread else 0)
            
            # Total
            total = home_runs + away_runs
            pred_total = home_stats['runs_mean'] + away_stats['runs_mean']
            targets['total'].append(1 if total > pred_total else 0)
            
            if len(features) % 1000 == 0:
                print(f"    Processed {len(features)} games...")
        
        print(f"  Created {len(features)} feature rows")
        return pd.DataFrame(features), targets, valid_idx


class V6MLBModel:
    """V6 MLB model."""
    
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
        path = MODELS_DIR / "v6_mlb_complete.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'calibrators': self.calibrators,
                'metrics': self.metrics,
                'feature_names': self.feature_names,
            }, f)
        
        # Save metrics JSON
        metrics_path = MODELS_DIR / "v6_mlb_metrics.json"
        json_metrics = {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv 
                           for kk, vv in v.items()} 
                       for k, v in self.metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"\n  Saved to: {path}")


def train_v6_mlb():
    """Train V6 MLB model."""
    print("\n" + "="*60)
    print("V6 MLB BEHAVIORAL PROXY MODEL")
    print("="*60)
    
    engine = MLBBehavioralEngine()
    
    df = engine.load_data()
    if df is None:
        return None
    
    histories = engine.build_histories(df)
    X, targets, valid_idx = engine.create_all_features(df, histories)
    
    print(f"\n  Features: {X.shape}")
    
    model = V6MLBModel()
    model.train_all(X, targets)
    
    print("\n" + "="*60)
    print("V6 MLB RESULTS")
    print("="*60)
    for bt, m in model.metrics.items():
        print(f"  {bt.upper():12} | Accuracy: {m['accuracy']:.1%} | AUC: {m['auc']:.4f}")
    
    model.save()
    return model


if __name__ == "__main__":
    model = train_v6_mlb()
