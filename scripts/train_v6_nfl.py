"""
V6 NFL Behavioral Proxy Model
==============================
Uses available NFL data:
- Scores, spread lines, over/under
- Weather (temperature, wind, humidity)
- Home/Away performance
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
DATA_DIR = BASE_DIR / "data" / "nfl"
MODELS_DIR = BASE_DIR / "models"


class NFLBehavioralEngine:
    """Feature engineering for NFL."""
    
    def load_data(self):
        """Load NFL spreadspoke data."""
        print("  Loading NFL data...")
        
        df = pd.read_csv(DATA_DIR / "spreadspoke_scores.csv")
        
        # Parse dates
        df['date'] = pd.to_datetime(df['schedule_date'], errors='coerce')
        
        # Filter to completed games with scores
        df = df.dropna(subset=['score_home', 'score_away', 'date'])
        df = df[df['schedule_season'] >= 2015]
        
        print(f"  Loaded {len(df)} games")
        print(f"  Seasons: {sorted(df['schedule_season'].unique())[-5:]}")
        
        return df.sort_values('date').reset_index(drop=True)
    
    def build_histories(self, df):
        """Build team game histories."""
        print("  Building team histories...")
        
        teams = set(df['team_home'].unique()) | set(df['team_away'].unique())
        histories = {t: [] for t in teams}
        
        for _, row in df.iterrows():
            home = row['team_home']
            away = row['team_away']
            date = row['date']
            
            home_score = row['score_home']
            away_score = row['score_away']
            total = home_score + away_score
            
            home_game = {
                'date': date,
                'is_home': True,
                'won': home_score > away_score,
                'pts': home_score,
                'pts_against': away_score,
                'margin': home_score - away_score,
                'total': total,
                'spread_line': row.get('spread_favorite', None),
                'over_under': row.get('over_under_line', None),
            }
            
            away_game = {
                'date': date,
                'is_home': False,
                'won': away_score > home_score,
                'pts': away_score,
                'pts_against': home_score,
                'margin': away_score - home_score,
                'total': total,
            }
            
            histories[home].append(home_game)
            histories[away].append(away_game)
        
        return histories
    
    def get_team_stats(self, history, n=10):  # NFL uses smaller window due to fewer games
        """Get stats from recent games."""
        if len(history) < 3:
            return None
        
        recent = history[-n:] if len(history) >= n else history
        
        def safe_mean(vals):
            valid = [v for v in vals if v is not None and not pd.isna(v) and v >= 0]
            return np.mean(valid) if valid else 0
        
        def safe_std(vals):
            valid = [v for v in vals if v is not None and not pd.isna(v) and v >= 0]
            return np.std(valid) if len(valid) > 1 else 0
        
        wins = [1 if g['won'] else 0 for g in recent]
        
        return {
            'pts_mean': safe_mean([g['pts'] for g in recent]),
            'pts_std': safe_std([g['pts'] for g in recent]),
            'pts_against_mean': safe_mean([g['pts_against'] for g in recent]),
            'margin_mean': safe_mean([g['margin'] for g in recent]),
            'total_mean': safe_mean([g['total'] for g in recent]),
            'win_rate': np.mean(wins),
            'last_3_wins': sum(wins[-3:]) if len(wins) >= 3 else sum(wins),
            'home_games': [g for g in recent if g['is_home']],
            'away_games': [g for g in recent if not g['is_home']],
            'games': len(recent),
        }
    
    def calculate_features(self, home_stats, away_stats, home_hist, away_hist, date, row):
        """Calculate features."""
        features = {}
        
        # === REST/FATIGUE (2) ===
        def calc_rest(history, d):
            if len(history) < 1:
                return 7
            last = history[-1]['date']
            return (d - last).days if d > last else 7
        
        home_rest = calc_rest(home_hist, date)
        away_rest = calc_rest(away_hist, date)
        features['rest_diff'] = (home_rest - away_rest) / 7
        features['home_rested'] = 1 if home_rest >= 7 else 0
        
        # === SCORING (4) ===
        features['pts_diff'] = (home_stats['pts_mean'] - away_stats['pts_mean']) / 30
        features['pts_against_diff'] = (away_stats['pts_against_mean'] - home_stats['pts_against_mean']) / 30
        features['margin_diff'] = (home_stats['margin_mean'] - away_stats['margin_mean']) / 20
        features['scoring_variance'] = home_stats['pts_std'] / 15 - away_stats['pts_std'] / 15
        
        # === WIN RATE (4) ===
        features['win_rate_diff'] = home_stats['win_rate'] - away_stats['win_rate']
        features['momentum_diff'] = (home_stats['last_3_wins'] - away_stats['last_3_wins']) / 3
        
        home_home_rate = np.mean([1 if g['won'] else 0 for g in home_stats['home_games']]) if home_stats['home_games'] else 0.5
        away_away_rate = np.mean([1 if g['won'] else 0 for g in away_stats['away_games']]) if away_stats['away_games'] else 0.5
        features['home_win_rate'] = home_home_rate
        features['away_road_rate'] = away_away_rate
        
        # === WEATHER (3) ===
        temp = row.get('weather_temperature', 65)
        wind = row.get('weather_wind_mph', 0)
        humidity = row.get('weather_humidity', 50)
        
        if pd.isna(temp):
            temp = 65
        if pd.isna(wind):
            wind = 0
        
        features['cold_game'] = 1 if temp < 35 else 0
        features['high_wind'] = 1 if wind > 15 else 0
        features['dome_game'] = 1 if row.get('stadium_neutral') == True else 0
        
        # === TOTALS (2) ===
        features['total_pts_diff'] = (home_stats['total_mean'] + away_stats['total_mean']) / 2 / 50
        features['defensive_quality'] = 1 - (home_stats['pts_against_mean'] + away_stats['pts_against_mean']) / 50
        
        return features
    
    def create_all_features(self, df, histories):
        """Create features for all games."""
        print("  Creating behavioral features...")
        
        features = []
        targets = {'moneyline': [], 'spread': [], 'total': []}
        valid_idx = []
        
        for idx, row in df.iterrows():
            home = row['team_home']
            away = row['team_away']
            date = row['date']
            
            home_hist = [g for g in histories.get(home, []) if g['date'] < date]
            away_hist = [g for g in histories.get(away, []) if g['date'] < date]
            
            if len(home_hist) < 3 or len(away_hist) < 3:
                continue
            
            home_stats = self.get_team_stats(home_hist, 10)
            away_stats = self.get_team_stats(away_hist, 10)
            
            if home_stats is None or away_stats is None:
                continue
            
            f = self.calculate_features(home_stats, away_stats, home_hist, away_hist, date, row)
            features.append(f)
            valid_idx.append(idx)
            
            # Targets
            home_score = row['score_home']
            away_score = row['score_away']
            
            targets['moneyline'].append(1 if home_score > away_score else 0)
            
            # Spread
            margin = home_score - away_score
            pred_spread = -(home_stats['margin_mean'] - away_stats['margin_mean']) * 0.3
            targets['spread'].append(1 if margin > pred_spread else 0)
            
            # Total
            total = home_score + away_score
            pred_total = (home_stats['total_mean'] + away_stats['total_mean']) / 2
            targets['total'].append(1 if total > pred_total else 0)
            
            if len(features) % 1000 == 0:
                print(f"    Processed {len(features)} games...")
        
        print(f"  Created {len(features)} feature rows")
        return pd.DataFrame(features), targets, valid_idx


class V6NFLModel:
    """V6 NFL model."""
    
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
            n_estimators=300, max_depth=4, learning_rate=0.01,
            reg_lambda=10.0, reg_alpha=2.0, subsample=0.7, colsample_bytree=0.6,
            random_state=42, n_jobs=-1, use_label_encoder=False
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.01,
            reg_lambda=10.0, reg_alpha=2.0, subsample=0.7, colsample_bytree=0.6,
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
        path = MODELS_DIR / "v6_nfl_complete.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'calibrators': self.calibrators,
                'metrics': self.metrics,
                'feature_names': self.feature_names,
            }, f)
        
        metrics_path = MODELS_DIR / "v6_nfl_metrics.json"
        json_metrics = {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv 
                           for kk, vv in v.items()} 
                       for k, v in self.metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"\n  Saved to: {path}")


def train_v6_nfl():
    """Train V6 NFL model."""
    print("\n" + "="*60)
    print("V6 NFL BEHAVIORAL PROXY MODEL")
    print("="*60)
    
    engine = NFLBehavioralEngine()
    
    df = engine.load_data()
    if df is None or len(df) == 0:
        return None
    
    histories = engine.build_histories(df)
    X, targets, valid_idx = engine.create_all_features(df, histories)
    
    print(f"\n  Features: {X.shape}")
    
    model = V6NFLModel()
    model.train_all(X, targets)
    
    print("\n" + "="*60)
    print("V6 NFL RESULTS")
    print("="*60)
    for bt, m in model.metrics.items():
        print(f"  {bt.upper():12} | Accuracy: {m['accuracy']:.1%} | AUC: {m['auc']:.4f}")
    
    model.save()
    return model


if __name__ == "__main__":
    model = train_v6_nfl()
