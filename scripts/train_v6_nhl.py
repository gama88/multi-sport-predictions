"""
V6 NHL Behavioral Proxy Model
==============================
Uses NHL's rich behavioral stats:
- Hits, shots, blocks (physicality)
- Giveaways, takeaways (puck control/discipline)
- Penalty minutes (discipline)
- FaceOff%, PowerPlay (special teams)
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
DATA_DIR = BASE_DIR / "data" / "nhl"
MODELS_DIR = BASE_DIR / "models"


class NHLBehavioralEngine:
    """Feature engineering for NHL behavioral proxy model."""
    
    def load_data(self):
        """Load and merge NHL game data."""
        print("  Loading NHL data...")
        
        # Game stats per team
        stats = pd.read_csv(DATA_DIR / "game_teams_stats.csv")
        
        # Game info for dates
        games = pd.read_csv(DATA_DIR / "game.csv")
        
        print(f"  Team stats: {len(stats):,} rows")
        print(f"  Games: {len(games):,} rows")
        
        # Merge to get dates
        stats = stats.merge(games[['game_id', 'date_time_GMT', 'home_team_id', 'away_team_id']], 
                           on='game_id', how='left')
        
        # Parse date
        stats['date'] = pd.to_datetime(stats['date_time_GMT'])
        
        # Split home/away
        home = stats[stats['HoA'] == 'home'].copy()
        away = stats[stats['HoA'] == 'away'].copy()
        
        # Rename columns
        home_cols = {c: f'{c}_home' for c in home.columns if c not in ['game_id', 'date', 'home_team_id', 'away_team_id']}
        away_cols = {c: f'{c}_away' for c in away.columns if c not in ['game_id', 'date', 'home_team_id', 'away_team_id']}
        
        home = home.rename(columns=home_cols)
        away = away.rename(columns=away_cols)
        
        # Merge home and away
        paired = home.merge(away, on=['game_id', 'date', 'home_team_id', 'away_team_id'], how='inner')
        
        print(f"  Paired games: {len(paired):,}")
        
        return paired.sort_values('date').reset_index(drop=True)
    
    def build_histories(self, df):
        """Build team histories for behavioral features."""
        print("  Building team histories...")
        
        teams = set(df['home_team_id'].unique()) | set(df['away_team_id'].unique())
        histories = {t: [] for t in teams}
        
        for _, row in df.iterrows():
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            date = row['date']
            
            # Home team game record
            home_game = {
                'date': date,
                'is_home': True,
                'won': row.get('won_home', False) == True,
                'goals': row.get('goals_home', 0) or 0,
                'goals_against': row.get('goals_away', 0) or 0,
                'shots': row.get('shots_home', 30) or 30,
                'hits': row.get('hits_home', 20) or 20,
                'pim': row.get('pim_home', 10) or 10,
                'giveaways': row.get('giveaways_home', 10) or 10,
                'takeaways': row.get('takeaways_home', 5) or 5,
                'blocked': row.get('blocked_home', 15) or 15,
                'faceoff_pct': row.get('faceOffWinPercentage_home', 50) or 50,
                'pp_goals': row.get('powerPlayGoals_home', 0) or 0,
                'pp_opps': row.get('powerPlayOpportunities_home', 3) or 3,
            }
            
            # Away team game record
            away_game = {
                'date': date,
                'is_home': False,
                'won': row.get('won_away', False) == True,
                'goals': row.get('goals_away', 0) or 0,
                'goals_against': row.get('goals_home', 0) or 0,
                'shots': row.get('shots_away', 30) or 30,
                'hits': row.get('hits_away', 20) or 20,
                'pim': row.get('pim_away', 10) or 10,
                'giveaways': row.get('giveaways_away', 10) or 10,
                'takeaways': row.get('takeaways_away', 5) or 5,
                'blocked': row.get('blocked_away', 15) or 15,
                'faceoff_pct': row.get('faceOffWinPercentage_away', 50) or 50,
                'pp_goals': row.get('powerPlayGoals_away', 0) or 0,
                'pp_opps': row.get('powerPlayOpportunities_away', 3) or 3,
            }
            
            histories[home_id].append(home_game)
            histories[away_id].append(away_game)
        
        return histories
    
    def get_team_stats(self, history, n=20):
        """Aggregate stats from last n games."""
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
            'goals_mean': safe_mean([g['goals'] for g in recent]),
            'goals_std': safe_std([g['goals'] for g in recent]),
            'goals_against_mean': safe_mean([g['goals_against'] for g in recent]),
            
            # Shots/Possession
            'shots_mean': safe_mean([g['shots'] for g in recent]),
            'faceoff_pct': safe_mean([g['faceoff_pct'] for g in recent]),
            
            # Physicality
            'hits_mean': safe_mean([g['hits'] for g in recent]),
            'blocked_mean': safe_mean([g['blocked'] for g in recent]),
            
            # Discipline
            'pim_mean': safe_mean([g['pim'] for g in recent]),
            'giveaways_mean': safe_mean([g['giveaways'] for g in recent]),
            'takeaways_mean': safe_mean([g['takeaways'] for g in recent]),
            
            # Special teams
            'pp_pct': safe_mean([g['pp_goals']/max(g['pp_opps'],1) for g in recent]),
            
            # Win rate
            'win_rate': np.mean(wins),
            'last_5_wins': sum(wins[-5:]) if len(wins) >= 5 else sum(wins),
            'home_games': [g for g in recent if g['is_home']],
            'away_games': [g for g in recent if not g['is_home']],
            'games': len(recent),
        }
    
    def calculate_behavioral_features(self, home_stats, away_stats, home_hist, away_hist, date):
        """Calculate NHL behavioral proxy features."""
        features = {}
        
        # === FATIGUE (4) ===
        def calc_fatigue(history, d):
            if len(history) < 2:
                return {'b2b': 0, 'g7d': 0.5, 'rest': 0.5}
            last = history[-1]['date']
            rest = (d - last).days if d > last else 1
            week_ago = d - timedelta(days=7)
            g7d = sum(1 for g in history if g['date'] >= week_ago and g['date'] < d)
            return {
                'b2b': 1.0 if rest <= 1 else 0.0,
                'g7d': min(g7d / 4.0, 1.0),
                'rest': min(rest / 7.0, 1.0),
            }
        
        home_fat = calc_fatigue(home_hist, date)
        away_fat = calc_fatigue(away_hist, date)
        
        features['fatigue_b2b_diff'] = away_fat['b2b'] - home_fat['b2b']
        features['fatigue_g7d_diff'] = away_fat['g7d'] - home_fat['g7d']
        features['fatigue_rest_diff'] = home_fat['rest'] - away_fat['rest']
        
        # === PHYSICALITY (3) - Unique to hockey ===
        features['hits_diff'] = (home_stats['hits_mean'] - away_stats['hits_mean']) / 30
        features['blocked_diff'] = (home_stats['blocked_mean'] - away_stats['blocked_mean']) / 20
        features['physical_index'] = features['hits_diff'] + features['blocked_diff']
        
        # === DISCIPLINE (3) - PIM, giveaways ===
        features['pim_diff'] = (away_stats['pim_mean'] - home_stats['pim_mean']) / 15  # Inverted - more PIM = bad
        features['giveaway_diff'] = (away_stats['giveaways_mean'] - home_stats['giveaways_mean']) / 15
        features['takeaway_diff'] = (home_stats['takeaways_mean'] - away_stats['takeaways_mean']) / 10
        
        # === POSSESSION (3) ===
        features['shots_diff'] = (home_stats['shots_mean'] - away_stats['shots_mean']) / 35
        features['faceoff_diff'] = (home_stats['faceoff_pct'] - away_stats['faceoff_pct']) / 100
        features['possession_index'] = features['shots_diff'] + features['faceoff_diff']
        
        # === SPECIAL TEAMS (2) ===
        features['pp_diff'] = home_stats['pp_pct'] - away_stats['pp_pct']
        
        # === SCORING (4) ===
        features['goals_diff'] = (home_stats['goals_mean'] - away_stats['goals_mean']) / 4
        features['goals_against_diff'] = (away_stats['goals_against_mean'] - home_stats['goals_against_mean']) / 4
        features['goal_variance'] = home_stats['goals_std'] / 3 - away_stats['goals_std'] / 3
        
        # === WIN RATE (3) ===
        features['win_rate_diff'] = home_stats['win_rate'] - away_stats['win_rate']
        features['momentum_diff'] = (home_stats['last_5_wins'] - away_stats['last_5_wins']) / 5
        
        home_home_rate = np.mean([1 if g['won'] else 0 for g in home_stats['home_games']]) if home_stats['home_games'] else 0.5
        away_away_rate = np.mean([1 if g['won'] else 0 for g in away_stats['away_games']]) if away_stats['away_games'] else 0.5
        features['home_win_rate'] = home_home_rate
        features['away_road_rate'] = away_away_rate
        
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
            
            f = self.calculate_behavioral_features(home_stats, away_stats, home_hist, away_hist, date)
            features.append(f)
            valid_idx.append(idx)
            
            # Targets
            home_goals = row.get('goals_home', 0) or 0
            away_goals = row.get('goals_away', 0) or 0
            targets['moneyline'].append(1 if home_goals > away_goals else 0)
            
            # Spread
            margin = home_goals - away_goals
            predicted_spread = -(home_stats['goals_mean'] - away_stats['goals_mean']) * 0.5
            targets['spread'].append(1 if margin > predicted_spread else 0)
            
            # Total
            total = home_goals + away_goals
            predicted_total = home_stats['goals_mean'] + away_stats['goals_mean']
            targets['total'].append(1 if total > predicted_total else 0)
            
            if len(features) % 5000 == 0:
                print(f"    Processed {len(features)} games...")
        
        print(f"  Created {len(features)} feature rows")
        return pd.DataFrame(features), targets, valid_idx


class V6NHLModel:
    """V6 NHL model with behavioral proxy features."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.metrics = {}
        self.feature_names = None
    
    def train_bet_style(self, X, y, bet_type='moneyline'):
        """Train for a specific bet type."""
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
        
        # Calibration
        cal = IsotonicRegression(out_of_bounds='clip')
        cal.fit(ensemble_p, y_test)
        self.calibrators[bet_type] = cal
        
        pred = (ensemble_p >= 0.5).astype(int)
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, ensemble_p)
        
        self.models[bet_type] = {'xgb': xgb_model, 'lgb': lgb_model}
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
        """Save the model."""
        path = MODELS_DIR / "v6_nhl_complete.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'calibrators': self.calibrators,
                'metrics': self.metrics,
                'feature_names': self.feature_names,
            }, f)
        
        # Also save metrics
        metrics_path = MODELS_DIR / "v6_nhl_metrics.json"
        json_metrics = {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv 
                           for kk, vv in v.items()} 
                       for k, v in self.metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"\n  Model saved to: {path}")


def train_v6_nhl():
    """Train V6 NHL model."""
    print("\n" + "="*60)
    print("V6 NHL BEHAVIORAL PROXY MODEL")
    print("="*60)
    
    engine = NHLBehavioralEngine()
    
    # Load data
    df = engine.load_data()
    
    # Build histories
    histories = engine.build_histories(df)
    
    # Create features
    X, targets, valid_idx = engine.create_all_features(df, histories)
    
    print(f"\n  Features: {X.shape}")
    print(f"  Columns: {list(X.columns)}")
    
    # Train
    model = V6NHLModel()
    model.train_all(X, targets)
    
    # Summary
    print("\n" + "="*60)
    print("V6 NHL RESULTS")
    print("="*60)
    for bt, m in model.metrics.items():
        print(f"  {bt.upper():12} | Accuracy: {m['accuracy']:.1%} | AUC: {m['auc']:.4f}")
    
    # Save
    model.save()
    
    return model


if __name__ == "__main__":
    model = train_v6_nhl()
