"""
V6 Complete NBA Model - All Bet Styles
=======================================
Trains V6 behavioral proxy model for:
- Moneyline (winner prediction)
- Spread (point spread coverage)
- Over/Under (total points)
- Contracts (prediction market probability)
- Parlay (combined leg confidence)

Uses XGBoost + LightGBM ensemble with behavioral proxy features.
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
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score, mean_absolute_error
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "nba"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


class BehavioralProxyEngine:
    """Feature engineering for behavioral proxy model."""
    
    def build_team_history(self, df):
        """Build game-by-game history for each team."""
        print("  Building team histories...")
        
        df = df.sort_values('GAME_DATE_EST').reset_index(drop=True)
        
        teams = set(df['HOME_TEAM_ID'].unique()) | set(df['VISITOR_TEAM_ID'].unique())
        histories = {t: [] for t in teams}
        
        for _, row in df.iterrows():
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            date = row['GAME_DATE_EST']
            
            home_game = {
                'date': date,
                'is_home': True,
                'won': row['HOME_TEAM_WINS'] == 1,
                'pts': row.get('PTS_home', 0) or 0,
                'opp_pts': row.get('PTS_away', 0) or 0,
                'fg_pct': row.get('FG_PCT_home', 0.45) or 0.45,
                'fg3_pct': row.get('FG3_PCT_home', 0.35) or 0.35,
                'ft_pct': row.get('FT_PCT_home', 0.75) or 0.75,
                'reb': row.get('REB_home', 40) or 40,
                'ast': row.get('AST_home', 24) or 24,
                'stl': row.get('STL_home', 7) or 7,
                'blk': row.get('BLK_home', 5) or 5,
                'tov': row.get('TOV_home', 14) or 14,
                'pf': row.get('PF_home', 20) or 20,
            }
            
            away_game = {
                'date': date,
                'is_home': False,
                'won': row['HOME_TEAM_WINS'] == 0,
                'pts': row.get('PTS_away', 0) or 0,
                'opp_pts': row.get('PTS_home', 0) or 0,
                'fg_pct': row.get('FG_PCT_away', 0.45) or 0.45,
                'fg3_pct': row.get('FG3_PCT_away', 0.35) or 0.35,
                'ft_pct': row.get('FT_PCT_away', 0.75) or 0.75,
                'reb': row.get('REB_away', 40) or 40,
                'ast': row.get('AST_away', 24) or 24,
                'stl': row.get('STL_away', 7) or 7,
                'blk': row.get('BLK_away', 5) or 5,
                'tov': row.get('TOV_away', 14) or 14,
                'pf': row.get('PF_away', 20) or 20,
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
            valid = [v for v in vals if v and v > 0]
            return np.mean(valid) if valid else 0
        
        def safe_std(vals):
            valid = [v for v in vals if v and v > 0]
            return np.std(valid) if len(valid) > 1 else 0
        
        wins = [1 if g['won'] else 0 for g in recent]
        
        return {
            'pts_mean': safe_mean([g['pts'] for g in recent]),
            'pts_std': safe_std([g['pts'] for g in recent]),
            'opp_pts_mean': safe_mean([g['opp_pts'] for g in recent]),
            'opp_pts_std': safe_std([g['opp_pts'] for g in recent]),
            'fg_pct': safe_mean([g['fg_pct'] for g in recent]),
            'fg3_pct': safe_mean([g['fg3_pct'] for g in recent]),
            'ft_pct': safe_mean([g['ft_pct'] for g in recent]),
            'reb': safe_mean([g['reb'] for g in recent]),
            'ast': safe_mean([g['ast'] for g in recent]),
            'stl': safe_mean([g['stl'] for g in recent]),
            'blk': safe_mean([g['blk'] for g in recent]),
            'blk_std': safe_std([g['blk'] for g in recent]),
            'tov': safe_mean([g['tov'] for g in recent]),
            'pf': safe_mean([g['pf'] for g in recent]),
            'win_rate': np.mean(wins),
            'last_5_wins': sum(wins[-5:]) if len(wins) >= 5 else sum(wins),
            'first_5_wins': sum(wins[:5]) if len(wins) >= 5 else sum(wins),
            'home_games': [g for g in recent if g['is_home']],
            'away_games': [g for g in recent if not g['is_home']],
            'games_played': len(recent),
        }
    
    def calculate_behavioral_features(self, home_stats, away_stats, home_history, away_history, game_date):
        """Calculate all behavioral proxy features."""
        
        features = {}
        
        # === FATIGUE PROXIES (5) ===
        def calc_fatigue(history, date):
            if len(history) < 2:
                return {'b2b': 0, 'g7d': 0.5, 'rest': 0.5, 'load': 0.5, 'away': 0.5}
            
            last = history[-1]['date']
            rest = (date - last).days if date > last else 1
            
            week_ago = date - timedelta(days=7)
            g7d = sum(1 for g in history if g['date'] >= week_ago and g['date'] < date)
            
            recent = history[-10:] if len(history) >= 10 else history
            away_load = sum(1 for g in recent if not g['is_home']) / len(recent)
            
            return {
                'b2b': 1.0 if rest <= 1 else 0.0,
                'g7d': min(g7d / 4.0, 1.0),
                'rest': min(rest / 7.0, 1.0),
                'load': min(len(recent) / 8.0, 1.0),
                'away': away_load,
            }
        
        home_fat = calc_fatigue(home_history, game_date)
        away_fat = calc_fatigue(away_history, game_date)
        
        features['fatigue_b2b_diff'] = away_fat['b2b'] - home_fat['b2b']
        features['fatigue_g7d_diff'] = away_fat['g7d'] - home_fat['g7d']
        features['fatigue_rest_diff'] = home_fat['rest'] - away_fat['rest']
        features['fatigue_load_diff'] = away_fat['load'] - home_fat['load']
        features['fatigue_away_diff'] = away_fat['away'] - home_fat['away']
        
        # === DEFENSIVE DISCIPLINE (4) ===
        features['def_pf_diff'] = (1 - home_stats['pf']/25) - (1 - away_stats['pf']/25)
        features['def_stl_pf_diff'] = (home_stats['stl']/max(home_stats['pf'],1)) - (away_stats['stl']/max(away_stats['pf'],1))
        features['def_blk_consistency'] = (1 - home_stats['blk_std']/5) - (1 - away_stats['blk_std']/5)
        features['def_variance_diff'] = (1 - home_stats['opp_pts_std']/20) - (1 - away_stats['opp_pts_std']/20)
        
        # === CLUTCH/PRESSURE (4) ===
        features['clutch_ft_diff'] = home_stats['ft_pct'] - away_stats['ft_pct']
        features['clutch_streak_diff'] = (home_stats['last_5_wins'] - away_stats['last_5_wins']) / 5
        home_momentum = (home_stats['last_5_wins'] - home_stats['first_5_wins']) / 5 + 0.5
        away_momentum = (away_stats['last_5_wins'] - away_stats['first_5_wins']) / 5 + 0.5
        features['clutch_momentum_diff'] = home_momentum - away_momentum
        features['clutch_variance_diff'] = home_stats['pts_std']/15 - away_stats['pts_std']/15
        
        # === SPACING/FLOW (4) ===
        home_ast_rate = home_stats['ast'] / max(home_stats['pts_mean']/2.2, 1)
        away_ast_rate = away_stats['ast'] / max(away_stats['pts_mean']/2.2, 1)
        features['flow_ast_diff'] = home_ast_rate - away_ast_rate
        features['flow_3pt_diff'] = home_stats['fg3_pct'] - away_stats['fg3_pct']
        features['flow_tov_diff'] = (1 - away_stats['tov']/20) - (1 - home_stats['tov']/20)
        features['flow_fg_diff'] = home_stats['fg_pct'] - away_stats['fg_pct']
        
        # === CHEMISTRY (3) ===
        features['chem_unselfish_diff'] = home_ast_rate - away_ast_rate
        features['chem_stability_diff'] = (1 - (home_stats['pts_std']+home_stats['opp_pts_std'])/30) - (1 - (away_stats['pts_std']+away_stats['opp_pts_std'])/30)
        features['chem_exp_diff'] = (home_stats['games_played'] - away_stats['games_played']) / 30
        
        # === BASE FEATURES ===
        features['win_rate_diff'] = home_stats['win_rate'] - away_stats['win_rate']
        features['pts_diff'] = (home_stats['pts_mean'] - away_stats['pts_mean']) / 10
        features['home_win_rate'] = np.mean([1 if g['won'] else 0 for g in home_stats['home_games']]) if home_stats['home_games'] else 0.5
        features['away_road_rate'] = np.mean([1 if g['won'] else 0 for g in away_stats['away_games']]) if away_stats['away_games'] else 0.5
        
        # === EXTRA STATS ===
        features['stl_diff'] = (home_stats['stl'] - away_stats['stl']) / 5
        features['blk_diff'] = (home_stats['blk'] - away_stats['blk']) / 3
        features['tov_diff'] = (away_stats['tov'] - home_stats['tov']) / 5
        features['reb_diff'] = (home_stats['reb'] - away_stats['reb']) / 10
        
        # === TOTAL POINTS FEATURES (for O/U) ===
        features['combined_pts_mean'] = (home_stats['pts_mean'] + away_stats['pts_mean']) / 220
        features['combined_pts_std'] = (home_stats['pts_std'] + away_stats['pts_std']) / 20
        features['pace_indicator'] = (home_stats['pts_mean'] + away_stats['opp_pts_mean']) / 220
        
        return features
    
    def create_all_features(self, df, histories):
        """Create features for all games."""
        print("  Creating behavioral features...")
        
        feature_rows = []
        targets = {'moneyline': [], 'spread': [], 'total': [], 'total_line': []}
        valid_idx = []
        
        for idx, row in df.iterrows():
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            date = row['GAME_DATE_EST']
            
            home_hist = [g for g in histories.get(home_id, []) if g['date'] < date]
            away_hist = [g for g in histories.get(away_id, []) if g['date'] < date]
            
            if len(home_hist) < 5 or len(away_hist) < 5:
                continue
            
            home_stats = self.get_team_stats(home_hist, 20)
            away_stats = self.get_team_stats(away_hist, 20)
            
            if home_stats is None or away_stats is None:
                continue
            
            features = self.calculate_behavioral_features(
                home_stats, away_stats, home_hist, away_hist, date
            )
            
            feature_rows.append(features)
            valid_idx.append(idx)
            
            # Targets
            targets['moneyline'].append(row['HOME_TEAM_WINS'])
            
            # Spread: Did home cover? Assume typical spread = -pts_diff * 0.8
            home_pts = row.get('PTS_home', 0) or 0
            away_pts = row.get('PTS_away', 0) or 0
            margin = home_pts - away_pts
            predicted_spread = -(home_stats['pts_mean'] - away_stats['pts_mean']) * 0.8
            targets['spread'].append(1 if margin > predicted_spread else 0)
            
            # Total
            total_pts = home_pts + away_pts
            predicted_total = home_stats['pts_mean'] + away_stats['pts_mean']
            targets['total'].append(1 if total_pts > predicted_total else 0)
            targets['total_line'].append(total_pts)
            
            if len(feature_rows) % 2000 == 0:
                print(f"    Processed {len(feature_rows)} games...")
        
        print(f"  Created {len(feature_rows)} feature rows")
        return pd.DataFrame(feature_rows), targets, valid_idx


class V6NBAModel:
    """
    Complete V6 model for all bet styles.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.metrics = {}
        self.feature_names = None
    
    def train_bet_style(self, X, y, bet_type='moneyline'):
        """Train model for a specific bet type."""
        print(f"\n  Training {bet_type.upper()} model...")
        
        # Scale
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[bet_type] = scaler
        self.feature_names = list(X.columns)
        
        # Split (80/20 time-based)
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = np.array(y[:split]), np.array(y[split:])
        
        print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.01,
            reg_lambda=5.0,
            reg_alpha=1.0,
            subsample=0.8,
            colsample_bytree=0.7,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.01,
            reg_lambda=5.0,
            reg_alpha=1.0,
            subsample=0.8,
            colsample_bytree=0.7,
            objective='binary',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        
        # Ensemble
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
        ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
        
        # Calibration (Isotonic)
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(ensemble_proba, y_test)
        calibrated_proba = calibrator.predict(ensemble_proba)
        self.calibrators[bet_type] = calibrator
        
        # Metrics
        pred = (ensemble_proba >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, ensemble_proba)
        brier = brier_score_loss(y_test, calibrated_proba)
        
        self.models[bet_type] = {'xgb': xgb_model, 'lgb': lgb_model}
        self.metrics[bet_type] = {
            'accuracy': accuracy,
            'auc': auc,
            'brier': brier,
            'test_size': len(y_test)
        }
        
        print(f"    Accuracy: {accuracy:.1%}")
        print(f"    AUC: {auc:.4f}")
        print(f"    Brier: {brier:.4f}")
        
        return accuracy
    
    def train_all(self, X, targets):
        """Train models for all bet styles."""
        print("\n" + "="*60)
        print("TRAINING V6 MODELS FOR ALL BET STYLES")
        print("="*60)
        
        results = {}
        
        # Moneyline
        results['moneyline'] = self.train_bet_style(X, targets['moneyline'], 'moneyline')
        
        # Spread
        results['spread'] = self.train_bet_style(X, targets['spread'], 'spread')
        
        # Total (O/U)
        results['total'] = self.train_bet_style(X, targets['total'], 'total')
        
        # Contracts (same as moneyline but calibrated for market prices)
        results['contracts'] = self.train_bet_style(X, targets['moneyline'], 'contracts')
        
        return results
    
    def predict(self, X, bet_type='moneyline'):
        """Make prediction for a bet type."""
        if bet_type not in self.models:
            return None
        
        X_scaled = self.scalers[bet_type].transform(X)
        
        xgb_proba = self.models[bet_type]['xgb'].predict_proba(X_scaled)[:, 1]
        lgb_proba = self.models[bet_type]['lgb'].predict_proba(X_scaled)[:, 1]
        ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
        
        # Calibrate
        calibrated = self.calibrators[bet_type].predict(ensemble_proba)
        
        return calibrated
    
    def calculate_parlay_confidence(self, leg_probabilities):
        """Calculate parlay confidence from individual leg probabilities."""
        # Combined probability (all must hit)
        combined = np.prod(leg_probabilities)
        
        # Adjust for correlation (legs can be correlated)
        # Use slight boost for 2-3 leg parlays, reduce for 4+
        n_legs = len(leg_probabilities)
        if n_legs <= 2:
            adjustment = 1.05
        elif n_legs == 3:
            adjustment = 1.0
        else:
            adjustment = 0.95
        
        adjusted = min(combined * adjustment, 0.99)
        return adjusted
    
    def save(self, path=None):
        """Save the trained model."""
        if path is None:
            path = MODELS_DIR / "v6_nba_complete.pkl"
        
        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'calibrators': self.calibrators,
            'metrics': self.metrics,
            'feature_names': self.feature_names,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\n  Model saved to: {path}")
        
        # Also save metrics as JSON
        metrics_path = MODELS_DIR / "v6_nba_metrics.json"
        json_metrics = {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv 
                           for kk, vv in v.items()} 
                       for k, v in self.metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"  Metrics saved to: {metrics_path}")
    
    @classmethod
    def load(cls, path=None):
        """Load a trained model."""
        if path is None:
            path = MODELS_DIR / "v6_nba_complete.pkl"
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.models = data['models']
        model.scalers = data['scalers']
        model.calibrators = data['calibrators']
        model.metrics = data['metrics']
        model.feature_names = data['feature_names']
        
        return model


def load_training_data():
    """Load the enhanced training data."""
    path = DATA_DIR / "nba_new" / "nba_training_games_20251205.csv"
    
    if not path.exists():
        print(f"Error: {path} not found")
        return None
    
    df = pd.read_csv(path)
    df['is_home'] = df['MATCHUP'].str.contains(' vs. ')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')
    
    home = df[df['is_home']].add_suffix('_home')
    away = df[~df['is_home']].add_suffix('_away')
    
    paired = home.merge(away, left_on='Game_ID_home', right_on='Game_ID_away', how='inner')
    
    games = pd.DataFrame({
        'GAME_DATE_EST': paired['GAME_DATE_home'],
        'HOME_TEAM_ID': paired['Team_ID_home'],
        'VISITOR_TEAM_ID': paired['Team_ID_away'],
        'PTS_home': paired['PTS_home'],
        'PTS_away': paired['PTS_away'],
        'FG_PCT_home': paired['FG_PCT_home'],
        'FG_PCT_away': paired['FG_PCT_away'],
        'FG3_PCT_home': paired.get('FG3_PCT_home', 0.35),
        'FG3_PCT_away': paired.get('FG3_PCT_away', 0.35),
        'FT_PCT_home': paired.get('FT_PCT_home', 0.75),
        'FT_PCT_away': paired.get('FT_PCT_away', 0.75),
        'REB_home': paired['REB_home'],
        'REB_away': paired['REB_away'],
        'AST_home': paired['AST_home'],
        'AST_away': paired['AST_away'],
        'STL_home': paired['STL_home'],
        'STL_away': paired['STL_away'],
        'BLK_home': paired['BLK_home'],
        'BLK_away': paired['BLK_away'],
        'TOV_home': paired['TOV_home'],
        'TOV_away': paired['TOV_away'],
        'PF_home': paired['PF_home'],
        'PF_away': paired['PF_away'],
        'HOME_TEAM_WINS': (paired['WL_home'] == 'W').astype(int),
    })
    
    return games.sort_values('GAME_DATE_EST').reset_index(drop=True)


def train_v6_complete():
    """Train the complete V6 NBA model."""
    print("\n" + "="*60)
    print("V6 COMPLETE NBA MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading enhanced training data...")
    games = load_training_data()
    
    if games is None:
        return None
    
    print(f"  Loaded {len(games)} games")
    print(f"  Date range: {games['GAME_DATE_EST'].min()} to {games['GAME_DATE_EST'].max()}")
    
    # Build features
    engine = BehavioralProxyEngine()
    histories = engine.build_team_history(games)
    X, targets, valid_idx = engine.create_all_features(games, histories)
    
    print(f"\n  Feature matrix: {X.shape}")
    
    # Train
    model = V6NBAModel()
    results = model.train_all(X, targets)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    for bet_type, metrics in model.metrics.items():
        print(f"\n  {bet_type.upper()}:")
        print(f"    Accuracy: {metrics['accuracy']:.1%}")
        print(f"    AUC:      {metrics['auc']:.4f}")
        print(f"    Brier:    {metrics['brier']:.4f}")
    
    # Parlay example
    print("\n  PARLAY CONFIDENCE EXAMPLE:")
    sample_legs = [0.66, 0.64, 0.62]  # 3 moneyline picks
    parlay_conf = model.calculate_parlay_confidence(sample_legs)
    print(f"    3-leg parlay with {[f'{p:.0%}' for p in sample_legs]}")
    print(f"    Raw combined: {np.prod(sample_legs):.1%}")
    print(f"    Adjusted:     {parlay_conf:.1%}")
    
    # Save
    model.save()
    
    return model


if __name__ == "__main__":
    model = train_v6_complete()
