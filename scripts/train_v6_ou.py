"""
V6-OU Specialized Over/Under Model
===================================
Optimized for predicting TOTAL POINTS, not winner.

Key Features:
- Pace/tempo indicators (possessions per game)
- Combined offensive efficiency
- Combined defensive efficiency  
- Recent O/U trends
- Scoring variance indicators
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
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "nba"
MODELS_DIR = BASE_DIR / "models"


class OverUnderFeatureEngine:
    """
    Specialized feature engineering for Over/Under predictions.
    Focus on TOTAL POINTS, not winner.
    """
    
    def build_team_history(self, df):
        """Build game history with focus on scoring data."""
        print("  Building team histories for O/U...")
        
        df = df.sort_values('GAME_DATE_EST').reset_index(drop=True)
        
        teams = set(df['HOME_TEAM_ID'].unique()) | set(df['VISITOR_TEAM_ID'].unique())
        histories = {t: [] for t in teams}
        
        for _, row in df.iterrows():
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            date = row['GAME_DATE_EST']
            
            home_pts = row.get('PTS_home', 0) or 0
            away_pts = row.get('PTS_away', 0) or 0
            total_pts = home_pts + away_pts
            
            # Home team record
            home_game = {
                'date': date,
                'is_home': True,
                'pts_scored': home_pts,
                'pts_allowed': away_pts,
                'total_pts': total_pts,
                'fg_pct': row.get('FG_PCT_home', 0.45) or 0.45,
                'fg3_pct': row.get('FG3_PCT_home', 0.35) or 0.35,
                'reb': row.get('REB_home', 40) or 40,
                'ast': row.get('AST_home', 24) or 24,
                'tov': row.get('TOV_home', 14) or 14,
                'pf': row.get('PF_home', 20) or 20,
            }
            
            # Away team record
            away_game = {
                'date': date,
                'is_home': False,
                'pts_scored': away_pts,
                'pts_allowed': home_pts,
                'total_pts': total_pts,
                'fg_pct': row.get('FG_PCT_away', 0.45) or 0.45,
                'fg3_pct': row.get('FG3_PCT_away', 0.35) or 0.35,
                'reb': row.get('REB_away', 40) or 40,
                'ast': row.get('AST_away', 24) or 24,
                'tov': row.get('TOV_away', 14) or 14,
                'pf': row.get('PF_away', 20) or 20,
            }
            
            histories[home_id].append(home_game)
            histories[away_id].append(away_game)
        
        return histories
    
    def get_ou_stats(self, history, n=20):
        """Get O/U focused stats from last n games."""
        if len(history) < 3:
            return None
        
        recent = history[-n:] if len(history) >= n else history
        
        def safe_mean(vals):
            valid = [v for v in vals if v and v > 0]
            return np.mean(valid) if valid else 0
        
        def safe_std(vals):
            valid = [v for v in vals if v and v > 0]
            return np.std(valid) if len(valid) > 1 else 0
        
        pts_scored = [g['pts_scored'] for g in recent if g['pts_scored'] > 0]
        pts_allowed = [g['pts_allowed'] for g in recent if g['pts_allowed'] > 0]
        totals = [g['total_pts'] for g in recent if g['total_pts'] > 0]
        
        # O/U trends - how often does this team go over?
        avg_total = safe_mean(totals)
        overs = sum(1 for t in totals if t > avg_total) / len(totals) if totals else 0.5
        
        return {
            # Scoring
            'pts_scored_mean': safe_mean(pts_scored),
            'pts_scored_std': safe_std(pts_scored),
            'pts_allowed_mean': safe_mean(pts_allowed),
            'pts_allowed_std': safe_std(pts_allowed),
            
            # Totals
            'total_pts_mean': safe_mean(totals),
            'total_pts_std': safe_std(totals),
            
            # Pace indicators
            'fg_pct': safe_mean([g['fg_pct'] for g in recent]),
            'fg3_pct': safe_mean([g['fg3_pct'] for g in recent]),
            'ast': safe_mean([g['ast'] for g in recent]),
            'tov': safe_mean([g['tov'] for g in recent]),
            'reb': safe_mean([g['reb'] for g in recent]),
            'pf': safe_mean([g['pf'] for g in recent]),
            
            # O/U tendency
            'over_rate': overs,
            
            # Recent form (last 5)
            'recent_pts_scored': safe_mean([g['pts_scored'] for g in recent[-5:]]),
            'recent_pts_allowed': safe_mean([g['pts_allowed'] for g in recent[-5:]]),
            'recent_total': safe_mean([g['total_pts'] for g in recent[-5:]]),
            
            # Trend (recent 5 vs overall)
            'scoring_trend': safe_mean([g['pts_scored'] for g in recent[-5:]]) - safe_mean(pts_scored),
            
            'games_played': len(recent),
        }
    
    def calculate_ou_features(self, home_stats, away_stats, game_date):
        """Calculate O/U specific features."""
        features = {}
        
        # === PACE/TEMPO FEATURES (8) ===
        # Combined pace indicator
        features['combined_pts_mean'] = (home_stats['pts_scored_mean'] + away_stats['pts_scored_mean']) / 220
        features['combined_pts_allowed'] = (home_stats['pts_allowed_mean'] + away_stats['pts_allowed_mean']) / 220
        features['expected_total'] = (home_stats['pts_scored_mean'] + away_stats['pts_allowed_mean'] + 
                                      away_stats['pts_scored_mean'] + home_stats['pts_allowed_mean']) / 2 / 220
        
        # Offensive efficiency
        features['combined_fg_pct'] = (home_stats['fg_pct'] + away_stats['fg_pct']) / 2
        features['combined_fg3_pct'] = (home_stats['fg3_pct'] + away_stats['fg3_pct']) / 2
        
        # Ball movement/pace
        features['combined_ast'] = (home_stats['ast'] + away_stats['ast']) / 50
        features['combined_tov'] = (home_stats['tov'] + away_stats['tov']) / 30
        
        # Fouling (more fouls = more FTs = more points)
        features['combined_pf'] = (home_stats['pf'] + away_stats['pf']) / 40
        
        # === VARIANCE FEATURES (4) ===
        features['home_pts_variance'] = home_stats['pts_scored_std'] / 15
        features['away_pts_variance'] = away_stats['pts_scored_std'] / 15
        features['combined_variance'] = (home_stats['total_pts_std'] + away_stats['total_pts_std']) / 20
        features['scoring_consistency'] = 1 - features['combined_variance']
        
        # === O/U TENDENCY FEATURES (4) ===
        features['home_over_rate'] = home_stats['over_rate']
        features['away_over_rate'] = away_stats['over_rate']
        features['combined_over_rate'] = (home_stats['over_rate'] + away_stats['over_rate']) / 2
        features['over_tendency'] = features['combined_over_rate'] - 0.5
        
        # === RECENT FORM FEATURES (6) ===
        features['home_recent_pts'] = home_stats['recent_pts_scored'] / 120
        features['away_recent_pts'] = away_stats['recent_pts_scored'] / 120
        features['home_recent_allowed'] = home_stats['recent_pts_allowed'] / 120
        features['away_recent_allowed'] = away_stats['recent_pts_allowed'] / 120
        features['combined_recent_total'] = (home_stats['recent_total'] + away_stats['recent_total']) / 2 / 220
        
        # Trend
        features['home_scoring_trend'] = home_stats['scoring_trend'] / 10
        features['away_scoring_trend'] = away_stats['scoring_trend'] / 10
        features['combined_trend'] = (features['home_scoring_trend'] + features['away_scoring_trend']) / 2
        
        # === HISTORICAL AVERAGE FEATURES (4) ===
        features['home_avg_total'] = home_stats['total_pts_mean'] / 220
        features['away_avg_total'] = away_stats['total_pts_mean'] / 220
        features['historical_expected'] = (home_stats['total_pts_mean'] + away_stats['total_pts_mean']) / 2 / 220
        
        # Defensive quality affects points
        features['defensive_quality'] = 1 - (home_stats['pts_allowed_mean'] + away_stats['pts_allowed_mean']) / 240
        
        return features
    
    def create_features(self, df, histories):
        """Create O/U features for all games."""
        print("  Creating O/U specialized features...")
        
        feature_rows = []
        targets = []
        target_lines = []
        valid_idx = []
        
        for idx, row in df.iterrows():
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            date = row['GAME_DATE_EST']
            
            home_hist = [g for g in histories.get(home_id, []) if g['date'] < date]
            away_hist = [g for g in histories.get(away_id, []) if g['date'] < date]
            
            if len(home_hist) < 5 or len(away_hist) < 5:
                continue
            
            home_stats = self.get_ou_stats(home_hist, 20)
            away_stats = self.get_ou_stats(away_hist, 20)
            
            if home_stats is None or away_stats is None:
                continue
            
            features = self.calculate_ou_features(home_stats, away_stats, date)
            
            # Calculate target - did game go OVER predicted total?
            home_pts = row.get('PTS_home', 0) or 0
            away_pts = row.get('PTS_away', 0) or 0
            actual_total = home_pts + away_pts
            
            # Predicted total from historical averages
            predicted_total = (home_stats['pts_scored_mean'] + away_stats['pts_allowed_mean'] +
                              away_stats['pts_scored_mean'] + home_stats['pts_allowed_mean']) / 2
            
            feature_rows.append(features)
            targets.append(1 if actual_total > predicted_total else 0)
            target_lines.append(actual_total)
            valid_idx.append(idx)
            
            if len(feature_rows) % 2000 == 0:
                print(f"    Processed {len(feature_rows)} games...")
        
        print(f"  Created {len(feature_rows)} O/U feature rows")
        return pd.DataFrame(feature_rows), targets, target_lines, valid_idx


class V6OverUnderModel:
    """Specialized O/U model with pace/tempo features."""
    
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = None
        self.calibrator = None
        self.feature_names = None
        self.metrics = {}
    
    def train(self, X, y):
        """Train the O/U model."""
        print("\n  Training V6-OU Specialized Model...")
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = list(X.columns)
        
        # Time-based split
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = np.array(y[:split]), np.array(y[split:])
        
        print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
        
        # XGBoost
        print("    Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
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
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # LightGBM
        print("    Training LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(
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
        self.lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        
        # Ensemble
        xgb_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X_test)[:, 1]
        ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
        
        # Calibration
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(ensemble_proba, y_test)
        calibrated = self.calibrator.predict(ensemble_proba)
        
        # Metrics
        pred = (ensemble_proba >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, ensemble_proba)
        brier = brier_score_loss(y_test, calibrated)
        
        xgb_acc = accuracy_score(y_test, (xgb_proba >= 0.5).astype(int))
        lgb_acc = accuracy_score(y_test, (lgb_proba >= 0.5).astype(int))
        
        self.metrics = {
            'accuracy': accuracy,
            'xgb_accuracy': xgb_acc,
            'lgb_accuracy': lgb_acc,
            'auc': auc,
            'brier': brier,
            'test_size': len(y_test)
        }
        
        print(f"\n  === V6-OU RESULTS ===")
        print(f"  XGBoost Accuracy:  {xgb_acc:.1%}")
        print(f"  LightGBM Accuracy: {lgb_acc:.1%}")
        print(f"  Ensemble Accuracy: {accuracy:.1%}")
        print(f"  AUC-ROC:           {auc:.4f}")
        print(f"  Brier Score:       {brier:.4f}")
        
        return accuracy
    
    def get_feature_importance(self):
        """Get combined feature importance."""
        xgb_imp = self.xgb_model.feature_importances_
        lgb_imp = self.lgb_model.feature_importances_
        combined = (xgb_imp + lgb_imp) / 2
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': combined
        }).sort_values('importance', ascending=False)
    
    def save(self, path=None):
        """Save the model."""
        if path is None:
            path = MODELS_DIR / "v6_ou_specialized.pkl"
        
        with open(path, 'wb') as f:
            pickle.dump({
                'xgb': self.xgb_model,
                'lgb': self.lgb_model,
                'scaler': self.scaler,
                'calibrator': self.calibrator,
                'feature_names': self.feature_names,
                'metrics': self.metrics
            }, f)
        
        print(f"\n  Model saved to: {path}")


def load_training_data():
    """Load enhanced NBA data."""
    path = DATA_DIR / "nba_new" / "nba_training_games_20251205.csv"
    
    if not path.exists():
        print(f"Error: {path} not found!")
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
        'REB_home': paired['REB_home'],
        'REB_away': paired['REB_away'],
        'AST_home': paired['AST_home'],
        'AST_away': paired['AST_away'],
        'TOV_home': paired['TOV_home'],
        'TOV_away': paired['TOV_away'],
        'PF_home': paired['PF_home'],
        'PF_away': paired['PF_away'],
    })
    
    return games.sort_values('GAME_DATE_EST').reset_index(drop=True)


def train_v6_ou():
    """Train the specialized O/U model."""
    print("\n" + "="*60)
    print("V6-OU SPECIALIZED OVER/UNDER MODEL")
    print("="*60)
    
    # Load data
    print("\nLoading training data...")
    games = load_training_data()
    
    if games is None:
        return None
    
    print(f"  Loaded {len(games)} games")
    
    # Build features
    engine = OverUnderFeatureEngine()
    histories = engine.build_team_history(games)
    X, targets, target_lines, valid_idx = engine.create_features(games, histories)
    
    print(f"\n  Feature matrix: {X.shape}")
    print(f"  Features: {list(X.columns)}")
    
    # Train
    model = V6OverUnderModel()
    accuracy = model.train(X, targets)
    
    # Feature importance
    print("\n  Top 15 Most Important Features:")
    importance = model.get_feature_importance()
    for _, row in importance.head(15).iterrows():
        print(f"    {row['feature']}: {row['importance']:.2f}")
    
    # Save
    model.save()
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"  V6 Behavioral O/U:  55.0%")
    print(f"  V6-OU Specialized:  {accuracy:.1%}")
    improvement = (accuracy - 0.55) * 100
    if improvement > 0:
        print(f"  IMPROVEMENT:        +{improvement:.1f}pp 🎉")
    
    return model


if __name__ == "__main__":
    model = train_v6_ou()
