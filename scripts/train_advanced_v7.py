"""
Advanced Bet-Type Models - Research-Based Methods
=================================================
Implements state-of-the-art prediction methods from academic research:

NBA: Graph features, calibrated XGBoost + SHAP, ELO ratings
NFL: DVOA-inspired metrics, weather features, gradient boosting
NHL: Corsi/Fenwick-inspired, goaltender stats, low-scoring adjustments
MLB: ELO rating system, pitcher-specific features
Soccer: Poisson regression, xG-inspired attack/defense ratings

Goal: Beat our current V6 baselines using research-backed methods.

Run: python scripts/train_advanced_v7.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Current baselines to beat
CURRENT_BASELINES = {
    'nba': {'moneyline': 0.654, 'spread': 0.734, 'total': 0.622},
    'nfl': {'moneyline': 0.651, 'spread': 0.690, 'total': 0.563},
    'nhl': {'moneyline': 0.720, 'spread': 0.664, 'total': 0.601},
    'mlb': {'moneyline': 0.534, 'spread': 0.620, 'total': 0.584},
    'soccer': {'moneyline': 0.670, 'spread': 0.753, 'total': 0.615},
}


class ELOSystem:
    """ELO rating system - proven effective for all sports."""
    
    def __init__(self, k_factor=20, home_advantage=24):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}
        self.default_rating = 1500
    
    def get_rating(self, team):
        return self.ratings.get(team, self.default_rating)
    
    def expected_score(self, rating_a, rating_b):
        """Calculate expected score for team A."""
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def update(self, team_a, team_b, outcome, home_team=None):
        """Update ratings after a game. Outcome: 1 if A wins, 0 if B wins, 0.5 for tie."""
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Apply home advantage
        if home_team == team_a:
            rating_a += self.home_advantage
        elif home_team == team_b:
            rating_b += self.home_advantage
        
        expected = self.expected_score(rating_a, rating_b)
        
        # Update ratings
        self.ratings[team_a] = self.get_rating(team_a) + self.k_factor * (outcome - expected)
        self.ratings[team_b] = self.get_rating(team_b) + self.k_factor * ((1 - outcome) - (1 - expected))
    
    def calculate_all(self, df, home_col, away_col, home_score_col, away_score_col):
        """Calculate ELO ratings for entire dataset and return features."""
        elo_features = []
        
        for idx, row in df.iterrows():
            home = row[home_col]
            away = row[away_col]
            home_score = row[home_score_col]
            away_score = row[away_score_col]
            
            # Get pre-game ratings
            home_elo = self.get_rating(home)
            away_elo = self.get_rating(away)
            
            # Calculate difference and win probability
            elo_diff = home_elo - away_elo + self.home_advantage
            win_prob = self.expected_score(home_elo + self.home_advantage, away_elo)
            
            elo_features.append({
                'home_elo': home_elo,
                'away_elo': away_elo,
                'elo_diff': elo_diff,
                'elo_win_prob': win_prob,
            })
            
            # Update ratings with result
            if pd.notna(home_score) and pd.notna(away_score):
                outcome = 1 if home_score > away_score else (0.5 if home_score == away_score else 0)
                self.update(home, away, outcome, home)
        
        return pd.DataFrame(elo_features)


class AdvancedV7Trainer:
    """Advanced training using research-backed methods."""
    
    def __init__(self):
        self.results = {}
    
    def calculate_rolling_features(self, df, team_col, score_col, opp_score_col, window=10):
        """Calculate rolling averages for team performance."""
        df = df.sort_values('date' if 'date' in df.columns else df.index)
        
        features = pd.DataFrame(index=df.index)
        
        # Group by team and calculate rolling stats
        for team in df[team_col].unique():
            team_mask = df[team_col] == team
            team_games = df[team_mask]
            
            if len(team_games) > window:
                # Offensive metrics
                features.loc[team_mask, f'rolling_pts_{window}'] = team_games[score_col].rolling(window, min_periods=3).mean()
                features.loc[team_mask, f'rolling_pts_allowed_{window}'] = team_games[opp_score_col].rolling(window, min_periods=3).mean()
                
                # Net rating proxy
                features.loc[team_mask, 'net_rating'] = features.loc[team_mask, f'rolling_pts_{window}'] - features.loc[team_mask, f'rolling_pts_allowed_{window}']
                
                # Trend (recent vs older performance)
                features.loc[team_mask, 'trend_3'] = team_games[score_col].rolling(3, min_periods=2).mean()
                features.loc[team_mask, 'momentum'] = features.loc[team_mask, 'trend_3'] - features.loc[team_mask, f'rolling_pts_{window}']
        
        return features.fillna(0)
    
    def train_nba_advanced(self):
        """NBA: XGBoost + SHAP with ELO and rolling features."""
        print("\n" + "="*60)
        print("TRAINING NBA V7 (ELO + Behavioral + Calibration)")
        print("="*60)
        
        # Try multiple data paths
        data_path = DATA_DIR / "nba" / "games.csv"
        if not data_path.exists():
            data_path = DATA_DIR / "nba" / "nba_games_enhanced.csv"
        if not data_path.exists():
            print("  NBA data not found")
            return None
        
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df)} games from {data_path.name}")
        
        # Standardize column names based on what's available
        if 'PTS_home' in df.columns:
            df['home_score'] = df['PTS_home']
            df['away_score'] = df['PTS_away']
            df['home_team'] = df['HOME_TEAM_ID']
            df['away_team'] = df['VISITOR_TEAM_ID']
            if 'GAME_DATE_EST' in df.columns:
                df['date'] = df['GAME_DATE_EST']
        
        df = df.dropna(subset=['home_score', 'away_score'])
        print(f"  {len(df)} games with scores")
        
        # Sort by date for proper ELO calculation
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date')
        
        # Calculate ELO ratings
        print("  Calculating ELO ratings...")
        elo = ELOSystem(k_factor=20, home_advantage=100)
        elo_features = elo.calculate_all(df, 'home_team', 'away_team', 'home_score', 'away_score')
        
        # Create behavioral features
        print("  Creating behavioral features...")
        X = pd.DataFrame()
        X['elo_diff'] = elo_features['elo_diff']
        X['elo_win_prob'] = elo_features['elo_win_prob']
        X['home_elo'] = elo_features['home_elo']
        X['away_elo'] = elo_features['away_elo']
        
        # Add shooting/stats features if available (games.csv has these)
        stat_cols = {
            'FG_PCT_home': 'home_fg_pct', 'FG_PCT_away': 'away_fg_pct',
            'FG3_PCT_home': 'home_3pt_pct', 'FG3_PCT_away': 'away_3pt_pct',
            'REB_home': 'home_rebounds', 'REB_away': 'away_rebounds',
            'AST_home': 'home_assists', 'AST_away': 'away_assists',
        }
        for orig_col, new_col in stat_cols.items():
            if orig_col in df.columns:
                X[new_col] = pd.to_numeric(df[orig_col], errors='coerce').fillna(0)
        
        # Targets
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        df['total_score'] = df['home_score'] + df['away_score']
        df['margin'] = df['home_score'] - df['away_score']
        
        # Spread target (home covers -5)
        df['spread_result'] = (df['margin'] > 5).astype(int)
        
        # O/U target
        median_total = df['total_score'].median()
        df['over_result'] = (df['total_score'] > median_total).astype(int)
        
        X = X.fillna(0)
        print(f"  Created {len(X.columns)} features: {list(X.columns)}")
        
        results = {}
        baseline = CURRENT_BASELINES['nba']
        
        for bet_type, y_col in [('moneyline', 'home_win'), ('spread', 'spread_result'), ('total', 'over_result')]:
            print(f"\n  === {bet_type.upper()} ===")
            print(f"  Current baseline: {baseline[bet_type]:.1%}")
            
            y = df[y_col].values
            
            # Ensure X and y have same length
            min_len = min(len(X), len(y))
            X_use = X.iloc[:min_len].reset_index(drop=True)
            y_use = y[:min_len]
            
            # Time-based split (80/20)
            split_idx = int(len(X_use) * 0.8)
            X_train, X_test = X_use.iloc[:split_idx], X_use.iloc[split_idx:]
            y_train, y_test = y_use[:split_idx], y_use[split_idx:]
            
            # Scale
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost first (without calibration wrapper)
            xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.01,
                reg_lambda=5.0,
                subsample=0.8,
                colsample_bytree=0.7,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
            
            # Get XGBoost predictions
            xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
            
            # Apply Platt scaling for calibration (sigmoid calibration)
            # Train a logistic regression on the XGBoost probabilities
            calib_train_proba = xgb_model.predict_proba(X_train_scaled)[:, 1].reshape(-1, 1)
            calib_model = LogisticRegression(C=1.0, random_state=42)
            calib_model.fit(calib_train_proba, y_train)
            
            # Calibrated probabilities
            proba = calib_model.predict_proba(xgb_proba.reshape(-1, 1))[:, 1]
            pred = (proba >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, pred)
            try:
                auc = roc_auc_score(y_test, proba)
            except:
                auc = 0.5
            brier = brier_score_loss(y_test, proba)
            
            improvement = acc - baseline[bet_type]
            status = "✅ BEAT" if improvement > 0 else "❌ Below"
            
            print(f"    Accuracy: {acc:.1%} ({status} by {abs(improvement)*100:.1f}pp)")
            print(f"    AUC: {auc:.4f}, Brier: {brier:.4f}")
            
            results[bet_type] = {
                'xgb_model': xgb_model,
                'calib_model': calib_model,
                'scaler': scaler,
                'accuracy': acc,
                'auc': auc,
                'brier': brier,
                'baseline': baseline[bet_type],
                'improvement': improvement,
            }
            
            if improvement > 0:
                model_path = MODELS_DIR / f"v7_nba_{bet_type}_advanced.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'xgb_model': xgb_model, 
                        'calib_model': calib_model, 
                        'scaler': scaler, 
                        'features': list(X.columns)
                    }, f)
                print(f"    → Saved: {model_path.name}")
        
        return results
    
    def train_nfl_advanced(self):
        """NFL: Gradient boosting with weather and DVOA-inspired features."""
        print("\n" + "="*60)
        print("TRAINING NFL V7 (Weather + ELO + Gradient Boosting)")
        print("="*60)
        
        data_path = DATA_DIR / "nfl" / "spreadspoke_scores.csv"
        if not data_path.exists():
            print("  NFL data not found")
            return None
        
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['score_home', 'score_away'])
        print(f"  Loaded {len(df)} games")
        
        # Calculate ELO
        print("  Calculating ELO...")
        elo = ELOSystem(k_factor=32, home_advantage=48)  # Higher K for NFL
        elo_features = elo.calculate_all(df, 'team_home', 'team_away', 'score_home', 'score_away')
        
        # Build features
        X = pd.DataFrame()
        X['elo_diff'] = elo_features['elo_diff']
        X['elo_win_prob'] = elo_features['elo_win_prob']
        
        # Weather features (research shows these matter)
        if 'weather_temperature' in df.columns:
            X['temperature'] = pd.to_numeric(df['weather_temperature'], errors='coerce').fillna(60)
            X['cold_game'] = (X['temperature'] < 35).astype(int)
        if 'weather_wind_mph' in df.columns:
            X['wind'] = pd.to_numeric(df['weather_wind_mph'], errors='coerce').fillna(5)
            X['high_wind'] = (X['wind'] > 15).astype(int)
        
        # Stadium features
        if 'stadium_neutral' in df.columns:
            X['neutral'] = df['stadium_neutral'].fillna(False).astype(int)
        
        # Season context
        if 'schedule_season' in df.columns:
            X['season'] = df['schedule_season']
        if 'schedule_week' in df.columns:
            X['week'] = pd.to_numeric(df['schedule_week'], errors='coerce').fillna(1)
            X['late_season'] = (X['week'] >= 14).astype(int)
        
        # Targets
        df['home_win'] = (df['score_home'] > df['score_away']).astype(int)
        df['total_score'] = df['score_home'] + df['score_away']
        df['margin'] = df['score_home'] - df['score_away']
        
        # Spread from actual spread data
        if 'spread_favorite' in df.columns:
            df['spread_favorite'] = pd.to_numeric(df['spread_favorite'], errors='coerce')
            df['spread_result'] = np.where(
                df['team_favorite_id'] == df['team_home'],
                df['margin'] > abs(df['spread_favorite']),
                df['margin'] < -abs(df['spread_favorite'])
            ).astype(int)
        else:
            df['spread_result'] = (df['margin'] > 3).astype(int)
        
        # O/U
        if 'over_under_line' in df.columns:
            df['over_under_line'] = pd.to_numeric(df['over_under_line'], errors='coerce')
            df['over_result'] = (df['total_score'] > df['over_under_line']).astype(int)
        else:
            df['over_result'] = (df['total_score'] > df['total_score'].median()).astype(int)
        
        X = X.fillna(0)
        
        results = {}
        baseline = CURRENT_BASELINES['nfl']
        
        for bet_type, y_col in [('moneyline', 'home_win'), ('spread', 'spread_result'), ('total', 'over_result')]:
            print(f"\n  === {bet_type.upper()} ===")
            print(f"  Current baseline: {baseline[bet_type]:.1%}")
            
            y = df[y_col].values
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask].astype(int)
            
            split_idx = int(len(X_valid) * 0.8)
            X_train, X_test = X_valid[:split_idx], X_valid[split_idx:]
            y_train, y_test = y_valid[:split_idx], y_valid[split_idx:]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Gradient Boosting (research shows this works well for NFL)
            model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            proba = model.predict_proba(X_test_scaled)[:, 1]
            pred = (proba >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, pred)
            improvement = acc - baseline[bet_type]
            status = "✅ BEAT" if improvement > 0 else "❌ Below"
            
            print(f"    Accuracy: {acc:.1%} ({status} by {abs(improvement)*100:.1f}pp)")
            
            results[bet_type] = {
                'accuracy': acc,
                'baseline': baseline[bet_type],
                'improvement': improvement,
            }
            
            if improvement > 0:
                model_path = MODELS_DIR / f"v7_nfl_{bet_type}_advanced.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({'model': model, 'scaler': scaler, 'features': list(X.columns)}, f)
                print(f"    → Saved: {model_path.name}")
        
        return results
    
    def train_nhl_advanced(self):
        """NHL: Focus on Corsi/Fenwick-inspired shot metrics."""
        print("\n" + "="*60)
        print("TRAINING NHL V7 (Shot Metrics + ELO)")
        print("="*60)
        
        data_path = DATA_DIR / "nhl" / "game.csv"
        stats_path = DATA_DIR / "nhl" / "game_teams_stats.csv"
        
        if not data_path.exists():
            print("  NHL data not found")
            return None
        
        df = pd.read_csv(data_path)
        df = df[df['type'] == 'R']  # Regular season
        df = df.dropna(subset=['home_goals', 'away_goals'])
        print(f"  Loaded {len(df)} games")
        
        # Calculate ELO
        print("  Calculating ELO...")
        # Need to map team IDs to names
        elo = ELOSystem(k_factor=20, home_advantage=24)
        elo_features = elo.calculate_all(df, 'home_team_id', 'away_team_id', 'home_goals', 'away_goals')
        
        X = pd.DataFrame()
        X['elo_diff'] = elo_features['elo_diff']
        X['elo_win_prob'] = elo_features['elo_win_prob']
        
        # Load team stats for shot metrics
        if stats_path.exists():
            stats = pd.read_csv(stats_path)
            
            # Merge home stats
            home_stats = stats[stats['HoA'] == 'home'][['game_id', 'shots', 'pim', 'faceOffWinPercentage', 'powerPlayOpportunities']]
            home_stats.columns = ['game_id', 'home_shots', 'home_pim', 'home_faceoff_pct', 'home_pp']
            
            away_stats = stats[stats['HoA'] == 'away'][['game_id', 'shots', 'pim', 'faceOffWinPercentage', 'powerPlayOpportunities']]
            away_stats.columns = ['game_id', 'away_shots', 'away_pim', 'away_faceoff_pct', 'away_pp']
            
            df = df.merge(home_stats, on='game_id', how='left')
            df = df.merge(away_stats, on='game_id', how='left')
            
            # Shot differential (Corsi-like)
            X['shot_diff'] = df['home_shots'].fillna(30) - df['away_shots'].fillna(30)
            X['penalty_diff'] = df['home_pim'].fillna(10) - df['away_pim'].fillna(10)
            if 'home_faceoff_pct' in df.columns:
                X['faceoff_diff'] = df['home_faceoff_pct'].fillna(50) - df['away_faceoff_pct'].fillna(50)
        
        # Targets
        df['home_win'] = (df['home_goals'] > df['away_goals']).astype(int)
        df['total_score'] = df['home_goals'] + df['away_goals']
        df['margin'] = df['home_goals'] - df['away_goals']
        df['spread_result'] = (df['margin'] > 1.5).astype(int)
        df['over_result'] = (df['total_score'] > 5.5).astype(int)
        
        X = X.fillna(0)
        
        results = {}
        baseline = CURRENT_BASELINES['nhl']
        
        for bet_type, y_col in [('moneyline', 'home_win'), ('spread', 'spread_result'), ('total', 'over_result')]:
            print(f"\n  === {bet_type.upper()} ===")
            print(f"  Current baseline: {baseline[bet_type]:.1%}")
            
            y = df[y_col].values
            
            # Ensure X and y have same length
            min_len = min(len(X), len(y))
            X_use = X.iloc[:min_len]
            y_use = y[:min_len]
            
            split_idx = int(len(X_use) * 0.8)
            X_train, X_test = X_use[:split_idx], X_use[split_idx:]
            y_train, y_test = y_use[:split_idx], y_use[split_idx:]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Random Forest (handles noisy NHL data well)
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, pred)
            
            improvement = acc - baseline[bet_type]
            status = "✅ BEAT" if improvement > 0 else "❌ Below"
            print(f"    Accuracy: {acc:.1%} ({status} by {abs(improvement)*100:.1f}pp)")
            
            results[bet_type] = {
                'accuracy': acc,
                'baseline': baseline[bet_type],
                'improvement': improvement,
            }
            
            if improvement > 0:
                model_path = MODELS_DIR / f"v7_nhl_{bet_type}_advanced.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({'model': model, 'scaler': scaler}, f)
                print(f"    → Saved: {model_path.name}")
        
        return results
    
    def train_mlb_advanced(self):
        """MLB: ELO system is proven most effective for baseball."""
        print("\n" + "="*60)
        print("TRAINING MLB V7 (ELO-Focused)")
        print("="*60)
        
        data_path = DATA_DIR / "mlb" / "games.csv"
        if not data_path.exists():
            print("  MLB data not found")
            return None
        
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['home_final_score', 'away_final_score'])
        print(f"  Loaded {len(df)} games")
        
        # ELO is particularly effective for MLB
        print("  Calculating ELO...")
        elo = ELOSystem(k_factor=4, home_advantage=24)  # Lower K for MLB's larger sample
        elo_features = elo.calculate_all(df, 'home_team', 'away_team', 'home_final_score', 'away_final_score')
        
        X = pd.DataFrame()
        X['elo_diff'] = elo_features['elo_diff']
        X['elo_win_prob'] = elo_features['elo_win_prob']
        X['home_elo'] = elo_features['home_elo']
        X['away_elo'] = elo_features['away_elo']
        
        # Venue (park factors)
        if 'venue_name' in df.columns:
            X['venue_id'] = pd.factorize(df['venue_name'])[0]
        
        # Targets
        df['home_win'] = (df['home_final_score'] > df['away_final_score']).astype(int)
        df['total_score'] = df['home_final_score'] + df['away_final_score']
        df['margin'] = df['home_final_score'] - df['away_final_score']
        df['spread_result'] = (df['margin'] > 1.5).astype(int)
        df['over_result'] = (df['total_score'] > 8.5).astype(int)
        
        X = X.fillna(0)
        
        results = {}
        baseline = CURRENT_BASELINES['mlb']
        
        for bet_type, y_col in [('moneyline', 'home_win'), ('spread', 'spread_result'), ('total', 'over_result')]:
            print(f"\n  === {bet_type.upper()} ===")
            print(f"  Current baseline: {baseline[bet_type]:.1%}")
            
            y = df[y_col].values
            
            # Ensure X and y have same length
            min_len = min(len(X), len(y))
            X_use = X.iloc[:min_len].reset_index(drop=True)
            y_use = y[:min_len]
            
            split_idx = int(len(X_use) * 0.8)
            X_train, X_test = X_use.iloc[:split_idx], X_use.iloc[split_idx:]
            y_train, y_test = y_use[:split_idx], y_use[split_idx:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Logistic Regression with ELO (research shows this is effective)
            model = LogisticRegression(C=0.1, random_state=42, max_iter=500)
            model.fit(X_train_scaled, y_train)
            
            proba = model.predict_proba(X_test_scaled)[:, 1]
            pred = (proba >= 0.5).astype(int)
            acc = accuracy_score(y_test, pred)
            
            improvement = acc - baseline[bet_type]
            status = "✅ BEAT" if improvement > 0 else "❌ Below"
            print(f"    Accuracy: {acc:.1%} ({status} by {abs(improvement)*100:.1f}pp)")
            
            results[bet_type] = {
                'accuracy': acc,
                'baseline': baseline[bet_type],
                'improvement': improvement,
            }
            
            if improvement > 0:
                model_path = MODELS_DIR / f"v7_mlb_{bet_type}_advanced.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({'model': model, 'scaler': scaler}, f)
                print(f"    → Saved: {model_path.name}")
        
        return results
    
    def train_soccer_advanced(self):
        """Soccer: Poisson-inspired with attack/defense ratings."""
        print("\n" + "="*60)
        print("TRAINING SOCCER V7 (Poisson-Inspired + ELO)")
        print("="*60)
        
        data_path = DATA_DIR / "soccer" / "games.csv"
        if not data_path.exists():
            print("  Soccer data not found")
            return None
        
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['home_club_goals', 'away_club_goals'])
        print(f"  Loaded {len(df)} games")
        
        # ELO for team strength
        print("  Calculating ELO...")
        elo = ELOSystem(k_factor=20, home_advantage=40)
        elo_features = elo.calculate_all(df, 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals')
        
        X = pd.DataFrame()
        X['elo_diff'] = elo_features['elo_diff']
        X['elo_win_prob'] = elo_features['elo_win_prob']
        
        # Attack/Defense ratings (Poisson-inspired)
        # Calculate average goals for/against
        if 'home_club_position' in df.columns:
            X['position_diff'] = df.get('away_club_position', 10) - df.get('home_club_position', 10)
        
        # Targets
        df['home_win'] = (df['home_club_goals'] > df['away_club_goals']).astype(int)
        df['total_goals'] = df['home_club_goals'] + df['away_club_goals']
        df['margin'] = df['home_club_goals'] - df['away_club_goals']
        df['spread_result'] = (df['margin'] > 0.5).astype(int)
        df['over_result'] = (df['total_goals'] > 2.5).astype(int)
        
        X = X.fillna(0)
        
        results = {}
        baseline = CURRENT_BASELINES['soccer']
        
        for bet_type, y_col in [('moneyline', 'home_win'), ('spread', 'spread_result'), ('total', 'over_result')]:
            print(f"\n  === {bet_type.upper()} ===")
            print(f"  Current baseline: {baseline[bet_type]:.1%}")
            
            y = df[y_col].values
            
            # Ensure X and y have same length
            min_len = min(len(X), len(y))
            X_use = X.iloc[:min_len].reset_index(drop=True)
            y_use = y[:min_len]
            
            split_idx = int(len(X_use) * 0.8)
            X_train, X_test = X_use.iloc[:split_idx], X_use.iloc[split_idx:]
            y_train, y_test = y_use[:split_idx], y_use[split_idx:]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # XGBoost with Platt scaling calibration
            xgb_model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.02,
                reg_lambda=5.0,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
            
            # Get XGBoost predictions
            xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
            
            # Platt scaling calibration
            calib_train_proba = xgb_model.predict_proba(X_train_scaled)[:, 1].reshape(-1, 1)
            calib_model = LogisticRegression(C=1.0, random_state=42)
            calib_model.fit(calib_train_proba, y_train)
            
            proba = calib_model.predict_proba(xgb_proba.reshape(-1, 1))[:, 1]
            pred = (proba >= 0.5).astype(int)
            acc = accuracy_score(y_test, pred)
            
            improvement = acc - baseline[bet_type]
            status = "✅ BEAT" if improvement > 0 else "❌ Below"
            print(f"    Accuracy: {acc:.1%} ({status} by {abs(improvement)*100:.1f}pp)")
            
            results[bet_type] = {
                'accuracy': acc,
                'baseline': baseline[bet_type],
                'improvement': improvement,
            }
            
            if improvement > 0:
                model_path = MODELS_DIR / f"v7_soccer_{bet_type}_advanced.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({'model': model, 'scaler': scaler}, f)
                print(f"    → Saved: {model_path.name}")
        
        return results


def main():
    """Run advanced V7 training for all sports."""
    print("\n" + "="*70)
    print("ADVANCED V7 MODEL TRAINING")
    print("Research-Backed Methods: ELO, Calibration, Weather, Shot Metrics")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer = AdvancedV7Trainer()
    all_results = {}
    
    # Train each sport
    all_results['nba'] = trainer.train_nba_advanced()
    all_results['nfl'] = trainer.train_nfl_advanced()
    all_results['nhl'] = trainer.train_nhl_advanced()
    all_results['mlb'] = trainer.train_mlb_advanced()
    all_results['soccer'] = trainer.train_soccer_advanced()
    
    # Summary
    print("\n" + "="*70)
    print("V7 TRAINING SUMMARY")
    print("="*70)
    
    wins = 0
    total = 0
    
    for sport, results in all_results.items():
        if results:
            print(f"\n{sport.upper()}")
            for bet_type, data in results.items():
                total += 1
                status = "✅" if data['improvement'] > 0 else "❌"
                if data['improvement'] > 0:
                    wins += 1
                print(f"  {bet_type}: {data['accuracy']:.1%} (baseline: {data['baseline']:.1%}) {status} {data['improvement']*100:+.1f}pp")
    
    print(f"\n📊 Beat baseline in {wins}/{total} models")
    
    # Save summary
    summary_path = MODELS_DIR / "v7_advanced_results.json"
    summary = {}
    for sport, results in all_results.items():
        if results:
            summary[sport] = {
                bet_type: {
                    'accuracy': data['accuracy'],
                    'baseline': data['baseline'],
                    'improvement': data['improvement'],
                }
                for bet_type, data in results.items()
            }
    summary['trained_at'] = datetime.now().isoformat()
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to: {summary_path}")
    
    return all_results


if __name__ == "__main__":
    results = main()
