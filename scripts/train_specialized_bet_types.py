"""
Specialized Bet-Type Models Training
====================================
Trains separate models for each bet type (Moneyline, Spread, O/U) for each sport.
Goal: Beat the V6 "complete" models by using specialized architectures.

Sports: NFL, NHL, MLB, Soccer
Bet Types: Moneyline, Spread, Over/Under

Run: python scripts/train_specialized_bet_types.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Current V6 baselines to beat
V6_BASELINES = {
    'nfl': {'moneyline': 0.651, 'spread': 0.652, 'total': 0.563},
    'nhl': {'moneyline': 0.720, 'spread': 0.591, 'total': 0.601},
    'mlb': {'moneyline': 0.534, 'spread': 0.556, 'total': 0.584},
    'soccer': {'moneyline': 0.643, 'spread': 0.753, 'total': 0.615},
}


class SpecializedBetTypeTrainer:
    """Train specialized models for each bet type."""
    
    def __init__(self, sport):
        self.sport = sport
        self.results = {}
        
    def load_nfl_data(self):
        """Load and prepare NFL data."""
        print(f"\n{'='*60}")
        print(f"LOADING NFL DATA")
        print(f"{'='*60}")
        
        # Try spreadspoke for game-level data with spreads
        games_path = DATA_DIR / "nfl" / "spreadspoke_scores.csv"
        if games_path.exists():
            df = pd.read_csv(games_path)
            print(f"  Loaded {len(df)} games from spreadspoke_scores.csv")
            
            # Clean and prepare
            df = df.dropna(subset=['score_home', 'score_away'])
            
            # Create targets
            df['home_win'] = (df['score_home'] > df['score_away']).astype(int)
            df['total_score'] = df['score_home'] + df['score_away']
            
            # Spread target (if spread column exists)
            if 'spread_favorite' in df.columns:
                # Determine if home covered the spread
                df['spread_result'] = np.where(
                    df['team_favorite_id'] == df['team_home'],
                    (df['score_home'] - df['score_away']) > abs(df['spread_favorite']),
                    (df['score_away'] - df['score_home']) > abs(df['spread_favorite'])
                ).astype(int)
            else:
                df['spread_result'] = df['home_win']  # Fallback
            
            # O/U target
            if 'over_under_line' in df.columns:
                df['over_under_line'] = pd.to_numeric(df['over_under_line'], errors='coerce')
                df['over_result'] = (df['total_score'] > df['over_under_line']).astype(int)
            else:
                median_total = df['total_score'].median()
                df['over_result'] = (df['total_score'] > median_total).astype(int)
            
            print(f"  Home win rate: {df['home_win'].mean():.1%}")
            print(f"  Over rate: {df['over_result'].mean():.1%}")
            
            return df
        return None
    
    def load_nhl_data(self):
        """Load and prepare NHL data."""
        print(f"\n{'='*60}")
        print(f"LOADING NHL DATA")
        print(f"{'='*60}")
        
        games_path = DATA_DIR / "nhl" / "game.csv"
        teams_path = DATA_DIR / "nhl" / "game_teams_stats.csv"
        
        if games_path.exists():
            games = pd.read_csv(games_path)
            print(f"  Loaded {len(games)} games")
            
            # Filter completed games
            games = games[games['type'] == 'R']  # Regular season
            games = games.dropna(subset=['home_goals', 'away_goals'])
            
            # Create targets
            games['home_win'] = (games['home_goals'] > games['away_goals']).astype(int)
            games['total_score'] = games['home_goals'] + games['away_goals']
            games['goal_diff'] = games['home_goals'] - games['away_goals']
            
            # Spread: Did home team win by more than 1.5 goals?
            games['spread_result'] = (games['goal_diff'] > 1.5).astype(int)
            
            # O/U: Did total exceed 5.5 goals (common NHL total)?
            games['over_result'] = (games['total_score'] > 5.5).astype(int)
            
            print(f"  Home win rate: {games['home_win'].mean():.1%}")
            print(f"  Over 5.5 rate: {games['over_result'].mean():.1%}")
            
            # Load team stats for features
            if teams_path.exists():
                team_stats = pd.read_csv(teams_path)
                # Aggregate to game level
                home_stats = team_stats[team_stats['HoA'] == 'home'].copy()
                away_stats = team_stats[team_stats['HoA'] == 'away'].copy()
                
                home_stats = home_stats.add_prefix('home_')
                away_stats = away_stats.add_prefix('away_')
                
                # Merge with games
                games = games.merge(home_stats, left_on='game_id', right_on='home_game_id', how='left')
                games = games.merge(away_stats, left_on='game_id', right_on='away_game_id', how='left')
            
            return games
        return None
    
    def load_mlb_data(self):
        """Load and prepare MLB data."""
        print(f"\n{'='*60}")
        print(f"LOADING MLB DATA")
        print(f"{'='*60}")
        
        games_path = DATA_DIR / "mlb" / "games.csv"
        
        if games_path.exists():
            df = pd.read_csv(games_path)
            print(f"  Loaded {len(df)} games")
            
            # Filter completed games
            df = df.dropna(subset=['home_final_score', 'away_final_score'])
            
            # Create targets
            df['home_win'] = (df['home_final_score'] > df['away_final_score']).astype(int)
            df['total_score'] = df['home_final_score'] + df['away_final_score']
            df['run_diff'] = df['home_final_score'] - df['away_final_score']
            
            # Spread: Did home team win by more than 1.5 runs?
            df['spread_result'] = (df['run_diff'] > 1.5).astype(int)
            
            # O/U: Did total exceed 8.5 runs (common MLB total)?
            df['over_result'] = (df['total_score'] > 8.5).astype(int)
            
            print(f"  Home win rate: {df['home_win'].mean():.1%}")
            print(f"  Over 8.5 rate: {df['over_result'].mean():.1%}")
            
            return df
        return None
    
    def load_soccer_data(self):
        """Load and prepare Soccer data."""
        print(f"\n{'='*60}")
        print(f"LOADING SOCCER DATA")
        print(f"{'='*60}")
        
        games_path = DATA_DIR / "soccer" / "games.csv"
        
        if games_path.exists():
            df = pd.read_csv(games_path)
            print(f"  Loaded {len(df)} games")
            
            # Filter completed games
            df = df.dropna(subset=['home_club_goals', 'away_club_goals'])
            
            # Create targets
            df['home_win'] = (df['home_club_goals'] > df['away_club_goals']).astype(int)
            df['total_goals'] = df['home_club_goals'] + df['away_club_goals']
            df['goal_diff'] = df['home_club_goals'] - df['away_club_goals']
            
            # Spread: Did home team win by more than 0.5 goals?
            df['spread_result'] = (df['goal_diff'] > 0.5).astype(int)
            
            # O/U: Did total exceed 2.5 goals?
            df['over_result'] = (df['total_goals'] > 2.5).astype(int)
            
            print(f"  Home win rate: {df['home_win'].mean():.1%}")
            print(f"  Over 2.5 rate: {df['over_result'].mean():.1%}")
            
            return df
        return None
    
    def create_features(self, df, sport):
        """Create features based on available columns - NO SCORE-BASED FEATURES (prevent leakage)."""
        features = pd.DataFrame()
        
        if sport == 'nfl':
            # NFL specific features - pre-game only
            if 'schedule_season' in df.columns:
                features['season'] = df['schedule_season']
            if 'schedule_week' in df.columns:
                features['week'] = pd.to_numeric(df['schedule_week'], errors='coerce').fillna(1)
            if 'stadium_neutral' in df.columns:
                features['neutral'] = df['stadium_neutral'].fillna(False).astype(int)
            if 'weather_temperature' in df.columns:
                features['temperature'] = pd.to_numeric(df['weather_temperature'], errors='coerce').fillna(70)
            if 'weather_wind_mph' in df.columns:
                features['wind'] = pd.to_numeric(df['weather_wind_mph'], errors='coerce').fillna(5)
            
        elif sport == 'nhl':
            # NHL: Use team-level aggregates NOT game scores
            if 'home_shots' in df.columns:
                features['shot_diff'] = df.get('home_shots', 30) - df.get('away_shots', 30)
            if 'home_pim' in df.columns:
                features['penalty_diff'] = df.get('home_pim', 10) - df.get('away_pim', 10)
            # NOTE: Removed 'avg_goals' as it uses actual game scores - data leakage!
                
        elif sport == 'mlb':
            # MLB: Venue is OK, but NOT scores
            if 'venue_name' in df.columns:
                features['venue_id'] = pd.factorize(df['venue_name'])[0]
            # Add date-based features instead
            if 'date' in df.columns:
                try:
                    dates = pd.to_datetime(df['date'], errors='coerce')
                    features['month'] = dates.dt.month.fillna(6)
                    features['day_of_week'] = dates.dt.dayofweek.fillna(3)
                except:
                    pass
            # NOTE: Removed 'run_environment' as it uses actual scores - data leakage!
                
        elif sport == 'soccer':
            # Soccer: Position differential is OK (pre-game info)
            if 'home_club_position' in df.columns:
                features['position_diff'] = df.get('away_club_position', 10) - df.get('home_club_position', 10)
            if 'date' in df.columns:
                try:
                    dates = pd.to_datetime(df['date'], errors='coerce')
                    features['month'] = dates.dt.month.fillna(6)
                except:
                    pass
            # NOTE: Removed 'goal_environment' as it uses actual scores - data leakage!
        
        # Common features: home advantage baseline
        features['home_bias'] = 1  # Constant for home team
        
        # Fill NaN and ensure at least some features
        features = features.fillna(0)
        
        if len(features.columns) < 2:
            # Add noise features to prevent single-feature failure
            np.random.seed(42)  # Consistent noise
            features['noise1'] = np.random.randn(len(df)) * 0.01
            features['noise2'] = np.random.randn(len(df)) * 0.01
        
        return features
    
    def train_specialized_model(self, X, y, bet_type):
        """Train a specialized model for a specific bet type."""
        print(f"\n  Training {bet_type.upper()} model...")
        
        # Time-based split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # XGBoost with bet-type specific tuning
        if bet_type == 'moneyline':
            # Moneyline: Focus on accuracy
            xgb_params = {
                'n_estimators': 500,
                'max_depth': 5,
                'learning_rate': 0.01,
                'reg_lambda': 5.0,
                'reg_alpha': 1.0,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
            }
        elif bet_type == 'spread':
            # Spread: Focus on margin prediction
            xgb_params = {
                'n_estimators': 600,
                'max_depth': 6,
                'learning_rate': 0.008,
                'reg_lambda': 8.0,
                'reg_alpha': 2.0,
                'subsample': 0.75,
                'colsample_bytree': 0.65,
            }
        else:  # total
            # O/U: Focus on scoring environment
            xgb_params = {
                'n_estimators': 700,
                'max_depth': 7,
                'learning_rate': 0.005,
                'reg_lambda': 10.0,
                'reg_alpha': 3.0,
                'subsample': 0.7,
                'colsample_bytree': 0.6,
            }
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            **xgb_params,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train_scaled, y_train, 
                     eval_set=[(X_test_scaled, y_test)],
                     verbose=False)
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            **xgb_params,
            objective='binary',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_train,
                     eval_set=[(X_test_scaled, y_test)])
        
        # Ensemble predictions
        xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
        lgb_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
        ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, ensemble_pred)
        
        try:
            auc = roc_auc_score(y_test, ensemble_proba)
        except:
            auc = 0.5
            
        try:
            brier = brier_score_loss(y_test, ensemble_proba)
        except:
            brier = 0.25
        
        print(f"    Accuracy: {accuracy:.1%}")
        print(f"    AUC-ROC:  {auc:.4f}")
        print(f"    Brier:    {brier:.4f}")
        
        return {
            'xgb_model': xgb_model,
            'lgb_model': lgb_model,
            'scaler': scaler,
            'accuracy': accuracy,
            'auc': auc,
            'brier': brier,
            'test_size': len(y_test),
            'feature_names': list(X.columns) if hasattr(X, 'columns') else None,
        }
    
    def train_all_bet_types(self):
        """Train specialized models for all bet types for a sport."""
        print(f"\n{'='*60}")
        print(f"TRAINING SPECIALIZED MODELS FOR {self.sport.upper()}")
        print(f"{'='*60}")
        
        # Load data based on sport
        if self.sport == 'nfl':
            df = self.load_nfl_data()
        elif self.sport == 'nhl':
            df = self.load_nhl_data()
        elif self.sport == 'mlb':
            df = self.load_mlb_data()
        elif self.sport == 'soccer':
            df = self.load_soccer_data()
        else:
            print(f"  Unknown sport: {self.sport}")
            return None
        
        if df is None or len(df) < 100:
            print(f"  Insufficient data for {self.sport}")
            return None
        
        # Create features
        X = self.create_features(df, self.sport)
        print(f"  Created {len(X.columns)} features: {list(X.columns)}")
        
        # Filter rows where we have valid targets
        valid_idx = ~(df['home_win'].isna() | df['spread_result'].isna() | df['over_result'].isna())
        X = X[valid_idx].reset_index(drop=True)
        df = df[valid_idx].reset_index(drop=True)
        
        print(f"  Training on {len(X)} samples")
        
        baselines = V6_BASELINES.get(self.sport, {})
        results = {}
        
        # Train each bet type
        for bet_type, target_col in [('moneyline', 'home_win'), 
                                      ('spread', 'spread_result'),
                                      ('total', 'over_result')]:
            y = df[target_col].values
            baseline = baselines.get(bet_type, 0.50)
            
            print(f"\n  === {bet_type.upper()} ===")
            print(f"  V6 Baseline: {baseline:.1%}")
            
            result = self.train_specialized_model(X, y, bet_type)
            result['baseline'] = baseline
            result['improvement'] = result['accuracy'] - baseline
            
            if result['accuracy'] > baseline:
                print(f"  🎉 BEAT BASELINE by {result['improvement']*100:.1f}pp!")
            else:
                print(f"  ❌ Below baseline by {-result['improvement']*100:.1f}pp")
            
            results[bet_type] = result
            
            # Save model if it beats baseline
            if result['accuracy'] > baseline:
                model_data = {
                    'xgb_model': result['xgb_model'],
                    'lgb_model': result['lgb_model'],
                    'scaler': result['scaler'],
                    'feature_names': result['feature_names'],
                    'accuracy': result['accuracy'],
                    'sport': self.sport,
                    'bet_type': bet_type,
                    'trained_at': datetime.now().isoformat(),
                }
                
                model_path = MODELS_DIR / f"v6_{self.sport}_{bet_type}_specialized.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                print(f"    ✅ Saved to: {model_path}")
        
        self.results = results
        return results


def train_all_sports():
    """Train specialized models for all sports."""
    print("\n" + "="*70)
    print("SPECIALIZED BET-TYPE MODEL TRAINING")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Goal: Beat V6 baselines with specialized models for each bet type")
    
    all_results = {}
    
    for sport in ['nfl', 'nhl', 'mlb', 'soccer']:
        trainer = SpecializedBetTypeTrainer(sport)
        results = trainer.train_all_bet_types()
        
        if results:
            all_results[sport] = results
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    wins = 0
    total = 0
    
    for sport, results in all_results.items():
        print(f"\n{sport.upper()}")
        for bet_type, data in results.items():
            total += 1
            status = "✅" if data['improvement'] > 0 else "❌"
            if data['improvement'] > 0:
                wins += 1
            print(f"  {bet_type}: {data['accuracy']:.1%} (baseline: {data['baseline']:.1%}) {status} {data['improvement']*100:+.1f}pp")
    
    print(f"\n📊 Beat baseline in {wins}/{total} models")
    
    # Save summary
    summary_path = MODELS_DIR / "specialized_bet_types_results.json"
    summary = {
        sport: {
            bet_type: {
                'accuracy': data['accuracy'],
                'baseline': data['baseline'],
                'improvement': data['improvement'],
                'auc': data['auc'],
                'brier': data['brier'],
                'test_size': data['test_size'],
            }
            for bet_type, data in results.items()
        }
        for sport, results in all_results.items()
    }
    summary['trained_at'] = datetime.now().isoformat()
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to: {summary_path}")
    
    return all_results


if __name__ == "__main__":
    results = train_all_sports()
