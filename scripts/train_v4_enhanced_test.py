"""
V4 Ensemble with Enhanced Data
==============================
Test V4 architecture (XGBoost + RF + MLP) with the new enhanced NBA data.
This helps us see if the improvement is from:
1. The new data (STL, BLK, TOV, PF)
2. The behavioral proxy feature engineering
3. The V6 architecture (XGBoost + LightGBM)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
import xgboost as xgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "nba"
MODELS_DIR = BASE_DIR / "models"


def load_enhanced_data():
    """Load the new enhanced NBA data."""
    print("\n=== Loading Enhanced NBA Data ===")
    
    new_games_path = DATA_DIR / "nba_new" / "nba_training_games_20251205.csv"
    
    if not new_games_path.exists():
        print(f"ERROR: {new_games_path} not found!")
        return None
    
    new_df = pd.read_csv(new_games_path)
    print(f"  Loaded: {len(new_df)} team-game records")
    
    # Identify home/away
    new_df['is_home'] = new_df['MATCHUP'].str.contains(' vs. ')
    new_df['GAME_DATE'] = pd.to_datetime(new_df['GAME_DATE'], format='%b %d, %Y')
    
    # Pair home and away
    home = new_df[new_df['is_home']].add_suffix('_home')
    away = new_df[~new_df['is_home']].add_suffix('_away')
    
    paired = home.merge(away, left_on='Game_ID_home', right_on='Game_ID_away', how='inner')
    
    games = pd.DataFrame({
        'GAME_DATE_EST': paired['GAME_DATE_home'],
        'GAME_ID': paired['Game_ID_home'],
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
    
    print(f"  Paired games: {len(games)}")
    return games


def build_v4_features(df):
    """Build V4-style features (ELO, momentum, rest) with the enhanced data."""
    print("  Building V4-style features...")
    
    df = df.sort_values('GAME_DATE_EST').reset_index(drop=True)
    
    # Build team histories
    teams = set(df['HOME_TEAM_ID'].unique()) | set(df['VISITOR_TEAM_ID'].unique())
    team_histories = {t: [] for t in teams}
    team_elo = {t: 1500 for t in teams}  # Starting ELO
    
    features = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        home_id = row['HOME_TEAM_ID']
        away_id = row['VISITOR_TEAM_ID']
        game_date = row['GAME_DATE_EST']
        
        home_history = [g for g in team_histories[home_id] if g['date'] < game_date]
        away_history = [g for g in team_histories[away_id] if g['date'] < game_date]
        
        if len(home_history) < 5 or len(away_history) < 5:
            # Still add to history
            home_game = {
                'date': game_date,
                'won': row['HOME_TEAM_WINS'] == 1,
                'pts': row['PTS_home'],
                'opp_pts': row['PTS_away'],
                'fg_pct': row['FG_PCT_home'],
                'stl': row['STL_home'],
                'blk': row['BLK_home'],
                'tov': row['TOV_home'],
                'ast': row['AST_home'],
            }
            away_game = {
                'date': game_date,
                'won': row['HOME_TEAM_WINS'] == 0,
                'pts': row['PTS_away'],
                'opp_pts': row['PTS_home'],
                'fg_pct': row['FG_PCT_away'],
                'stl': row['STL_away'],
                'blk': row['BLK_away'],
                'tov': row['TOV_away'],
                'ast': row['AST_away'],
            }
            team_histories[home_id].append(home_game)
            team_histories[away_id].append(away_game)
            
            # Update ELO
            if row['HOME_TEAM_WINS'] == 1:
                team_elo[home_id] += 20
                team_elo[away_id] -= 20
            else:
                team_elo[home_id] -= 20
                team_elo[away_id] += 20
            continue
        
        # Calculate V4 features
        def get_stats(history, n=20):
            recent = history[-n:] if len(history) >= n else history
            pts = [g['pts'] for g in recent if g['pts'] > 0]
            wins = [1 if g['won'] else 0 for g in recent]
            fg_pct = [g['fg_pct'] for g in recent if g['fg_pct'] > 0]
            stl = [g['stl'] for g in recent if g['stl'] > 0]
            blk = [g['blk'] for g in recent if g['blk'] > 0]
            tov = [g['tov'] for g in recent if g['tov'] > 0]
            ast = [g['ast'] for g in recent if g['ast'] > 0]
            return {
                'pts_mean': np.mean(pts) if pts else 100,
                'win_rate': np.mean(wins),
                'fg_pct': np.mean(fg_pct) if fg_pct else 0.45,
                'stl_mean': np.mean(stl) if stl else 7,
                'blk_mean': np.mean(blk) if blk else 5,
                'tov_mean': np.mean(tov) if tov else 14,
                'ast_mean': np.mean(ast) if ast else 24,
                'last_5_wins': sum([1 if g['won'] else 0 for g in recent[-5:]]),
            }
        
        home_stats = get_stats(home_history)
        away_stats = get_stats(away_history)
        
        # Rest days
        home_rest = (game_date - home_history[-1]['date']).days
        away_rest = (game_date - away_history[-1]['date']).days
        
        # V4-style feature vector
        game_features = {
            # ELO
            'elo_diff': (team_elo[home_id] - team_elo[away_id]) / 400,
            'home_elo': team_elo[home_id] / 2000,
            'away_elo': team_elo[away_id] / 2000,
            
            # Win rates
            'win_rate_diff': home_stats['win_rate'] - away_stats['win_rate'],
            'home_win_rate': home_stats['win_rate'],
            'away_win_rate': away_stats['win_rate'],
            
            # Momentum
            'momentum_diff': (home_stats['last_5_wins'] - away_stats['last_5_wins']) / 5,
            'home_momentum': home_stats['last_5_wins'] / 5,
            'away_momentum': away_stats['last_5_wins'] / 5,
            
            # Rest
            'rest_diff': (home_rest - away_rest) / 7,
            'home_rest': min(home_rest / 7, 1),
            'away_rest': min(away_rest / 7, 1),
            
            # Scoring
            'pts_diff': (home_stats['pts_mean'] - away_stats['pts_mean']) / 10,
            'fg_pct_diff': home_stats['fg_pct'] - away_stats['fg_pct'],
            
            # NEW: Enhanced stats 
            'stl_diff': (home_stats['stl_mean'] - away_stats['stl_mean']) / 5,
            'blk_diff': (home_stats['blk_mean'] - away_stats['blk_mean']) / 3,
            'tov_diff': (away_stats['tov_mean'] - home_stats['tov_mean']) / 5,  # Inverted
            'ast_diff': (home_stats['ast_mean'] - away_stats['ast_mean']) / 5,
        }
        
        features.append(game_features)
        valid_indices.append(idx)
        
        # Update histories
        home_game = {
            'date': game_date,
            'won': row['HOME_TEAM_WINS'] == 1,
            'pts': row['PTS_home'],
            'opp_pts': row['PTS_away'],
            'fg_pct': row['FG_PCT_home'],
            'stl': row['STL_home'],
            'blk': row['BLK_home'],
            'tov': row['TOV_home'],
            'ast': row['AST_home'],
        }
        away_game = {
            'date': game_date,
            'won': row['HOME_TEAM_WINS'] == 0,
            'pts': row['PTS_away'],
            'opp_pts': row['PTS_home'],
            'fg_pct': row['FG_PCT_away'],
            'stl': row['STL_away'],
            'blk': row['BLK_away'],
            'tov': row['TOV_away'],
            'ast': row['AST_away'],
        }
        team_histories[home_id].append(home_game)
        team_histories[away_id].append(away_game)
        
        # Update ELO
        if row['HOME_TEAM_WINS'] == 1:
            team_elo[home_id] += 20
            team_elo[away_id] -= 20
        else:
            team_elo[home_id] -= 20
            team_elo[away_id] += 20
        
        if len(features) % 2000 == 0:
            print(f"    Processed {len(features)} games...")
    
    print(f"  Created {len(features)} feature vectors")
    return pd.DataFrame(features), valid_indices


def train_v4_with_enhanced_data():
    """Train V4 ensemble (XGB + RF + MLP) with enhanced data."""
    print("\n" + "="*60)
    print("V4 ENSEMBLE WITH ENHANCED DATA")
    print("="*60)
    
    # Load data
    games = load_enhanced_data()
    if games is None:
        return None
    
    games['target'] = games['HOME_TEAM_WINS'].astype(int)
    games = games.sort_values('GAME_DATE_EST').reset_index(drop=True)
    
    print(f"\n  Total games: {len(games)}")
    print(f"  Date range: {games['GAME_DATE_EST'].min()} to {games['GAME_DATE_EST'].max()}")
    print(f"  Home win rate: {games['target'].mean():.1%}")
    
    # Build features
    X, valid_indices = build_v4_features(games)
    y = games.loc[valid_indices, 'target'].values
    
    print(f"\n  Features: {X.shape}")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train V4 Ensemble
    print("\n  Training V4 Ensemble...")
    
    # XGBoost
    print("    Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Random Forest
    print("    Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # MLP
    print("    Training MLP...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp_model.fit(X_train, y_train)
    
    # Get probabilities
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    mlp_proba = mlp_model.predict_proba(X_test)[:, 1]
    
    # V4 weighted ensemble (45% XGB, 35% RF, 20% MLP)
    ensemble_proba = 0.45 * xgb_proba + 0.35 * rf_proba + 0.20 * mlp_proba
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, ensemble_pred)
    auc = roc_auc_score(y_test, ensemble_proba)
    brier = brier_score_loss(y_test, ensemble_proba)
    logloss = log_loss(y_test, ensemble_proba)
    
    xgb_acc = accuracy_score(y_test, (xgb_proba >= 0.5).astype(int))
    rf_acc = accuracy_score(y_test, (rf_proba >= 0.5).astype(int))
    mlp_acc = accuracy_score(y_test, (mlp_proba >= 0.5).astype(int))
    
    print(f"\n  {'='*50}")
    print(f"  V4 ENSEMBLE + ENHANCED DATA RESULTS")
    print(f"  {'='*50}")
    print(f"  XGBoost Accuracy:  {xgb_acc:.1%}")
    print(f"  RF Accuracy:       {rf_acc:.1%}")
    print(f"  MLP Accuracy:      {mlp_acc:.1%}")
    print(f"  Ensemble Accuracy: {accuracy:.1%}")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  Brier Score:       {brier:.4f}")
    print(f"  Log Loss:          {logloss:.4f}")
    
    print(f"\n  {'='*50}")
    print(f"  COMPARISON")
    print(f"  {'='*50}")
    print(f"  V4 (old data):        62.0%")
    print(f"  V4 (enhanced data):   {accuracy:.1%}")
    print(f"  V6 Enhanced:          66.0%")
    
    v4_improvement = (accuracy - 0.62) * 100
    v6_vs_v4 = (0.66 - accuracy) * 100
    
    if v4_improvement > 0:
        print(f"\n  V4 data improvement:  +{v4_improvement:.1f}pp")
    print(f"  V6 vs V4 (same data): +{v6_vs_v4:.1f}pp (architecture advantage)")
    
    return {
        'accuracy': float(accuracy),
        'xgb_accuracy': float(xgb_acc),
        'rf_accuracy': float(rf_acc),
        'mlp_accuracy': float(mlp_acc),
        'auc': float(auc),
        'brier': float(brier),
    }


if __name__ == "__main__":
    results = train_v4_with_enhanced_data()
