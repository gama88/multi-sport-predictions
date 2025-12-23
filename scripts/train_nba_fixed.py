"""
Fixed NBA Model Training - No Data Leakage
Only uses PRE-GAME features (not in-game stats)

The problem: Using fgm, fga, pts, reb, ast, etc. causes 100% accuracy 
because these are POST-GAME stats that directly predict the outcome.

Solution: Use historical rolling averages that would be known BEFORE the game.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def create_pregame_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create pre-game features using ONLY historical data (no leakage)."""
    print("  📊 Creating pre-game features (no data leakage)...")
    
    # Sort by date first
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)
    
    # In-game stats columns (CANNOT use these - they're post-game)
    postgame_cols = [
        'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct',
        'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb',
        'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'plus_minus',
        'wl', 'min', 'video_available'
    ]
    
    # Team stats history for rolling averages
    team_stats = {}
    
    # Pre-game features to create
    pregame_data = []
    
    for idx, row in df.iterrows():
        home_team = row['team_id_home']
        away_team = row['team_id_away']
        
        # Initialize team histories if needed
        if home_team not in team_stats:
            team_stats[home_team] = {'pts': [], 'wins': [], 'games': 0}
        if away_team not in team_stats:
            team_stats[away_team] = {'pts': [], 'wins': [], 'games': 0}
        
        # Calculate PRE-GAME features (using history BEFORE this game)
        home_hist = team_stats[home_team]
        away_hist = team_stats[away_team]
        
        # Rolling averages (last 10 games)
        home_pts_avg = np.mean(home_hist['pts'][-10:]) if home_hist['pts'] else 100
        away_pts_avg = np.mean(away_hist['pts'][-10:]) if away_hist['pts'] else 100
        
        home_win_pct = np.mean(home_hist['wins'][-20:]) if home_hist['wins'] else 0.5
        away_win_pct = np.mean(away_hist['wins'][-20:]) if away_hist['wins'] else 0.5
        
        # Games played (season fatigue)
        home_games = home_hist['games']
        away_games = away_hist['games']
        
        # Calculate REST DAYS
        if 'last_game_date' not in home_hist:
            home_rest = 7
        else:
            home_rest = (row['game_date'] - home_hist['last_game_date']).days
        
        if 'last_game_date' not in away_hist:
            away_rest = 7
        else:
            away_rest = (row['game_date'] - away_hist['last_game_date']).days
        
        # ELO-like rating
        home_elo = home_hist.get('elo', 1500)
        away_elo = away_hist.get('elo', 1500)
        
        # Create feature row
        features = {
            'home_pts_avg': home_pts_avg,
            'away_pts_avg': away_pts_avg,
            'pts_diff_avg': home_pts_avg - away_pts_avg,
            'home_win_pct': home_win_pct,
            'away_win_pct': away_win_pct,
            'win_pct_diff': home_win_pct - away_win_pct,
            'home_games_played': home_games,
            'away_games_played': away_games,
            'home_rest_days': min(home_rest, 14),  # Cap at 14
            'away_rest_days': min(away_rest, 14),
            'rest_differential': home_rest - away_rest,
            'home_b2b': 1 if home_rest <= 1 else 0,
            'away_b2b': 1 if away_rest <= 1 else 0,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo,
            'is_home': 1,  # Home advantage
            'day_of_week': row['game_date'].dayofweek,
            'month': row['game_date'].month,
        }
        
        # Target: did home team win?
        home_pts = row.get('pts_home', 0)
        away_pts = row.get('pts_away', 0)
        home_win = 1 if home_pts > away_pts else 0
        features['target'] = home_win
        
        pregame_data.append(features)
        
        # UPDATE history AFTER the game (for next games)
        team_stats[home_team]['pts'].append(home_pts)
        team_stats[home_team]['wins'].append(home_win)
        team_stats[home_team]['games'] += 1
        team_stats[home_team]['last_game_date'] = row['game_date']
        
        team_stats[away_team]['pts'].append(away_pts)
        team_stats[away_team]['wins'].append(1 - home_win)
        team_stats[away_team]['games'] += 1
        team_stats[away_team]['last_game_date'] = row['game_date']
        
        # Update ELO
        k = 20
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        team_stats[home_team]['elo'] = home_elo + k * (home_win - expected_home)
        team_stats[away_team]['elo'] = away_elo + k * ((1 - home_win) - (1 - expected_home))
    
    return pd.DataFrame(pregame_data)


def train_nba_fixed():
    """Train NBA model without data leakage."""
    print("\n" + "="*60)
    print("🏀 NBA MODEL TRAINING - NO DATA LEAKAGE")
    print("="*60)
    
    # Load data
    nba_file = DATA_DIR / "nba" / "game.csv"
    df = pd.read_csv(nba_file, low_memory=False)
    print(f"  📊 Loaded {len(df):,} games")
    
    # Create pre-game features
    df_features = create_pregame_features(df)
    
    # Skip first few games per team (not enough history)
    df_features = df_features[df_features['home_games_played'] >= 5]
    df_features = df_features[df_features['away_games_played'] >= 5]
    
    print(f"  📊 {len(df_features):,} games with sufficient history")
    
    # Split features and target
    feature_cols = [c for c in df_features.columns if c != 'target']
    X = df_features[feature_cols]
    y = df_features['target']
    
    print(f"  📊 Features: {len(feature_cols)}")
    print(f"  📊 Home win rate: {y.mean():.1%}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print(f"\n  🎯 Training XGBoost (no leakage)...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    print(f"\n  📊 RESULTS (No Data Leakage):")
    print(f"     Accuracy: {accuracy:.1%}")
    print(f"     CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    print(f"     Precision: {precision:.1%}")
    print(f"     Recall: {recall:.1%}")
    print(f"     AUC: {auc:.3f}")
    print(f"     Brier Score: {brier:.3f}")
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n  🎯 Top Features:")
    for feat, imp in top_features:
        print(f"     • {feat}: {imp:.4f}")
    
    # Save model
    model_dir = MODELS_DIR / "nba"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "moneyline_noleak_v3.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'features': feature_cols,
            'metrics': {
                'accuracy': accuracy,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'brier_score': brier,
            },
            'top_features': top_features,
            'version': '3.0-noleak',
            'trained_at': datetime.now().isoformat(),
        }, f)
    
    print(f"\n  ✅ Saved to {model_path}")
    
    # Also run fatigue analysis now
    print("\n" + "-"*40)
    print("  🏋️ FATIGUE IMPACT ANALYSIS")
    print("-"*40)
    
    # Train without fatigue features
    non_fatigue = [c for c in feature_cols if not any(f in c for f in 
        ['rest', 'b2b', 'day_of_week', 'month'])]
    
    X_base = df_features[non_fatigue]
    X_train_b, X_test_b, _, _ = train_test_split(X_base, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler_b = RobustScaler()
    X_train_b_scaled = scaler_b.fit_transform(X_train_b)
    X_test_b_scaled = scaler_b.transform(X_test_b)
    
    model_base = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    model_base.fit(X_train_b_scaled, y_train, verbose=False)
    
    y_pred_base = model_base.predict(X_test_b_scaled)
    acc_base = accuracy_score(y_test, y_pred_base)
    
    improvement = accuracy - acc_base
    print(f"  With fatigue features: {accuracy:.1%}")
    print(f"  Without fatigue: {acc_base:.1%}")
    print(f"  📈 Fatigue improvement: {improvement:+.1%}")
    
    return {
        'accuracy': accuracy,
        'accuracy_baseline': acc_base,
        'fatigue_improvement': improvement,
        'auc': auc,
        'top_features': top_features,
    }


if __name__ == "__main__":
    train_nba_fixed()
