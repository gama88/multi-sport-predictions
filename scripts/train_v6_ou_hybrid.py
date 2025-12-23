"""
V6-OU HYBRID Model
==================
Combines behavioral proxy features WITH pace/tempo features.
The hypothesis: O/U needs BOTH team quality AND scoring pace.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "nba"


def load_training_data():
    """Load enhanced NBA data."""
    path = DATA_DIR / "nba_new" / "nba_training_games_20251205.csv"
    
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
    })
    
    return games.sort_values('GAME_DATE_EST').reset_index(drop=True)


def build_histories(df):
    """Build comprehensive team histories."""
    df = df.sort_values('GAME_DATE_EST').reset_index(drop=True)
    
    teams = set(df['HOME_TEAM_ID'].unique()) | set(df['VISITOR_TEAM_ID'].unique())
    histories = {t: [] for t in teams}
    
    for _, row in df.iterrows():
        home_id = row['HOME_TEAM_ID']
        away_id = row['VISITOR_TEAM_ID']
        date = row['GAME_DATE_EST']
        
        home_pts = row.get('PTS_home', 0) or 0
        away_pts = row.get('PTS_away', 0) or 0
        
        home_game = {
            'date': date,
            'pts_scored': home_pts,
            'pts_allowed': away_pts,
            'total': home_pts + away_pts,
            'fg_pct': row.get('FG_PCT_home', 0.45) or 0.45,
            'ast': row.get('AST_home', 24) or 24,
            'stl': row.get('STL_home', 7) or 7,
            'blk': row.get('BLK_home', 5) or 5,
            'tov': row.get('TOV_home', 14) or 14,
            'pf': row.get('PF_home', 20) or 20,
            'reb': row.get('REB_home', 40) or 40,
        }
        
        away_game = {
            'date': date,
            'pts_scored': away_pts,
            'pts_allowed': home_pts,
            'total': home_pts + away_pts,
            'fg_pct': row.get('FG_PCT_away', 0.45) or 0.45,
            'ast': row.get('AST_away', 24) or 24,
            'stl': row.get('STL_away', 7) or 7,
            'blk': row.get('BLK_away', 5) or 5,
            'tov': row.get('TOV_away', 14) or 14,
            'pf': row.get('PF_away', 20) or 20,
            'reb': row.get('REB_away', 40) or 40,
        }
        
        histories[home_id].append(home_game)
        histories[away_id].append(away_game)
    
    return histories


def get_stats(history, n=20):
    """Get comprehensive stats."""
    if len(history) < 3:
        return None
    
    recent = history[-n:] if len(history) >= n else history
    
    def safe_mean(vals):
        valid = [v for v in vals if v and v > 0]
        return np.mean(valid) if valid else 0
    
    def safe_std(vals):
        valid = [v for v in vals if v and v > 0]
        return np.std(valid) if len(valid) > 1 else 0
    
    totals = [g['total'] for g in recent]
    avg_total = safe_mean(totals)
    
    return {
        'pts_scored': safe_mean([g['pts_scored'] for g in recent]),
        'pts_allowed': safe_mean([g['pts_allowed'] for g in recent]),
        'pts_std': safe_std([g['pts_scored'] for g in recent]),
        'total_mean': avg_total,
        'total_std': safe_std(totals),
        'fg_pct': safe_mean([g['fg_pct'] for g in recent]),
        'ast': safe_mean([g['ast'] for g in recent]),
        'stl': safe_mean([g['stl'] for g in recent]),
        'blk': safe_mean([g['blk'] for g in recent]),
        'tov': safe_mean([g['tov'] for g in recent]),
        'pf': safe_mean([g['pf'] for g in recent]),
        'reb': safe_mean([g['reb'] for g in recent]),
        'over_rate': sum(1 for t in totals if t > avg_total) / len(totals),
        'recent_pts': safe_mean([g['pts_scored'] for g in recent[-5:]]),
        'recent_allowed': safe_mean([g['pts_allowed'] for g in recent[-5:]]),
    }


def create_hybrid_features(df, histories):
    """Create HYBRID features - both behavioral + O/U specific."""
    print("  Creating hybrid features...")
    
    features = []
    targets = []
    valid_idx = []
    
    for idx, row in df.iterrows():
        home_id = row['HOME_TEAM_ID']
        away_id = row['VISITOR_TEAM_ID']
        date = row['GAME_DATE_EST']
        
        home_hist = [g for g in histories.get(home_id, []) if g['date'] < date]
        away_hist = [g for g in histories.get(away_id, []) if g['date'] < date]
        
        if len(home_hist) < 5 or len(away_hist) < 5:
            continue
        
        h = get_stats(home_hist, 20)
        a = get_stats(away_hist, 20)
        
        if h is None or a is None:
            continue
        
        f = {}
        
        # === PACE FEATURES (Combined scoring)===
        f['combined_pts_scored'] = (h['pts_scored'] + a['pts_scored']) / 220
        f['combined_pts_allowed'] = (h['pts_allowed'] + a['pts_allowed']) / 220
        f['expected_total'] = (h['pts_scored'] + a['pts_allowed'] + a['pts_scored'] + h['pts_allowed']) / 2 / 220
        f['combined_total_avg'] = (h['total_mean'] + a['total_mean']) / 2 / 220
        
        # === EFFICIENCY ===
        f['combined_fg_pct'] = (h['fg_pct'] + a['fg_pct']) / 2
        f['combined_ast'] = (h['ast'] + a['ast']) / 50
        f['combined_tov'] = (h['tov'] + a['tov']) / 30
        f['combined_pf'] = (h['pf'] + a['pf']) / 40  # More fouls = more FTs
        
        # === BEHAVIORAL (Defensive quality affects total) ===
        f['combined_stl'] = (h['stl'] + a['stl']) / 20
        f['combined_blk'] = (h['blk'] + a['blk']) / 12
        f['defensive_quality'] = f['combined_stl'] + f['combined_blk']
        
        # === O/U TENDENCY ===
        f['home_over_rate'] = h['over_rate']
        f['away_over_rate'] = a['over_rate']
        f['combined_over_rate'] = (h['over_rate'] + a['over_rate']) / 2
        
        # === VARIANCE (High variance = unpredictable totals) ===
        f['home_variance'] = h['pts_std'] / 15
        f['away_variance'] = a['pts_std'] / 15
        f['combined_variance'] = (h['total_std'] + a['total_std']) / 20
        
        # === RECENT FORM ===
        f['home_recent_scoring'] = h['recent_pts'] / 120
        f['away_recent_scoring'] = a['recent_pts'] / 120
        f['home_recent_defense'] = 1 - h['recent_allowed'] / 120
        f['away_recent_defense'] = 1 - a['recent_allowed'] / 120
        
        # === MATCHUP QUALITY ===
        # Good offense vs bad defense = more points
        f['home_off_vs_away_def'] = h['pts_scored'] / max(a['pts_allowed'], 80)
        f['away_off_vs_home_def'] = a['pts_scored'] / max(h['pts_allowed'], 80)
        f['matchup_quality'] = (f['home_off_vs_away_def'] + f['away_off_vs_home_def']) / 2
        
        features.append(f)
        valid_idx.append(idx)
        
        # Target
        actual_total = (row.get('PTS_home', 0) or 0) + (row.get('PTS_away', 0) or 0)
        predicted_total = (h['pts_scored'] + a['pts_allowed'] + a['pts_scored'] + h['pts_allowed']) / 2
        targets.append(1 if actual_total > predicted_total else 0)
        
        if len(features) % 2000 == 0:
            print(f"    Processed {len(features)} games...")
    
    print(f"  Created {len(features)} feature rows")
    return pd.DataFrame(features), targets, valid_idx


def train_hybrid():
    """Train hybrid O/U model."""
    print("\n" + "="*60)
    print("V6-OU HYBRID MODEL (Behavioral + Pace)")
    print("="*60)
    
    # Load data
    games = load_training_data()
    print(f"  Loaded {len(games)} games")
    
    # Build
    histories = build_histories(games)
    X, y, valid_idx = create_hybrid_features(games, histories)
    
    print(f"\n  Features: {X.shape}")
    
    # Scale and split
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = np.array(y[:split]), np.array(y[split:])
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train XGBoost
    print("\n  Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,  # Shallower to prevent overfitting
        learning_rate=0.01,
        reg_lambda=10.0,  # More regularization
        reg_alpha=2.0,
        subsample=0.7,
        colsample_bytree=0.6,
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Train LightGBM
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.01,
        reg_lambda=10.0,
        reg_alpha=2.0,
        subsample=0.7,
        colsample_bytree=0.6,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    # Ensemble
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
    
    pred = (ensemble_proba >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, ensemble_proba)
    
    xgb_acc = accuracy_score(y_test, (xgb_proba >= 0.5).astype(int))
    lgb_acc = accuracy_score(y_test, (lgb_proba >= 0.5).astype(int))
    
    print(f"\n  === V6-OU HYBRID RESULTS ===")
    print(f"  XGBoost:  {xgb_acc:.1%}")
    print(f"  LightGBM: {lgb_acc:.1%}")
    print(f"  Ensemble: {accuracy:.1%}")
    print(f"  AUC:      {auc:.4f}")
    
    # Feature importance
    print(f"\n  Top 10 Features:")
    combined_imp = (xgb_model.feature_importances_ + lgb_model.feature_importances_) / 2
    for i, (feat, imp) in enumerate(sorted(zip(X.columns, combined_imp), key=lambda x: -x[1])[:10]):
        print(f"    {feat}: {imp:.2f}")
    
    print(f"\n  === COMPARISON ===")
    print(f"  V6 Behavioral O/U: 55.0%")
    print(f"  V6-OU Specialized: 52.2%")
    print(f"  V6-OU HYBRID:      {accuracy:.1%}")
    
    return accuracy


if __name__ == "__main__":
    train_hybrid()
