"""
V13 Specialized Models Per Bet Type
====================================
Key insight from experiments: Different approaches work best for different bet types!

- MONEYLINE: V6 approach (behavioral + outcome features)  
- SPREAD: V6 approach (already 73.4%)
- TOTAL/O/U: New specialized approach (cumulative stats + pace features)

This script trains the OPTIMAL model for each bet type.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "nba"
MODELS_DIR = BASE_DIR / "models"


def load_data():
    """Load NBA data."""
    print("\n📊 Loading NBA data...")
    df = pd.read_csv(DATA_DIR / "games.csv")
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['game_date_est'])
    df = df.dropna(subset=['pts_home', 'pts_away'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['date'].dt.year >= 2015]
    print(f"  Loaded {len(df)} games (2015+)")
    return df


def get_team_stats(df, team_id, idx, window=15):
    """Get team stats for feature engineering."""
    prev = df.iloc[:idx]
    home_mask = prev['home_team_id'] == team_id
    away_mask = prev['visitor_team_id'] == team_id
    team_games = prev[home_mask | away_mask].tail(window)
    
    if len(team_games) < 5:
        return None
    
    pts_for, pts_against = [], []
    fg_pct, fg3_pct, ft_pct = [], [], []
    reb, ast = [], []
    wins = []
    
    for _, g in team_games.iterrows():
        is_home = g['home_team_id'] == team_id
        if is_home:
            pts_for.append(g['pts_home'])
            pts_against.append(g['pts_away'])
            fg_pct.append(g.get('fg_pct_home', 0.45))
            fg3_pct.append(g.get('fg3_pct_home', 0.35))
            ft_pct.append(g.get('ft_pct_home', 0.75))
            reb.append(g.get('reb_home', 42))
            ast.append(g.get('ast_home', 22))
            wins.append(1 if g['pts_home'] > g['pts_away'] else 0)
        else:
            pts_for.append(g['pts_away'])
            pts_against.append(g['pts_home'])
            fg_pct.append(g.get('fg_pct_away', 0.45))
            fg3_pct.append(g.get('fg3_pct_away', 0.35))
            ft_pct.append(g.get('ft_pct_away', 0.75))
            reb.append(g.get('reb_away', 42))
            ast.append(g.get('ast_away', 22))
            wins.append(1 if g['pts_away'] > g['pts_home'] else 0)
    
    n = len(pts_for)
    
    return {
        # Outcome-based
        'win_pct': np.mean(wins),
        'pts_mean': np.mean(pts_for),
        'pts_against': np.mean(pts_against),
        'net_rating': np.mean(pts_for) - np.mean(pts_against),
        
        # Behavioral
        'fg_pct': np.mean(fg_pct),
        'fg3_pct': np.mean(fg3_pct),
        'ft_pct': np.mean(ft_pct),
        'reb': np.mean(reb),
        'ast': np.mean(ast),
        
        # Variability
        'pts_std': np.std(pts_for),
        'pts_against_std': np.std(pts_against),
        
        # Pace proxy
        'pace': np.mean(pts_for) + np.mean(pts_against),
        
        # Trend
        'pts_trend': np.mean(pts_for[-5:]) - np.mean(pts_for) if n >= 5 else 0,
        'win_trend': np.mean(wins[-5:]) - np.mean(wins) if n >= 5 else 0,
        
        'games': n
    }


def create_features_for_bet_type(df, bet_type):
    """
    Create SPECIALIZED features based on bet type.
    Each bet type gets optimized features.
    """
    print(f"\n🔧 Creating features for {bet_type.upper()}...")
    
    features = []
    targets = []
    
    for idx in range(len(df)):
        if idx < 150:
            continue
        
        row = df.iloc[idx]
        home_id = row['home_team_id']
        away_id = row['visitor_team_id']
        
        home = get_team_stats(df, home_id, idx)
        away = get_team_stats(df, away_id, idx)
        
        if home is None or away is None:
            continue
        
        # Calculate target first
        home_pts = row['pts_home']
        away_pts = row['pts_away']
        total = home_pts + away_pts
        
        if bet_type == 'moneyline':
            # MONEYLINE features: Focus on WIN factors
            f = {
                'win_pct_diff': home['win_pct'] - away['win_pct'],
                'net_rating_diff': home['net_rating'] - away['net_rating'],
                'pts_diff': home['pts_mean'] - away['pts_mean'],
                'fg_pct_diff': home['fg_pct'] - away['fg_pct'],
                'fg3_pct_diff': home['fg3_pct'] - away['fg3_pct'],
                'reb_diff': home['reb'] - away['reb'],
                'ast_diff': home['ast'] - away['ast'],
                'win_trend_diff': home['win_trend'] - away['win_trend'],
                'home_win_pct': home['win_pct'],
                'away_win_pct': away['win_pct'],
                'home_net': home['net_rating'],
                'away_net': away['net_rating'],
            }
            target = 1.0 if home_pts > away_pts else 0.0
            
        elif bet_type == 'spread':
            # SPREAD features: Focus on MARGIN factors
            f = {
                'net_rating_diff': home['net_rating'] - away['net_rating'],
                'pts_diff': home['pts_mean'] - away['pts_mean'],
                'defense_diff': away['pts_against'] - home['pts_against'],  # Lower is better
                'consistency': 1 / (1 + home['pts_std'] + away['pts_std']),
                'fg_pct_diff': home['fg_pct'] - away['fg_pct'],
                'reb_diff': home['reb'] - away['reb'],
                'pts_trend_diff': home['pts_trend'] - away['pts_trend'],
                'home_net': home['net_rating'],
                'away_net': away['net_rating'],
                'mismatch': abs(home['net_rating'] - away['net_rating']),
            }
            # Target: Cover implied spread
            implied_spread = (home['net_rating'] - away['net_rating']) * 0.4
            target = 1.0 if (home_pts - away_pts) > implied_spread else 0.0
            
        elif bet_type == 'total':
            # TOTAL/O-U features: Focus on PACE and SCORING
            # This is where other models beat V6!
            f = {
                # Pace features (KEY for totals!)
                'combined_pace': (home['pace'] + away['pace']) / 2,
                'home_pace': home['pace'],
                'away_pace': away['pace'],
                'pace_diff': abs(home['pace'] - away['pace']),
                
                # Scoring features
                'combined_pts': (home['pts_mean'] + away['pts_mean']),
                'combined_pts_against': (home['pts_against'] + away['pts_against']),
                'total_fg_pct': (home['fg_pct'] + away['fg_pct']) / 2,
                'total_fg3_pct': (home['fg3_pct'] + away['fg3_pct']) / 2,
                
                # Variability (more variance = harder to predict)
                'total_variance': home['pts_std'] + away['pts_std'],
                
                # Defense allowing points
                'defense_allowing': home['pts_against'] + away['pts_against'],
                
                # Trend in scoring
                'pts_trend_combined': home['pts_trend'] + away['pts_trend'],
                
                # Expected total
                'expected_total': (home['pts_mean'] + away['pts_mean']) * 0.98,
                
                # Game pace proxy
                'rebounding_total': home['reb'] + away['reb'],
            }
            # Target: Over predicted total
            expected = (home['pts_mean'] + away['pts_mean']) * 0.98
            target = 1.0 if total > expected else 0.0
        
        else:  # contracts (same as moneyline)
            f = {
                'win_pct_diff': home['win_pct'] - away['win_pct'],
                'net_rating_diff': home['net_rating'] - away['net_rating'],
                'pts_diff': home['pts_mean'] - away['pts_mean'],
                'fg_pct_diff': home['fg_pct'] - away['fg_pct'],
                'home_win_pct': home['win_pct'],
                'away_win_pct': away['win_pct'],
            }
            target = 1.0 if home_pts > away_pts else 0.0
        
        features.append(f)
        targets.append(target)
        
        if len(features) % 3000 == 0:
            print(f"    Processed {len(features)} games...")
    
    X = pd.DataFrame(features)
    y = np.array(targets)
    
    print(f"  Created {len(X)} samples with {len(X.columns)} {bet_type}-specific features")
    return X, y


def train_specialized_model(X, y, bet_type):
    """Train specialized model for this bet type."""
    print(f"\n🧠 Training specialized {bet_type.upper()} model...")
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    split = int(len(X) * 0.85)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Customize hyperparams per bet type
    if bet_type == 'total':
        # Lower regularization for totals (more signal)
        xgb_params = dict(n_estimators=600, max_depth=5, learning_rate=0.015,
                         reg_lambda=3.0, reg_alpha=0.5)
    else:
        # Standard params for moneyline/spread
        xgb_params = dict(n_estimators=500, max_depth=4, learning_rate=0.02,
                         reg_lambda=5.0, reg_alpha=1.0)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        **xgb_params,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        **xgb_params,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    # Ensemble
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    ensemble = 0.5 * xgb_pred + 0.5 * lgb_pred
    
    # Calibrate
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(ensemble, y_test)
    
    # Evaluate
    acc = accuracy_score(y_test, (ensemble > 0.5).astype(int))
    auc = roc_auc_score(y_test, ensemble)
    
    xgb_acc = accuracy_score(y_test, (xgb_pred > 0.5).astype(int))
    lgb_acc = accuracy_score(y_test, (lgb_pred > 0.5).astype(int))
    
    print(f"  XGB: {xgb_acc:.1%}, LGB: {lgb_acc:.1%}, Ensemble: {acc:.1%} (AUC: {auc:.4f})")
    
    return {
        'xgb': xgb_model,
        'lgb': lgb_model,
        'scaler': scaler,
        'calibrator': calibrator,
        'accuracy': acc,
        'auc': auc,
        'features': list(X.columns)
    }


def main():
    print("\n" + "="*60)
    print("🎯 V13 SPECIALIZED MODELS PER BET TYPE")
    print("="*60)
    print("Creating the BEST model for each bet type")
    
    df = load_data()
    
    results = {}
    bet_types = ['moneyline', 'spread', 'total']
    
    for bet_type in bet_types:
        X, y = create_features_for_bet_type(df, bet_type)
        result = train_specialized_model(X, y, bet_type)
        results[bet_type] = result
    
    # Add contracts (uses moneyline model)
    results['contracts'] = results['moneyline']
    
    # Summary
    print("\n" + "="*60)
    print("📊 V13 SPECIALIZED RESULTS vs V6 GENERAL")
    print("="*60)
    
    v6 = {'moneyline': 0.654, 'spread': 0.734, 'total': 0.55}
    
    for bt in bet_types:
        v6_acc = v6[bt]
        v13_acc = results[bt]['accuracy']
        diff = (v13_acc - v6_acc) * 100
        symbol = '↑' if diff > 0 else '↓'
        marker = ' 🎉' if diff > 0 else ''
        print(f"  {bt.upper():10} | V6: {v6_acc:.1%} → V13: {v13_acc:.1%} | {symbol}{abs(diff):.1f}pp{marker}")
    
    # Save combined model
    with open(MODELS_DIR / "v13_specialized.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Save metrics
    metrics = {bt: {'accuracy': r['accuracy'], 'auc': r['auc'], 'features': r['features']} 
               for bt, r in results.items() if bt != 'contracts'}
    with open(MODELS_DIR / "v13_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n  Saved to: {MODELS_DIR / 'v13_specialized.pkl'}")
    
    return results


if __name__ == "__main__":
    results = main()
