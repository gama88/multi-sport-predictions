"""
Multi-Sport Specialized Model Testing
======================================
Tests specialized models per bet type for ALL sports.
Goal: Find the optimal approach for each sport + bet type combination.
"""

import pandas as pd
import numpy as np
from pathlib import Path
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
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# V6 baselines for comparison
V6_BASELINES = {
    'nba': {'moneyline': 0.654, 'spread': 0.734, 'total': 0.55},
    'nfl': {'moneyline': 0.651, 'spread': 0.652, 'total': 0.53},
    'nhl': {'moneyline': 0.512, 'spread': 0.591, 'total': 0.56},
    'mlb': {'moneyline': 0.532, 'spread': 0.556, 'total': 0.53},
    'soccer': {'moneyline': 0.643, 'spread': 0.753, 'total': 0.55},
    'tennis': {'moneyline': 0.628},
}


def load_sport_data(sport):
    """Load data for a specific sport."""
    path = DATA_DIR / sport
    
    if sport == 'nba':
        df = pd.read_csv(path / "games.csv")
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['game_date_est'])
        df = df.dropna(subset=['pts_home', 'pts_away'])
        df = df[df['date'].dt.year >= 2015]
    
    elif sport == 'nfl':
        df = pd.read_csv(path / "spreadspoke_scores.csv")
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['schedule_date'])
        df = df.dropna(subset=['score_home', 'score_away'])
        df = df[df['date'].dt.year >= 2015]
        # Rename to standard columns
        df['pts_home'] = df['score_home']
        df['pts_away'] = df['score_away']
        df['home_team_id'] = df['team_home']
        df['visitor_team_id'] = df['team_away']
    
    elif sport == 'nhl':
        df = pd.read_csv(path / "game.csv")
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date_time_gmt'])
        df = df.dropna(subset=['home_goals', 'away_goals'])
        df = df[df['date'].dt.year >= 2015]
        df['pts_home'] = df['home_goals']
        df['pts_away'] = df['away_goals']
        df['home_team_id'] = df['home_team_id']
        df['visitor_team_id'] = df['away_team_id']
    
    elif sport == 'mlb':
        df = pd.read_csv(path / "games.csv")
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['home_final_score', 'away_final_score'])
        df = df[df['date'].dt.year >= 2015]
        df['pts_home'] = df['home_final_score']
        df['pts_away'] = df['away_final_score']
        df['home_team_id'] = df['home_team']
        df['visitor_team_id'] = df['away_team']
    
    elif sport == 'soccer':
        df = pd.read_csv(path / "games.csv")
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['home_club_goals', 'away_club_goals'])
        df = df[df['date'].dt.year >= 2015]
        df['pts_home'] = df['home_club_goals']
        df['pts_away'] = df['away_club_goals']
        df['home_team_id'] = df['home_club_id']
        df['visitor_team_id'] = df['away_club_id']
    
    elif sport == 'tennis':
        df = pd.read_csv(path / "atp_matches.csv")
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['winner_id', 'loser_id'])
        # Tennis is player-based, different format
        return df.sort_values('date').reset_index(drop=True), 'tennis_special'
    
    else:
        return None, None
    
    return df.sort_values('date').reset_index(drop=True), 'standard'


def get_team_stats(df, team_id, idx, window=15):
    """Universal team stats getter."""
    prev = df.iloc[:idx]
    home_mask = prev['home_team_id'] == team_id
    away_mask = prev['visitor_team_id'] == team_id
    team_games = prev[home_mask | away_mask].tail(window)
    
    if len(team_games) < 3:
        return None
    
    pts_for, pts_against = [], []
    wins = []
    
    for _, g in team_games.iterrows():
        is_home = g['home_team_id'] == team_id
        if is_home:
            pts_for.append(g['pts_home'])
            pts_against.append(g['pts_away'])
            wins.append(1 if g['pts_home'] > g['pts_away'] else 0)
        else:
            pts_for.append(g['pts_away'])
            pts_against.append(g['pts_home'])
            wins.append(1 if g['pts_away'] > g['pts_home'] else 0)
    
    n = len(pts_for)
    
    return {
        'win_pct': np.mean(wins),
        'pts_mean': np.mean(pts_for),
        'pts_against': np.mean(pts_against),
        'net_rating': np.mean(pts_for) - np.mean(pts_against),
        'pts_std': np.std(pts_for) if n > 1 else 0,
        'pace': np.mean(pts_for) + np.mean(pts_against),
        'pts_trend': np.mean(pts_for[-3:]) - np.mean(pts_for) if n >= 3 else 0,
        'games': n
    }


def create_features(df, sport, bet_type):
    """Create features optimized for sport + bet type combination."""
    features = []
    targets = []
    
    for idx in range(len(df)):
        if idx < 100:
            continue
        
        row = df.iloc[idx]
        home = get_team_stats(df, row['home_team_id'], idx)
        away = get_team_stats(df, row['visitor_team_id'], idx)
        
        if home is None or away is None:
            continue
        
        home_pts = row['pts_home']
        away_pts = row['pts_away']
        
        if bet_type == 'moneyline':
            f = {
                'win_pct_diff': home['win_pct'] - away['win_pct'],
                'net_diff': home['net_rating'] - away['net_rating'],
                'pts_diff': home['pts_mean'] - away['pts_mean'],
                'home_win_pct': home['win_pct'],
                'away_win_pct': away['win_pct'],
                'trend_diff': home['pts_trend'] - away['pts_trend'],
            }
            target = 1.0 if home_pts > away_pts else 0.0
            
        elif bet_type == 'spread':
            f = {
                'net_diff': home['net_rating'] - away['net_rating'],
                'pts_diff': home['pts_mean'] - away['pts_mean'],
                'defense_diff': away['pts_against'] - home['pts_against'],
                'consistency': 1 / (1 + home['pts_std'] + away['pts_std']),
                'mismatch': abs(home['net_rating'] - away['net_rating']),
            }
            implied = (home['net_rating'] - away['net_rating']) * 0.4
            target = 1.0 if (home_pts - away_pts) > implied else 0.0
            
        elif bet_type == 'total':
            f = {
                'combined_pace': (home['pace'] + away['pace']) / 2,
                'combined_pts': home['pts_mean'] + away['pts_mean'],
                'defense_allowing': home['pts_against'] + away['pts_against'],
                'total_variance': home['pts_std'] + away['pts_std'],
                'pts_trend_comb': home['pts_trend'] + away['pts_trend'],
            }
            expected = (home['pts_mean'] + away['pts_mean']) * 0.98
            target = 1.0 if (home_pts + away_pts) > expected else 0.0
        
        features.append(f)
        targets.append(target)
    
    if not features:
        return None, None
    
    return pd.DataFrame(features), np.array(targets)


def train_model(X, y):
    """Train and evaluate model."""
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    split = int(len(X) * 0.85)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    
    if len(X_test) < 50:
        return None
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        reg_lambda=5.0, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        reg_lambda=5.0, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    ensemble = 0.5 * xgb_pred + 0.5 * lgb_pred
    
    acc = accuracy_score(y_test, (ensemble > 0.5).astype(int))
    
    return {'accuracy': acc, 'test_size': len(y_test)}


def test_tennis_moneyline(df):
    """Special handling for tennis (player-based)."""
    # Build player histories
    features = []
    targets = []
    
    for idx in range(len(df)):
        if idx < 100:
            continue
        
        row = df.iloc[idx]
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        
        # Get history
        prev = df.iloc[:idx]
        
        def get_player_stats(player_id):
            as_winner = prev[prev['winner_id'] == player_id]
            as_loser = prev[prev['loser_id'] == player_id]
            total = len(as_winner) + len(as_loser)
            if total < 3:
                return None
            return {
                'win_pct': len(as_winner) / total,
                'matches': total,
                'rank': row.get('winner_rank', 100) if player_id == winner_id else row.get('loser_rank', 100)
            }
        
        w = get_player_stats(winner_id)
        l = get_player_stats(loser_id)
        
        if w is None or l is None:
            continue
        
        # Random assignment of p1/p2
        if np.random.random() > 0.5:
            p1, p2 = w, l
            target = 1.0
        else:
            p1, p2 = l, w
            target = 0.0
        
        features.append({
            'win_pct_diff': p1['win_pct'] - p2['win_pct'],
            'rank_diff': (p2['rank'] - p1['rank']) / 100 if p2['rank'] < 9999 and p1['rank'] < 9999 else 0,
            'p1_win_pct': p1['win_pct'],
            'p2_win_pct': p2['win_pct'],
        })
        targets.append(target)
    
    if not features:
        return None
    
    return train_model(pd.DataFrame(features), np.array(targets))


def main():
    print("\n" + "="*70)
    print("🏆 MULTI-SPORT SPECIALIZED MODEL TESTING")
    print("="*70)
    print("Testing optimal approach for each sport + bet type combination\n")
    
    results = {}
    
    sports = ['nba', 'nfl', 'nhl', 'mlb', 'soccer', 'tennis']
    bet_types = ['moneyline', 'spread', 'total']
    
    for sport in sports:
        print(f"\n{'='*50}")
        print(f"🏅 {sport.upper()}")
        print('='*50)
        
        df, data_type = load_sport_data(sport)
        if df is None:
            print(f"  ⚠️  No data found for {sport}")
            continue
        
        print(f"  Loaded {len(df)} games")
        results[sport] = {}
        
        for bet_type in bet_types:
            # Skip bet types not applicable for this sport
            if sport == 'tennis' and bet_type in ['spread', 'total']:
                continue
            if sport not in V6_BASELINES or bet_type not in V6_BASELINES.get(sport, {}):
                continue
            
            print(f"\n  📊 {bet_type.upper()}...")
            
            if sport == 'tennis':
                result = test_tennis_moneyline(df)
            else:
                X, y = create_features(df, sport, bet_type)
                if X is None or len(X) < 200:
                    print(f"    ⚠️  Not enough data")
                    continue
                result = train_model(X, y)
            
            if result is None:
                print(f"    ⚠️  Training failed")
                continue
            
            v6 = V6_BASELINES[sport][bet_type]
            specialized = result['accuracy']
            diff = (specialized - v6) * 100
            symbol = '↑' if diff > 0 else '↓'
            marker = ' 🎉' if diff > 2 else (' ✓' if diff > 0 else '')
            
            print(f"    V6: {v6:.1%} → Specialized: {specialized:.1%} | {symbol}{abs(diff):.1f}pp{marker}")
            
            results[sport][bet_type] = {
                'v6': v6,
                'specialized': specialized,
                'improvement': diff,
                'best': 'specialized' if diff > 0 else 'v6'
            }
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY: BEST MODEL PER SPORT + BET TYPE")
    print("="*70)
    
    improvements = []
    
    for sport, bet_types_dict in results.items():
        for bt, data in bet_types_dict.items():
            best = data['best']
            best_acc = data['specialized'] if best == 'specialized' else data['v6']
            improvement = data['improvement']
            
            marker = '🎉' if improvement > 2 else ('✓' if improvement > 0 else '')
            print(f"  {sport.upper():8} {bt:10} | Best: {best:12} | {best_acc:.1%} {marker}")
            
            if improvement > 0:
                improvements.append((sport, bt, improvement))
    
    if improvements:
        print(f"\n🏆 IMPROVEMENTS FOUND:")
        for sport, bt, imp in sorted(improvements, key=lambda x: -x[2]):
            print(f"  {sport.upper()} {bt}: +{imp:.1f}pp with specialized approach")
    
    # Save results
    with open(MODELS_DIR / "multi_sport_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Saved to: {MODELS_DIR / 'multi_sport_comparison.json'}")


if __name__ == "__main__":
    main()
