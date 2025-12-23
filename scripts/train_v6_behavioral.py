"""
V6 Behavioral Proxy Basketball Model
=====================================
Target: 66.5%+ accuracy

Key Innovation: "Behavioral Proxy" features that capture what an AI watching games would observe:
- Is the team tired? (fatigue proxies)
- Are they disciplined defensively? (defensive discipline)
- Do they perform under pressure? (clutch/pressure)
- Do they share the ball well? (spacing/flow)
- Is the team chemistry good? (chemistry indicators)

Architecture:
- XGBoost + LightGBM ensemble (50/50 weighted)
- RobustScaler for outlier-resistant scaling
- Time-based train/test split (80/20)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "nba"
MODELS_DIR = BASE_DIR / "models"


class BehavioralProxyFeatureEngine:
    """
    Creates 20 "behavioral proxy" features as DIFFERENTIALS (home - away).
    These capture what an AI watching games would observe about team behavior.
    """
    
    def __init__(self):
        self.team_stats_cache = {}
        self.team_game_dates = {}
        
    def build_team_history(self, df):
        """Build historical stats for each team game-by-game."""
        print("  Building team history cache...")
        
        # Sort by date
        df = df.sort_values('GAME_DATE_EST').reset_index(drop=True)
        
        # Initialize team histories
        teams = set(df['HOME_TEAM_ID'].unique()) | set(df['VISITOR_TEAM_ID'].unique())
        team_histories = {team: [] for team in teams}
        
        for idx, row in df.iterrows():
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            game_date = pd.to_datetime(row['GAME_DATE_EST'])
            
            # Home team stats
            home_game = {
                'date': game_date,
                'pts': row.get('PTS_home', 0) or 0,
                'fg_pct': row.get('FG_PCT_home', 0) or 0,
                'fg3_pct': row.get('FG3_PCT_home', 0) or 0,
                'ft_pct': row.get('FT_PCT_home', 0) or 0,
                'reb': row.get('REB_home', 0) or 0,
                'ast': row.get('AST_home', 0) or 0,
                'is_home': True,
                'won': row.get('HOME_TEAM_WINS', 0) == 1,
                'opp_pts': row.get('PTS_away', 0) or 0,
            }
            
            # Away team stats
            away_game = {
                'date': game_date,
                'pts': row.get('PTS_away', 0) or 0,
                'fg_pct': row.get('FG_PCT_away', 0) or 0,
                'fg3_pct': row.get('FG3_PCT_away', 0) or 0,
                'ft_pct': row.get('FT_PCT_away', 0) or 0,
                'reb': row.get('REB_away', 0) or 0,
                'ast': row.get('AST_away', 0) or 0,
                'is_home': False,
                'won': row.get('HOME_TEAM_WINS', 0) == 0,
                'opp_pts': row.get('PTS_home', 0) or 0,
            }
            
            team_histories[home_id].append(home_game)
            team_histories[away_id].append(away_game)
        
        return team_histories
    
    def get_team_stats(self, team_history, n_games=20):
        """Get aggregated stats from last n games."""
        if len(team_history) < 3:
            return None
            
        recent = team_history[-n_games:] if len(team_history) >= n_games else team_history
        
        pts = [g['pts'] for g in recent if g['pts'] > 0]
        fg_pct = [g['fg_pct'] for g in recent if g['fg_pct'] > 0]
        fg3_pct = [g['fg3_pct'] for g in recent if g['fg3_pct'] > 0]
        ft_pct = [g['ft_pct'] for g in recent if g['ft_pct'] > 0]
        reb = [g['reb'] for g in recent if g['reb'] > 0]
        ast = [g['ast'] for g in recent if g['ast'] > 0]
        wins = [1 if g['won'] else 0 for g in recent]
        opp_pts = [g['opp_pts'] for g in recent if g['opp_pts'] > 0]
        home_games = [g for g in recent if g['is_home']]
        away_games = [g for g in recent if not g['is_home']]
        
        if len(pts) < 3:
            return None
        
        return {
            'pts_mean': np.mean(pts),
            'pts_std': np.std(pts),
            'fg_pct_mean': np.mean(fg_pct) if fg_pct else 0.45,
            'fg3_pct_mean': np.mean(fg3_pct) if fg3_pct else 0.35,
            'ft_pct_mean': np.mean(ft_pct) if ft_pct else 0.75,
            'reb_mean': np.mean(reb) if reb else 40,
            'ast_mean': np.mean(ast) if ast else 24,
            'win_rate': np.mean(wins),
            'games_played': len(recent),
            'opp_pts_mean': np.mean(opp_pts) if opp_pts else 110,
            'opp_pts_std': np.std(opp_pts) if opp_pts else 10,
            'home_win_rate': np.mean([1 if g['won'] else 0 for g in home_games]) if home_games else 0.5,
            'away_win_rate': np.mean([1 if g['won'] else 0 for g in away_games]) if away_games else 0.5,
            'last_5_wins': sum([1 if g['won'] else 0 for g in recent[-5:]]) if len(recent) >= 5 else sum(wins),
            'first_5_wins': sum([1 if g['won'] else 0 for g in recent[:5]]) if len(recent) >= 5 else sum(wins),
        }
    
    def calculate_fatigue_proxies(self, team_history, current_date):
        """
        FATIGUE PROXIES (5 features):
        - avg_minutes_load: normalized
        - back_to_back: consecutive days
        - games_in_7_days: games last week
        - travel_distance: (approximated by away game count)
        - rest_days: days since last game
        """
        if len(team_history) < 2:
            return {'fatigue_back_to_back': 0, 'fatigue_games_7d': 0.5, 
                    'fatigue_rest_days': 0.5, 'fatigue_recent_load': 0.5, 'fatigue_away_load': 0.5}
        
        last_game = team_history[-1]
        last_date = last_game['date']
        
        # Back-to-back detection
        days_rest = (current_date - last_date).days if current_date > last_date else 1
        back_to_back = 1.0 if days_rest <= 1 else 0.0
        
        # Games in last 7 days
        week_ago = current_date - timedelta(days=7)
        games_7d = sum(1 for g in team_history if g['date'] >= week_ago and g['date'] < current_date)
        games_7d_norm = min(games_7d / 4.0, 1.0)
        
        # Rest days normalized
        rest_norm = min(days_rest / 7.0, 1.0)
        
        # Recent load (games in last 14 days)
        two_weeks = current_date - timedelta(days=14)
        recent_games = sum(1 for g in team_history if g['date'] >= two_weeks and g['date'] < current_date)
        recent_load = min(recent_games / 8.0, 1.0)
        
        # Away game load (approximation for travel)
        recent = team_history[-10:] if len(team_history) >= 10 else team_history
        away_count = sum(1 for g in recent if not g['is_home'])
        away_load = away_count / len(recent)
        
        return {
            'fatigue_back_to_back': back_to_back,
            'fatigue_games_7d': games_7d_norm,
            'fatigue_rest_days': rest_norm,
            'fatigue_recent_load': recent_load,
            'fatigue_away_load': away_load
        }
    
    def calculate_defensive_discipline(self, stats):
        """
        DEFENSIVE DISCIPLINE (4 features):
        - pf_per_game: lower fouls = better discipline
        - stl_to_pf_ratio: steals / fouls
        - blk_consistency: consistent blocking
        - def_variance: consistent defense (low points allowed variance)
        """
        if stats is None:
            return {'def_pts_allowed': 0.5, 'def_consistency': 0.5, 
                    'def_variance': 0.5, 'def_margin': 0.5}
        
        # Points allowed (normalized)
        pts_allowed = stats.get('opp_pts_mean', 110)
        pts_allowed_norm = 1.0 - min(pts_allowed / 130, 1.0)  # Lower is better
        
        # Defensive consistency (low variance in points allowed)
        opp_pts_std = stats.get('opp_pts_std', 10)
        def_consistency = 1.0 - min(opp_pts_std / 20.0, 1.0)
        
        # Defensive variance
        def_variance = 1.0 - min(opp_pts_std / 15.0, 1.0)
        
        # Net margin
        pts_scored = stats.get('pts_mean', 110)
        margin = (pts_scored - pts_allowed) / 20.0  # Normalized around 0
        margin_norm = min(max(margin + 0.5, 0), 1.0)  # Shift to 0-1
        
        return {
            'def_pts_allowed': pts_allowed_norm,
            'def_consistency': def_consistency,
            'def_variance': def_variance,
            'def_margin': margin_norm
        }
    
    def calculate_clutch_pressure(self, stats, team_history):
        """
        CLUTCH/PRESSURE (4 features):
        - ft_pct_composure: FT% as mental indicator
        - win_streak: recent wins
        - momentum: trend in win rate
        - close_game_rate: variance indicator
        """
        if stats is None:
            return {'clutch_ft_pct': 0.75, 'clutch_win_streak': 0.5,
                    'clutch_momentum': 0.5, 'clutch_variance': 0.5}
        
        # FT% as composure indicator (mental fortitude under pressure)
        ft_pct = stats.get('ft_pct_mean', 0.75)
        ft_composure = min(ft_pct, 1.0)
        
        # Win streak (last 5 games)
        last_5_wins = stats.get('last_5_wins', 2.5)
        win_streak_norm = last_5_wins / 5.0
        
        # Momentum (improvement trend)
        first_5 = stats.get('first_5_wins', 2.5)
        last_5 = stats.get('last_5_wins', 2.5)
        momentum = (last_5 - first_5) / 5.0 + 0.5  # Centered at 0.5
        momentum = min(max(momentum, 0), 1.0)
        
        # Close game indicator (high variance = more close games)
        pts_std = stats.get('pts_std', 10)
        close_game_rate = min(pts_std / 15.0, 1.0)
        
        return {
            'clutch_ft_pct': ft_composure,
            'clutch_win_streak': win_streak_norm,
            'clutch_momentum': momentum,
            'clutch_variance': close_game_rate
        }
    
    def calculate_spacing_flow(self, stats):
        """
        SPACING/FLOW (4 features):
        - assist_rate: ball movement
        - three_pt_rate: floor spacing
        - tov_rate: ball security (lower = better)
        - offensive_flow: scoring consistency
        """
        if stats is None:
            return {'flow_assist_rate': 0.5, 'flow_3pt_rate': 0.35,
                    'flow_efficiency': 0.5, 'flow_consistency': 0.5}
        
        # Assist rate
        ast_mean = stats.get('ast_mean', 24)
        ast_rate = min(ast_mean / 35.0, 1.0)
        
        # 3-point rate (from 3pt percentage as proxy)
        fg3_pct = stats.get('fg3_pct_mean', 0.35)
        three_rate = min(fg3_pct, 1.0)
        
        # Offensive efficiency (FG%)
        fg_pct = stats.get('fg_pct_mean', 0.45)
        efficiency = min(fg_pct * 2, 1.0)  # Scale up
        
        # Offensive consistency (low variance in scoring)
        pts_std = stats.get('pts_std', 10)
        consistency = 1.0 - min(pts_std / 20.0, 1.0)
        
        return {
            'flow_assist_rate': ast_rate,
            'flow_3pt_rate': three_rate,
            'flow_efficiency': efficiency,
            'flow_consistency': consistency
        }
    
    def calculate_team_chemistry(self, stats, team_history):
        """
        TEAM CHEMISTRY (3 features):
        - ast_per_fgm: unselfish play
        - plus_minus_consistency: stable performance
        - team_experience: games played this season
        """
        if stats is None:
            return {'chem_unselfishness': 0.5, 'chem_stability': 0.5, 'chem_experience': 0.5}
        
        # Unselfishness (ast to scoring ratio)
        ast = stats.get('ast_mean', 24)
        pts = stats.get('pts_mean', 110)
        # Approximate FGM from pts (rough estimate: pts/2.2)
        fgm_approx = pts / 2.2
        unselfish = min(ast / fgm_approx, 1.0) if fgm_approx > 0 else 0.5
        
        # Stability (consistent point margin)
        opp_std = stats.get('opp_pts_std', 10)
        pts_std = stats.get('pts_std', 10)
        combined_std = (opp_std + pts_std) / 2
        stability = 1.0 - min(combined_std / 15.0, 1.0)
        
        # Experience (games played)
        games = stats.get('games_played', 15)
        experience = min(games / 30.0, 1.0)
        
        return {
            'chem_unselfishness': unselfish,
            'chem_stability': stability,
            'chem_experience': experience
        }
    
    def create_behavioral_features(self, df, team_histories):
        """Create all behavioral proxy features for each game."""
        print("  Creating behavioral proxy features...")
        
        features = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            game_date = pd.to_datetime(row['GAME_DATE_EST'])
            
            # Get team histories UP TO this game (no data leakage)
            home_history = [g for g in team_histories[home_id] if g['date'] < game_date]
            away_history = [g for g in team_histories[away_id] if g['date'] < game_date]
            
            # Need at least 5 games of history
            if len(home_history) < 5 or len(away_history) < 5:
                continue
            
            # Get aggregated stats
            home_stats = self.get_team_stats(home_history, n_games=20)
            away_stats = self.get_team_stats(away_history, n_games=20)
            
            if home_stats is None or away_stats is None:
                continue
            
            # Calculate all behavioral proxies
            home_fatigue = self.calculate_fatigue_proxies(home_history, game_date)
            away_fatigue = self.calculate_fatigue_proxies(away_history, game_date)
            
            home_defense = self.calculate_defensive_discipline(home_stats)
            away_defense = self.calculate_defensive_discipline(away_stats)
            
            home_clutch = self.calculate_clutch_pressure(home_stats, home_history)
            away_clutch = self.calculate_clutch_pressure(away_stats, away_history)
            
            home_flow = self.calculate_spacing_flow(home_stats)
            away_flow = self.calculate_spacing_flow(away_stats)
            
            home_chem = self.calculate_team_chemistry(home_stats, home_history)
            away_chem = self.calculate_team_chemistry(away_stats, away_history)
            
            # BUILD FEATURE VECTOR AS DIFFERENTIALS (home - away)
            game_features = {}
            
            # Fatigue proxies (5) - LOWER fatigue for home is BETTER
            for key in home_fatigue:
                # Invert so higher = better for home
                if 'back_to_back' in key or 'games_7d' in key or 'recent_load' in key or 'away_load' in key:
                    # For these, lower is better, so we want home to have lower values
                    game_features[f'{key}_diff'] = away_fatigue[key] - home_fatigue[key]
                else:
                    # For rest_days, higher is better
                    game_features[f'{key}_diff'] = home_fatigue[key] - away_fatigue[key]
            
            # Defensive discipline (4)
            for key in home_defense:
                game_features[f'{key}_diff'] = home_defense[key] - away_defense[key]
            
            # Clutch/pressure (4)
            for key in home_clutch:
                game_features[f'{key}_diff'] = home_clutch[key] - away_clutch[key]
            
            # Spacing/flow (4)
            for key in home_flow:
                game_features[f'{key}_diff'] = home_flow[key] - away_flow[key]
            
            # Team chemistry (3)
            for key in home_chem:
                game_features[f'{key}_diff'] = home_chem[key] - away_chem[key]
            
            # BASE FEATURES (5)
            game_features['win_rate_diff'] = home_stats['win_rate'] - away_stats['win_rate']
            game_features['form_diff'] = (home_stats['last_5_wins'] - away_stats['last_5_wins']) / 5.0
            game_features['pts_diff'] = (home_stats['pts_mean'] - away_stats['pts_mean']) / 10.0
            game_features['home_win_rate'] = home_stats['home_win_rate']
            game_features['away_road_rate'] = away_stats['away_win_rate']
            
            # SEQUENCE FEATURES (additional)
            # Mean stats from last 20 games
            game_features['home_pts_20'] = home_stats['pts_mean'] / 130
            game_features['away_pts_20'] = away_stats['pts_mean'] / 130
            game_features['home_fg_pct'] = home_stats['fg_pct_mean']
            game_features['away_fg_pct'] = away_stats['fg_pct_mean']
            
            # Recent form (last 5)
            home_5 = self.get_team_stats(home_history, n_games=5)
            away_5 = self.get_team_stats(away_history, n_games=5)
            if home_5 and away_5:
                game_features['recent_pts_diff'] = (home_5['pts_mean'] - away_5['pts_mean']) / 10.0
                game_features['recent_form_diff'] = home_5['win_rate'] - away_5['win_rate']
            else:
                game_features['recent_pts_diff'] = 0
                game_features['recent_form_diff'] = 0
            
            features.append(game_features)
            valid_indices.append(idx)
            
            if len(features) % 5000 == 0:
                print(f"    Processed {len(features)} games...")
        
        print(f"  Created features for {len(features)} games")
        return pd.DataFrame(features), valid_indices


class V6BehavioralProxyModel:
    """
    XGBoost + LightGBM ensemble with behavioral proxy features.
    """
    
    def __init__(self):
        self.feature_engine = BehavioralProxyFeatureEngine()
        self.scaler = RobustScaler()
        self.xgb_model = None
        self.lgb_model = None
        self.feature_names = None
        
    def train(self, X, y):
        """Train XGBoost + LightGBM ensemble."""
        print("\n  Training XGBoost + LightGBM Ensemble...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = X.columns.tolist()
        
        # Time-based split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
        
        # XGBoost model
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
        self.xgb_model.fit(X_train, y_train, 
                          eval_set=[(X_test, y_test)],
                          verbose=False)
        
        # LightGBM model
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
        self.lgb_model.fit(X_train, y_train,
                          eval_set=[(X_test, y_test)])
        
        # Ensemble predictions (50/50 weighted average)
        xgb_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X_test)[:, 1]
        ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, ensemble_pred)
        auc = roc_auc_score(y_test, ensemble_proba)
        brier = brier_score_loss(y_test, ensemble_proba)
        logloss = log_loss(y_test, ensemble_proba)
        
        # Individual model metrics
        xgb_acc = accuracy_score(y_test, (xgb_proba >= 0.5).astype(int))
        lgb_acc = accuracy_score(y_test, (lgb_proba >= 0.5).astype(int))
        
        print(f"\n  === V6 BEHAVIORAL PROXY RESULTS ===")
        print(f"  XGBoost Accuracy:  {xgb_acc:.1%}")
        print(f"  LightGBM Accuracy: {lgb_acc:.1%}")
        print(f"  Ensemble Accuracy: {accuracy:.1%}")
        print(f"  AUC-ROC:           {auc:.4f}")
        print(f"  Brier Score:       {brier:.4f}")
        print(f"  Log Loss:          {logloss:.4f}")
        
        return {
            'accuracy': accuracy,
            'xgb_accuracy': xgb_acc,
            'lgb_accuracy': lgb_acc,
            'auc': auc,
            'brier': brier,
            'logloss': logloss,
            'test_size': len(y_test)
        }
    
    def get_feature_importance(self):
        """Get combined feature importance."""
        if self.xgb_model is None or self.lgb_model is None:
            return None
        
        xgb_imp = self.xgb_model.feature_importances_
        lgb_imp = self.lgb_model.feature_importances_
        
        combined = (xgb_imp + lgb_imp) / 2
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': combined
        }).sort_values('importance', ascending=False)
        
        return importance_df


def train_v6_basketball():
    """Train V6 Behavioral Proxy model for basketball."""
    print("\n" + "="*60)
    print("V6 BEHAVIORAL PROXY BASKETBALL MODEL")
    print("="*60)
    
    # Load data
    print("\nLoading NBA data...")
    games_path = DATA_DIR / "games.csv"
    
    if not games_path.exists():
        print(f"ERROR: {games_path} not found!")
        return None
    
    df = pd.read_csv(games_path)
    print(f"  Loaded {len(df)} games")
    
    # Parse dates
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
    
    # Filter valid games (completed with scores)
    df = df[df['GAME_STATUS_TEXT'] == 'Final'].copy()
    df = df.dropna(subset=['PTS_home', 'PTS_away'])
    
    # Create target
    df['target'] = (df['HOME_TEAM_WINS'] == 1).astype(int)
    
    print(f"  Valid games: {len(df)}")
    print(f"  Date range: {df['GAME_DATE_EST'].min()} to {df['GAME_DATE_EST'].max()}")
    print(f"  Home win rate: {df['target'].mean():.1%}")
    
    # Sort by date
    df = df.sort_values('GAME_DATE_EST').reset_index(drop=True)
    
    # Build features
    print("\nBuilding behavioral proxy features...")
    engine = BehavioralProxyFeatureEngine()
    team_histories = engine.build_team_history(df)
    
    X, valid_indices = engine.create_behavioral_features(df, team_histories)
    y = df.loc[valid_indices, 'target'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features created: {list(X.columns)}")
    
    # Train model
    model = V6BehavioralProxyModel()
    results = model.train(X, y)
    
    # Feature importance
    print("\n  Top 10 Most Important Features:")
    importance = model.get_feature_importance()
    if importance is not None:
        for idx, row in importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # Save results
    results_path = MODELS_DIR / "v6_basketball_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in results.items()}, f, indent=2)
    
    print(f"\n  Results saved to: {results_path}")
    
    # Compare to baseline
    print("\n" + "="*60)
    print("COMPARISON TO V4 BASELINE")
    print("="*60)
    print(f"  V4 Ensemble NBA Accuracy: 62.0%")
    print(f"  V6 Behavioral Proxy:      {results['accuracy']:.1%}")
    improvement = (results['accuracy'] - 0.62) * 100
    if improvement > 0:
        print(f"  IMPROVEMENT:              +{improvement:.1f} percentage points! 🎉")
    else:
        print(f"  Difference:               {improvement:.1f} percentage points")
    
    return results


if __name__ == "__main__":
    results = train_v6_basketball()
