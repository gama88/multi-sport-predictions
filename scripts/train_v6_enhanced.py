"""
V6 Behavioral Proxy Basketball Model - ENHANCED VERSION
========================================================
Combines existing NBA data with new detailed 2025 data.
Uses full stats: STL, BLK, TOV, PF, PLUS_MINUS, etc.

Target: 66.5%+ accuracy
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


def load_and_combine_data():
    """Load and combine old and new NBA data."""
    print("\n=== Loading and Combining NBA Data ===")
    
    # Load new detailed data (has STL, BLK, TOV, PF)
    new_games_path = DATA_DIR / "nba_new" / "nba_training_games_20251205.csv"
    
    if new_games_path.exists():
        new_df = pd.read_csv(new_games_path)
        print(f"  New data: {len(new_df)} team-game records")
        print(f"  Columns: {list(new_df.columns)}")
        
        # The new data is per-team per-game, we need to pair home/away
        # MATCHUP format: "ATL vs. BOS" (home) or "ATL @ BOS" (away)
        new_df['is_home'] = new_df['MATCHUP'].str.contains(' vs. ')
        
        # Parse date
        new_df['GAME_DATE'] = pd.to_datetime(new_df['GAME_DATE'], format='%b %d, %Y')
        
        # Create paired games (home vs away)
        home_games = new_df[new_df['is_home']].copy()
        away_games = new_df[~new_df['is_home']].copy()
        
        # Merge home and away on Game_ID
        home_games = home_games.add_suffix('_home')
        away_games = away_games.add_suffix('_away')
        
        paired = home_games.merge(
            away_games, 
            left_on='Game_ID_home', 
            right_on='Game_ID_away',
            how='inner'
        )
        
        print(f"  Paired games: {len(paired)}")
        
        # Create standardized format
        games = pd.DataFrame({
            'GAME_DATE_EST': paired['GAME_DATE_home'],
            'GAME_ID': paired['Game_ID_home'],
            'HOME_TEAM_ID': paired['Team_ID_home'],
            'VISITOR_TEAM_ID': paired['Team_ID_away'],
            'HOME_TEAM_ABBR': paired['TEAM_ABBR_home'],
            'AWAY_TEAM_ABBR': paired['TEAM_ABBR_away'],
            # Home stats
            'PTS_home': paired['PTS_home'],
            'FGM_home': paired['FGM_home'],
            'FGA_home': paired['FGA_home'],
            'FG_PCT_home': paired['FG_PCT_home'],
            'FG3M_home': paired['FG3M_home'] if 'FG3M_home' in paired.columns else 0,
            'FG3A_home': paired['FG3A_home'] if 'FG3A_home' in paired.columns else 0,
            'FG3_PCT_home': paired['FG3_PCT_home'] if 'FG3_PCT_home' in paired.columns else 0,
            'FT_PCT_home': paired['FT_PCT_home'] if 'FT_PCT_home' in paired.columns else 0,
            'REB_home': paired['REB_home'],
            'OREB_home': paired['OREB_home'] if 'OREB_home' in paired.columns else 0,
            'DREB_home': paired['DREB_home'] if 'DREB_home' in paired.columns else 0,
            'AST_home': paired['AST_home'],
            'STL_home': paired['STL_home'],
            'BLK_home': paired['BLK_home'],
            'TOV_home': paired['TOV_home'],
            'PF_home': paired['PF_home'],
            # Away stats
            'PTS_away': paired['PTS_away'],
            'FGM_away': paired['FGM_away'],
            'FGA_away': paired['FGA_away'],
            'FG_PCT_away': paired['FG_PCT_away'],
            'FG3M_away': paired['FG3M_away'] if 'FG3M_away' in paired.columns else 0,
            'FG3A_away': paired['FG3A_away'] if 'FG3A_away' in paired.columns else 0,
            'FG3_PCT_away': paired['FG3_PCT_away'] if 'FG3_PCT_away' in paired.columns else 0,
            'FT_PCT_away': paired['FT_PCT_away'] if 'FT_PCT_away' in paired.columns else 0,
            'REB_away': paired['REB_away'],
            'OREB_away': paired['OREB_away'] if 'OREB_away' in paired.columns else 0,
            'DREB_away': paired['DREB_away'] if 'DREB_away' in paired.columns else 0,
            'AST_away': paired['AST_away'],
            'STL_away': paired['STL_away'],
            'BLK_away': paired['BLK_away'],
            'TOV_away': paired['TOV_away'],
            'PF_away': paired['PF_away'],
            # Target
            'HOME_TEAM_WINS': (paired['WL_home'] == 'W').astype(int),
            'SEASON': paired['SEASON_home']
        })
        
        print(f"  Final paired: {len(games)} games")
        return games
    else:
        print(f"  ERROR: New data not found at {new_games_path}")
        return None


class BehavioralProxyV6:
    """
    Enhanced behavioral proxy features using full stats:
    STL, BLK, TOV, PF, AST, REB, FGM/FGA, etc.
    """
    
    def build_team_history(self, df):
        """Build historical stats for each team game-by-game."""
        print("  Building team history cache with full stats...")
        
        df = df.sort_values('GAME_DATE_EST').reset_index(drop=True)
        
        teams = set(df['HOME_TEAM_ID'].unique()) | set(df['VISITOR_TEAM_ID'].unique())
        team_histories = {team: [] for team in teams}
        
        for idx, row in df.iterrows():
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            game_date = row['GAME_DATE_EST']
            
            # Home team game record (FULL STATS)
            home_game = {
                'date': game_date,
                'is_home': True,
                'won': row['HOME_TEAM_WINS'] == 1,
                'pts': row.get('PTS_home', 0) or 0,
                'opp_pts': row.get('PTS_away', 0) or 0,
                'fgm': row.get('FGM_home', 0) or 0,
                'fga': row.get('FGA_home', 0) or 0,
                'fg_pct': row.get('FG_PCT_home', 0) or 0,
                'fg3_pct': row.get('FG3_PCT_home', 0) or 0,
                'ft_pct': row.get('FT_PCT_home', 0) or 0,
                'reb': row.get('REB_home', 0) or 0,
                'oreb': row.get('OREB_home', 0) or 0,
                'dreb': row.get('DREB_home', 0) or 0,
                'ast': row.get('AST_home', 0) or 0,
                'stl': row.get('STL_home', 0) or 0,
                'blk': row.get('BLK_home', 0) or 0,
                'tov': row.get('TOV_home', 0) or 0,
                'pf': row.get('PF_home', 0) or 0,
            }
            
            # Away team game record
            away_game = {
                'date': game_date,
                'is_home': False,
                'won': row['HOME_TEAM_WINS'] == 0,
                'pts': row.get('PTS_away', 0) or 0,
                'opp_pts': row.get('PTS_home', 0) or 0,
                'fgm': row.get('FGM_away', 0) or 0,
                'fga': row.get('FGA_away', 0) or 0,
                'fg_pct': row.get('FG_PCT_away', 0) or 0,
                'fg3_pct': row.get('FG3_PCT_away', 0) or 0,
                'ft_pct': row.get('FT_PCT_away', 0) or 0,
                'reb': row.get('REB_away', 0) or 0,
                'oreb': row.get('OREB_away', 0) or 0,
                'dreb': row.get('DREB_away', 0) or 0,
                'ast': row.get('AST_away', 0) or 0,
                'stl': row.get('STL_away', 0) or 0,
                'blk': row.get('BLK_away', 0) or 0,
                'tov': row.get('TOV_away', 0) or 0,
                'pf': row.get('PF_away', 0) or 0,
            }
            
            team_histories[home_id].append(home_game)
            team_histories[away_id].append(away_game)
        
        return team_histories
    
    def get_team_stats(self, history, n_games=20):
        """Aggregate stats from last n games."""
        if len(history) < 3:
            return None
        
        recent = history[-n_games:] if len(history) >= n_games else history
        
        def safe_mean(values):
            valid = [v for v in values if v is not None and v > 0]
            return np.mean(valid) if valid else 0
        
        def safe_std(values):
            valid = [v for v in values if v is not None and v > 0]
            return np.std(valid) if len(valid) > 1 else 0
        
        wins = [1 if g['won'] else 0 for g in recent]
        
        return {
            # Basic stats
            'pts_mean': safe_mean([g['pts'] for g in recent]),
            'pts_std': safe_std([g['pts'] for g in recent]),
            'opp_pts_mean': safe_mean([g['opp_pts'] for g in recent]),
            'opp_pts_std': safe_std([g['opp_pts'] for g in recent]),
            'fg_pct_mean': safe_mean([g['fg_pct'] for g in recent]),
            'fg3_pct_mean': safe_mean([g['fg3_pct'] for g in recent]),
            'ft_pct_mean': safe_mean([g['ft_pct'] for g in recent]),
            
            # Rebounds
            'reb_mean': safe_mean([g['reb'] for g in recent]),
            'oreb_mean': safe_mean([g['oreb'] for g in recent]),
            'dreb_mean': safe_mean([g['dreb'] for g in recent]),
            
            # Playmaking
            'ast_mean': safe_mean([g['ast'] for g in recent]),
            'tov_mean': safe_mean([g['tov'] for g in recent]),
            'ast_to_tov': safe_mean([g['ast'] for g in recent]) / max(safe_mean([g['tov'] for g in recent]), 1),
            
            # Defense
            'stl_mean': safe_mean([g['stl'] for g in recent]),
            'blk_mean': safe_mean([g['blk'] for g in recent]),
            'blk_std': safe_std([g['blk'] for g in recent]),
            
            # Discipline
            'pf_mean': safe_mean([g['pf'] for g in recent]),
            'stl_to_pf': safe_mean([g['stl'] for g in recent]) / max(safe_mean([g['pf'] for g in recent]), 1),
            
            # Win rates
            'win_rate': np.mean(wins),
            'last_5_wins': sum(wins[-5:]) if len(wins) >= 5 else sum(wins),
            'first_5_wins': sum(wins[:5]) if len(wins) >= 5 else sum(wins),
            'games_played': len(recent),
            
            # Home/away splits
            'home_games': [g for g in recent if g['is_home']],
            'away_games': [g for g in recent if not g['is_home']],
            
            # Efficiency
            'fgm_mean': safe_mean([g['fgm'] for g in recent]),
            'fga_mean': safe_mean([g['fga'] for g in recent]),
        }
    
    def calculate_fatigue_proxies(self, history, current_date):
        """5 Fatigue proxy features."""
        if len(history) < 2:
            return {f'fatigue_{i}': 0.5 for i in range(5)}
        
        last_date = history[-1]['date']
        days_rest = (current_date - last_date).days if current_date > last_date else 1
        
        # Back-to-back
        back_to_back = 1.0 if days_rest <= 1 else 0.0
        
        # Games in last 7 days
        week_ago = current_date - timedelta(days=7)
        games_7d = sum(1 for g in history if g['date'] >= week_ago and g['date'] < current_date)
        
        # Recent away game load (travel fatigue proxy)
        recent_10 = history[-10:] if len(history) >= 10 else history
        away_load = sum(1 for g in recent_10 if not g['is_home']) / len(recent_10)
        
        # Average minutes load (PTS as proxy for intensity)
        avg_pts = np.mean([g['pts'] for g in recent_10 if g['pts'] > 0])
        intensity = avg_pts / 120.0  # Normalized
        
        return {
            'fatigue_back_to_back': back_to_back,
            'fatigue_games_7d': min(games_7d / 4.0, 1.0),
            'fatigue_rest_days': min(days_rest / 7.0, 1.0),
            'fatigue_away_load': away_load,
            'fatigue_intensity': min(intensity, 1.0),
        }
    
    def calculate_defensive_discipline(self, stats):
        """4 Defensive discipline features using STL, BLK, PF."""
        if stats is None:
            return {f'def_{i}': 0.5 for i in range(4)}
        
        # Personal fouls (lower = better discipline)
        pf = stats.get('pf_mean', 20)
        pf_discipline = 1.0 - min(pf / 25.0, 1.0)
        
        # Steals to fouls ratio (higher = cleaner active defense)
        stl_to_pf = stats.get('stl_to_pf', 0.5)
        stl_pf_ratio = min(stl_to_pf / 1.0, 1.0)
        
        # Block consistency
        blk_std = stats.get('blk_std', 2)
        blk_consistency = 1.0 - min(blk_std / 5.0, 1.0)
        
        # Defensive variance (points allowed consistency)
        opp_pts_std = stats.get('opp_pts_std', 10)
        def_variance = 1.0 - min(opp_pts_std / 20.0, 1.0)
        
        return {
            'def_pf_discipline': pf_discipline,
            'def_stl_pf_ratio': stl_pf_ratio,
            'def_blk_consistency': blk_consistency,
            'def_variance': def_variance,
        }
    
    def calculate_clutch_pressure(self, stats):
        """4 Clutch/pressure features."""
        if stats is None:
            return {f'clutch_{i}': 0.5 for i in range(4)}
        
        # FT% as composure indicator
        ft_pct = stats.get('ft_pct_mean', 0.75)
        
        # Win streak momentum
        last_5 = stats.get('last_5_wins', 2.5)
        first_5 = stats.get('first_5_wins', 2.5)
        win_streak = last_5 / 5.0
        momentum = (last_5 - first_5) / 5.0 + 0.5
        momentum = min(max(momentum, 0), 1.0)
        
        # Close game indicator
        pts_std = stats.get('pts_std', 10)
        variance = min(pts_std / 15.0, 1.0)
        
        return {
            'clutch_ft_composure': min(ft_pct, 1.0),
            'clutch_win_streak': win_streak,
            'clutch_momentum': momentum,
            'clutch_variance': variance,
        }
    
    def calculate_spacing_flow(self, stats):
        """4 Spacing/flow features using AST, TOV."""
        if stats is None:
            return {f'flow_{i}': 0.5 for i in range(4)}
        
        # Assist rate
        ast = stats.get('ast_mean', 24)
        fgm = stats.get('fgm_mean', 40)
        assist_rate = min(ast / max(fgm, 1) / 0.7, 1.0)
        
        # 3-point tendency
        fg3_pct = stats.get('fg3_pct_mean', 0.35)
        three_rate = min(fg3_pct, 1.0)
        
        # Turnover rate (lower = better)
        tov = stats.get('tov_mean', 14)
        tov_rate = 1.0 - min(tov / 20.0, 1.0)
        
        # Offensive flow (efficiency consistency)
        fg_pct = stats.get('fg_pct_mean', 0.45)
        flow = min(fg_pct * 2, 1.0)
        
        return {
            'flow_assist_rate': assist_rate,
            'flow_3pt_rate': three_rate,
            'flow_tov_rate': tov_rate,
            'flow_efficiency': flow,
        }
    
    def calculate_chemistry(self, stats):
        """3 Team chemistry features."""
        if stats is None:
            return {f'chem_{i}': 0.5 for i in range(3)}
        
        # Ast to FGM ratio (unselfish play)
        ast = stats.get('ast_mean', 24)
        fgm = stats.get('fgm_mean', 40)
        unselfishness = min(ast / max(fgm, 1), 1.0)
        
        # Performance stability
        pts_std = stats.get('pts_std', 10)
        opp_std = stats.get('opp_pts_std', 10)
        stability = 1.0 - min((pts_std + opp_std) / 30.0, 1.0)
        
        # Experience
        games = stats.get('games_played', 15)
        experience = min(games / 30.0, 1.0)
        
        return {
            'chem_unselfishness': unselfishness,
            'chem_stability': stability,
            'chem_experience': experience,
        }
    
    def create_all_features(self, df, team_histories):
        """Create all 20 behavioral proxy features + base features."""
        print("  Creating behavioral proxy features with full stats...")
        
        features = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            game_date = row['GAME_DATE_EST']
            
            # Get history UP TO this game (no leakage)
            home_history = [g for g in team_histories.get(home_id, []) if g['date'] < game_date]
            away_history = [g for g in team_histories.get(away_id, []) if g['date'] < game_date]
            
            if len(home_history) < 5 or len(away_history) < 5:
                continue
            
            # Get aggregated stats
            home_stats = self.get_team_stats(home_history, 20)
            away_stats = self.get_team_stats(away_history, 20)
            
            home_stats_5 = self.get_team_stats(home_history, 5)
            away_stats_5 = self.get_team_stats(away_history, 5)
            
            if home_stats is None or away_stats is None:
                continue
            
            # Calculate all behavioral proxies
            home_fatigue = self.calculate_fatigue_proxies(home_history, game_date)
            away_fatigue = self.calculate_fatigue_proxies(away_history, game_date)
            
            home_def = self.calculate_defensive_discipline(home_stats)
            away_def = self.calculate_defensive_discipline(away_stats)
            
            home_clutch = self.calculate_clutch_pressure(home_stats)
            away_clutch = self.calculate_clutch_pressure(away_stats)
            
            home_flow = self.calculate_spacing_flow(home_stats)
            away_flow = self.calculate_spacing_flow(away_stats)
            
            home_chem = self.calculate_chemistry(home_stats)
            away_chem = self.calculate_chemistry(away_stats)
            
            # BUILD DIFFERENTIAL FEATURES (home - away)
            game_features = {}
            
            # Fatigue (5) - inverted so less fatigue for home is better
            for key in home_fatigue:
                if 'rest_days' in key:
                    game_features[f'{key}_diff'] = home_fatigue[key] - away_fatigue[key]
                else:
                    game_features[f'{key}_diff'] = away_fatigue[key] - home_fatigue[key]
            
            # Defensive discipline (4)
            for key in home_def:
                game_features[f'{key}_diff'] = home_def[key] - away_def[key]
            
            # Clutch (4)
            for key in home_clutch:
                game_features[f'{key}_diff'] = home_clutch[key] - away_clutch[key]
            
            # Flow (4)
            for key in home_flow:
                game_features[f'{key}_diff'] = home_flow[key] - away_flow[key]
            
            # Chemistry (3)
            for key in home_chem:
                game_features[f'{key}_diff'] = home_chem[key] - away_chem[key]
            
            # === BASE FEATURES (5) ===
            game_features['win_rate_diff'] = home_stats['win_rate'] - away_stats['win_rate']
            game_features['pts_diff'] = (home_stats['pts_mean'] - away_stats['pts_mean']) / 10.0
            game_features['home_win_rate'] = np.mean([1 if g['won'] else 0 for g in home_stats['home_games']]) if home_stats['home_games'] else 0.5
            game_features['away_road_rate'] = np.mean([1 if g['won'] else 0 for g in away_stats['away_games']]) if away_stats['away_games'] else 0.5
            
            # Recent form
            if home_stats_5 and away_stats_5:
                game_features['recent_form_diff'] = home_stats_5['win_rate'] - away_stats_5['win_rate']
                game_features['recent_pts_diff'] = (home_stats_5['pts_mean'] - away_stats_5['pts_mean']) / 10.0
            else:
                game_features['recent_form_diff'] = 0
                game_features['recent_pts_diff'] = 0
            
            # === EXTRA STATS FEATURES (new with STL, BLK, TOV) ===
            game_features['stl_diff'] = (home_stats['stl_mean'] - away_stats['stl_mean']) / 10.0
            game_features['blk_diff'] = (home_stats['blk_mean'] - away_stats['blk_mean']) / 5.0
            game_features['tov_diff'] = (away_stats['tov_mean'] - home_stats['tov_mean']) / 10.0  # Inverted
            game_features['ast_tov_diff'] = home_stats['ast_to_tov'] - away_stats['ast_to_tov']
            game_features['reb_diff'] = (home_stats['reb_mean'] - away_stats['reb_mean']) / 10.0
            game_features['oreb_diff'] = (home_stats['oreb_mean'] - away_stats['oreb_mean']) / 5.0
            
            features.append(game_features)
            valid_indices.append(idx)
            
            if len(features) % 2000 == 0:
                print(f"    Processed {len(features)} games...")
        
        print(f"  Created features for {len(features)} games")
        return pd.DataFrame(features), valid_indices


def train_v6_enhanced():
    """Train enhanced V6 model with combined data."""
    print("\n" + "="*60)
    print("V6 BEHAVIORAL PROXY - ENHANCED (Full Stats)")
    print("="*60)
    
    # Load combined data
    games = load_and_combine_data()
    
    if games is None or len(games) == 0:
        print("ERROR: No data loaded!")
        return None
    
    # Create target
    games['target'] = games['HOME_TEAM_WINS'].astype(int)
    
    print(f"\n  Total games: {len(games)}")
    print(f"  Date range: {games['GAME_DATE_EST'].min()} to {games['GAME_DATE_EST'].max()}")
    print(f"  Home win rate: {games['target'].mean():.1%}")
    
    # Sort by date
    games = games.sort_values('GAME_DATE_EST').reset_index(drop=True)
    
    # Build features
    print("\nBuilding enhanced behavioral features...")
    engine = BehavioralProxyV6()
    team_histories = engine.build_team_history(games)
    
    X, valid_indices = engine.create_all_features(games, team_histories)
    y = games.loc[valid_indices, 'target'].values
    
    print(f"\nFeature matrix: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-based split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train XGBoost
    print("\n  Training XGBoost...")
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
    
    # Train LightGBM
    print("  Training LightGBM...")
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
    
    # Ensemble (50/50)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, ensemble_pred)
    auc = roc_auc_score(y_test, ensemble_proba)
    brier = brier_score_loss(y_test, ensemble_proba)
    logloss = log_loss(y_test, ensemble_proba)
    
    xgb_acc = accuracy_score(y_test, (xgb_proba >= 0.5).astype(int))
    lgb_acc = accuracy_score(y_test, (lgb_proba >= 0.5).astype(int))
    
    print(f"\n  {'='*50}")
    print(f"  V6 ENHANCED RESULTS")
    print(f"  {'='*50}")
    print(f"  XGBoost Accuracy:  {xgb_acc:.1%}")
    print(f"  LightGBM Accuracy: {lgb_acc:.1%}")
    print(f"  Ensemble Accuracy: {accuracy:.1%}")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  Brier Score:       {brier:.4f}")
    print(f"  Log Loss:          {logloss:.4f}")
    
    # Feature importance
    print(f"\n  Top 15 Most Important Features:")
    combined_imp = (xgb_model.feature_importances_ + lgb_model.feature_importances_) / 2
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': combined_imp
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(15).iterrows():
        print(f"    {row['feature']}: {row['importance']:.2f}")
    
    # Comparison
    print(f"\n  {'='*50}")
    print(f"  COMPARISON")
    print(f"  {'='*50}")
    print(f"  V4 Baseline:       62.0%")
    print(f"  V6 Basic:          62.4%")
    print(f"  V6 Enhanced:       {accuracy:.1%}")
    
    improvement = (accuracy - 0.62) * 100
    if improvement > 0:
        print(f"  IMPROVEMENT:       +{improvement:.1f}pp 🎉")
    
    # Save results
    import json
    results = {
        'accuracy': float(accuracy),
        'xgb_accuracy': float(xgb_acc),
        'lgb_accuracy': float(lgb_acc),
        'auc': float(auc),
        'brier': float(brier),
        'logloss': float(logloss),
        'test_size': len(y_test),
        'train_size': len(y_train),
    }
    
    with open(MODELS_DIR / "v6_enhanced_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = train_v6_enhanced()
