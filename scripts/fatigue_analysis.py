"""
Enhanced Feature Engineering with Fatigue Analysis
Key fatigue features:
1. Days since last game (rest days)
2. Back-to-back games
3. Games played in last 7 days
4. Travel/away game streaks
5. Season fatigue (games played so far)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


class FatigueFeatureEngineer:
    """Create fatigue-related features for sports betting."""
    
    @staticmethod
    def calculate_rest_days(df: pd.DataFrame, date_col: str, team_col: str) -> pd.DataFrame:
        """Calculate days since last game for each team."""
        df = df.sort_values(date_col)
        
        # Track last game date per team
        last_game = {}
        rest_days = []
        
        for _, row in df.iterrows():
            team = row[team_col]
            current_date = pd.to_datetime(row[date_col])
            
            if team in last_game:
                days = (current_date - last_game[team]).days
                rest_days.append(days)
            else:
                rest_days.append(7)  # Default for first game
            
            last_game[team] = current_date
        
        return rest_days
    
    @staticmethod
    def calculate_back_to_back(rest_days: list) -> list:
        """Identify back-to-back games (1 day rest)."""
        return [1 if r <= 1 else 0 for r in rest_days]
    
    @staticmethod
    def calculate_games_in_window(df: pd.DataFrame, date_col: str, team_col: str, window_days: int = 7) -> list:
        """Count games played in last N days."""
        df = df.sort_values(date_col)
        
        # Build game history per team
        team_games = {}
        games_in_window = []
        
        for _, row in df.iterrows():
            team = row[team_col]
            current_date = pd.to_datetime(row[date_col])
            
            if team not in team_games:
                team_games[team] = []
            
            # Count games in window
            cutoff = current_date - timedelta(days=window_days)
            recent = [d for d in team_games[team] if d >= cutoff]
            games_in_window.append(len(recent))
            
            # Add current game to history
            team_games[team].append(current_date)
        
        return games_in_window
    
    @staticmethod
    def calculate_season_fatigue(df: pd.DataFrame, date_col: str, team_col: str) -> list:
        """Count total games played in season (cumulative fatigue)."""
        df = df.sort_values(date_col)
        
        team_count = {}
        season_games = []
        
        for _, row in df.iterrows():
            team = row[team_col]
            
            if team not in team_count:
                team_count[team] = 0
            
            season_games.append(team_count[team])
            team_count[team] += 1
        
        return season_games
    
    @staticmethod
    def calculate_away_streak(df: pd.DataFrame, home_col: str, team_col: str) -> list:
        """Calculate consecutive away games (travel fatigue)."""
        # Simplified - would need more complex logic with actual team tracking
        is_home = df[home_col] if home_col in df.columns else None
        if is_home is not None:
            return [0 if h else 1 for h in is_home]
        return [0] * len(df)


def add_fatigue_features(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Add fatigue features based on sport-specific data structure."""
    fe = FatigueFeatureEngineer()
    
    # Detect date column
    date_cols = [c for c in df.columns if any(d in c.lower() for d in ['date', 'game_date', 'time'])]
    
    # Detect team columns
    home_cols = [c for c in df.columns if 'home' in c.lower() and ('team' in c.lower() or 'name' in c.lower() or 'id' in c.lower())]
    away_cols = [c for c in df.columns if 'away' in c.lower() or 'visitor' in c.lower() or 'road' in c.lower()]
    
    if not date_cols:
        print(f"    ⚠️ No date column found")
        return df
    
    date_col = date_cols[0]
    
    # Try to parse dates
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
    except:
        print(f"    ⚠️ Could not parse dates")
        return df
    
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Add fatigue features if we have team info
    if home_cols:
        home_col = home_cols[0]
        
        # Rest days for home team
        print(f"    📊 Adding rest days for home team...")
        df['home_rest_days'] = fe.calculate_rest_days(df, date_col, home_col)
        df['home_b2b'] = fe.calculate_back_to_back(df['home_rest_days'].tolist())
        df['home_games_7d'] = fe.calculate_games_in_window(df, date_col, home_col, 7)
        df['home_season_games'] = fe.calculate_season_fatigue(df, date_col, home_col)
        
        # Rest categories
        df['home_well_rested'] = (df['home_rest_days'] >= 3).astype(int)
        df['home_tired'] = (df['home_rest_days'] <= 1).astype(int)
    
    if away_cols:
        away_col = away_cols[0]
        
        # Rest days for away team
        print(f"    📊 Adding rest days for away team...")
        df['away_rest_days'] = fe.calculate_rest_days(df, date_col, away_col)
        df['away_b2b'] = fe.calculate_back_to_back(df['away_rest_days'].tolist())
        df['away_games_7d'] = fe.calculate_games_in_window(df, date_col, away_col, 7)
        df['away_season_games'] = fe.calculate_season_fatigue(df, date_col, away_col)
        
        # Rest categories  
        df['away_well_rested'] = (df['away_rest_days'] >= 3).astype(int)
        df['away_tired'] = (df['away_rest_days'] <= 1).astype(int)
    
    # Rest differential (key betting feature!)
    if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
        df['rest_differential'] = df['home_rest_days'] - df['away_rest_days']
        df['home_rest_advantage'] = (df['rest_differential'] > 0).astype(int)
        df['away_rest_advantage'] = (df['rest_differential'] < 0).astype(int)
        
        # B2B matchups
        df['home_b2b_vs_rested'] = ((df['home_b2b'] == 1) & (df['away_rest_days'] >= 2)).astype(int)
        df['away_b2b_vs_rested'] = ((df['away_b2b'] == 1) & (df['home_rest_days'] >= 2)).astype(int)
    
    # Day of week (weekend games different)
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Month (season timing)
    df['month'] = df[date_col].dt.month
    
    return df


def train_with_fatigue_features(sport: str) -> dict:
    """Train model with fatigue features and compare to baseline."""
    sport_dir = DATA_DIR / sport
    model_dir = MODELS_DIR / sport
    model_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🏋️ FATIGUE ANALYSIS: {sport.upper()}")
    print('='*60)
    
    # Find best data file (same logic as training script)
    csv_files = list(sport_dir.glob("*.csv"))
    priority_patterns = ['games.csv', 'game.csv', 'spreadspoke_scores', 'cbb.csv', 'club_games']
    
    best_file = None
    best_size = 0
    
    for f in csv_files:
        name = f.name.lower()
        if 'espn_recent' in name:
            continue
        for pattern in priority_patterns:
            if pattern in name:
                if f.stat().st_size > best_size:
                    best_file = f
                    best_size = f.stat().st_size
                break
        if best_file is None and 'game' in name:
            if f.stat().st_size > best_size:
                best_file = f
                best_size = f.stat().st_size
    
    if best_file is None:
        print(f"  ⚠️ No suitable data file found")
        return {}
    
    # Load data
    try:
        df = pd.read_csv(best_file, low_memory=False)
        print(f"  📊 Loaded {best_file.name}: {len(df):,} rows")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {}
    
    # Add fatigue features
    print(f"  🏋️ Adding fatigue features...")
    df_with_fatigue = add_fatigue_features(df.copy(), sport)
    
    # Count new features
    base_cols = set(df.columns)
    new_cols = set(df_with_fatigue.columns) - base_cols
    print(f"  ✅ Added {len(new_cols)} fatigue features")
    
    # Prepare features and target
    numeric_cols = df_with_fatigue.select_dtypes(include=[np.number]).columns.tolist()
    
    # Find score columns
    score_cols = [c for c in numeric_cols if any(s in c.lower() for s in ['score', 'pts', 'goals', 'runs'])]
    
    if len(score_cols) < 2:
        print(f"  ⚠️ No score columns found for target")
        return {}
    
    home_col, away_col = score_cols[0], score_cols[1]
    
    # Create target
    home_scores = df_with_fatigue[home_col].fillna(0)
    away_scores = df_with_fatigue[away_col].fillna(0)
    y = (home_scores > away_scores).astype(int)
    
    # Feature columns (exclude scores)
    feature_cols = [c for c in numeric_cols if c not in score_cols]
    X = df_with_fatigue[feature_cols].fillna(0)
    
    # Identify fatigue features
    fatigue_features = [c for c in feature_cols if any(f in c.lower() for f in 
        ['rest', 'b2b', 'tired', 'fatigue', 'games_7d', 'season_games', 'weekend', 'day_of_week', 'month'])]
    
    print(f"  📊 Total features: {len(feature_cols)} | Fatigue: {len(fatigue_features)}")
    
    if len(X) < 100:
        print(f"  ⚠️ Insufficient data")
        return {}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with ALL features
    print(f"\n  🎯 Training with fatigue features...")
    model_full = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model_full.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    
    y_pred_full = model_full.predict(X_test_scaled)
    y_proba_full = model_full.predict_proba(X_test_scaled)[:, 1]
    
    acc_full = accuracy_score(y_test, y_pred_full)
    auc_full = roc_auc_score(y_test, y_proba_full)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_full = cross_val_score(model_full, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    print(f"     Accuracy: {acc_full:.1%} | CV: {cv_full.mean():.1%} ± {cv_full.std():.1%}")
    print(f"     AUC: {auc_full:.3f}")
    
    # Train baseline (without fatigue features)
    print(f"\n  🎯 Training baseline (no fatigue)...")
    baseline_cols = [c for c in feature_cols if c not in fatigue_features]
    
    if len(baseline_cols) >= 3:
        X_baseline = X[baseline_cols]
        X_train_base, X_test_base, _, _ = train_test_split(X_baseline, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler_base = RobustScaler()
        X_train_base_scaled = scaler_base.fit_transform(X_train_base)
        X_test_base_scaled = scaler_base.transform(X_test_base)
        
        model_base = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model_base.fit(X_train_base_scaled, y_train, verbose=False)
        y_pred_base = model_base.predict(X_test_base_scaled)
        acc_base = accuracy_score(y_test, y_pred_base)
        
        cv_base = cross_val_score(model_base, X_train_base_scaled, y_train, cv=cv, scoring='accuracy')
        
        print(f"     Accuracy: {acc_base:.1%} | CV: {cv_base.mean():.1%} ± {cv_base.std():.1%}")
        
        # Compare
        improvement = acc_full - acc_base
        print(f"\n  📈 FATIGUE IMPACT: {improvement:+.1%} accuracy")
    else:
        acc_base = 0.5
        improvement = acc_full - 0.5
    
    # Get fatigue feature importance
    feature_importance = dict(zip(feature_cols, model_full.feature_importances_))
    fatigue_importance = {f: feature_importance.get(f, 0) for f in fatigue_features}
    top_fatigue = sorted(fatigue_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\n  🏋️ Top Fatigue Features:")
    for feat, imp in top_fatigue:
        print(f"     • {feat}: {imp:.4f}")
    
    # Save improved model
    model_path = model_dir / "moneyline_fatigue_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model_full,
            'scaler': scaler,
            'features': feature_cols,
            'fatigue_features': fatigue_features,
            'metrics': {
                'accuracy': acc_full,
                'accuracy_cv': cv_full.mean(),
                'auc': auc_full,
                'baseline_accuracy': acc_base,
                'fatigue_improvement': improvement,
            },
            'top_fatigue_features': top_fatigue,
            'sport': sport,
            'trained_at': datetime.now().isoformat(),
        }, f)
    
    print(f"  ✅ Saved fatigue model to {model_path.name}")
    
    return {
        'accuracy_with_fatigue': acc_full,
        'accuracy_baseline': acc_base,
        'improvement': improvement,
        'auc': auc_full,
        'top_fatigue_features': top_fatigue,
    }


def analyze_all_sports():
    """Run fatigue analysis on all sports."""
    print("\n" + "🏋️ " * 20)
    print("   FATIGUE FEATURE ANALYSIS")
    print("   Does rest/fatigue improve predictions?")
    print("🏋️ " * 20)
    
    sports = ['nba', 'nfl', 'nhl', 'mlb', 'soccer']
    results = {}
    
    for sport in sports:
        sport_dir = DATA_DIR / sport
        if sport_dir.exists() and any(sport_dir.glob("*.csv")):
            result = train_with_fatigue_features(sport)
            if result:
                results[sport] = result
    
    # Summary
    print("\n" + "="*60)
    print("📊 FATIGUE ANALYSIS SUMMARY")
    print("="*60)
    
    for sport, res in results.items():
        imp = res['improvement']
        imp_str = f"+{imp:.1%}" if imp > 0 else f"{imp:.1%}"
        status = "✅" if imp > 0 else "⚠️"
        print(f"\n{sport.upper()}: {status} {imp_str}")
        print(f"   With fatigue: {res['accuracy_with_fatigue']:.1%}")
        print(f"   Baseline: {res['accuracy_baseline']:.1%}")
        if res['top_fatigue_features']:
            top_feat = res['top_fatigue_features'][0][0]
            print(f"   Best fatigue feature: {top_feat}")
    
    # Save summary
    summary = {
        'analysis': 'Fatigue Feature Impact',
        'results': {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv 
                       for kk, vv in v.items() if kk != 'top_fatigue_features'}
                   for k, v in results.items()},
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(MODELS_DIR / "fatigue_analysis.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Analysis saved to {MODELS_DIR / 'fatigue_analysis.json'}")
    
    return results


if __name__ == "__main__":
    analyze_all_sports()
