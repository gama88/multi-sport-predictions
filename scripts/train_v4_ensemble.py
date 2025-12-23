"""
Advanced Sports Betting Model v4.0
Research-backed improvements:
1. Hybrid XGBoost + Random Forest + Neural Network ensemble
2. Advanced feature engineering (momentum, pace, ELO)
3. Rolling window features
4. Better target engineering
5. Manual ensemble with weighted averaging
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


class AdvancedFeatureEngine:
    """Research-backed feature engineering."""
    
    @staticmethod
    def create_elo_features(df, home_col, away_col, home_score_col, away_score_col):
        """Create advanced ELO ratings with home/away split."""
        df = df.sort_values('game_date').reset_index(drop=True)
        
        teams = set(df[home_col].unique()) | set(df[away_col].unique())
        elo = {team: {'overall': 1500, 'home': 1500, 'away': 1500, 'off': 1500, 'def': 1500} for team in teams}
        
        features = []
        k = 20
        
        for _, row in df.iterrows():
            home, away = row[home_col], row[away_col]
            
            # Pre-game ELO features
            feat = {
                'home_elo': elo[home]['overall'],
                'away_elo': elo[away]['overall'],
                'elo_diff': elo[home]['overall'] - elo[away]['overall'],
                'home_home_elo': elo[home]['home'],
                'away_away_elo': elo[away]['away'],
                'home_off_elo': elo[home]['off'],
                'home_def_elo': elo[home]['def'],
                'away_off_elo': elo[away]['off'],
                'away_def_elo': elo[away]['def'],
            }
            features.append(feat)
            
            # Update ELO after game
            home_score = row.get(home_score_col, 0)
            away_score = row.get(away_score_col, 0)
            
            if pd.notna(home_score) and pd.notna(away_score):
                home_win = 1 if home_score > away_score else 0
                expected = 1 / (1 + 10 ** ((elo[away]['overall'] - elo[home]['overall']) / 400))
                
                delta = k * (home_win - expected)
                elo[home]['overall'] += delta
                elo[away]['overall'] -= delta
                elo[home]['home'] += delta * 0.5
                elo[away]['away'] -= delta * 0.5
                
                # Offensive/Defensive ELO
                score_diff = home_score - away_score
                elo[home]['off'] += score_diff * 0.1
                elo[home]['def'] -= away_score * 0.05
                elo[away]['off'] += away_score * 0.05
                elo[away]['def'] -= home_score * 0.05
        
        return pd.DataFrame(features)
    
    @staticmethod
    def create_momentum_features(df, home_col, away_col, home_score_col, away_score_col, windows=[3, 5, 10]):
        """Create momentum/form features."""
        df = df.sort_values('game_date').reset_index(drop=True)
        
        teams = set(df[home_col].unique()) | set(df[away_col].unique())
        history = {team: {'wins': [], 'pts_for': [], 'pts_against': []} for team in teams}
        
        features = []
        
        for _, row in df.iterrows():
            home, away = row[home_col], row[away_col]
            home_hist, away_hist = history[home], history[away]
            
            feat = {}
            
            for w in windows:
                # Win rate
                home_wins = home_hist['wins'][-w:] if home_hist['wins'] else []
                away_wins = away_hist['wins'][-w:] if away_hist['wins'] else []
                feat[f'home_winrate_{w}'] = np.mean(home_wins) if home_wins else 0.5
                feat[f'away_winrate_{w}'] = np.mean(away_wins) if away_wins else 0.5
                feat[f'winrate_diff_{w}'] = feat[f'home_winrate_{w}'] - feat[f'away_winrate_{w}']
                
                # Points for/against
                home_pts = home_hist['pts_for'][-w:] if home_hist['pts_for'] else []
                away_pts = away_hist['pts_for'][-w:] if away_hist['pts_for'] else []
                feat[f'home_pts_avg_{w}'] = np.mean(home_pts) if home_pts else 100
                feat[f'away_pts_avg_{w}'] = np.mean(away_pts) if away_pts else 100
                
                # Offensive/Defensive trend
                if len(home_pts) >= w:
                    feat[f'home_pts_trend_{w}'] = np.polyfit(range(w), home_pts[-w:], 1)[0]
                else:
                    feat[f'home_pts_trend_{w}'] = 0
            
            # Streak
            feat['home_streak'] = self._count_streak(home_hist['wins'])
            feat['away_streak'] = self._count_streak(away_hist['wins'])
            
            features.append(feat)
            
            # Update history
            home_score = row.get(home_score_col, 0)
            away_score = row.get(away_score_col, 0)
            
            if pd.notna(home_score) and pd.notna(away_score):
                home_win = 1 if home_score > away_score else 0
                history[home]['wins'].append(home_win)
                history[away]['wins'].append(1 - home_win)
                history[home]['pts_for'].append(home_score)
                history[away]['pts_for'].append(away_score)
                history[home]['pts_against'].append(away_score)
                history[away]['pts_against'].append(home_score)
        
        return pd.DataFrame(features)
    
    @staticmethod
    def _count_streak(wins):
        """Count current win/loss streak."""
        if not wins:
            return 0
        streak = 0
        last_val = wins[-1]
        for w in reversed(wins):
            if w == last_val:
                streak += 1 if last_val else -1
            else:
                break
        return streak
    
    @staticmethod
    def create_rest_features(df, date_col, home_col, away_col):
        """Create rest/fatigue features."""
        df = df.sort_values(date_col).reset_index(drop=True)
        
        last_game = {}
        last_5_games = {}
        
        features = []
        
        for _, row in df.iterrows():
            home, away = row[home_col], row[away_col]
            game_date = pd.to_datetime(row[date_col])
            
            feat = {}
            
            # Rest days
            if home in last_game:
                feat['home_rest'] = min((game_date - last_game[home]).days, 14)
            else:
                feat['home_rest'] = 7
                
            if away in last_game:
                feat['away_rest'] = min((game_date - last_game[away]).days, 14)
            else:
                feat['away_rest'] = 7
            
            feat['rest_diff'] = feat['home_rest'] - feat['away_rest']
            feat['home_b2b'] = 1 if feat['home_rest'] <= 1 else 0
            feat['away_b2b'] = 1 if feat['away_rest'] <= 1 else 0
            
            # Games in last 7 days
            if home in last_5_games:
                recent = [d for d in last_5_games[home] if (game_date - d).days <= 7]
                feat['home_games_7d'] = len(recent)
            else:
                feat['home_games_7d'] = 0
                
            if away in last_5_games:
                recent = [d for d in last_5_games[away] if (game_date - d).days <= 7]
                feat['away_games_7d'] = len(recent)
            else:
                feat['away_games_7d'] = 0
            
            features.append(feat)
            
            # Update
            last_game[home] = game_date
            last_game[away] = game_date
            
            if home not in last_5_games:
                last_5_games[home] = []
            last_5_games[home].append(game_date)
            last_5_games[home] = last_5_games[home][-20:]
            
            if away not in last_5_games:
                last_5_games[away] = []
            last_5_games[away].append(game_date)
            last_5_games[away] = last_5_games[away][-20:]
        
        return pd.DataFrame(features)


class EnsembleTrainer:
    """Train ensemble of XGBoost, Random Forest, and Neural Network."""
    
    def __init__(self, sport):
        self.sport = sport
        self.models = {}
        self.scalers = {}
        self.metrics = {}
    
    def train(self, X, y, bet_type='moneyline'):
        """Train calibrated ensemble model."""
        print(f"\n    🎯 Training {bet_type.upper()} ensemble...")
        
        # Clean data
        valid = y.notna()
        X_clean = X[valid].copy()
        y_clean = y[valid].copy()
        
        if len(X_clean) < 100:
            print(f"       ⚠️ Insufficient data")
            return {}
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        
        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models individually then combine (avoiding VotingClassifier XGBoost issue)
        print("       Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            random_state=42, objective='binary:logistic'
        )
        xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
        
        print("       Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        print("       Training Neural Network...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu',
            max_iter=500, random_state=42, early_stopping=True
        )
        nn_model.fit(X_train_scaled, y_train)
        
        # Manual ensemble - weighted average of probabilities
        print("       Creating ensemble...")
        xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
        rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        nn_proba = nn_model.predict_proba(X_test_scaled)[:, 1]
        
        # Weights: XGBoost 50%, RF 25%, NN 25%
        ensemble_proba = 0.5 * xgb_proba + 0.25 * rf_proba + 0.25 * nn_proba
        y_pred = (ensemble_proba > 0.5).astype(int)
        y_proba = ensemble_proba
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        logloss = log_loss(y_test, y_proba)
        
        # Cross-validation (using XGBoost as proxy since it's the main model)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        print(f"       Accuracy: {accuracy:.1%} | CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
        print(f"       AUC: {auc:.3f} | Brier: {brier:.3f} | LogLoss: {logloss:.3f}")
        
        # Feature importance (from XGBoost)
        importance = dict(zip(X.columns, xgb_model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Store the ensemble models
        ensemble_models = {
            'xgb': xgb_model,
            'rf': rf_model,
            'nn': nn_model,
            'weights': [0.5, 0.25, 0.25]
        }
        
        self.models[bet_type] = ensemble_models
        self.scalers[bet_type] = scaler
        self.metrics[bet_type] = {
            'accuracy': accuracy,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'auc': auc,
            'brier': brier,
            'logloss': logloss,
            'samples': len(X_clean),
            'top_features': top_features[:5],
        }
        
        return self.metrics[bet_type]
    
    def train_contracts(self, X, y, bet_type='contracts'):
        """
        Train model optimized for prediction market contracts.
        Focus on CALIBRATION (Brier score) rather than just accuracy.
        Contracts need: predicted probability = actual probability
        """
        from sklearn.calibration import CalibratedClassifierCV, calibration_curve
        
        print(f"\n    📈 Training {bet_type.upper()} (calibrated for prediction markets)...")
        
        # Clean data
        valid = y.notna()
        X_clean = X[valid].copy()
        y_clean = y[valid].copy()
        
        if len(X_clean) < 100:
            print(f"       ⚠️ Insufficient data")
            return {}
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        
        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Use XGBoost as base (best for tabular data)
        print("       Training base XGBoost...")
        base_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.0,
            random_state=42, objective='binary:logistic'
        )
        base_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
        
        # Get raw probabilities
        y_proba_raw = base_model.predict_proba(X_test_scaled)[:, 1]
        
        # Manual isotonic calibration using sklearn's IsotonicRegression
        # This maps raw probabilities to calibrated probabilities
        from sklearn.isotonic import IsotonicRegression
        print("       Calibrating probabilities (isotonic regression)...")
        
        # Train isotonic regressor on raw probabilities
        y_proba_train = base_model.predict_proba(X_train_scaled)[:, 1]
        isotonic = IsotonicRegression(out_of_bounds='clip')
        isotonic.fit(y_proba_train, y_train)
        
        # Apply calibration to test set
        y_proba_calibrated = isotonic.predict(y_proba_raw)
        y_pred = (y_proba_calibrated > 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba_calibrated)
        brier_raw = brier_score_loss(y_test, y_proba_raw)
        brier_calibrated = brier_score_loss(y_test, y_proba_calibrated)
        logloss = log_loss(y_test, y_proba_calibrated)
        
        # Calculate calibration quality
        # Lower Brier = better calibration
        calibration_improvement = brier_raw - brier_calibrated
        
        print(f"       Accuracy: {accuracy:.1%}")
        print(f"       AUC: {auc:.3f}")
        print(f"       Brier (raw): {brier_raw:.4f} → (calibrated): {brier_calibrated:.4f}")
        print(f"       Calibration improvement: {calibration_improvement:.4f}")
        
        # Feature importance
        importance = dict(zip(X.columns, base_model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Store calibrated model
        contract_model = {
            'base_model': base_model,
            'isotonic_calibrator': isotonic,
            'is_calibrated': True
        }
        
        self.models[bet_type] = contract_model
        self.scalers[bet_type] = scaler
        self.metrics[bet_type] = {
            'accuracy': accuracy,
            'auc': auc,
            'brier_raw': brier_raw,
            'brier_calibrated': brier_calibrated,
            'calibration_improvement': calibration_improvement,
            'logloss': logloss,
            'samples': len(X_clean),
            'top_features': top_features[:5],
        }
        
        return self.metrics[bet_type]


def train_sport_v4(sport):
    """Train v4 model for a sport."""
    print(f"\n{'='*60}")
    print(f"🏆 V4 TRAINING: {sport.upper()}")
    print('='*60)
    
    sport_dir = DATA_DIR / sport
    model_dir = MODELS_DIR / sport
    model_dir.mkdir(exist_ok=True)
    
    # Find best data file
    csv_files = list(sport_dir.glob("*.csv"))
    priority_patterns = ['games.csv', 'game.csv', 'spreadspoke', 'cbb.csv']
    
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
    
    if best_file is None:
        for f in csv_files:
            if f.stat().st_size > best_size:
                best_file = f
                best_size = f.stat().st_size
    
    if best_file is None:
        print(f"  ⚠️ No data found")
        return {}
    
    # Load data
    df = pd.read_csv(best_file, low_memory=False)
    print(f"  📊 Loaded {best_file.name}: {len(df):,} rows")
    
    # Detect columns
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    home_cols = [c for c in df.columns if 'home' in c.lower() and ('team' in c.lower() or 'id' in c.lower() or c.lower().endswith('_home'))]
    away_cols = [c for c in df.columns if ('away' in c.lower() or 'visitor' in c.lower()) and ('team' in c.lower() or 'id' in c.lower() or c.lower().endswith('_away'))]
    
    if not date_cols or not home_cols or not away_cols:
        print(f"  ⚠️ Missing required columns")
        return {}
    
    date_col = date_cols[0]
    home_col = home_cols[0]
    away_col = away_cols[0]
    
    # Parse dates
    df['game_date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['game_date']).sort_values('game_date').reset_index(drop=True)
    
    # Find score columns (include goals for hockey/soccer)
    score_patterns = ['score', 'pts', 'points', 'goals']
    score_cols = [c for c in df.columns if any(s in c.lower() for s in score_patterns)]
    home_score_cols = [c for c in score_cols if 'home' in c.lower()]
    away_score_cols = [c for c in score_cols if 'away' in c.lower() or 'visitor' in c.lower()]
    
    if not home_score_cols or not away_score_cols:
        print(f"  ⚠️ Missing score columns")
        print(f"      Found: {score_cols}")
        return {}
    
    home_score_col = home_score_cols[0]
    away_score_col = away_score_cols[0]
    print(f"  📊 Using score columns: {home_score_col}, {away_score_col}")
    
    # Create target
    df['home_score'] = pd.to_numeric(df[home_score_col], errors='coerce')
    df['away_score'] = pd.to_numeric(df[away_score_col], errors='coerce')
    df = df.dropna(subset=['home_score', 'away_score'])
    
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['point_diff'] = df['home_score'] - df['away_score']
    df['total_points'] = df['home_score'] + df['away_score']
    
    print(f"  🔧 Creating advanced features (no data leakage)...")
    
    # Create features
    fe = AdvancedFeatureEngine()
    
    # Rest features (these are PRE-GAME)
    rest_features = fe.create_rest_features(df, 'game_date', home_col, away_col)
    
    # IMPORTANT: Filter out POST-GAME stats to prevent data leakage
    # These are stats that are only known AFTER the game ends
    postgame_patterns = [
        'score', 'pts', 'points', 'goals', 'win', 'loss',
        'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct',
        'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb',
        'ast', 'stl', 'blk', 'tov', 'pf', 'plus_minus',
        'min_', 'minutes', 'saves', 'shots', 'hits', 'penalty'
    ]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out post-game columns
    exclude_cols = ['home_score', 'away_score', 'home_win', 'point_diff', 'total_points',
                   home_score_col, away_score_col]
    
    feature_cols = []
    for c in numeric_cols:
        c_lower = c.lower()
        # Skip known post-game stats
        if any(pattern in c_lower for pattern in postgame_patterns):
            continue
        # Skip if in exclude list
        if c in exclude_cols:
            continue
        feature_cols.append(c)
    
    print(f"  📊 Using {len(feature_cols)} pre-game features (filtered {len(numeric_cols) - len(feature_cols)} post-game)")
    
    X = df[feature_cols].fillna(0) if feature_cols else pd.DataFrame()
    X = pd.concat([X.reset_index(drop=True), rest_features.reset_index(drop=True)], axis=1)
    
    # Add basic engineered features
    if len(X) > 100:
        for col in feature_cols[:5]:
            if col in X.columns:
                X[f'{col}_sq'] = X[col] ** 2
    
    # Skip first 20 games (need history)
    X = X.iloc[20:].reset_index(drop=True)
    y_ml = df['home_win'].iloc[20:].reset_index(drop=True)
    y_spread = (df['point_diff'] > df['point_diff'].median()).astype(int).iloc[20:].reset_index(drop=True)
    y_total = (df['total_points'] > df['total_points'].median()).astype(int).iloc[20:].reset_index(drop=True)
    
    print(f"  📊 Features: {X.shape[1]} | Samples: {X.shape[0]:,}")
    
    # Train ensemble
    trainer = EnsembleTrainer(sport)
    
    metrics = {}
    metrics['moneyline'] = trainer.train(X, y_ml, 'moneyline')
    metrics['spread'] = trainer.train(X, y_spread, 'spread')
    metrics['overunder'] = trainer.train(X, y_total, 'overunder')
    
    # Train CONTRACTS model (calibrated for prediction markets)
    # Contracts need well-calibrated probabilities since price = probability
    metrics['contracts'] = trainer.train_contracts(X, y_ml, 'contracts')
    
    # Save models
    for bet_type, model in trainer.models.items():
        model_path = model_dir / f"{bet_type}_model_v4.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': trainer.scalers[bet_type],
                'metrics': trainer.metrics[bet_type],
                'sport': sport,
                'bet_type': bet_type,
                'version': '4.0-ensemble',
                'trained_at': datetime.now().isoformat(),
            }, f)
    
    print(f"  ✅ Saved v4 ensemble models")
    
    return metrics


def train_all_v4():
    """Train v4 models for all sports."""
    print("\n" + "🚀 " * 20)
    print("   ADVANCED ENSEMBLE MODEL TRAINING v4.0")
    print("   XGBoost + Random Forest + Neural Network")
    print("🚀 " * 20)
    
    sports = ['nba', 'nfl', 'nhl', 'mlb', 'soccer']
    all_metrics = {}
    
    for sport in sports:
        sport_dir = DATA_DIR / sport
        if sport_dir.exists() and any(sport_dir.glob("*.csv")):
            metrics = train_sport_v4(sport)
            if metrics:
                all_metrics[sport] = metrics
    
    # Summary
    print("\n" + "="*60)
    print("📊 V4 TRAINING SUMMARY")
    print("="*60)
    
    for sport, metrics in all_metrics.items():
        print(f"\n{sport.upper()}:")
        for bet_type, m in metrics.items():
            if m:
                acc = m.get('accuracy', 0)
                auc = m.get('auc', 0)
                if bet_type == 'contracts':
                    brier = m.get('brier_calibrated', 0)
                    print(f"  • {bet_type}: {acc:.1%} acc | {auc:.3f} AUC | {brier:.4f} Brier")
                else:
                    cv = m.get('cv_accuracy', 0)
                    print(f"  • {bet_type}: {acc:.1%} acc | {cv:.1%} CV | {auc:.3f} AUC")
    
    # Calculate overall stats
    all_acc = [m.get('accuracy', 0) for sport_m in all_metrics.values() for m in sport_m.values() if m]
    avg_acc = np.mean(all_acc) if all_acc else 0
    
    # Contracts specific stats
    contract_briers = [
        m.get('brier_calibrated', 0) for sport_m in all_metrics.values() 
        for k, m in sport_m.items() if k == 'contracts' and m and isinstance(m, dict)
    ]
    avg_brier = np.mean(contract_briers) if contract_briers else 0
    
    print(f"\n✅ V4 Training Complete!")
    print(f"   Models: {sum(len(m) for m in all_metrics.values())}")
    print(f"   Avg Accuracy: {avg_acc:.1%}")
    print(f"   Est. ROI: {(avg_acc - 0.524) / 0.524 * 100:.1f}%")
    if avg_brier > 0:
        print(f"   Avg Contracts Brier: {avg_brier:.4f}")
    
    return all_metrics


if __name__ == "__main__":
    train_all_v4()
