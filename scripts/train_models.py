"""
Train XGBoost models for all sports and bet types.
Supports: Moneyline, Spread/ATS, Over/Under, and Parlay optimization.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("Installing xgboost...")
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    import xgboost as xgb
    XGB_AVAILABLE = True


# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


class SportModelTrainer:
    """Train prediction models for a specific sport."""
    
    def __init__(self, sport: str):
        self.sport = sport
        self.data_dir = DATA_DIR / sport
        self.model_dir = MODELS_DIR / sport
        self.model_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.metrics = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for training."""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"  ⚠️ No data files found for {self.sport}")
            return None
        
        # Load the main games file
        for f in csv_files:
            if 'game' in f.name.lower() or f.name == 'cbb.csv' or 'scores' in f.name.lower():
                try:
                    df = pd.read_csv(f)
                    print(f"  📊 Loaded {f.name}: {len(df):,} rows")
                    return df
                except Exception as e:
                    print(f"  ❌ Error loading {f.name}: {e}")
        
        # Fallback to first CSV
        try:
            df = pd.read_csv(csv_files[0])
            print(f"  📊 Loaded {csv_files[0].name}: {len(df):,} rows")
            return df
        except:
            return None
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Extract features and targets from data."""
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            return None, None, None
        
        # Try to identify score columns for target
        score_cols = [c for c in numeric_cols if 'score' in c.lower() or 'pts' in c.lower() or 'goals' in c.lower()]
        
        if len(score_cols) >= 2:
            # Create target: home team wins
            home_col = score_cols[0]
            away_col = score_cols[1]
            
            # Check if columns have valid data
            if df[home_col].notna().sum() > 100 and df[away_col].notna().sum() > 100:
                target_ml = (df[home_col] > df[away_col]).astype(int)
                target_total = df[home_col] + df[away_col]
                target_spread = df[home_col] - df[away_col]
                
                # Feature columns (exclude score columns for prediction)
                feature_cols = [c for c in numeric_cols if c not in score_cols]
                
                if len(feature_cols) >= 2:
                    X = df[feature_cols].fillna(0)
                    return X, target_ml, {'total': target_total, 'spread': target_spread}
        
        # Fallback: use first numeric column as target
        target = df[numeric_cols[0]]
        target_binary = (target > target.median()).astype(int)
        X = df[numeric_cols[1:]].fillna(0)
        
        return X, target_binary, None
    
    def train_moneyline_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train moneyline (winner) prediction model."""
        print(f"    🎯 Training Moneyline model...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else 0.5,
        }
        
        print(f"       Accuracy: {metrics['accuracy']:.1%}")
        
        self.models['moneyline'] = model
        self.scalers['moneyline'] = scaler
        self.metrics['moneyline'] = metrics
        
        return metrics
    
    def train_spread_model(self, X: pd.DataFrame, y_spread: pd.Series) -> dict:
        """Train spread/ATS prediction model."""
        print(f"    📊 Training Spread model...")
        
        # Binary target: does home team cover a typical spread?
        median_spread = y_spread.median()
        y = (y_spread > median_spread).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
        }
        
        print(f"       Accuracy: {metrics['accuracy']:.1%}")
        
        self.models['spread'] = model
        self.scalers['spread'] = scaler
        self.metrics['spread'] = metrics
        
        return metrics
    
    def train_overunder_model(self, X: pd.DataFrame, y_total: pd.Series) -> dict:
        """Train over/under prediction model."""
        print(f"    📈 Training Over/Under model...")
        
        # Binary target: is total over median?
        median_total = y_total.median()
        y = (y_total > median_total).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
        }
        
        print(f"       Accuracy: {metrics['accuracy']:.1%}")
        
        self.models['overunder'] = model
        self.scalers['overunder'] = scaler
        self.metrics['overunder'] = metrics
        
        return metrics
    
    def save_models(self):
        """Save trained models to disk."""
        for bet_type, model in self.models.items():
            model_path = self.model_dir / f"{bet_type}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': self.scalers.get(bet_type),
                    'metrics': self.metrics.get(bet_type),
                    'sport': self.sport,
                    'bet_type': bet_type,
                    'trained_at': datetime.now().isoformat(),
                }, f)
        
        # Save summary
        summary_path = self.model_dir / "model_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'sport': self.sport,
                'models_trained': list(self.models.keys()),
                'metrics': {k: {mk: float(mv) for mk, mv in v.items()} 
                           for k, v in self.metrics.items()},
                'trained_at': datetime.now().isoformat(),
            }, f, indent=2)
        
        print(f"  ✅ Saved {len(self.models)} models to {self.model_dir}")
    
    def train_all(self) -> dict:
        """Train all bet type models."""
        print(f"\n{'='*60}")
        print(f"🏆 Training {self.sport.upper()} Models")
        print('='*60)
        
        df = self.load_data()
        if df is None:
            return {}
        
        X, y_ml, extras = self.prepare_features(df)
        
        if X is None or len(X) < 100:
            print(f"  ⚠️ Insufficient data for training")
            return {}
        
        # Train moneyline model
        self.train_moneyline_model(X, y_ml)
        
        # Train spread model if we have spread data
        if extras and 'spread' in extras:
            self.train_spread_model(X, extras['spread'])
        
        # Train over/under model if we have total data
        if extras and 'total' in extras:
            self.train_overunder_model(X, extras['total'])
        
        # Save all models
        self.save_models()
        
        return self.metrics


def train_all_sports():
    """Train models for all sports."""
    print("\n" + "🎰 " * 20)
    print("   MULTI-SPORT PREDICTION MODEL TRAINING")
    print("🎰 " * 20)
    
    sports = ['nba', 'nfl', 'nhl', 'mlb', 'ncaa_basketball', 'soccer']
    
    all_metrics = {}
    
    for sport in sports:
        sport_dir = DATA_DIR / sport
        if sport_dir.exists() and any(sport_dir.glob("*.csv")):
            trainer = SportModelTrainer(sport)
            metrics = trainer.train_all()
            if metrics:
                all_metrics[sport] = metrics
        else:
            print(f"\n⚠️ Skipping {sport} - no data directory")
    
    # Save overall summary
    summary = {
        'sports_trained': list(all_metrics.keys()),
        'total_models': sum(len(m) for m in all_metrics.values()),
        'results': all_metrics,
        'trained_at': datetime.now().isoformat(),
    }
    
    summary_path = MODELS_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    
    for sport, metrics in all_metrics.items():
        print(f"\n{sport.upper()}:")
        for bet_type, m in metrics.items():
            acc = m.get('accuracy', 0)
            print(f"  • {bet_type}: {acc:.1%} accuracy")
    
    print(f"\n✅ Training complete! Models saved to: {MODELS_DIR}")
    print(f"📋 Summary saved to: {summary_path}")
    
    return all_metrics


if __name__ == "__main__":
    train_all_sports()
