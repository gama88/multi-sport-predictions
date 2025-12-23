"""
NBA Exploratory Data Analysis - Generate insights and visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parents[4]))

# Output directory for graphs
GRAPH_DIR = Path(__file__).parent / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")


def load_data():
    """Load NBA data from various sources."""
    # This will be updated once datasets are downloaded
    print("📊 Loading NBA data...")
    
    # Placeholder - will load from kagglehub downloads
    data_path = Path(__file__).parents[4] / "data" / "nba"
    
    if not data_path.exists():
        print(f"  ⚠️ Data directory not found: {data_path}")
        print("  Run: python scripts/download_datasets.py --sport nba")
        return None
    
    # Look for CSV files
    csv_files = list(data_path.glob("*.csv"))
    if csv_files:
        print(f"  Found {len(csv_files)} CSV files")
        return {f.stem: pd.read_csv(f) for f in csv_files}
    
    return None


def plot_win_distribution(df: pd.DataFrame, save: bool = True):
    """Plot home vs away win distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate win percentages if we have the data
    if 'home_win' in df.columns:
        home_wins = df['home_win'].sum()
        total_games = len(df)
        away_wins = total_games - home_wins
        
        labels = ['Home Wins', 'Away Wins']
        sizes = [home_wins, away_wins]
        colors = ['#3498db', '#e74c3c']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=(0.05, 0))
        ax.set_title('NBA Home vs Away Win Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        path = GRAPH_DIR / "win_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  📈 Saved: {path}")
    
    plt.close()


def plot_scoring_trends(df: pd.DataFrame, save: bool = True):
    """Plot scoring trends over seasons."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'season' in df.columns and 'home_score' in df.columns:
        season_avg = df.groupby('season').agg({
            'home_score': 'mean',
            'away_score': 'mean'
        }).reset_index()
        
        ax.plot(season_avg['season'], season_avg['home_score'], 
                marker='o', label='Home Score', linewidth=2)
        ax.plot(season_avg['season'], season_avg['away_score'], 
                marker='s', label='Away Score', linewidth=2)
        
        ax.set_xlabel('Season', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title('NBA Scoring Trends by Season', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        path = GRAPH_DIR / "scoring_trends.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  📈 Saved: {path}")
    
    plt.close()


def plot_feature_correlation(df: pd.DataFrame, save: bool = True):
    """Plot feature correlation heatmap."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 2:
        # Calculate correlation matrix
        corr = df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        path = GRAPH_DIR / "feature_correlation.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  📈 Saved: {path}")
    
    plt.close()


def plot_point_differential(df: pd.DataFrame, save: bool = True):
    """Plot point differential distribution."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'home_score' in df.columns and 'away_score' in df.columns:
        df['point_diff'] = df['home_score'] - df['away_score']
        
        ax.hist(df['point_diff'], bins=50, edgecolor='black', alpha=0.7,
               color='#9b59b6')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Tie')
        ax.axvline(df['point_diff'].mean(), color='yellow', linestyle='-', 
                   linewidth=2, label=f'Mean: {df["point_diff"].mean():.1f}')
        
        ax.set_xlabel('Point Differential (Home - Away)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('NBA Point Differential Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        path = GRAPH_DIR / "point_differential.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  📈 Saved: {path}")
    
    plt.close()


def generate_sample_graphs():
    """Generate sample graphs with synthetic data for demo."""
    print("📊 Generating sample NBA graphs...")
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_games = 1000
    
    df = pd.DataFrame({
        'home_score': np.random.normal(108, 12, n_games).astype(int),
        'away_score': np.random.normal(105, 12, n_games).astype(int),
        'season': np.random.choice(range(2018, 2024), n_games),
        'home_win_pct': np.random.uniform(0.3, 0.7, n_games),
        'away_win_pct': np.random.uniform(0.3, 0.7, n_games),
    })
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    # Generate all plots
    plot_win_distribution(df)
    plot_scoring_trends(df)
    plot_point_differential(df)
    plot_feature_correlation(df)
    
    print(f"\n✅ Generated {len(list(GRAPH_DIR.glob('*.png')))} graphs in {GRAPH_DIR}")


def main():
    print("🏀 NBA Exploratory Data Analysis")
    print("=" * 50)
    
    # Try to load real data first
    data = load_data()
    
    if data is None:
        print("\n📝 No data found, generating sample graphs...")
        generate_sample_graphs()
    else:
        # Generate graphs with real data
        for name, df in data.items():
            print(f"\n📊 Analyzing: {name}")
            plot_win_distribution(df)
            plot_scoring_trends(df)
            plot_point_differential(df)
            plot_feature_correlation(df)


if __name__ == "__main__":
    main()
