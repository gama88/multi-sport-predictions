"""
NFL Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GRAPH_DIR = SCRIPT_DIR.parent / "graphs"
DATA_DIR = Path(__file__).parents[5] / "data" / "nfl"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('dark_background')
sns.set_palette("husl")


def load_data():
    """Load NFL game data."""
    scores_file = DATA_DIR / "spreadspoke_scores.csv"
    if scores_file.exists():
        return pd.read_csv(scores_file)
    return None


def plot_score_distribution(df):
    """Plot score distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df['score_home'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='#3498db')
    axes[0].set_title('NFL Home Team Scores', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Points')
    
    axes[1].hist(df['score_away'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='#e74c3c')
    axes[1].set_title('NFL Away Team Scores', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Points')
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: score_distribution.png")


def plot_spread_analysis(df):
    """Analyze point spreads vs actual results."""
    if 'spread_favorite' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df['actual_spread'] = df['score_home'] - df['score_away']
    ax.scatter(df['spread_favorite'], df['actual_spread'], alpha=0.3, s=10)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Vegas Spread')
    ax.set_ylabel('Actual Point Differential')
    ax.set_title('NFL Spread vs Actual Results', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'spread_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: spread_analysis.png")


def main():
    print("🏈 NFL Exploratory Data Analysis")
    print("=" * 50)
    
    df = load_data()
    if df is None:
        print("  ❌ No data found")
        return
    
    print(f"  📊 Loaded {len(df):,} games")
    plot_score_distribution(df)
    plot_spread_analysis(df)
    print(f"\n  ✅ Graphs saved to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
