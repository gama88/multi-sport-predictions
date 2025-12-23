"""
NHL Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GRAPH_DIR = SCRIPT_DIR.parent / "graphs"
DATA_DIR = Path(__file__).parents[5] / "data" / "nhl"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('dark_background')
sns.set_palette("husl")


def load_data():
    """Load NHL game data."""
    game_file = DATA_DIR / "game.csv"
    if game_file.exists():
        return pd.read_csv(game_file)
    return None


def plot_goal_distribution(df):
    """Plot goal distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df['home_goals'].dropna(), bins=15, edgecolor='black', alpha=0.7, color='#3498db')
    axes[0].set_title('NHL Home Team Goals', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Goals')
    
    axes[1].hist(df['away_goals'].dropna(), bins=15, edgecolor='black', alpha=0.7, color='#e74c3c')
    axes[1].set_title('NHL Away Team Goals', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Goals')
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'goal_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: goal_distribution.png")


def plot_overtime_analysis(df):
    """Analyze overtime games."""
    if 'outcome' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    outcome_counts = df['outcome'].value_counts()
    ax.bar(outcome_counts.index, outcome_counts.values, color=['#3498db', '#e74c3c', '#27ae60'])
    ax.set_title('NHL Game Outcomes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Count')
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'outcome_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: outcome_analysis.png")


def main():
    print("🏒 NHL Exploratory Data Analysis")
    print("=" * 50)
    
    df = load_data()
    if df is None:
        print("  ❌ No data found")
        return
    
    print(f"  📊 Loaded {len(df):,} games")
    plot_goal_distribution(df)
    plot_overtime_analysis(df)
    print(f"\n  ✅ Graphs saved to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
