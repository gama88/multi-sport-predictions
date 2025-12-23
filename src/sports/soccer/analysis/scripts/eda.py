"""
Soccer/Football Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GRAPH_DIR = SCRIPT_DIR.parent / "graphs"
DATA_DIR = Path(__file__).parents[5] / "data" / "soccer"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('dark_background')
sns.set_palette("husl")


def load_data():
    """Load soccer game data."""
    games_file = DATA_DIR / "games.csv"
    if games_file.exists():
        return pd.read_csv(games_file)
    return None


def plot_goal_distribution(df):
    """Plot goals per game distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'home_club_goals' in df.columns:
        axes[0].hist(df['home_club_goals'].dropna(), bins=10, edgecolor='black', alpha=0.7, color='#3498db')
        axes[0].set_title('Soccer: Home Team Goals', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Goals')
        
        axes[1].hist(df['away_club_goals'].dropna(), bins=10, edgecolor='black', alpha=0.7, color='#e74c3c')
        axes[1].set_title('Soccer: Away Team Goals', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Goals')
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'goal_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: goal_distribution.png")


def plot_home_advantage(df):
    """Analyze home team advantage."""
    if 'home_club_goals' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    home_wins = (df['home_club_goals'] > df['away_club_goals']).sum()
    draws = (df['home_club_goals'] == df['away_club_goals']).sum()
    away_wins = (df['away_club_goals'] > df['home_club_goals']).sum()
    
    ax.pie([home_wins, draws, away_wins], 
           labels=['Home Win', 'Draw', 'Away Win'],
           colors=['#3498db', '#95a5a6', '#e74c3c'],
           autopct='%1.1f%%',
           startangle=90)
    ax.set_title('Soccer: Match Outcomes', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'home_advantage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: home_advantage.png")


def main():
    print("⚽ Soccer Exploratory Data Analysis")
    print("=" * 50)
    
    df = load_data()
    if df is None:
        print("  ❌ No data found")
        return
    
    print(f"  📊 Loaded {len(df):,} matches")
    plot_goal_distribution(df)
    plot_home_advantage(df)
    print(f"\n  ✅ Graphs saved to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
