"""
NBA Exploratory Data Analysis
Generates visualizations and statistical summaries for NBA data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
GRAPH_DIR = SCRIPT_DIR.parent / "graphs"
DATA_DIR = Path(__file__).parents[5] / "data" / "nba"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('dark_background')
sns.set_palette("husl")


def load_data():
    """Load NBA game data."""
    games_file = DATA_DIR / "games.csv"
    if games_file.exists():
        return pd.read_csv(games_file)
    return None


def plot_score_distribution(df):
    """Plot home vs away score distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df['PTS_home'].dropna(), bins=40, edgecolor='black', alpha=0.7, color='#3498db')
    axes[0].axvline(df['PTS_home'].mean(), color='yellow', linestyle='--', linewidth=2)
    axes[0].set_title('NBA Home Team Scores', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Points')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(df['PTS_away'].dropna(), bins=40, edgecolor='black', alpha=0.7, color='#e74c3c')
    axes[1].axvline(df['PTS_away'].mean(), color='yellow', linestyle='--', linewidth=2)
    axes[1].set_title('NBA Away Team Scores', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Points')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: score_distribution.png")


def plot_home_advantage(df):
    """Analyze home court advantage."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    home_wins = (df['PTS_home'] > df['PTS_away']).sum()
    away_wins = (df['PTS_away'] > df['PTS_home']).sum()
    
    ax.pie([home_wins, away_wins], 
           labels=['Home Wins', 'Away Wins'],
           colors=['#3498db', '#e74c3c'],
           autopct='%1.1f%%',
           startangle=90,
           explode=(0.05, 0))
    ax.set_title('NBA Home Court Advantage', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'home_advantage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: home_advantage.png")


def plot_season_trends(df):
    """Plot scoring trends by season."""
    if 'SEASON' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    season_avg = df.groupby('SEASON').agg({
        'PTS_home': 'mean',
        'PTS_away': 'mean'
    }).reset_index()
    
    ax.plot(season_avg['SEASON'], season_avg['PTS_home'], marker='o', label='Home', linewidth=2)
    ax.plot(season_avg['SEASON'], season_avg['PTS_away'], marker='s', label='Away', linewidth=2)
    ax.set_xlabel('Season')
    ax.set_ylabel('Average Points')
    ax.set_title('NBA Scoring Trends by Season', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'season_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: season_trends.png")


def main():
    print("🏀 NBA Exploratory Data Analysis")
    print("=" * 50)
    
    df = load_data()
    
    if df is None:
        print("  ❌ No data found. Run download_data.bat first.")
        return
    
    print(f"  📊 Loaded {len(df):,} games")
    
    plot_score_distribution(df)
    plot_home_advantage(df)
    plot_season_trends(df)
    
    print(f"\n  ✅ Graphs saved to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
