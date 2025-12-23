"""
NCAA Basketball Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GRAPH_DIR = SCRIPT_DIR.parent / "graphs"
DATA_DIR = Path(__file__).parents[5] / "data" / "ncaa_basketball"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('dark_background')
sns.set_palette("husl")


def load_data():
    """Load NCAA basketball data."""
    cbb_file = DATA_DIR / "cbb.csv"
    if cbb_file.exists():
        return pd.read_csv(cbb_file)
    return None


def plot_conference_analysis(df):
    """Analyze performance by conference."""
    if 'CONF' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    conf_wins = df.groupby('CONF')['W'].mean().sort_values(ascending=False).head(15)
    ax.bar(range(len(conf_wins)), conf_wins.values, color='#3498db', edgecolor='black')
    ax.set_xticks(range(len(conf_wins)))
    ax.set_xticklabels(conf_wins.index, rotation=45, ha='right')
    ax.set_title('NCAA Basketball: Average Wins by Conference', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Wins')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'conference_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: conference_analysis.png")


def plot_seed_vs_wins(df):
    """Analyze tournament seed vs regular season wins."""
    if 'SEED' not in df.columns or 'W' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter to tournament teams
    tourney = df[df['SEED'].notna()]
    
    ax.scatter(tourney['SEED'], tourney['W'], alpha=0.5, s=50)
    ax.set_xlabel('Tournament Seed')
    ax.set_ylabel('Regular Season Wins')
    ax.set_title('NCAA Tournament Seed vs Regular Season Wins', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'seed_vs_wins.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: seed_vs_wins.png")


def main():
    print("🏀 NCAA Basketball Exploratory Data Analysis")
    print("=" * 50)
    
    df = load_data()
    if df is None:
        print("  ❌ No data found")
        return
    
    print(f"  📊 Loaded {len(df):,} team-seasons")
    plot_conference_analysis(df)
    plot_seed_vs_wins(df)
    print(f"\n  ✅ Graphs saved to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
