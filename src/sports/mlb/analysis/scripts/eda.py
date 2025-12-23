"""
MLB Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GRAPH_DIR = SCRIPT_DIR.parent / "graphs"
DATA_DIR = Path(__file__).parents[5] / "data" / "mlb"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('dark_background')
sns.set_palette("husl")


def load_batting_data():
    """Load MLB batting data."""
    batting_file = DATA_DIR / "Batting.csv"
    if batting_file.exists():
        return pd.read_csv(batting_file)
    return None


def plot_batting_trends(df):
    """Plot batting statistics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Calculate batting average by year
    yearly = df.groupby('yearID').agg({
        'H': 'sum',
        'AB': 'sum',
        'HR': 'sum',
        'SO': 'sum'
    }).reset_index()
    yearly['BA'] = yearly['H'] / yearly['AB']
    
    # Batting average trend
    axes[0, 0].plot(yearly['yearID'], yearly['BA'], marker='o', linewidth=2)
    axes[0, 0].set_title('MLB Batting Average Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Batting Average')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Home runs trend
    axes[0, 1].plot(yearly['yearID'], yearly['HR'], marker='s', linewidth=2, color='#e74c3c')
    axes[0, 1].set_title('MLB Total Home Runs by Year', fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Home Runs')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Strikeouts trend
    axes[1, 0].plot(yearly['yearID'], yearly['SO'], marker='^', linewidth=2, color='#9b59b6')
    axes[1, 0].set_title('MLB Total Strikeouts by Year', fontweight='bold')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Strikeouts')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recent years zoom
    recent = yearly[yearly['yearID'] >= 2000]
    axes[1, 1].bar(recent['yearID'], recent['HR'], color='#3498db', edgecolor='black')
    axes[1, 1].set_title('MLB Home Runs (2000+)', fontweight='bold')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Home Runs')
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'batting_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: batting_trends.png")


def main():
    print("⚾ MLB Exploratory Data Analysis")
    print("=" * 50)
    
    df = load_batting_data()
    if df is None:
        print("  ❌ No data found")
        return
    
    print(f"  📊 Loaded {len(df):,} batting records")
    plot_batting_trends(df)
    print(f"\n  ✅ Graphs saved to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
