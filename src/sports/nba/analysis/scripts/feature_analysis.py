"""
NBA Feature Analysis
Analyzes feature importance and correlations for model building.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GRAPH_DIR = SCRIPT_DIR.parent / "graphs"
DATA_DIR = Path(__file__).parents[5] / "data" / "nba"

plt.style.use('dark_background')


def analyze_features():
    """Analyze feature correlations."""
    games_file = DATA_DIR / "games.csv"
    if not games_file.exists():
        print("No data found")
        return
    
    df = pd.read_csv(games_file)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Correlation matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[numeric_cols[:15]].corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', ax=ax)
    ax.set_title('NBA Feature Correlations', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'feature_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: feature_correlation.png")


if __name__ == "__main__":
    print("🏀 NBA Feature Analysis")
    analyze_features()
