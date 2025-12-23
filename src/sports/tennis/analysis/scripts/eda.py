"""
Tennis Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GRAPH_DIR = SCRIPT_DIR.parent / "graphs"
DATA_DIR = Path(__file__).parents[5] / "data" / "tennis"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('dark_background')


def plot_surface_win_rates():
    """Analyze win rates by surface type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
    home_advantage = [52.1, 54.3, 51.8, 50.5]  # Sample data
    
    colors = ['#3498db', '#e67e22', '#27ae60', '#9b59b6']
    ax.bar(surfaces, home_advantage, color=colors, edgecolor='black')
    ax.axhline(50, color='red', linestyle='--', linewidth=2, label='50% baseline')
    ax.set_ylabel('Home Win %')
    ax.set_title('Tennis: Win Rate by Surface', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / 'surface_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: surface_analysis.png")


def main():
    print("🎾 Tennis Exploratory Data Analysis")
    print("=" * 50)
    plot_surface_win_rates()
    print(f"\n  ✅ Graphs saved to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
