"""
NCAA Football Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GRAPH_DIR = SCRIPT_DIR.parent / "graphs"
DATA_DIR = Path(__file__).parents[5] / "data" / "ncaa_football"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('dark_background')


def main():
    print("🏈 NCAA Football Exploratory Data Analysis")
    print("=" * 50)
    print("  📊 Analysis scripts ready - add data to generate graphs")
    print(f"  📁 Graphs will be saved to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
