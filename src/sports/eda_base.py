"""
Base EDA Template - Reusable analysis functions for all sports.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


# Set global style
plt.style.use('dark_background')
sns.set_palette("husl")


class BaseSportEDA(ABC):
    """Base class for sport-specific EDA."""

    def __init__(self, sport_id: str, sport_name: str):
        self.sport_id = sport_id
        self.sport_name = sport_name
        
        # Paths
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path.parents[2] / "data" / sport_id
        self.graph_path = self.base_path / sport_id / "analysis" / "graphs"
        
        # Ensure directories exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.graph_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_data(self) -> Optional[pd.DataFrame]:
        """Load sport-specific data."""
        pass

    @abstractmethod
    def get_home_away_columns(self) -> Tuple[str, str]:
        """Return (home_score_col, away_score_col)."""
        pass

    def save_plot(self, fig: plt.Figure, name: str) -> Path:
        """Save a plot to the graphs directory."""
        path = self.graph_path / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        print(f"  📈 Saved: {path.name}")
        return path

    def plot_score_distribution(self, df: pd.DataFrame) -> Path:
        """Plot score distribution for home and away teams."""
        home_col, away_col = self.get_home_away_columns()
        
        if home_col not in df.columns or away_col not in df.columns:
            print(f"  ⚠️ Columns {home_col}, {away_col} not found")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Home scores
        axes[0].hist(df[home_col].dropna(), bins=40, edgecolor='black', 
                    alpha=0.7, color='#3498db')
        axes[0].set_xlabel('Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'{self.sport_name} Home Scores', fontsize=14, fontweight='bold')
        axes[0].axvline(df[home_col].mean(), color='yellow', linestyle='--', 
                       linewidth=2, label=f'Mean: {df[home_col].mean():.1f}')
        axes[0].legend()

        # Away scores
        axes[1].hist(df[away_col].dropna(), bins=40, edgecolor='black', 
                    alpha=0.7, color='#e74c3c')
        axes[1].set_xlabel('Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'{self.sport_name} Away Scores', fontsize=14, fontweight='bold')
        axes[1].axvline(df[away_col].mean(), color='yellow', linestyle='--', 
                       linewidth=2, label=f'Mean: {df[away_col].mean():.1f}')
        axes[1].legend()

        plt.suptitle(f'{self.sport_name} Score Distribution Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self.save_plot(fig, 'score_distribution')

    def plot_home_advantage(self, df: pd.DataFrame) -> Path:
        """Plot home team advantage analysis."""
        home_col, away_col = self.get_home_away_columns()
        
        if home_col not in df.columns or away_col not in df.columns:
            return None
        
        df = df.dropna(subset=[home_col, away_col])
        home_wins = (df[home_col] > df[away_col]).sum()
        away_wins = (df[away_col] > df[home_col]).sum()
        ties = (df[home_col] == df[away_col]).sum()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        labels = ['Home Wins', 'Away Wins', 'Ties'] if ties > 0 else ['Home Wins', 'Away Wins']
        sizes = [home_wins, away_wins, ties] if ties > 0 else [home_wins, away_wins]
        colors = ['#3498db', '#e74c3c', '#95a5a6']
        
        axes[0].pie(sizes, labels=labels, colors=colors[:len(sizes)], 
                   autopct='%1.1f%%', startangle=90, explode=[0.05] * len(sizes))
        axes[0].set_title('Home vs Away Win Distribution', fontsize=14, fontweight='bold')
        
        # Home advantage metrics
        home_win_pct = home_wins / (home_wins + away_wins) * 100
        avg_margin = (df[home_col] - df[away_col]).mean()
        
        metrics = ['Home Win %', 'Avg Home Margin']
        values = [home_win_pct, avg_margin]
        colors_bar = ['#27ae60' if v > 0 or v > 50 else '#e74c3c' for v in values]
        
        bars = axes[1].bar(metrics, values, color=colors_bar, edgecolor='white')
        axes[1].set_ylabel('Value', fontsize=12)
        axes[1].set_title('Home Advantage Metrics', fontsize=14, fontweight='bold')
        axes[1].axhline(50, color='white', linestyle='--', alpha=0.5)
        
        for bar, val in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'{self.sport_name} Home Advantage Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self.save_plot(fig, 'home_advantage')

    def plot_correlation_matrix(self, df: pd.DataFrame, 
                                columns: Optional[List[str]] = None) -> Path:
        """Plot correlation matrix for numeric columns."""
        if columns:
            cols = [c for c in columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(cols) < 2:
            print("  ⚠️ Not enough numeric columns for correlation")
            return None
        
        # Limit to 20 columns for readability
        cols = cols[:20]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        corr = df[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=len(cols) <= 10, cmap='RdBu_r',
                   center=0, square=True, linewidths=0.5, ax=ax,
                   fmt='.2f' if len(cols) <= 10 else None)
        
        ax.set_title(f'{self.sport_name} Feature Correlation Matrix', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return self.save_plot(fig, 'correlation_matrix')

    def plot_scoring_trends(self, df: pd.DataFrame, 
                           date_col: str = 'date',
                           freq: str = 'M') -> Path:
        """Plot scoring trends over time."""
        home_col, away_col = self.get_home_away_columns()
        
        if date_col not in df.columns:
            print(f"  ⚠️ Date column '{date_col}' not found")
            return None
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, home_col, away_col])
        
        # Aggregate by period
        df.set_index(date_col, inplace=True)
        agg = df[[home_col, away_col]].resample(freq).mean()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(agg.index, agg[home_col], marker='o', label='Home Score', 
               linewidth=2, markersize=4)
        ax.plot(agg.index, agg[away_col], marker='s', label='Away Score', 
               linewidth=2, markersize=4)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title(f'{self.sport_name} Scoring Trends Over Time', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self.save_plot(fig, 'scoring_trends')

    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics."""
        home_col, away_col = self.get_home_away_columns()
        
        stats = {
            'total_games': len(df),
            'date_range': None,
            'home_score_mean': df[home_col].mean() if home_col in df.columns else None,
            'away_score_mean': df[away_col].mean() if away_col in df.columns else None,
            'home_win_pct': None,
        }
        
        if home_col in df.columns and away_col in df.columns:
            home_wins = (df[home_col] > df[away_col]).sum()
            total = home_wins + (df[away_col] > df[home_col]).sum()
            stats['home_win_pct'] = home_wins / total * 100 if total > 0 else None
        
        return stats

    def run_full_analysis(self, df: Optional[pd.DataFrame] = None) -> List[Path]:
        """Run full EDA and generate all graphs."""
        print(f"\n🏆 {self.sport_name} Exploratory Data Analysis")
        print("=" * 50)
        
        if df is None:
            df = self.load_data()
        
        if df is None or len(df) == 0:
            print("  ⚠️ No data available. Run download_datasets.py first.")
            return []
        
        print(f"  📊 Loaded {len(df):,} records")
        
        # Generate all plots
        graphs = []
        
        print("\n📈 Generating visualizations...")
        
        # Score distribution
        path = self.plot_score_distribution(df)
        if path:
            graphs.append(path)
        
        # Home advantage
        path = self.plot_home_advantage(df)
        if path:
            graphs.append(path)
        
        # Correlation matrix
        path = self.plot_correlation_matrix(df)
        if path:
            graphs.append(path)
        
        # Scoring trends
        path = self.plot_scoring_trends(df)
        if path:
            graphs.append(path)
        
        # Summary stats
        stats = self.generate_summary_stats(df)
        print(f"\n📋 Summary Statistics:")
        for key, value in stats.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  • {key}: {value:.2f}")
                else:
                    print(f"  • {key}: {value}")
        
        print(f"\n✅ Generated {len(graphs)} graphs in {self.graph_path}")
        
        return graphs
