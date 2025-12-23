"""
Run analysis for all sports and generate graphs.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


SPORTS = ['nba', 'nfl', 'nhl', 'mlb', 'ncaa_basketball', 'ncaa_football', 'tennis', 'soccer']


def generate_sample_data(sport: str, n_games: int = 1000) -> pd.DataFrame:
    """Generate sample data for a sport when real data unavailable."""
    np.random.seed(42)
    
    # Sport-specific score distributions
    score_params = {
        'nba': {'home_mean': 108, 'away_mean': 105, 'std': 12},
        'nfl': {'home_mean': 24, 'away_mean': 21, 'std': 10},
        'nhl': {'home_mean': 3.2, 'away_mean': 2.8, 'std': 1.5},
        'mlb': {'home_mean': 4.5, 'away_mean': 4.2, 'std': 2.5},
        'ncaa_basketball': {'home_mean': 72, 'away_mean': 68, 'std': 10},
        'ncaa_football': {'home_mean': 28, 'away_mean': 24, 'std': 14},
        'tennis': {'home_mean': 2, 'away_mean': 1, 'std': 0.8},  # Sets won
        'soccer': {'home_mean': 1.5, 'away_mean': 1.1, 'std': 1.2},  # Goals
    }
    
    params = score_params.get(sport, score_params['nba'])
    
    df = pd.DataFrame({
        'home_score': np.random.normal(params['home_mean'], params['std'], n_games).astype(int),
        'away_score': np.random.normal(params['away_mean'], params['std'], n_games).astype(int),
        'date': pd.date_range(end=pd.Timestamp.now(), periods=n_games, freq='D'),
        'home_team': np.random.choice([f'Team_{i}' for i in range(30)], n_games),
        'away_team': np.random.choice([f'Team_{i}' for i in range(30)], n_games),
    })
    
    # Add some features
    df['home_win_pct'] = np.random.uniform(0.3, 0.7, n_games)
    df['away_win_pct'] = np.random.uniform(0.3, 0.7, n_games)
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['point_diff'] = df['home_score'] - df['away_score']
    
    return df


def run_sport_analysis(sport: str, use_sample: bool = False):
    """Run analysis for a specific sport."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('dark_background')
    sns.set_palette("husl")
    
    print(f"\n{'='*60}")
    print(f"🏆 {sport.upper()} Analysis")
    print('='*60)
    
    # Setup paths
    base_path = Path(__file__).parent.parent / "src" / "sports" / sport
    graph_path = base_path / "analysis" / "graphs"
    graph_path.mkdir(parents=True, exist_ok=True)
    
    # Load or generate data
    data_path = Path(__file__).parent.parent / "data" / sport
    
    if data_path.exists() and not use_sample:
        csv_files = list(data_path.glob("**/*.csv"))
        if csv_files:
            print(f"  📊 Found {len(csv_files)} CSV files")
            df = pd.read_csv(csv_files[0])
        else:
            print("  📝 No CSV files found, using sample data")
            df = generate_sample_data(sport)
    else:
        print("  📝 Using sample data for demonstration")
        df = generate_sample_data(sport)
    
    print(f"  📈 Loaded {len(df):,} records")
    
    # Generate graphs
    graphs_generated = 0
    
    # 1. Score Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'home_score' in df.columns:
        axes[0].hist(df['home_score'].dropna(), bins=40, edgecolor='black', 
                    alpha=0.7, color='#3498db')
        axes[0].axvline(df['home_score'].mean(), color='yellow', linestyle='--', linewidth=2)
        axes[0].set_title(f'{sport.upper()} Home Scores', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Score')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(df['away_score'].dropna(), bins=40, edgecolor='black', 
                    alpha=0.7, color='#e74c3c')
        axes[1].axvline(df['away_score'].mean(), color='yellow', linestyle='--', linewidth=2)
        axes[1].set_title(f'{sport.upper()} Away Scores', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Score')
        axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    fig.savefig(graph_path / 'score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    graphs_generated += 1
    print(f"  📈 Generated: score_distribution.png")
    
    # 2. Home Advantage Pie Chart
    if 'home_score' in df.columns and 'away_score' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        home_wins = (df['home_score'] > df['away_score']).sum()
        away_wins = (df['away_score'] > df['home_score']).sum()
        
        ax.pie([home_wins, away_wins], 
               labels=['Home Wins', 'Away Wins'],
               colors=['#3498db', '#e74c3c'],
               autopct='%1.1f%%',
               startangle=90,
               explode=(0.05, 0))
        ax.set_title(f'{sport.upper()} Home vs Away Wins', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(graph_path / 'home_advantage.png', dpi=150, bbox_inches='tight')
        plt.close()
        graphs_generated += 1
        print(f"  📈 Generated: home_advantage.png")
    
    # 3. Point Differential
    if 'home_score' in df.columns and 'away_score' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        point_diff = df['home_score'] - df['away_score']
        
        ax.hist(point_diff, bins=50, edgecolor='black', alpha=0.7, color='#9b59b6')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Tie Line')
        ax.axvline(point_diff.mean(), color='yellow', linestyle='-', linewidth=2, 
                  label=f'Mean: {point_diff.mean():.1f}')
        ax.set_title(f'{sport.upper()} Point Differential Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Point Differential (Home - Away)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(graph_path / 'point_differential.png', dpi=150, bbox_inches='tight')
        plt.close()
        graphs_generated += 1
        print(f"  📈 Generated: point_differential.png")
    
    # 4. Correlation Matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 3:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr = df[numeric_cols[:15]].corr()
        sns.heatmap(corr, annot=len(numeric_cols) <= 8, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, ax=ax, fmt='.2f')
        ax.set_title(f'{sport.upper()} Feature Correlations', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(graph_path / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        graphs_generated += 1
        print(f"  📈 Generated: correlation_matrix.png")
    
    # 5. Scoring by Team (if applicable)
    if 'home_team' in df.columns and 'home_score' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        team_scores = df.groupby('home_team')['home_score'].mean().sort_values(ascending=False).head(15)
        
        bars = ax.bar(range(len(team_scores)), team_scores.values, color='#3498db', edgecolor='black')
        ax.set_xticks(range(len(team_scores)))
        ax.set_xticklabels(team_scores.index, rotation=45, ha='right')
        ax.set_title(f'{sport.upper()} Average Home Score by Team (Top 15)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Score')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(graph_path / 'team_scoring.png', dpi=150, bbox_inches='tight')
        plt.close()
        graphs_generated += 1
        print(f"  📈 Generated: team_scoring.png")
    
    print(f"\n  ✅ Generated {graphs_generated} graphs in {graph_path}")
    
    return graphs_generated


def main():
    parser = argparse.ArgumentParser(description="Run analysis for all sports")
    parser.add_argument(
        "--sport",
        choices=SPORTS + ['all'],
        default='all',
        help="Sport to analyze"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample data instead of downloaded data"
    )
    args = parser.parse_args()

    print("📊 Multi-Sport Analysis Runner")
    print("=" * 60)

    # Determine sports to analyze
    if args.sport == 'all':
        sports = SPORTS
    else:
        sports = [args.sport]

    total_graphs = 0
    for sport in sports:
        try:
            graphs = run_sport_analysis(sport, use_sample=args.sample)
            total_graphs += graphs
        except Exception as e:
            print(f"  ❌ Error analyzing {sport}: {e}")

    print("\n" + "=" * 60)
    print(f"✅ Analysis complete! Generated {total_graphs} total graphs.")


if __name__ == "__main__":
    main()
