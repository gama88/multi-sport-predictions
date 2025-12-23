"""
NBA Feature Engineering - Create features for NBA predictions.
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class NBAFeatureEngineer:
    """Creates features for NBA game predictions."""

    def __init__(self):
        self.rolling_windows = [5, 10, 20]  # Games to look back
        self.team_stats_cache: Dict[str, pd.DataFrame] = {}

    def create_all_features(
        self,
        games_df: pd.DataFrame,
        team_stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create all features for prediction."""
        df = games_df.copy()
        
        # Basic team stats
        df = self._add_basic_stats(df, team_stats_df)
        
        # Rolling performance
        df = self._add_rolling_stats(df)
        
        # Rest days
        df = self._add_rest_features(df)
        
        # Head-to-head
        df = self._add_h2h_features(df)
        
        # Advanced metrics
        df = self._add_advanced_metrics(df)
        
        # Situational
        df = self._add_situational_features(df)
        
        return df

    def _add_basic_stats(
        self, df: pd.DataFrame, team_stats_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add basic team statistics."""
        # Merge home team stats
        home_stats = team_stats_df.add_prefix('home_')
        df = df.merge(
            home_stats,
            left_on='home_team_id',
            right_on='home_team_id',
            how='left'
        )
        
        # Merge away team stats
        away_stats = team_stats_df.add_prefix('away_')
        df = df.merge(
            away_stats,
            left_on='away_team_id',
            right_on='away_team_id',
            how='left'
        )
        
        return df

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling performance statistics."""
        for window in self.rolling_windows:
            # Points scored
            df[f'home_pts_last{window}'] = df.groupby('home_team_id')['home_score'].rolling(
                window, min_periods=1
            ).mean().reset_index(drop=True)
            
            df[f'away_pts_last{window}'] = df.groupby('away_team_id')['away_score'].rolling(
                window, min_periods=1
            ).mean().reset_index(drop=True)
            
            # Win rate
            df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
            df[f'home_winrate_last{window}'] = df.groupby('home_team_id')['home_win'].rolling(
                window, min_periods=1
            ).mean().reset_index(drop=True)
        
        return df

    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rest and travel features."""
        df = df.sort_values(['date']).reset_index(drop=True)
        
        # Calculate days since last game
        for team_type in ['home', 'away']:
            team_col = f'{team_type}_team_id'
            df[f'{team_type}_rest_days'] = df.groupby(team_col)['date'].diff().dt.days.fillna(3)
            
            # Back-to-back flag
            df[f'{team_type}_is_b2b'] = (df[f'{team_type}_rest_days'] == 1).astype(int)
            
            # 3 games in 4 nights
            df[f'{team_type}_3in4'] = 0  # Simplified, would need more complex logic
        
        # Rest advantage
        df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
        
        return df

    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head historical features."""
        # This would require historical matchup data
        # Simplified placeholder
        df['h2h_home_wins'] = 0
        df['h2h_total_games'] = 0
        df['h2h_avg_margin'] = 0
        
        return df

    def _add_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced basketball metrics."""
        # Four Factors
        if 'home_efg_pct' not in df.columns:
            df['home_efg_pct'] = 0.5
            df['away_efg_pct'] = 0.5
        
        if 'home_tov_pct' not in df.columns:
            df['home_tov_pct'] = 0.14
            df['away_tov_pct'] = 0.14
        
        if 'home_reb_pct' not in df.columns:
            df['home_reb_pct'] = 0.5
            df['away_reb_pct'] = 0.5
        
        if 'home_ft_rate' not in df.columns:
            df['home_ft_rate'] = 0.2
            df['away_ft_rate'] = 0.2
        
        # Calculate four factors differential
        df['efg_diff'] = df['home_efg_pct'] - df['away_efg_pct']
        df['tov_diff'] = df['away_tov_pct'] - df['home_tov_pct']  # Lower is better
        df['reb_diff'] = df['home_reb_pct'] - df['away_reb_pct']
        df['ft_rate_diff'] = df['home_ft_rate'] - df['away_ft_rate']
        
        return df

    def _add_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational features."""
        # Day of week
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Month of season (early, mid, late)
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['season_phase'] = df['month'].apply(
            lambda x: 0 if x in [10, 11, 12] else (1 if x in [1, 2] else 2)
        )
        
        return df

    def get_feature_list(self) -> List[str]:
        """Get list of all features used."""
        features = [
            # Basic stats
            'home_win_pct', 'away_win_pct',
            'home_ppg', 'away_ppg',
            'home_opp_ppg', 'away_opp_ppg',
            
            # Rolling stats
            'home_pts_last5', 'home_pts_last10', 'home_pts_last20',
            'away_pts_last5', 'away_pts_last10', 'away_pts_last20',
            'home_winrate_last5', 'home_winrate_last10', 'home_winrate_last20',
            
            # Rest
            'home_rest_days', 'away_rest_days',
            'home_is_b2b', 'away_is_b2b',
            'rest_advantage',
            
            # H2H
            'h2h_home_wins', 'h2h_total_games', 'h2h_avg_margin',
            
            # Advanced
            'home_off_rating', 'away_off_rating',
            'home_def_rating', 'away_def_rating',
            'home_net_rating', 'away_net_rating',
            'home_pace', 'away_pace',
            'efg_diff', 'tov_diff', 'reb_diff', 'ft_rate_diff',
            
            # Situational
            'is_weekend', 'season_phase',
        ]
        return features
