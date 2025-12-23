"""
Configuration management for sports predictions.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json


@dataclass
class SportConfig:
    """Configuration for a specific sport."""
    sport_id: str
    sport_name: str
    enabled: bool = True
    model_path: str = ""
    refresh_interval_seconds: int = 60
    api_source: str = "espn"


@dataclass
class AppConfig:
    """Main application configuration."""
    # Database
    database_path: str = "data/sports.db"
    
    # Models
    models_dir: str = "models"
    
    # Refresh intervals
    live_refresh_seconds: int = 30
    upcoming_refresh_seconds: int = 300
    
    # Sports configuration
    sports: Dict[str, SportConfig] = field(default_factory=dict)
    
    # API keys (optional)
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # UI settings
    theme: str = "dark"
    show_predictions: bool = True
    show_confidence: bool = True
    
    @classmethod
    def load(cls, path: str = "config/config.json") -> "AppConfig":
        """Load configuration from file."""
        config_path = Path(path)
        
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            
            # Parse sports configs
            sports = {}
            for sport_id, sport_data in data.get('sports', {}).items():
                sports[sport_id] = SportConfig(sport_id=sport_id, **sport_data)
            
            return cls(
                database_path=data.get('database_path', cls.database_path),
                models_dir=data.get('models_dir', cls.models_dir),
                live_refresh_seconds=data.get('live_refresh_seconds', cls.live_refresh_seconds),
                upcoming_refresh_seconds=data.get('upcoming_refresh_seconds', cls.upcoming_refresh_seconds),
                sports=sports,
                api_keys=data.get('api_keys', {}),
                theme=data.get('theme', cls.theme),
                show_predictions=data.get('show_predictions', cls.show_predictions),
                show_confidence=data.get('show_confidence', cls.show_confidence),
            )
        
        # Return default config
        return cls.default()

    @classmethod
    def default(cls) -> "AppConfig":
        """Create default configuration."""
        sports = {
            'nba': SportConfig(
                sport_id='nba',
                sport_name='NBA Basketball',
                refresh_interval_seconds=30,
            ),
            'ncaa_basketball': SportConfig(
                sport_id='ncaa_basketball',
                sport_name='NCAA Basketball',
                refresh_interval_seconds=30,
            ),
            'nfl': SportConfig(
                sport_id='nfl',
                sport_name='NFL Football',
                refresh_interval_seconds=60,
            ),
            'ncaa_football': SportConfig(
                sport_id='ncaa_football',
                sport_name='NCAA Football',
                refresh_interval_seconds=60,
            ),
            'nhl': SportConfig(
                sport_id='nhl',
                sport_name='NHL Hockey',
                refresh_interval_seconds=30,
            ),
            'mlb': SportConfig(
                sport_id='mlb',
                sport_name='MLB Baseball',
                refresh_interval_seconds=30,
            ),
        }
        
        return cls(sports=sports)

    def save(self, path: str = "config/config.json") -> None:
        """Save configuration to file."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'database_path': self.database_path,
            'models_dir': self.models_dir,
            'live_refresh_seconds': self.live_refresh_seconds,
            'upcoming_refresh_seconds': self.upcoming_refresh_seconds,
            'sports': {
                sport_id: {
                    'sport_name': sport.sport_name,
                    'enabled': sport.enabled,
                    'model_path': sport.model_path,
                    'refresh_interval_seconds': sport.refresh_interval_seconds,
                    'api_source': sport.api_source,
                }
                for sport_id, sport in self.sports.items()
            },
            'api_keys': self.api_keys,
            'theme': self.theme,
            'show_predictions': self.show_predictions,
            'show_confidence': self.show_confidence,
        }
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_sport(self, sport_id: str) -> Optional[SportConfig]:
        """Get configuration for a specific sport."""
        return self.sports.get(sport_id)

    def get_enabled_sports(self) -> List[SportConfig]:
        """Get list of enabled sports."""
        return [s for s in self.sports.values() if s.enabled]
