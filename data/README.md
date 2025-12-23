# Data directory for multi-sport predictions

This directory stores:
- `sports.db` - SQLite database with games, predictions, and team stats
- Sport-specific subdirectories for raw data

## Directory Structure

```
data/
├── sports.db           # Main SQLite database
├── nba/                # NBA-specific data
├── ncaa_basketball/    # NCAA Basketball data
├── nfl/                # NFL data
├── ncaa_football/      # NCAA Football data
├── nhl/                # NHL data
└── mlb/                # MLB data
```

## Data Sources

- ESPN API for live scores and schedules
- Basketball Reference for historical NBA data
- Sports Reference for other sports
- KenPom for NCAA basketball ratings
