# Trained Models

This directory stores trained ML models for each sport.

## Directory Structure

```
models/
├── nba/
│   ├── model_v1.0.pkl      # Current production model
│   └── model_v1.1.pkl      # Experimental model
├── ncaa_basketball/
├── nfl/
├── ncaa_football/
├── nhl/
└── mlb/
```

## Model Versioning

- `v1.0` - Baseline XGBoost models
- `v1.1` - Enhanced with player props (coming soon)
- `v2.0` - Deep learning models (planned)

## Training

To train a model:

```bash
python -m src.sports.nba.train --seasons 3 --output models/nba/model_v1.0.pkl
```
