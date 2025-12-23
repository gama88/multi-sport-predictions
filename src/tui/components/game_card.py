"""
Game Card Component - Displays individual game information.
"""
from textual.widgets import Static
from textual.containers import Horizontal, Vertical
from textual.app import ComposeResult
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class GameData:
    """Data structure for a game."""
    game_id: str
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    quarter: Optional[str] = None
    time_remaining: Optional[str] = None
    status: str = "scheduled"  # scheduled, live, final
    start_time: Optional[datetime] = None
    prediction_home_win: Optional[float] = None
    prediction_spread: Optional[float] = None
    prediction_total: Optional[float] = None


class GameCard(Static):
    """A card component displaying game information."""

    DEFAULT_CSS = """
    GameCard {
        height: auto;
        min-height: 5;
        margin: 0 0 1 0;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }
    
    GameCard.live {
        border: solid #e74c3c;
        background: $surface-darken-1;
    }
    
    GameCard.final {
        border: solid $accent;
    }
    
    .game-header {
        height: 1;
        margin-bottom: 1;
    }
    
    .game-status {
        dock: left;
        padding: 0 1;
    }
    
    .game-status.live {
        background: #e74c3c;
        color: white;
        text-style: bold;
    }
    
    .game-status.final {
        background: $accent;
        color: white;
    }
    
    .game-status.scheduled {
        background: $primary;
        color: white;
    }
    
    .game-time {
        dock: right;
        color: $text-muted;
    }
    
    .teams-container {
        height: auto;
    }
    
    .team-row {
        height: 1;
    }
    
    .team-name {
        width: 1fr;
    }
    
    .team-name.winner {
        text-style: bold;
        color: #27ae60;
    }
    
    .team-score {
        width: 4;
        text-align: right;
        text-style: bold;
    }
    
    .prediction-row {
        height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    
    .prediction-value {
        color: #9b59b6;
        text-style: bold;
    }
    
    .confidence-high {
        color: #27ae60;
    }
    
    .confidence-medium {
        color: #f39c12;
    }
    
    .confidence-low {
        color: #e74c3c;
    }
    """

    def __init__(self, game: GameData, **kwargs):
        super().__init__(**kwargs)
        self.game = game
        if game.status == "live":
            self.add_class("live")
        elif game.status == "final":
            self.add_class("final")

    def compose(self) -> ComposeResult:
        """Compose the game card layout."""
        with Vertical(classes="game-content"):
            # Header with status and time
            with Horizontal(classes="game-header"):
                status_class = f"game-status {self.game.status}"
                if self.game.status == "live":
                    status_text = f"🔴 {self.game.quarter} {self.game.time_remaining}"
                elif self.game.status == "final":
                    status_text = "FINAL"
                else:
                    status_text = "SCHEDULED"
                yield Static(status_text, classes=status_class)
                
                if self.game.start_time:
                    time_str = self.game.start_time.strftime("%I:%M %p")
                    yield Static(time_str, classes="game-time")

            # Teams and scores
            with Vertical(classes="teams-container"):
                # Away team
                with Horizontal(classes="team-row"):
                    away_class = "team-name"
                    if self.game.status == "final" and self.game.away_score and self.game.home_score:
                        if self.game.away_score > self.game.home_score:
                            away_class += " winner"
                    yield Static(self.game.away_team, classes=away_class)
                    if self.game.away_score is not None:
                        yield Static(str(self.game.away_score), classes="team-score")

                # Home team
                with Horizontal(classes="team-row"):
                    home_class = "team-name"
                    if self.game.status == "final" and self.game.away_score and self.game.home_score:
                        if self.game.home_score > self.game.away_score:
                            home_class += " winner"
                    yield Static(f"{self.game.home_team} (H)", classes=home_class)
                    if self.game.home_score is not None:
                        yield Static(str(self.game.home_score), classes="team-score")

            # Prediction row
            if self.game.prediction_home_win is not None:
                with Horizontal(classes="prediction-row"):
                    confidence = self.game.prediction_home_win * 100
                    if confidence >= 70:
                        conf_class = "confidence-high"
                    elif confidence >= 55:
                        conf_class = "confidence-medium"
                    else:
                        conf_class = "confidence-low"
                    
                    prediction_text = f"Win Prob: "
                    yield Static(prediction_text)
                    yield Static(f"{confidence:.1f}%", classes=f"prediction-value {conf_class}")
