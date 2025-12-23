"""
Stats Panel Component - Displays model performance statistics.
"""
from textual.widgets import Static, Sparkline
from textual.containers import Vertical, Horizontal, Grid
from textual.app import ComposeResult
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelStats:
    """Statistics for a prediction model."""
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    roi: float = 0.0
    recent_accuracy: List[float] = None  # Last N predictions accuracy
    streak: int = 0  # Positive = win streak, negative = loss streak


class StatsPanel(Static):
    """Panel showing model performance statistics."""

    DEFAULT_CSS = """
    StatsPanel {
        height: auto;
        padding: 1;
        border: solid $accent;
        background: $surface;
        margin: 1;
    }
    
    .stats-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    .stats-grid {
        grid-size: 3;
        grid-gutter: 1 2;
        height: auto;
    }
    
    .stat-box {
        height: 3;
        padding: 0 1;
        border: solid $primary-darken-2;
        text-align: center;
    }
    
    .stat-value {
        text-style: bold;
        color: $text;
    }
    
    .stat-label {
        color: $text-muted;
    }
    
    .stat-positive {
        color: #27ae60;
    }
    
    .stat-negative {
        color: #e74c3c;
    }
    
    .sparkline-container {
        margin-top: 1;
        height: 3;
    }
    
    .sparkline-label {
        color: $text-muted;
        margin-bottom: 0;
    }
    
    Sparkline {
        height: 2;
    }
    """

    def __init__(self, stats: Optional[ModelStats] = None, sport_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self.stats = stats or ModelStats()
        self.sport_name = sport_name

    def compose(self) -> ComposeResult:
        """Compose the stats panel."""
        with Vertical():
            yield Static(f"📈 {self.sport_name} Model Performance", classes="stats-title")
            
            with Grid(classes="stats-grid"):
                # Accuracy
                with Vertical(classes="stat-box"):
                    acc_class = "stat-positive" if self.stats.accuracy >= 0.55 else "stat-negative"
                    yield Static(f"{self.stats.accuracy*100:.1f}%", classes=f"stat-value {acc_class}")
                    yield Static("Accuracy", classes="stat-label")
                
                # Total Predictions
                with Vertical(classes="stat-box"):
                    yield Static(str(self.stats.total_predictions), classes="stat-value")
                    yield Static("Predictions", classes="stat-label")
                
                # ROI
                with Vertical(classes="stat-box"):
                    roi_class = "stat-positive" if self.stats.roi >= 0 else "stat-negative"
                    roi_sign = "+" if self.stats.roi >= 0 else ""
                    yield Static(f"{roi_sign}{self.stats.roi:.1f}%", classes=f"stat-value {roi_class}")
                    yield Static("ROI", classes="stat-label")
                
                # Precision
                with Vertical(classes="stat-box"):
                    yield Static(f"{self.stats.precision*100:.1f}%", classes="stat-value")
                    yield Static("Precision", classes="stat-label")
                
                # Correct
                with Vertical(classes="stat-box"):
                    yield Static(str(self.stats.correct_predictions), classes="stat-value")
                    yield Static("Correct", classes="stat-label")
                
                # Streak
                with Vertical(classes="stat-box"):
                    if self.stats.streak >= 0:
                        streak_text = f"🔥 {self.stats.streak}W"
                        streak_class = "stat-positive"
                    else:
                        streak_text = f"❄️ {abs(self.stats.streak)}L"
                        streak_class = "stat-negative"
                    yield Static(streak_text, classes=f"stat-value {streak_class}")
                    yield Static("Streak", classes="stat-label")
            
            # Recent accuracy sparkline
            if self.stats.recent_accuracy:
                with Vertical(classes="sparkline-container"):
                    yield Static("Recent Performance", classes="sparkline-label")
                    yield Sparkline(
                        self.stats.recent_accuracy,
                        summary_function=max,
                    )

    def update_stats(self, stats: ModelStats) -> None:
        """Update stats and refresh display."""
        self.stats = stats
        self.refresh()
