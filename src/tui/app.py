"""
Multi-Sport Predictions TUI - Main Application
"""
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane, Static, DataTable, Label
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.binding import Binding
from textual import work
from datetime import datetime
import asyncio

from .components.game_card import GameCard
from .components.prediction_panel import PredictionPanel
from .components.stats_panel import StatsPanel
from .screens.sport_screen import SportScreen


class SportsPredictionsApp(App):
    """Main TUI Application for Multi-Sport Predictions."""

    CSS = """
    Screen {
        background: $surface;
    }
    
    TabbedContent {
        height: 100%;
    }
    
    TabPane {
        padding: 1;
    }
    
    .sport-header {
        height: 3;
        background: $primary;
        color: $text;
        text-align: center;
        padding: 1;
        text-style: bold;
    }
    
    .live-indicator {
        background: #e74c3c;
        color: white;
        padding: 0 1;
        text-style: bold;
    }
    
    .live-indicator.active {
        background: #27ae60;
    }
    
    .game-container {
        height: 100%;
        padding: 1;
    }
    
    .predictions-panel {
        border: solid $primary;
        padding: 1;
        margin: 1;
        height: auto;
    }
    
    .upcoming-games {
        border: solid $secondary;
        padding: 1;
        margin: 1;
    }
    
    .stats-panel {
        border: solid $accent;
        padding: 1;
        margin: 1;
        height: 8;
    }
    
    DataTable {
        height: auto;
        max-height: 20;
    }
    
    .section-title {
        text-style: bold;
        color: $primary;
        padding: 0 0 1 0;
    }
    
    .prediction-high {
        color: #27ae60;
        text-style: bold;
    }
    
    .prediction-medium {
        color: #f39c12;
    }
    
    .prediction-low {
        color: #e74c3c;
    }
    
    .score {
        text-style: bold;
        color: $text;
    }
    
    .team-name {
        width: 20;
    }
    
    .refresh-indicator {
        dock: right;
        padding: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("p", "toggle_predictions", "Predictions"),
        Binding("l", "toggle_live", "Live Scores"),
        Binding("tab", "next_tab", "Next Sport", show=False),
        Binding("shift+tab", "prev_tab", "Prev Sport", show=False),
        Binding("d", "toggle_dark", "Dark Mode"),
    ]

    TITLE = "🏆 Multi-Sport Predictions"
    SUB_TITLE = "Live Predictions • Upcoming Games • ML-Powered"

    def __init__(self):
        super().__init__()
        self.show_predictions = True
        self.show_live = True
        self.last_refresh = datetime.now()

    def compose(self) -> ComposeResult:
        """Create the main layout."""
        yield Header(show_clock=True)
        
        with TabbedContent():
            # NBA Tab
            with TabPane("🏀 NBA", id="nba"):
                yield SportScreen(sport_id="nba", sport_name="NBA Basketball")
            
            # NCAA Basketball Tab
            with TabPane("🏀 NCAA", id="ncaa_basketball"):
                yield SportScreen(sport_id="ncaa_basketball", sport_name="NCAA Basketball")
            
            # NFL Tab
            with TabPane("🏈 NFL", id="nfl"):
                yield SportScreen(sport_id="nfl", sport_name="NFL Football")
            
            # NCAA Football Tab
            with TabPane("🏈 CFB", id="ncaa_football"):
                yield SportScreen(sport_id="ncaa_football", sport_name="NCAA Football")
            
            # NHL Tab
            with TabPane("🏒 NHL", id="nhl"):
                yield SportScreen(sport_id="nhl", sport_name="NHL Hockey")
            
            # MLB Tab
            with TabPane("⚾ MLB", id="mlb"):
                yield SportScreen(sport_id="mlb", sport_name="MLB Baseball")
            
            # Tennis Tab
            with TabPane("🎾 Tennis", id="tennis"):
                yield SportScreen(sport_id="tennis", sport_name="Tennis")
            
            # Soccer Tab
            with TabPane("⚽ Soccer", id="soccer"):
                yield SportScreen(sport_id="soccer", sport_name="Soccer/Football")

        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.refresh_all_data()

    @work(exclusive=True)
    async def refresh_all_data(self) -> None:
        """Refresh data for all sports."""
        # This will be implemented to fetch live data
        self.last_refresh = datetime.now()
        self.notify("Data refreshed!", title="Refresh")

    def action_refresh(self) -> None:
        """Manual refresh action."""
        self.refresh_all_data()

    def action_toggle_predictions(self) -> None:
        """Toggle predictions visibility."""
        self.show_predictions = not self.show_predictions
        status = "shown" if self.show_predictions else "hidden"
        self.notify(f"Predictions {status}")

    def action_toggle_live(self) -> None:
        """Toggle live scores visibility."""
        self.show_live = not self.show_live
        status = "shown" if self.show_live else "hidden"
        self.notify(f"Live scores {status}")

    def action_next_tab(self) -> None:
        """Switch to next sport tab."""
        tabbed = self.query_one(TabbedContent)
        tabbed.action_next_tab()

    def action_prev_tab(self) -> None:
        """Switch to previous sport tab."""
        tabbed = self.query_one(TabbedContent)
        tabbed.action_previous_tab()


def main():
    """Entry point for the TUI application."""
    app = SportsPredictionsApp()
    app.run()


if __name__ == "__main__":
    main()
