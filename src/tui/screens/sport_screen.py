"""
Sport Screen - Simplified version that directly displays games.
"""
import httpx
from textual.widgets import Static
from textual.containers import Vertical, ScrollableContainer
from textual.app import ComposeResult
from textual import work
from datetime import datetime


class SportScreen(Static):
    """Display sport games directly without nested tabs."""

    DEFAULT_CSS = """
    SportScreen {
        height: 100%;
        width: 100%;
        padding: 1;
    }
    
    .games-content {
        height: 100%;
        width: 100%;
    }
    """

    SPORT_ENDPOINTS = {
        'nba': '/basketball/nba/scoreboard',
        'ncaa_basketball': '/basketball/mens-college-basketball/scoreboard',
        'nfl': '/football/nfl/scoreboard',
        'ncaa_football': '/football/college-football/scoreboard',
        'nhl': '/hockey/nhl/scoreboard',
        'mlb': '/baseball/mlb/scoreboard',
        'tennis': '/tennis/atp/scoreboard',
        'soccer': '/soccer/eng.1/scoreboard',
    }

    def __init__(self, sport_id: str, sport_name: str, **kwargs):
        super().__init__(**kwargs)
        self.sport_id = sport_id
        self.sport_name = sport_name

    def compose(self) -> ComposeResult:
        with ScrollableContainer(classes="games-content"):
            yield Static(f"[bold]Loading {self.sport_name} games...[/]", id=f"games-{self.sport_id}")

    def on_mount(self) -> None:
        self.fetch_games()

    @work(exclusive=True)
    async def fetch_games(self) -> None:
        """Fetch and display games from ESPN."""
        endpoint = self.SPORT_ENDPOINTS.get(self.sport_id, '')
        if not endpoint:
            self.query_one(f"#games-{self.sport_id}", Static).update("[red]Sport not supported[/]")
            return
        
        url = f"https://site.api.espn.com/apis/site/v2/sports{endpoint}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                data = response.json()
            
            content = self._format_games(data)
            self.query_one(f"#games-{self.sport_id}", Static).update(content)
            
        except Exception as e:
            self.query_one(f"#games-{self.sport_id}", Static).update(f"[red]Error: {e}[/]")

    def _format_games(self, data: dict) -> str:
        """Format ESPN data into display text."""
        events = data.get('events', [])
        
        if not events:
            return f"[dim]No {self.sport_name} games scheduled today[/]"
        
        lines = [
            f"[bold cyan]{'═' * 50}[/]",
            f"[bold]🏆 {self.sport_name.upper()} - TODAY'S GAMES[/]",
            f"[bold cyan]{'═' * 50}[/]",
            "",
        ]
        
        for event in events:
            try:
                comp = event.get('competitions', [{}])[0]
                competitors = comp.get('competitors', [])
                
                if len(competitors) < 2:
                    continue
                
                # Get teams
                home = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
                away = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])
                
                home_team = home.get('team', {}).get('displayName', 'TBD')
                away_team = away.get('team', {}).get('displayName', 'TBD')
                home_score = home.get('score', '0')
                away_score = away.get('score', '0')
                
                home_record = ''
                away_record = ''
                if home.get('records'):
                    home_record = home['records'][0].get('summary', '')
                if away.get('records'):
                    away_record = away['records'][0].get('summary', '')
                
                # Get status
                status_data = event.get('status', {})
                status_type = status_data.get('type', {}).get('name', '')
                period = status_data.get('period', '')
                clock = status_data.get('displayClock', '')
                
                # Get time
                date_str = event.get('date', '')
                try:
                    game_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    time_str = game_time.strftime('%I:%M %p')
                except:
                    time_str = 'TBD'
                
                # Get broadcast
                broadcasts = comp.get('broadcasts', [])
                broadcast = ''
                if broadcasts and broadcasts[0].get('names'):
                    broadcast = broadcasts[0]['names'][0]
                
                # Format based on status
                if 'FINAL' in status_type:
                    status_icon = "[green]✅ FINAL[/]"
                    score_line = f"[bold]{away_score}[/] - [bold]{home_score}[/]"
                elif 'IN_PROGRESS' in status_type:
                    status_icon = f"[bold red]🔴 LIVE - {period}Q {clock}[/]"
                    score_line = f"[bold yellow]{away_score}[/] - [bold yellow]{home_score}[/]"
                else:
                    status_icon = f"[cyan]📅 {time_str}[/]"
                    score_line = "vs"
                
                # Build the game display
                lines.append(f"┌{'─' * 48}┐")
                lines.append(f"│ {status_icon:<46} │")
                lines.append(f"├{'─' * 48}┤")
                lines.append(f"│ [bold]{away_team:<25}[/] {away_record:>8} │")
                lines.append(f"│ {'':>15}{score_line:^16}{'':>15} │")
                lines.append(f"│ [bold]{home_team:<25}[/] {home_record:>8} │")
                if broadcast:
                    lines.append(f"├{'─' * 48}┤")
                    lines.append(f"│ 📺 {broadcast:<44} │")
                lines.append(f"└{'─' * 48}┘")
                lines.append("")
                
            except Exception as e:
                continue
        
        if len(lines) <= 4:
            return f"[dim]No games found for {self.sport_name}[/]"
        
        return "\n".join(lines)
