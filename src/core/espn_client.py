"""
ESPN API Client - Fetch live games, scores, and schedules for all sports.
Uses httpx for proper async HTTP requests.
"""
import httpx
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Game:
    """Universal game representation."""
    id: str
    sport: str
    home_team: str
    away_team: str
    home_score: Optional[int]
    away_score: Optional[int]
    status: str  # scheduled, live, final
    start_time: datetime
    venue: str = ""
    broadcast: str = ""
    home_record: str = ""
    away_record: str = ""
    period: str = ""
    clock: str = ""
    odds_spread: Optional[float] = None
    odds_total: Optional[float] = None


class ESPNClient:
    """ESPN API client for all sports."""
    
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"
    
    SPORT_ENDPOINTS = {
        'nba': '/basketball/nba/scoreboard',
        'ncaa_basketball': '/basketball/mens-college-basketball/scoreboard',
        'nfl': '/football/nfl/scoreboard',
        'ncaa_football': '/football/college-football/scoreboard',
        'nhl': '/hockey/nhl/scoreboard',
        'mlb': '/baseball/mlb/scoreboard',
        'tennis': '/tennis/atp/scoreboard',
        'soccer': '/soccer/eng.1/scoreboard',  # Premier League
    }
    
    async def fetch_games(self, sport: str, date: Optional[datetime] = None) -> List[Game]:
        """Fetch games for a sport on a specific date."""
        endpoint = self.SPORT_ENDPOINTS.get(sport)
        if not endpoint:
            return []
        
        url = f"{self.BASE_URL}{endpoint}"
        
        params = {}
        if date:
            params['dates'] = date.strftime('%Y%m%d')
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                if response.status_code != 200:
                    return []
                data = response.json()
                return self._parse_games(data, sport)
        except Exception as e:
            print(f"ESPN API Error ({sport}): {e}")
            return []
    
    def _parse_games(self, data: Dict, sport: str) -> List[Game]:
        """Parse ESPN API response into Game objects."""
        games = []
        
        events = data.get('events', [])
        for event in events:
            try:
                competition = event.get('competitions', [{}])[0]
                competitors = competition.get('competitors', [])
                
                if len(competitors) < 2:
                    continue
                
                # Find home and away teams
                home = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
                away = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])
                
                home_team = home.get('team', {}).get('displayName', 'TBD')
                away_team = away.get('team', {}).get('displayName', 'TBD')
                
                home_score = int(home.get('score', 0)) if home.get('score') else None
                away_score = int(away.get('score', 0)) if away.get('score') else None
                
                # Get status
                status_data = event.get('status', {})
                status_type = status_data.get('type', {}).get('name', 'STATUS_SCHEDULED')
                
                if 'FINAL' in status_type:
                    status = 'final'
                elif 'IN_PROGRESS' in status_type or status_type == 'STATUS_IN_PROGRESS':
                    status = 'live'
                else:
                    status = 'scheduled'
                
                # Parse start time
                start_str = event.get('date', '')
                try:
                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                except:
                    start_time = datetime.now()
                
                # Get records
                home_record = home.get('records', [{}])[0].get('summary', '') if home.get('records') else ''
                away_record = away.get('records', [{}])[0].get('summary', '') if away.get('records') else ''
                
                # Get venue and broadcast
                venue = competition.get('venue', {}).get('fullName', '')
                broadcasts = competition.get('broadcasts', [])
                broadcast = ''
                if broadcasts and len(broadcasts) > 0:
                    names = broadcasts[0].get('names', [])
                    broadcast = names[0] if names else ''
                
                # Get period/clock for live games
                period = str(status_data.get('period', '')) if status_data.get('period') else ''
                clock = status_data.get('displayClock', '')
                
                # Get odds if available
                odds = competition.get('odds', [{}])[0] if competition.get('odds') else {}
                spread = odds.get('spread')
                total = odds.get('overUnder')
                
                games.append(Game(
                    id=event.get('id', ''),
                    sport=sport,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    status=status,
                    start_time=start_time,
                    venue=venue[:50] if venue else '',
                    broadcast=broadcast,
                    home_record=home_record,
                    away_record=away_record,
                    period=period,
                    clock=clock,
                    odds_spread=float(spread) if spread else None,
                    odds_total=float(total) if total else None,
                ))
            except Exception as e:
                continue
        
        return games


# Test function
async def test():
    client = ESPNClient()
    
    for sport in ['nba', 'nfl', 'nhl']:
        print(f"\n{'='*50}")
        print(f"📊 {sport.upper()} Games Today")
        print('='*50)
        
        games = await client.fetch_games(sport)
        
        if not games:
            print("  No games scheduled")
            continue
        
        for game in games[:5]:
            status_icon = "🔴" if game.status == 'live' else "✅" if game.status == 'final' else "📅"
            
            if game.status == 'live' or game.status == 'final':
                score = f"{game.away_score} - {game.home_score}"
            else:
                score = game.start_time.strftime('%I:%M %p')
            
            print(f"  {status_icon} {game.away_team} @ {game.home_team}: {score}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test())
