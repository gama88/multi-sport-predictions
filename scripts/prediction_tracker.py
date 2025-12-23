"""
Prediction Tracker - Manages prediction history and results
Automatically checks game results and updates the CSV
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
HISTORY_FILE = os.path.join(DATA_DIR, 'prediction_history.csv')

def load_history():
    """Load prediction history from CSV"""
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=['date', 'sport', 'bet_type', 'pick', 'confidence', 'odds', 'game_id', 'result', 'actual_score', 'resolved_date'])

def save_history(df):
    """Save prediction history to CSV"""
    df.to_csv(HISTORY_FILE, index=False)
    print(f"✅ Saved {len(df)} predictions to {HISTORY_FILE}")

def add_prediction(sport, bet_type, pick, confidence, odds, game_id=None):
    """Add a new prediction to history"""
    df = load_history()
    
    new_pred = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'sport': sport.lower(),
        'bet_type': bet_type.lower(),
        'pick': pick,
        'confidence': round(confidence, 3),
        'odds': odds,
        'game_id': game_id or f"{sport}_{datetime.now().timestamp()}",
        'result': 'pending',
        'actual_score': '',
        'resolved_date': ''
    }
    
    df = pd.concat([df, pd.DataFrame([new_pred])], ignore_index=True)
    save_history(df)
    print(f"📌 Added prediction: {pick} ({sport} {bet_type})")
    return df

def update_result(game_id, result, actual_score=''):
    """Update the result of a prediction"""
    df = load_history()
    
    mask = df['game_id'] == game_id
    if mask.any():
        df.loc[mask, 'result'] = result
        df.loc[mask, 'actual_score'] = actual_score
        df.loc[mask, 'resolved_date'] = datetime.now().strftime('%Y-%m-%d')
        save_history(df)
        print(f"✅ Updated {game_id}: {result}")
    else:
        print(f"❌ Game ID not found: {game_id}")
    
    return df

def get_stats():
    """Calculate stats from prediction history"""
    df = load_history()
    
    resolved = df[df['result'].isin(['win', 'loss', 'push'])]
    wins = len(resolved[resolved['result'] == 'win'])
    losses = len(resolved[resolved['result'] == 'loss'])
    pushes = len(resolved[resolved['result'] == 'push'])
    pending = len(df[df['result'] == 'pending'])
    
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    
    # Stats by sport
    sport_stats = {}
    for sport in df['sport'].unique():
        sport_df = resolved[resolved['sport'] == sport]
        sw = len(sport_df[sport_df['result'] == 'win'])
        sl = len(sport_df[sport_df['result'] == 'loss'])
        st = sw + sl
        sport_stats[sport] = {
            'wins': sw,
            'losses': sl,
            'win_rate': round(sw / st * 100, 1) if st > 0 else 0
        }
    
    # Stats by bet type
    type_stats = {}
    for bt in df['bet_type'].unique():
        bt_df = resolved[resolved['bet_type'] == bt]
        tw = len(bt_df[bt_df['result'] == 'win'])
        tl = len(bt_df[bt_df['result'] == 'loss'])
        tt = tw + tl
        type_stats[bt] = {
            'wins': tw,
            'losses': tl,
            'win_rate': round(tw / tt * 100, 1) if tt > 0 else 0
        }
    
    return {
        'total_predictions': len(df),
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'pending': pending,
        'win_rate': round(win_rate, 1),
        'by_sport': sport_stats,
        'by_type': type_stats
    }

def check_espn_results(sport='nba'):
    """Check ESPN for finished games and update results"""
    sport_map = {
        'nba': 'basketball/nba',
        'nfl': 'football/nfl',
        'mlb': 'baseball/mlb',
        'nhl': 'hockey/nhl',
        'soccer': 'soccer/eng.1'
    }
    
    if sport not in sport_map:
        print(f"Unknown sport: {sport}")
        return
    
    # Get yesterday and today's games
    dates = [
        datetime.now().strftime('%Y%m%d'),
        (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    ]
    
    df = load_history()
    pending = df[df['result'] == 'pending']
    
    for date in dates:
        url = f"https://site.api.espn.com/apis/site/v2/sports/{sport_map[sport]}/scoreboard?dates={date}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            
            data = resp.json()
            events = data.get('events', [])
            
            for event in events:
                game_id = event.get('id')
                status = event.get('status', {}).get('type', {}).get('name', '')
                
                if 'FINAL' not in status:
                    continue
                
                # Check if we have a pending prediction for this game
                game_preds = pending[pending['game_id'].astype(str) == str(game_id)]
                if game_preds.empty:
                    continue
                
                # Get scores
                comp = event.get('competitions', [{}])[0]
                competitors = comp.get('competitors', [])
                home = next((c for c in competitors if c.get('homeAway') == 'home'), {})
                away = next((c for c in competitors if c.get('homeAway') == 'away'), {})
                
                home_score = int(home.get('score', 0))
                away_score = int(away.get('score', 0))
                home_name = home.get('team', {}).get('displayName', '')
                away_name = away.get('team', {}).get('displayName', '')
                home_won = home_score > away_score
                
                score_str = f"{home_name} {home_score} - {away_score} {away_name}"
                
                for idx, pred in game_preds.iterrows():
                    result = 'loss'
                    pick = pred['pick']
                    bet_type = pred['bet_type']
                    
                    if bet_type == 'moneyline':
                        # Check if picked team won
                        if home_name in pick or home.get('team', {}).get('abbreviation', '') in pick:
                            result = 'win' if home_won else 'loss'
                        elif away_name in pick or away.get('team', {}).get('abbreviation', '') in pick:
                            result = 'win' if not home_won else 'loss'
                    
                    # Update the prediction
                    df.loc[idx, 'result'] = result
                    df.loc[idx, 'actual_score'] = score_str
                    df.loc[idx, 'resolved_date'] = datetime.now().strftime('%Y-%m-%d')
                    print(f"✅ Resolved: {pick} -> {result}")
        
        except Exception as e:
            print(f"Error checking {sport}: {e}")
    
    save_history(df)

def print_summary():
    """Print a summary of prediction history"""
    stats = get_stats()
    
    print("\n" + "="*50)
    print("📊 PREDICTION HISTORY SUMMARY")
    print("="*50)
    print(f"\n🎯 Overall Record: {stats['wins']}-{stats['losses']}-{stats['pushes']}")
    print(f"📈 Win Rate: {stats['win_rate']}%")
    print(f"⏳ Pending: {stats['pending']}")
    print(f"📋 Total Predictions: {stats['total_predictions']}")
    
    print("\n📊 By Sport:")
    for sport, s in stats['by_sport'].items():
        print(f"  {sport.upper()}: {s['wins']}-{s['losses']} ({s['win_rate']}%)")
    
    print("\n📊 By Bet Type:")
    for bt, s in stats['by_type'].items():
        print(f"  {bt.title()}: {s['wins']}-{s['losses']} ({s['win_rate']}%)")
    
    print("\n" + "="*50)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == 'stats':
            print_summary()
        
        elif cmd == 'check':
            sports = sys.argv[2:] if len(sys.argv) > 2 else ['nba', 'nfl', 'mlb', 'nhl']
            for sport in sports:
                print(f"\n🔍 Checking {sport.upper()} results...")
                check_espn_results(sport)
            print_summary()
        
        elif cmd == 'add':
            if len(sys.argv) >= 6:
                sport, bet_type, pick, conf = sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
                odds = sys.argv[6] if len(sys.argv) > 6 else '-110'
                add_prediction(sport, bet_type, pick, conf, odds)
            else:
                print("Usage: python prediction_tracker.py add <sport> <bet_type> <pick> <confidence> [odds]")
        
        elif cmd == 'update':
            if len(sys.argv) >= 4:
                game_id, result = sys.argv[2], sys.argv[3]
                score = sys.argv[4] if len(sys.argv) > 4 else ''
                update_result(game_id, result, score)
            else:
                print("Usage: python prediction_tracker.py update <game_id> <win|loss|push> [score]")
    else:
        print_summary()
