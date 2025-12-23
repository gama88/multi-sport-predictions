"""
Search GitHub for sports datasets and prediction projects.
"""
import subprocess
import json
from pathlib import Path


def search_github(query: str, limit: int = 10) -> list:
    """Search GitHub for repositories."""
    try:
        result = subprocess.run(
            ['gh', 'search', 'repos', query, '--limit', str(limit),
             '--json', 'name,description,url,stargazersCount', '--sort', 'stars'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return []
    except Exception as e:
        print(f"  Error searching '{query}': {e}")
        return []


# Search queries for each sport
GITHUB_QUERIES = {
    'nba': ['nba prediction machine learning', 'nba-api python', 'basketball analytics'],
    'nfl': ['nfl prediction model', 'nfl analytics python', 'football machine learning'],
    'nhl': ['nhl prediction', 'hockey analytics python', 'nhl machine learning'],
    'mlb': ['mlb prediction', 'baseball analytics sabermetrics', 'baseball machine learning'],
    'ncaa_basketball': ['march madness prediction', 'ncaa basketball machine learning', 'bracketology'],
    'ncaa_football': ['college football prediction', 'cfb analytics', 'college football machine learning'],
}


def main():
    print("🔍 Searching GitHub for Sports Prediction Projects")
    print("=" * 60)
    
    all_results = {}
    
    for sport, queries in GITHUB_QUERIES.items():
        print(f"\n🏆 {sport.upper()}")
        print("-" * 40)
        
        sport_repos = []
        seen_urls = set()
        
        for query in queries:
            print(f"  Searching: '{query}'...")
            results = search_github(query)
            
            for repo in results:
                if repo.get('url') not in seen_urls:
                    seen_urls.add(repo.get('url'))
                    sport_repos.append({
                        'name': repo.get('name'),
                        'description': repo.get('description', '')[:100] if repo.get('description') else '',
                        'url': repo.get('url'),
                        'stars': repo.get('stargazersCount', 0)
                    })
        
        # Sort by stars
        sport_repos.sort(key=lambda x: x['stars'], reverse=True)
        all_results[sport] = sport_repos[:10]  # Top 10 per sport
        
        print(f"  Found {len(sport_repos)} repositories")
        for repo in sport_repos[:3]:
            print(f"    ⭐ {repo['stars']} - {repo['name']}")
    
    # Save results
    output_path = Path("data/github_repos.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
