#!/usr/bin/env python3
"""
Chess.com PGN Downloader
Downloads all games in PGN format from Chess.com accounts.
"""

import requests
import os
import time
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Chess.com usernames to download
ACCOUNTS = {
    "childhood": "abdulrahmansoli",
    "current": "abdulrahmansoliman2"
}

OUTPUT_DIR = "PGN-data"
HEADERS = {"User-Agent": "Chess-ML-Pipeline/1.0"}
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 1.5
MAX_RETRIES = 3


def create_session() -> requests.Session:
    """Create a session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_game_archives(session: requests.Session, username: str) -> list:
    """Get list of monthly game archive URLs for a user."""
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    try:
        response = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            print(f"Error fetching archives for {username}: {response.status_code}")
            return []
        data = response.json()
        return data.get("archives", [])
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {username}: {e}")
        return []


def download_pgn_for_month(session: requests.Session, archive_url: str) -> str:
    """Download PGN data for a specific month archive."""
    pgn_url = f"{archive_url}/pgn"
    try:
        response = session.get(pgn_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            print(f"HTTP {response.status_code}", end=" ")
            return ""
        return response.text
    except requests.exceptions.Timeout:
        print("TIMEOUT", end=" ")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"ERROR: {e}", end=" ")
        return ""


def download_all_games(session: requests.Session, username: str, label: str) -> str:
    """Download all games for a user and return combined PGN."""
    print(f"\n{'='*50}")
    print(f"Downloading games for: {username} ({label})")
    print(f"{'='*50}")
    
    archives = get_game_archives(session, username)
    
    if not archives:
        print(f"No archives found for {username}")
        return ""
    
    print(f"Found {len(archives)} monthly archives")
    
    all_pgn = []
    failed_months = []
    
    for i, archive_url in enumerate(archives, 1):
        parts = archive_url.split("/")
        year_month = f"{parts[-2]}/{parts[-1]}"
        
        print(f"  [{i}/{len(archives)}] Downloading {year_month}...", end=" ", flush=True)
        
        pgn_data = download_pgn_for_month(session, archive_url)
        
        if pgn_data:
            all_pgn.append(pgn_data)
            game_count = pgn_data.count("[Event ")
            print(f"{game_count} games")
        else:
            failed_months.append(year_month)
            print("failed - will retry")
        
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Retry failed months
    if failed_months:
        print(f"\nRetrying {len(failed_months)} failed months...")
        time.sleep(3)
        
        for year_month in failed_months:
            archive_url = f"https://api.chess.com/pub/player/{username}/games/{year_month.replace('/', '/')}"
            print(f"  Retrying {year_month}...", end=" ", flush=True)
            
            pgn_data = download_pgn_for_month(session, archive_url)
            if pgn_data:
                all_pgn.append(pgn_data)
                game_count = pgn_data.count("[Event ")
                print(f"{game_count} games")
            else:
                print("still failed")
            
            time.sleep(DELAY_BETWEEN_REQUESTS * 2)
    
    return "\n\n".join(all_pgn)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Chess.com PGN Downloader")
    print(f"Download date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timeout: {REQUEST_TIMEOUT}s | Delay: {DELAY_BETWEEN_REQUESTS}s | Retries: {MAX_RETRIES}")
    
    session = create_session()
    total_games = 0
    
    for label, username in ACCOUNTS.items():
        pgn_data = download_all_games(session, username, label)
        
        if pgn_data:
            filename = f"chess_com_{label}_{username}.pgn"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(pgn_data)
            
            game_count = pgn_data.count("[Event ")
            total_games += game_count
            
            print(f"\nSaved {game_count} games to: {filepath}")
        else:
            print(f"\nNo games downloaded for {username}")
    
    print(f"\n{'='*50}")
    print(f"COMPLETE: Downloaded {total_games} total games")
    print(f"Files saved in: {OUTPUT_DIR}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
