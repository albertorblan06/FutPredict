import os
import sys
import json
import sqlite3
import pandas as pd
import warnings

# Add parent directory to path so we can import futpredict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from futpredict.config import DB_PATH

# Suppress the "NoAuthWarning: credentials were not supplied" warning from StatsBomb
warnings.filterwarnings("ignore", module="statsbombpy")

try:
    from statsbombpy import sb
except ImportError:
    sb = None

try:
    import kagglehub
except ImportError:
    kagglehub = None


def get_connection():
    return sqlite3.connect(DB_PATH)


def compute_statsbomb_time_bins(events, match, team_name):
    # We want 6 time bins: 0-15, 15-30, 30-45, 45-60, 60-75, 75-90
    bins_possession = [0] * 6
    bins_corners = [0] * 6
    bins_cards = [0] * 6
    bins_sot = [0] * 6
    
    total_events_in_bin = [0] * 6
    
    for _, row in events.iterrows():
        minute = row.get("minute", 0)
        # Cap at 89 for the 6th bin
        minute = min(minute, 89)
        bin_idx = minute // 15
        if bin_idx > 5:
            continue
            
        team_event = row.get("team") == team_name
        
        # Possession approximation (count passes for this team vs total passes)
        if row.get("type") == "Pass":
            total_events_in_bin[bin_idx] += 1
            if team_event:
                bins_possession[bin_idx] += 1
                if row.get("pass_type") == "Corner":
                    bins_corners[bin_idx] += 1
        
        # Shots on Target
        if row.get("type") == "Shot" and team_event:
            outcome = row.get("shot_outcome", "")
            if outcome in ["Goal", "Saved", "Saved To Post"]:
                bins_sot[bin_idx] += 1
                
        # Cards
        if row.get("type") == "Foul Committed" and team_event:
            if pd.notna(row.get("foul_committed_card")):
                bins_cards[bin_idx] += 1
        elif row.get("type") == "Bad Behaviour" and team_event:
            if pd.notna(row.get("bad_behaviour_card")):
                bins_cards[bin_idx] += 1

    # Convert possession counts to percentages
    possession_pct = []
    for i in range(6):
        if total_events_in_bin[i] > 0:
            possession_pct.append(round((bins_possession[i] / total_events_in_bin[i]) * 100, 1))
        else:
            possession_pct.append(50.0) # default

    return {
        "possession_bins": json.dumps(possession_pct),
        "corners_bins": json.dumps(bins_corners),
        "cards_bins": json.dumps(bins_cards),
        "sot_bins": json.dumps(bins_sot),
        "total_possession": round(sum(bins_possession) / sum(total_events_in_bin) * 100, 1) if sum(total_events_in_bin) > 0 else 50,
        "total_corners": sum(bins_corners),
        "total_cards": sum(bins_cards),
        "total_sot": sum(bins_sot)
    }


def import_statsbomb():
    if sb is None:
        print("statsbombpy not installed. Skipping.")
        return
        
    print("Fetching StatsBomb World Cup Matches...")
    conn = get_connection()
    c = conn.cursor()
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # 2018 (season_id=3) and 2022 (season_id=106)
    for season_id in [3, 106]:
        matches = sb.matches(competition_id=43, season_id=season_id)
        print(f"Found {len(matches)} matches for season {season_id}.")
        
        def fetch_events(match):
            m_id = match['match_id']
            try:
                evts = sb.events(match_id=m_id)
                return match, evts
            except Exception as e:
                return match, None
                
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_events, match): match for _, match in matches.iterrows()}
            
            for future in as_completed(futures):
                match, events = future.result()
                date = match['match_date']
                home_team = match['home_team']
                away_team = match['away_team']
                print(f"  -> Processed {home_team} vs {away_team} ({date})")
                
                if events is None:
                    print(f"     Failed to fetch events.")
                    continue
                
                home_stats = compute_statsbomb_time_bins(events, match, home_team)
                away_stats = compute_statsbomb_time_bins(events, match, away_team)
                
                c.execute("""
                    INSERT INTO advanced_stats (
                        match_date, home_team, away_team, 
                        home_possession, away_possession, 
                        home_corners, away_corners, 
                        home_cards, away_cards, 
                        home_sot, away_sot,
                        home_possession_bins, away_possession_bins,
                        home_corners_bins, away_corners_bins,
                        home_cards_bins, away_cards_bins,
                        home_sot_bins, away_sot_bins
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date, home_team, away_team,
                    home_stats['total_possession'], away_stats['total_possession'],
                    home_stats['total_corners'], away_stats['total_corners'],
                    home_stats['total_cards'], away_stats['total_cards'],
                    home_stats['total_sot'], away_stats['total_sot'],
                    home_stats['possession_bins'], away_stats['possession_bins'],
                    home_stats['corners_bins'], away_stats['corners_bins'],
                    home_stats['cards_bins'], away_stats['cards_bins'],
                    home_stats['sot_bins'], away_stats['sot_bins']
                ))
                conn.commit()


def import_kaggle():
    if kagglehub is None:
        print("kagglehub not installed. Skipping.")
        return
        
    print("Downloading Kaggle 2026 World Cup Match Data...")
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_1c4df5dc4a81d0380d6a72749cef25f6"
    
    try:
        path = kagglehub.dataset_download("swaptr/fifa-wc-2026-matches")
        csv_path = os.path.join(path, "matches.csv")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return
        
    df = pd.read_csv(csv_path)
    conn = get_connection()
    c = conn.cursor()
    
    count = 0
    for _, row in df.iterrows():
        # Kaggle 2026 matches.csv structure
        date = row.get("date")
        home = row.get("home_team")
        away = row.get("away_team")
        
        if pd.isna(date) or pd.isna(home) or pd.isna(away):
            continue
            
        home_pos = row.get("home_possession", 50)
        away_pos = row.get("away_possession", 50)
        
        home_corners = row.get("home_corners", 0)
        away_corners = row.get("away_corners", 0)
        
        home_sot = row.get("home_sot", 0)
        away_sot = row.get("away_sot", 0)
        
        home_cards = row.get("home_cards_yellow", 0) + row.get("home_cards_red", 0)
        away_cards = row.get("away_cards_yellow", 0) + row.get("away_cards_red", 0)
        
        c.execute("""
            INSERT INTO advanced_stats (
                match_date, home_team, away_team, 
                home_possession, away_possession, 
                home_corners, away_corners, 
                home_cards, away_cards, 
                home_sot, away_sot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date, home, away,
            home_pos, away_pos,
            home_corners, away_corners,
            home_cards, away_cards,
            home_sot, away_sot
        ))
        count += 1
    
    conn.commit()
    print(f"Inserted {count} Kaggle matches into advanced_stats.")


def update_advanced_stats(conn):
    """Entry point for predict.py --update-data"""
    import_statsbomb()
    import_kaggle()

if __name__ == "__main__":
    conn = get_connection()
    update_advanced_stats(conn)
