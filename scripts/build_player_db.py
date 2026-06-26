"""
Downloads the Kaggle FIFA World Cup 2026 Player Performance dataset
and ingests it into the SQLite database.
"""
import os
import sqlite3
import pandas as pd
import kagglehub
from futpredict.config import DB_PATH

def ingest_player_stats():
    print("Downloading Kaggle Player Performance Dataset...")
    try:
        path = kagglehub.dataset_download("rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset")
    except Exception as e:
        print(f"   ✗ Failed to download dataset: {e}")
        return

    csv_path = None
    for f in os.listdir(path):
        if f.endswith(".csv"):
            csv_path = os.path.join(path, f)
            break
            
    if not csv_path:
        print("   ✗ No CSV file found in the dataset.")
        return

    print(f"   ✓ Downloaded player dataset.")
    print("   ⚙  Processing player stats...")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"   ✗ Failed to read CSV: {e}")
        return

    # Basic cleaning
    # Convert match_date to string ISO format if not already
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # We only need the core columns to keep the DB fast and small
    core_cols = [
        'player_id', 'player_name', 'team', 'opponent_team', 'match_date',
        'position', 'minutes_played', 'goals', 'assists', 'shots_on_target',
        'expected_goals_xg', 'expected_assists_xa',
        'successful_passes', 'total_passes', 
        'defensive_actions', 'saves', 'goals_conceded',
        'offensive_contribution', 'defensive_contribution', 'player_rating'
    ]
    
    # Check if all columns exist
    missing_cols = [c for c in core_cols if c not in df.columns]
    if missing_cols:
        print(f"   ⚠ Warning: Missing columns in dataset: {missing_cols}")
        core_cols = [c for c in core_cols if c in df.columns]
        
    df = df[core_cols]

    # Insert into DB
    conn = sqlite3.connect(DB_PATH)
    try:
        # We will completely overwrite the table on update
        df.to_sql("player_stats", conn, if_exists="replace", index=False)
        
        # Create indices for fast querying
        conn.execute("CREATE INDEX IF NOT EXISTS idx_player_team ON player_stats(team)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_player_date ON player_stats(match_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_player_name ON player_stats(player_name)")
        conn.commit()
        print(f"   ✓ Inserted {len(df):,} player match records into database.")
    except Exception as e:
        print(f"   ✗ Failed to write to DB: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    ingest_player_stats()
