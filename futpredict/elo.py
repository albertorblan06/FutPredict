import sqlite3
import datetime
import math
from tqdm import tqdm
from collections import defaultdict
from .config import DB_PATH
from .data import get_connection

def calculate_k_factor(tournament, is_neutral=False):
    """Determine the K-factor based on match importance."""
    t = (tournament or "").lower()
    
    if "world cup" in t and "qualif" not in t:
        return 60
    if any(kw in t for kw in ("copa amér", "copa amer", "uefa euro", "european championship", "african cup", "asian cup")):
        return 50 if "qualif" not in t else 40
    if "qualif" in t:
        return 30
    if "nations league" in t:
        return 30
    if "gold cup" in t or "concacaf" in t:
        return 40
    if "friendly" in t:
        return 15
    return 20

def calculate_elo_history(conn, force=False):
    """
    Calculate Elo ratings for all teams chronologically.
    Stores results in the 'elo_history' table.
    """
    if not force:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='elo_history'")
        if cur.fetchone():
            # Check if populated
            cur = conn.execute("SELECT COUNT(*) FROM elo_history")
            if cur.fetchone()[0] > 1000:
                return

    print("   ⚙  Calculating Historical Elo Ratings...")
    
    conn.execute("DROP TABLE IF EXISTS elo_history")
    conn.execute("""
        CREATE TABLE elo_history (
            match_date TEXT,
            team TEXT,
            elo_before REAL,
            elo_after REAL,
            PRIMARY KEY (match_date, team)
        )
    """)
    
    # Initialize all teams to 1500
    team_elos = defaultdict(lambda: 1500.0)
    
    cur = conn.execute("""
        SELECT date, home_team, away_team, home_score, away_score, tournament, neutral
        FROM matches
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY date ASC
    """)
    matches = cur.fetchall()
    
    inserts = []
    
    for date_str, home, away, hs, as_, tourn, neut in tqdm(matches, desc="   Elo Processing", disable=True):
        elo_h = team_elos[home]
        elo_a = team_elos[away]
        
        # Expected outcomes
        e_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400.0))
        e_a = 1 / (1 + 10 ** ((elo_h - elo_a) / 400.0))
        
        # Actual outcomes
        s_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        s_a = 1.0 if as_ > hs else (0.5 if hs == as_ else 0.0)
        
        # Margin of Victory Multiplier (World Football Elo style)
        gd = abs(hs - as_)
        if gd <= 1:
            g_mult = 1.0
        elif gd == 2:
            g_mult = 1.5
        else:
            g_mult = (11.0 + gd) / 8.0
            
        k = calculate_k_factor(tourn, neut)
        
        # Elo updates
        change_h = k * g_mult * (s_h - e_h)
        change_a = k * g_mult * (s_a - e_a)
        
        elo_h_after = elo_h + change_h
        elo_a_after = elo_a + change_a
        
        inserts.append((date_str, home, elo_h, elo_h_after))
        inserts.append((date_str, away, elo_a, elo_a_after))
        
        team_elos[home] = elo_h_after
        team_elos[away] = elo_a_after
        
        # Batch insert
        if len(inserts) >= 5000:
            conn.executemany("INSERT OR REPLACE INTO elo_history VALUES (?, ?, ?, ?)", inserts)
            inserts = []
            
    if inserts:
        conn.executemany("INSERT OR REPLACE INTO elo_history VALUES (?, ?, ?, ?)", inserts)
        
    conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_elo_team_date ON elo_history(team, match_date)")
    print(f"   ✓  Elo system generated for {len(team_elos)} teams.")

def get_elo(conn, team, match_date=None):
    """
    Get Elo rating for a team on a specific date. 
    If date is None, gets the most recent rating.
    """
    if match_date:
        cur = conn.execute("""
            SELECT elo_before FROM elo_history
            WHERE team = ? AND match_date <= ?
            ORDER BY match_date DESC LIMIT 1
        """, (team, match_date))
    else:
        cur = conn.execute("""
            SELECT elo_after FROM elo_history
            WHERE team = ?
            ORDER BY match_date DESC LIMIT 1
        """, (team,))
        
    row = cur.fetchone()
    if row:
        return row[0]
    return 1500.0 # Default starting Elo
