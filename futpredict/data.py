"""
Data acquisition and SQLite database setup.
Downloads match results and historical rankings, imports into DB.
"""
import csv
import os
import sqlite3
from urllib.request import urlopen
from .config import (DATA_DIR, DB_PATH, RESULTS_CSV, RANKINGS_HIST_CSV, SHOOTOUTS_CSV,
                     RESULTS_URL, RANKINGS_URL, SHOOTOUTS_URL)


def download_csv(url, filepath):
    """Download CSV from URL. Returns True on success, falls back to cache."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        print(f"   ⬇  Downloading {os.path.basename(filepath)}...")
        response = urlopen(url, timeout=60)
        data = response.read()
        with open(filepath, "wb") as f:
            f.write(data)
        size_kb = len(data) / 1024
        print(f"   ✓  Downloaded ({size_kb:,.0f} KB)")
        return True
    except Exception as e:
        if os.path.exists(filepath):
            print(f"   ⚠  Download failed, using cached file: {e}")
            return True
        print(f"   ✗  Download failed: {e}")
        return False


def get_connection():
    """Get a SQLite connection with optimized pragmas."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def setup_database(conn):
    """Create (or recreate) tables for fresh data import."""
    conn.executescript("""
        DROP TABLE IF EXISTS matches;
        DROP TABLE IF EXISTS fifa_rankings;
        DROP TABLE IF EXISTS shootouts;

        CREATE TABLE matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_score INTEGER NOT NULL,
            away_score INTEGER NOT NULL,
            tournament TEXT,
            city TEXT,
            country TEXT,
            neutral TEXT,
            advance_team TEXT
        );

        CREATE TABLE shootouts (
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            winner TEXT NOT NULL
        );

        CREATE TABLE fifa_rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team TEXT NOT NULL,
            total_points REAL,
            date TEXT NOT NULL,
            team_short TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            team_a TEXT,
            team_b TEXT,
            venue TEXT,
            lambda_a REAL,
            lambda_b REAL,
            win_a REAL,
            draw REAL,
            win_b REAL,
            most_likely_score TEXT,
            simulations INTEGER
        );

        CREATE TABLE IF NOT EXISTS advanced_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            
            -- Targets (Full Match Aggregates)
            home_possession REAL,
            away_possession REAL,
            home_corners INTEGER,
            away_corners INTEGER,
            home_cards INTEGER,
            away_cards INTEGER,
            home_sot INTEGER,
            away_sot INTEGER,
            
            -- Features (Time-Bins as JSON arrays)
            home_possession_bins TEXT,
            away_possession_bins TEXT,
            home_corners_bins TEXT,
            away_corners_bins TEXT,
            home_cards_bins TEXT,
            away_cards_bins TEXT,
            home_sot_bins TEXT,
            away_sot_bins TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_matches_home ON matches(home_team, date);
        CREATE INDEX IF NOT EXISTS idx_matches_away ON matches(away_team, date);
        CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
        CREATE INDEX IF NOT EXISTS idx_rankings_team ON fifa_rankings(team, date);
        CREATE INDEX IF NOT EXISTS idx_advanced_stats_date ON advanced_stats(match_date);
        CREATE INDEX IF NOT EXISTS idx_advanced_stats_teams ON advanced_stats(home_team, away_team);
    """)


def import_matches(conn, filepath=None):
    """Bulk-import match results CSV into SQLite."""
    filepath = filepath or RESULTS_CSV
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            try:
                batch.append((
                    row["date"],
                    row["home_team"].strip(),
                    row["away_team"].strip(),
                    int(row["home_score"]),
                    int(row["away_score"]),
                    row.get("tournament", "").strip(),
                    row.get("city", "").strip(),
                    row.get("country", "").strip(),
                    row.get("neutral", "FALSE").strip(),
                ))
                count += 1
            except (ValueError, KeyError):
                continue
            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT INTO matches (date, home_team, away_team, "
                    "home_score, away_score, tournament, city, country, neutral) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                batch = []
        if batch:
            conn.executemany(
                "INSERT INTO matches (date, home_team, away_team, "
                "home_score, away_score, tournament, city, country, neutral) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
    
    # After importing, resolve advance_team using shootouts
    conn.execute("""
        UPDATE matches
        SET advance_team =
          CASE
            WHEN home_score > away_score THEN home_team
            WHEN away_score > home_score THEN away_team
            ELSE (
              SELECT winner
              FROM shootouts
              WHERE shootouts.date = matches.date
                AND shootouts.home_team = matches.home_team
                AND shootouts.away_team = matches.away_team
            )
          END
    """)
    conn.commit()
    return count

def import_shootouts(conn, filepath=None):
    """Bulk-import penalty shootouts CSV into SQLite."""
    filepath = filepath or SHOOTOUTS_CSV
    if not os.path.exists(filepath):
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            try:
                batch.append((
                    row["date"],
                    row["home_team"].strip(),
                    row["away_team"].strip(),
                    row["winner"].strip()
                ))
                count += 1
            except (ValueError, KeyError):
                continue
            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT INTO shootouts (date, home_team, away_team, winner) "
                    "VALUES (?, ?, ?, ?)", batch)
                batch = []
        if batch:
            conn.executemany(
                "INSERT INTO shootouts (date, home_team, away_team, winner) "
                "VALUES (?, ?, ?, ?)", batch)
    conn.commit()
    return count


def import_historical_rankings(conn, filepath=None):
    """Bulk-import historical FIFA rankings CSV into SQLite."""
    filepath = filepath or RANKINGS_HIST_CSV
    if not os.path.exists(filepath):
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            try:
                pts = row.get("total_points", "")
                if pts in ("NA", ""):
                    continue
                batch.append((
                    row["team"].strip(),
                    float(pts),
                    row["date"].strip(),
                    row.get("team_short", "").strip(),
                ))
                count += 1
            except (ValueError, KeyError):
                continue
            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT INTO fifa_rankings (team, total_points, date, "
                    "team_short) VALUES (?, ?, ?, ?)", batch)
                batch = []
        if batch:
            conn.executemany(
                "INSERT INTO fifa_rankings (team, total_points, date, "
                "team_short) VALUES (?, ?, ?, ?)", batch)
    conn.commit()
    return count


def download_all():
    """Download all data sources. Returns (ok_results, ok_rankings, ok_shootouts)."""
    ok_results = download_csv(RESULTS_URL, RESULTS_CSV)
    ok_rankings = download_csv(RANKINGS_URL, RANKINGS_HIST_CSV)
    ok_shootouts = download_csv(SHOOTOUTS_URL, SHOOTOUTS_CSV)
    return ok_results, ok_rankings, ok_shootouts
