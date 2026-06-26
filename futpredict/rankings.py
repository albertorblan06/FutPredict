"""
FIFA rankings — loaded from CSV into SQLite, not hardcoded.
Handles both the June 2026 snapshot and historical rankings.
"""
import csv
import os
from .config import RANKINGS_2026_CSV
from .names import get_all_names


# In-memory cache (loaded once from CSV)
_RANKING_POINTS = {}   # team → points
_RANK_POSITION = {}    # team → rank (1-indexed)
_MEDIAN_FIFA = 1100.0  # default fallback


def load_rankings_from_csv(filepath=None):
    """Load the June 2026 FIFA rankings from CSV into memory."""
    global _RANKING_POINTS, _RANK_POSITION, _MEDIAN_FIFA
    filepath = filepath or RANKINGS_2026_CSV
    if not os.path.exists(filepath):
        return False
    _RANKING_POINTS.clear()
    _RANK_POSITION.clear()
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = row["team"].strip()
            pts = float(row["points"])
            rank = int(row["rank"])
            _RANKING_POINTS[team] = pts
            _RANK_POSITION[team] = rank
    # Compute median
    all_pts = sorted(_RANKING_POINTS.values())
    if all_pts:
        _MEDIAN_FIFA = all_pts[len(all_pts) // 2]
    return True


def import_rankings_2026_to_db(conn, filepath=None):
    """Import June 2026 rankings into the SQLite rankings table."""
    filepath = filepath or RANKINGS_2026_CSV
    if not os.path.exists(filepath):
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append((
                row["team"].strip(),
                float(row["points"]),
                "2026-06-19",
                "",
            ))
            count += 1
        if batch:
            conn.executemany(
                "INSERT INTO fifa_rankings (team, total_points, date, team_short) "
                "VALUES (?, ?, ?, ?)", batch)
    conn.commit()
    return count


def get_fifa_points(team_name):
    """Get FIFA ranking points for a team, trying all name variants."""
    variants = get_all_names(team_name)
    for name in variants:
        if name in _RANKING_POINTS:
            return _RANKING_POINTS[name], name
    return None, None


def get_fifa_rank(team_name):
    """Get FIFA rank position for a team."""
    variants = get_all_names(team_name)
    for name in variants:
        if name in _RANK_POSITION:
            return _RANK_POSITION[name]
    return None


def get_median_fifa():
    """Get the global median FIFA points."""
    return _MEDIAN_FIFA
