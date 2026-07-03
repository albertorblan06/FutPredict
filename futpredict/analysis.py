"""
Analysis module — tournament classification, recent form, and head-to-head.
Extracted from the monolith with imports from the new package structure.
"""
import datetime
import math
from .config import LOOKBACK_MATCHES
from .rankings import get_fifa_points, get_median_fifa
from .names import get_all_names
from .weight_optimizer import (
    get_weights, tournament_weight_from_learned, LearnedWeights,
)


def get_tournament_weight(tournament_name, weights=None):
    """Classify tournament and return weight (0.0-1.0).

    Uses learned weights if calibrated, otherwise falls back to defaults.
    The optional `weights` parameter allows the optimizer to inject
    trial-specific weights during evaluation without side effects.
    """
    return tournament_weight_from_learned(tournament_name, weights=weights)


def _lookup_historical_points(conn, team_name, match_date):
    """Look up a team's FIFA points from historical rankings DB."""
    variants = get_all_names(team_name)
    for name in variants:
        cur = conn.execute("""
            SELECT total_points FROM fifa_rankings
            WHERE team = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, (name, match_date))
        row = cur.fetchone()
        if row and row[0]:
            return row[0]
    return None


def get_recent_form(conn, team_db_name, lookback=None, reference_date=None,
                    weights=None):
    """
    Analyze the team's last `lookback` matches with exponential time decay
    and opponent-strength weighting.
    """
    weights = weights or get_weights()
    lookback = lookback or LOOKBACK_MATCHES
    if reference_date is None:
        reference_date = datetime.date.today().isoformat()

    cur = conn.execute("""
        SELECT date, home_team, away_team, home_score, away_score, tournament
        FROM matches
        WHERE (home_team = ? OR away_team = ?) AND date <= ?
        ORDER BY date DESC LIMIT ?
    """, (team_db_name, team_db_name, reference_date, lookback))
    rows = cur.fetchall()
    if not rows:
        return None

    ref = datetime.date.fromisoformat(reference_date)
    decay_lambda = math.log(2) / weights.decay_half_life
    median = get_median_fifa()

    total_w = 0.0
    w_gf = w_ga = w_wins = w_draws = w_losses = 0.0
    opp_strengths = []
    recent_results = []

    for date_str, home, away, hs, as_, tournament in rows:
        try:
            match_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue
        days_ago = (ref - match_date).days
        if days_ago < 0:
            continue

        time_weight = math.exp(-decay_lambda * days_ago)
        tourn_weight = get_tournament_weight(tournament, weights=weights)
        weight = time_weight * tourn_weight

        is_home = (home == team_db_name)
        gf = hs if is_home else as_
        ga = as_ if is_home else hs
        opponent = away if is_home else home

        opp_pts, _ = get_fifa_points(opponent)
        if opp_pts is None:
            opp_pts = _lookup_historical_points(conn, opponent, date_str)
        if opp_pts is None:
            opp_pts = median

        opp_strength = opp_pts / median
        opp_strengths.append(opp_strength)
        adj_weight = weight * opp_strength

        w_gf += adj_weight * gf
        w_ga += adj_weight * ga
        total_w += adj_weight

        if gf > ga:
            w_wins += adj_weight
        elif gf == ga:
            w_draws += adj_weight
        else:
            w_losses += adj_weight

        if len(recent_results) < 10:
            res = "W" if gf > ga else ("D" if gf == ga else "L")
            recent_results.append({
                "date": date_str, "opponent": opponent,
                "gf": gf, "ga": ga, "result": res,
                "tournament": tournament,
            })

    if total_w == 0:
        return None

    if not recent_results:
        cs_rate = fts_rate = 0.0
        avg_tot = 2.6
    else:
        recent_5 = recent_results[:5]
        cs_rate = sum(1 for r in recent_5 if r["ga"] == 0) / len(recent_5)
        fts_rate = sum(1 for r in recent_5 if r["gf"] == 0) / len(recent_5)
        avg_tot = sum(r["gf"] + r["ga"] for r in recent_5) / len(recent_5)

    return {
        "weighted_gf": w_gf / total_w,
        "weighted_ga": w_ga / total_w,
        "win_rate": w_wins / total_w,
        "draw_rate": w_draws / total_w,
        "loss_rate": w_losses / total_w,
        "matches_analyzed": len(rows),
        "recent_results": recent_results,
        "avg_opponent_strength": (sum(opp_strengths) / len(opp_strengths)
                                  if opp_strengths else 1.0),
        "raw_wins": sum(1 for r in recent_results if r["result"] == "W"),
        "raw_draws": sum(1 for r in recent_results if r["result"] == "D"),
        "raw_losses": sum(1 for r in recent_results if r["result"] == "L"),
        "cs_rate": cs_rate,
        "fts_rate": fts_rate,
        "avg_tot": avg_tot,
    }


def get_head_to_head(conn, team_a_db, team_b_db, reference_date=None,
                     weights=None):
    """Analyze all historical matches between team A and team B."""
    weights = weights or get_weights()
    if reference_date is None:
        reference_date = datetime.date.today().isoformat()
        
    cur = conn.execute("""
        SELECT date, home_team, away_team, home_score, away_score, tournament
        FROM matches
        WHERE ((home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?))
          AND date <= ?
        ORDER BY date DESC
    """, (team_a_db, team_b_db, team_b_db, team_a_db, reference_date))
    rows = cur.fetchall()

    result = {
        "total_matches": 0, "wins_a": 0, "draws": 0, "wins_b": 0,
        "h2h_lambda_a": None, "h2h_lambda_b": None,
        "is_significant": False, "chi2": 0, "p_value": 1.0,
        "recent_h2h": [],
    }
    if not rows:
        return result

    ref = datetime.date.fromisoformat(reference_date)
    decay_lambda = math.log(2) / weights.h2h_decay_half_life

    total_w = w_gf_a = w_gf_b = 0.0
    wins_a = draws = wins_b = 0

    for date_str, home, away, hs, as_, tournament in rows:
        try:
            match_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue

        days_ago = (ref - match_date).days
        weight = math.exp(-decay_lambda * max(0, days_ago))

        if home == team_a_db:
            gf_a, gf_b = hs, as_
        else:
            gf_a, gf_b = as_, hs

        w_gf_a += weight * gf_a
        w_gf_b += weight * gf_b
        total_w += weight

        if gf_a > gf_b:
            wins_a += 1
        elif gf_a == gf_b:
            draws += 1
        else:
            wins_b += 1

        if len(result["recent_h2h"]) < 8:
            res = "A" if gf_a > gf_b else ("D" if gf_a == gf_b else "B")
            result["recent_h2h"].append({
                "date": date_str, "gf_a": gf_a, "gf_b": gf_b,
                "result": res, "tournament": tournament,
            })

    total = wins_a + draws + wins_b
    result["total_matches"] = total
    result["wins_a"] = wins_a
    result["draws"] = draws
    result["wins_b"] = wins_b

    if total_w > 0:
        result["h2h_lambda_a"] = w_gf_a / total_w
        result["h2h_lambda_b"] = w_gf_b / total_w

    if total >= 5:
        expected = total / 3.0
        chi2 = (((wins_a - expected) ** 2 +
                 (draws - expected) ** 2 +
                 (wins_b - expected) ** 2) / expected)
        p_value = math.exp(-chi2 / 2)
        result["chi2"] = chi2
        result["p_value"] = p_value
        result["is_significant"] = p_value < 0.10

    return result


def compute_global_avg_goals(conn, years_back=4):
    """Compute the global average goals per team per match over recent years."""
    cutoff = (datetime.date.today() -
              datetime.timedelta(days=years_back * 365)).isoformat()
    cur = conn.execute("""
        SELECT AVG(home_score + away_score) / 2.0, COUNT(*)
        FROM matches WHERE date >= ?
    """, (cutoff,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0], row[1]
    return 1.35, 0
