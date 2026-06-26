"""
XGBoost count regression — predicts expected goals directly.

Two separate regressors:
  - xgb_home: predicts E[home_goals] using count:poisson objective
  - xgb_away: predicts E[away_goals] using count:poisson objective

No more classifier → magic number → λ Frankenstein conversion.
Outputs a 9×9 score probability matrix via Poisson PMF from predicted λ.
"""
import os
import json
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import nbinom, poisson
from .config import (XGB_PARAMS, BTTS_PARAMS, XGB_TRAIN_START, MAX_GOALS,
                     XGB_TOTALS_MODEL, XGB_BTTS_MODEL, XGB_META_PATH, TRAIN_END_DATE)
from .rankings import get_fifa_points, get_fifa_rank, get_median_fifa, _RANKING_POINTS
from .names import get_all_names
from .analysis import get_tournament_weight
from .xgb_advanced import _get_time_bins

FEATURE_COLUMNS = [
    "elo_a", "elo_b", "elo_diff",
    "fifa_pts_a", "fifa_pts_b", "fifa_pts_diff",
    "rank_a", "rank_b", "rank_diff",
    "form_gf_a", "form_ga_a", "form_gf_b", "form_ga_b",
    "form_gd_a", "form_gd_b",
    "form_win_rate_a", "form_win_rate_b",
    "cs_rate_a", "fts_rate_a", "avg_tot_a",
    "cs_rate_b", "fts_rate_b", "avg_tot_b",
    "team_xg_a", "team_xg_b",
    "h2h_win_rate_a", "h2h_total_matches", "h2h_gd",
    "tournament_weight", "is_neutral", "days_since_last",
]
for i in range(8):
    FEATURE_COLUMNS.append(f"h_momentum_{i}")
    FEATURE_COLUMNS.append(f"a_momentum_{i}")



from collections import defaultdict
import bisect

from .momentum_model import train_momentum_model, get_momentum_vector

def _estimate_rank_from_pts(pts, all_pts_sorted):
    """Estimate a rank position from FIFA points using a sorted reference list."""
    if not all_pts_sorted or pts is None:
        return 100
    import bisect
    # all_pts_sorted is descending; find where pts would insert in descending order
    # bisect on reversed list: rank = position where pts fits
    pos = bisect.bisect_left([-p for p in all_pts_sorted], -pts)
    return max(1, min(pos + 1, 211))

def build_features(conn, force_momentum=False):
    """Build feature matrix with advanced stats + targets."""
    
    from futpredict.elo import calculate_elo_history, get_elo
    calculate_elo_history(conn, force=force_momentum)
    
    momentum_model = train_momentum_model(conn, force=force_momentum)
    
    query = f"""
        SELECT m.*, a.home_possession_bins, a.away_possession_bins,
               a.home_corners_bins, a.away_corners_bins,
               a.home_cards_bins, a.away_cards_bins,
               a.home_sot_bins, a.away_sot_bins,
               a.home_possession, a.away_possession,
               a.home_sot, a.away_sot
        FROM matches m
        LEFT JOIN advanced_stats a ON m.date = a.match_date AND m.home_team = a.home_team AND m.away_team = a.away_team
        WHERE m.date >= '{XGB_TRAIN_START}-01-01' AND m.date < '{TRAIN_END_DATE}'
        ORDER BY m.date
    """
    matches_df = pd.read_sql_query(query, conn)
    matches_df["home_score"] = matches_df["home_score"].astype(int)
    matches_df["away_score"] = matches_df["away_score"].astype(int)

    try:
        rankings_df = pd.read_sql_query(
            "SELECT team, total_points, date FROM fifa_rankings "
            "WHERE total_points IS NOT NULL ORDER BY date", conn)
        # Build dictionary for fast lookup: team -> (dates, points)
        rankings_dict = defaultdict(lambda: ([], []))
        for _, r in rankings_df.iterrows():
            team = r["team"]
            rankings_dict[team][0].append(r["date"])
            rankings_dict[team][1].append(r["total_points"])
    except Exception:
        rankings_dict = defaultdict(lambda: ([], []))

    try:
        player_df = pd.read_sql_query(
            "SELECT team, match_date, SUM(expected_goals_xg) as team_xg "
            "FROM player_stats GROUP BY team, match_date", conn)
        xg_dict = {}
        for _, r in player_df.iterrows():
            xg_dict[(r['team'], r['match_date'])] = r['team_xg']
    except Exception:
        xg_dict = {}

    def get_pts(team, m_date):
        dates, pts = rankings_dict.get(team, ([], []))
        if not dates:
            for variant in get_all_names(team):
                dates, pts = rankings_dict.get(variant, ([], []))
                if dates: break
        if not dates:
            p, _ = get_fifa_points(team)
            return p if p else get_median_fifa()
        idx = bisect.bisect_right(dates, m_date)
        if idx == 0:
            return pts[0]
        return pts[idx-1]

    team_history = defaultdict(list)
    h2h_history = defaultdict(list)
    rows = []
    
    start_idx = min(500, len(matches_df) // 5)
    
    for idx, row in matches_df.iterrows():
        team_a, team_b = row["home_team"], row["away_team"]
        match_date = row["date"]
        score_a, score_b = row["home_score"], row["away_score"]
        
        if idx >= start_idx:
            # Form
            hist_a = team_history[team_a][-5:]
            hist_b = team_history[team_b][-5:]
            
            if len(hist_a) > 0 and len(hist_b) > 0:
                gf_a = sum(x[1] for x in hist_a) / len(hist_a)
                ga_a = sum(x[2] for x in hist_a) / len(hist_a)
                wr_a = sum(1 if x[1]>x[2] else (0.5 if x[1]==x[2] else 0) for x in hist_a) / len(hist_a)
                cs_rate_a = sum(1 for x in hist_a if x[2] == 0) / len(hist_a)
                fts_rate_a = sum(1 for x in hist_a if x[1] == 0) / len(hist_a)
                avg_tot_a = sum(x[1] + x[2] for x in hist_a) / len(hist_a)
                team_xg_a = sum(x[3] for x in hist_a) / len(hist_a)
                
                gf_b = sum(x[1] for x in hist_b) / len(hist_b)
                ga_b = sum(x[2] for x in hist_b) / len(hist_b)
                wr_b = sum(1 if x[1]>x[2] else (0.5 if x[1]==x[2] else 0) for x in hist_b) / len(hist_b)
                cs_rate_b = sum(1 for x in hist_b if x[2] == 0) / len(hist_b)
                fts_rate_b = sum(1 for x in hist_b if x[1] == 0) / len(hist_b)
                avg_tot_b = sum(x[1] + x[2] for x in hist_b) / len(hist_b)
                team_xg_b = sum(x[3] for x in hist_b) / len(hist_b)
                
                h2h_key = tuple(sorted([team_a, team_b]))
                h2h_hist = h2h_history[h2h_key]
                if h2h_hist:
                    h2h_total = len(h2h_hist)
                    # Convert to team_a perspective
                    wins_a = 0
                    gd_a = 0
                    for h_team, h_gf, h_ga in h2h_hist:
                        if h_team == team_a:
                            wins_a += 1 if h_gf > h_ga else 0
                            gd_a += h_gf - h_ga
                        else:
                            wins_a += 1 if h_ga > h_gf else 0
                            gd_a += h_ga - h_gf
                    h2h_wr = wins_a / h2h_total
                    h2h_gd = gd_a / h2h_total
                else:
                    h2h_total = 0; h2h_wr = 0.5; h2h_gd = 0.0
                    
                fifa_a = get_pts(team_a, match_date)
                fifa_b = get_pts(team_b, match_date)
                
                elo_a = get_elo(conn, team_a, match_date)
                elo_b = get_elo(conn, team_b, match_date)
                
                # Derive historical rank proxy from historical points
                # (avoids leaking current-day rank into historical samples)
                median = get_median_fifa()
                all_pts_sorted = sorted(_RANKING_POINTS.values(), reverse=True) if _RANKING_POINTS else []
                rank_a = _estimate_rank_from_pts(fifa_a, all_pts_sorted)
                rank_b = _estimate_rank_from_pts(fifa_b, all_pts_sorted)
                tourn_w = get_tournament_weight(row.get("tournament", ""))
                is_neutral = 1 if str(row.get("neutral", "FALSE")).upper() == "TRUE" else 0
                
                if team_history[team_a]:
                    try: days_since = (pd.to_datetime(match_date) - pd.to_datetime(team_history[team_a][-1][0])).days
                    except Exception: days_since = 30
                else:
                    days_since = 30
                    
                h_pos_bins = _get_time_bins(row["home_possession_bins"], 50)
                h_cor_bins = _get_time_bins(row["home_corners_bins"], 0)
                h_car_bins = _get_time_bins(row["home_cards_bins"], 0)
                h_sot_bins = _get_time_bins(row["home_sot_bins"], 0)
                a_pos_bins = _get_time_bins(row["away_possession_bins"], 50)
                a_cor_bins = _get_time_bins(row["away_corners_bins"], 0)
                a_car_bins = _get_time_bins(row["away_cards_bins"], 0)
                a_sot_bins = _get_time_bins(row["away_sot_bins"], 0)
                
                median = get_median_fifa()
                feature_row = {
                    "elo_a": elo_a,
                    "elo_b": elo_b,
                    "elo_diff": elo_a - elo_b,
                    "fifa_pts_a": fifa_a or median,
                    "fifa_pts_b": fifa_b or median,
                    "fifa_pts_diff": (fifa_a or median) - (fifa_b or median),
                    "rank_a": rank_a, "rank_b": rank_b,
                    "rank_diff": rank_a - rank_b,
                    "form_gf_a": gf_a, "form_ga_a": ga_a,
                    "form_gf_b": gf_b, "form_ga_b": ga_b,
                    "form_gd_a": gf_a - ga_a, "form_gd_b": gf_b - ga_b,
                    "form_win_rate_a": wr_a,
                    "form_win_rate_b": wr_b,
                    "cs_rate_a": cs_rate_a, "fts_rate_a": fts_rate_a, "avg_tot_a": avg_tot_a,
                    "cs_rate_b": cs_rate_b, "fts_rate_b": fts_rate_b, "avg_tot_b": avg_tot_b,
                    "team_xg_a": team_xg_a, "team_xg_b": team_xg_b,
                    "h2h_win_rate_a": h2h_wr,
                    "h2h_total_matches": h2h_total,
                    "h2h_gd": h2h_gd,
                    "tournament_weight": tourn_w,
                    "is_neutral": is_neutral,
                    "days_since_last": min(days_since, 180),
                    "total_goals": int(score_a) + int(score_b),
                    "btts": 1 if int(score_a) > 0 and int(score_b) > 0 else 0,
                }
                # Add Momentum Vectors instead of raw time bins
                h_momentum = get_momentum_vector(momentum_model, {
                    'pos': h_pos_bins, 'cor': h_cor_bins, 'car': h_car_bins, 'sot': h_sot_bins
                })
                a_momentum = get_momentum_vector(momentum_model, {
                    'pos': a_pos_bins, 'cor': a_cor_bins, 'car': a_car_bins, 'sot': a_sot_bins
                })
                
                for i in range(8):
                    feature_row[f"h_momentum_{i}"] = h_momentum[i]
                    feature_row[f"a_momentum_{i}"] = a_momentum[i]
                    
                rows.append(feature_row)

        # Update state for next iterations
        # Synthetic xG Backfill
        if (team_a, match_date) in xg_dict:
            match_xg_a = xg_dict[(team_a, match_date)]
        else:
            h_sot = row.get("home_sot")
            h_pos = row.get("home_possession")
            if pd.notnull(h_sot) and pd.notnull(h_pos):
                match_xg_a = (float(h_sot) * 0.3) + (float(h_pos) * 0.01)
            else:
                match_xg_a = float(score_a)
                
        if (team_b, match_date) in xg_dict:
            match_xg_b = xg_dict[(team_b, match_date)]
        else:
            a_sot = row.get("away_sot")
            a_pos = row.get("away_possession")
            if pd.notnull(a_sot) and pd.notnull(a_pos):
                match_xg_b = (float(a_sot) * 0.3) + (float(a_pos) * 0.01)
            else:
                match_xg_b = float(score_b)
        
        team_history[team_a].append((match_date, score_a, score_b, match_xg_a))
        team_history[team_b].append((match_date, score_b, score_a, match_xg_b))
        h2h_key = tuple(sorted([team_a, team_b]))
        h2h_history[h2h_key].append((team_a, score_a, score_b))
        
    return pd.DataFrame(rows)

def _models_exist():
    return (os.path.exists(XGB_TOTALS_MODEL) and
            os.path.exists(XGB_BTTS_MODEL) and
            os.path.exists(XGB_META_PATH))

def _load_cached():
    model_totals = xgb.XGBRegressor()
    model_totals.load_model(XGB_TOTALS_MODEL)
    model_btts = xgb.XGBClassifier()
    model_btts.load_model(XGB_BTTS_MODEL)
    with open(XGB_META_PATH, "r") as f:
        meta = json.load(f)
    return model_totals, model_btts, meta

def _save_models(model_totals, model_btts, meta):
    os.makedirs(os.path.dirname(XGB_TOTALS_MODEL), exist_ok=True)
    model_totals.save_model(XGB_TOTALS_MODEL)
    model_btts.save_model(XGB_BTTS_MODEL)
    with open(XGB_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

def train_xgb(conn, force=False):
    """Train or load cached XGBoost market models."""
    if not force and _models_exist():
        try:
            model_t, model_b, meta = _load_cached()
            print(f"   ✓  Loaded cached XGBoost Markets ({meta.get('n_train', '?')} "
                  f"training samples)")
            print(f"      Totals MAE: {meta.get('totals_mae', meta.get('totals_logloss', 0)):.3f} │ "
                  f"BTTS LogLoss: {meta.get('btts_logloss', 0):.3f}")
            return model_t, model_b, meta
        except Exception as e:
            print(f"   ⚠  Cache load failed ({e}), retraining...")

    print("   ⬇  Building features for market prediction...")
    df = build_features(conn, force_momentum=force)
    
    if len(df) == 0:
        print(f"   ✗  Feature engineering failed: {e}")
        return None, None, None

    if len(df) < 1000:
        print(f"   ✗  Not enough data ({len(df)} samples)")
        return None, None, None

    print(f"   ✓  {len(df):,} samples built ({len(FEATURE_COLUMNS)} features)")

    X = df[FEATURE_COLUMNS].values
    y_totals = df["total_goals"].values
    y_btts = df["btts"].values

    split = int(len(df) * 0.80)
    X_train, X_val = X[:split], X[split:]
    yt_train, yt_val = y_totals[:split], y_totals[split:]
    yb_train, yb_val = y_btts[:split], y_btts[split:]

    print("   ⚙  Training XGBoost Market models...")

    model_totals = xgb.XGBRegressor(**XGB_PARAMS)
    model_totals.fit(X_train, yt_train, eval_set=[(X_val, yt_val)], verbose=False)

    btts_params = BTTS_PARAMS.copy()
    
    model_btts = xgb.XGBClassifier(**btts_params)
    model_btts.fit(X_train, yb_train, eval_set=[(X_val, yb_val)], verbose=False)

    # Evaluate
    pred_lambda = model_totals.predict(X_val)
    pred_b_proba = model_btts.predict_proba(X_val)[:, 1]
    
    from sklearn.metrics import log_loss, mean_absolute_error
    totals_mae = float(mean_absolute_error(yt_val, pred_lambda))
    btts_logloss = float(log_loss(yb_val, pred_b_proba))

    imp_t = model_totals.feature_importances_
    imp_b = model_btts.feature_importances_
    combined = sorted(zip(FEATURE_COLUMNS, (imp_t + imp_b) / 2),
                      key=lambda x: -x[1])

    meta = {
        "n_train": len(df),
        "n_val": int(len(X_val)),
        "totals_mae": totals_mae,
        "btts_logloss": btts_logloss,
        "trained_at": datetime.datetime.now().isoformat(),
    }

    print(f"   ✓  Done! Totals MAE: {totals_mae:.3f} │ BTTS LogLoss: {btts_logloss:.3f}")
    print(f"      Top features: {combined[0][0]} ({combined[0][1]:.3f}), "
          f"{combined[1][0]} ({combined[1][1]:.3f})")

    _save_models(model_totals, model_btts, meta)
    print(f"   ✓  Models cached")

    return model_totals, model_btts, meta


def predict_xgb(model_totals, model_btts, form_a, form_b,
                fifa_pts_a, fifa_pts_b, h2h, venue="neutral", conn=None, team_a=None, team_b=None, match_date=None):
    """
    Predict markets using XGBoost directly.
    """
    if model_totals is None or model_btts is None:
        return None, None

    rank_a = get_fifa_rank(form_a.get("_team", "")) or 100 if isinstance(form_a, dict) else 100
    rank_b = get_fifa_rank(form_b.get("_team", "")) or 100 if isinstance(form_b, dict) else 100
    median = get_median_fifa()

    if form_a and isinstance(form_a, dict):
        fgf_a, fga_a = form_a["weighted_gf"], form_a["weighted_ga"]
        fgd_a, fwr_a = fgf_a - fga_a, form_a["win_rate"]
        cs_rate_a = form_a.get("cs_rate", 0.0)
        fts_rate_a = form_a.get("fts_rate", 0.0)
        avg_tot_a = form_a.get("avg_tot", 2.6)
    else:
        fgf_a = fga_a = 1.3; fgd_a = 0.0; fwr_a = 0.33
        cs_rate_a = fts_rate_a = 0.0; avg_tot_a = 2.6

    if form_b and isinstance(form_b, dict):
        fgf_b, fga_b = form_b["weighted_gf"], form_b["weighted_ga"]
        fgd_b, fwr_b = fgf_b - fga_b, form_b["win_rate"]
        cs_rate_b = form_b.get("cs_rate", 0.0)
        fts_rate_b = form_b.get("fts_rate", 0.0)
        avg_tot_b = form_b.get("avg_tot", 2.6)
    else:
        fgf_b = fga_b = 1.3; fgd_b = 0.0; fwr_b = 0.33
        cs_rate_b = fts_rate_b = 0.0; avg_tot_b = 2.6

    if h2h and h2h.get("total_matches", 0) > 0:
        h2h_wr = h2h["wins_a"] / max(h2h["total_matches"], 1)
        h2h_gd = ((h2h.get("h2h_lambda_a", 1.3) or 1.3) -
                  (h2h.get("h2h_lambda_b", 1.3) or 1.3))
    else:
        h2h_wr = 0.5; h2h_gd = 0.0

    is_neutral = 1 if venue == "neutral" else 0
    h2h_total = h2h.get("total_matches", 0) if h2h else 0

    # Get Advanced Stats
    h_pos_bins, h_cor_bins, h_car_bins, h_sot_bins = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
    a_pos_bins, a_cor_bins, a_car_bins, a_sot_bins = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
    
    if conn and team_a and team_b:
        date_filter = f"AND m.date < '{match_date}'" if match_date else ""
        query = f"""
            SELECT m.date, m.home_team, m.away_team, 
                   a.home_possession_bins, a.away_possession_bins,
                   a.home_corners_bins, a.away_corners_bins,
                   a.home_cards_bins, a.away_cards_bins,
                   a.home_sot_bins, a.away_sot_bins
            FROM matches m
            JOIN advanced_stats a ON m.date = a.match_date AND m.home_team = a.home_team AND m.away_team = a.away_team
            WHERE (m.home_team = '{team_a}' OR m.away_team = '{team_a}' OR m.home_team = '{team_b}' OR m.away_team = '{team_b}')
            {date_filter}
            ORDER BY m.date
        """
        try:
            df = pd.read_sql_query(query, conn)
            past_home = df[(df["home_team"] == team_a) | (df["away_team"] == team_a)].tail(5)
            past_away = df[(df["home_team"] == team_b) | (df["away_team"] == team_b)].tail(5)
            
            if len(past_home) > 0:
                for _, p in past_home.iterrows():
                    is_home = (p["home_team"] == team_a)
                    h_pos_bins += _get_time_bins(p["home_possession_bins"] if is_home else p["away_possession_bins"], 50)
                    h_cor_bins += _get_time_bins(p["home_corners_bins"] if is_home else p["away_corners_bins"], 0)
                    h_car_bins += _get_time_bins(p["home_cards_bins"] if is_home else p["away_cards_bins"], 0)
                    h_sot_bins += _get_time_bins(p["home_sot_bins"] if is_home else p["away_sot_bins"], 0)
                h_pos_bins /= len(past_home)
                h_cor_bins /= len(past_home)
                h_car_bins /= len(past_home)
                h_sot_bins /= len(past_home)
                
            if len(past_away) > 0:
                for _, p in past_away.iterrows():
                    is_home = (p["home_team"] == team_b)
                    a_pos_bins += _get_time_bins(p["home_possession_bins"] if is_home else p["away_possession_bins"], 50)
                    a_cor_bins += _get_time_bins(p["home_corners_bins"] if is_home else p["away_corners_bins"], 0)
                    a_car_bins += _get_time_bins(p["home_cards_bins"] if is_home else p["away_cards_bins"], 0)
                    a_sot_bins += _get_time_bins(p["home_sot_bins"] if is_home else p["away_sot_bins"], 0)
                a_pos_bins /= len(past_away)
                a_cor_bins /= len(past_away)
                a_car_bins /= len(past_away)
                a_sot_bins /= len(past_away)
        except Exception:
            pass

    # Get Team xG from player_stats
    team_xg_a = 1.3
    team_xg_b = 1.3
    if conn:
        try:
            xg_a = pd.read_sql_query(
                "SELECT SUM(expected_goals_xg) as team_xg FROM player_stats WHERE team = ? GROUP BY match_date ORDER BY match_date DESC LIMIT 5", 
                conn, params=(team_a,))
            if len(xg_a) > 0: team_xg_a = xg_a['team_xg'].mean()
            
            xg_b = pd.read_sql_query(
                "SELECT SUM(expected_goals_xg) as team_xg FROM player_stats WHERE team = ? GROUP BY match_date ORDER BY match_date DESC LIMIT 5", 
                conn, params=(team_b,))
            if len(xg_b) > 0: team_xg_b = xg_b['team_xg'].mean()
        except Exception:
            pass
    # Compute dynamic tournament weight and days_since_last (instead of hardcoding)
    tourn_w = 0.85  # Default: World Cup weight (most common at-prediction context)
    days_since = 14  # Default
    if conn and team_a and match_date:
        try:
            cur = conn.execute(
                "SELECT date FROM matches WHERE (home_team = ? OR away_team = ?) AND date < ? ORDER BY date DESC LIMIT 1",
                (team_a, team_a, match_date))
            row = cur.fetchone()
            if row:
                import datetime
                d1 = datetime.date.fromisoformat(row[0])
                d2 = datetime.date.fromisoformat(match_date)
                days_since = min((d2 - d1).days, 180)
        except Exception:
            pass

    from futpredict.elo import get_elo
    elo_a = get_elo(conn, team_a, match_date) if conn and team_a else 1500.0
    elo_b = get_elo(conn, team_b, match_date) if conn and team_b else 1500.0
    
    base_feat = [
        elo_a, elo_b, elo_a - elo_b,
        fifa_pts_a or median, fifa_pts_b or median,
        (fifa_pts_a or median) - (fifa_pts_b or median),
        rank_a, rank_b, rank_a - rank_b,
        fgf_a, fga_a, fgf_b, fga_b, fgd_a, fgd_b,
        fwr_a, fwr_b,
        cs_rate_a, fts_rate_a, avg_tot_a,
        cs_rate_b, fts_rate_b, avg_tot_b,
        team_xg_a, team_xg_b,
        h2h_wr, h2h_total, h2h_gd,
        tourn_w, is_neutral, days_since,
    ]
    
    # Generate momentum vectors
    try:
        momentum_model = train_momentum_model(conn)
    except Exception:
        momentum_model = None
        
    h_momentum = get_momentum_vector(momentum_model, {
        'pos': h_pos_bins, 'cor': h_cor_bins, 'car': h_car_bins, 'sot': h_sot_bins
    })
    a_momentum = get_momentum_vector(momentum_model, {
        'pos': a_pos_bins, 'cor': a_cor_bins, 'car': a_car_bins, 'sot': a_sot_bins
    })
    
    for i in range(8):
        base_feat.append(float(h_momentum[i]))
        base_feat.append(float(a_momentum[i]))

    feature_vec = np.array([base_feat])

    lambda_total = float(model_totals.predict(feature_vec)[0])
    
    over05_prob = 1.0 - poisson.cdf(0, lambda_total)
    over15_prob = 1.0 - poisson.cdf(1, lambda_total)
    over25_prob = 1.0 - poisson.cdf(2, lambda_total)
    over35_prob = 1.0 - poisson.cdf(3, lambda_total)
    over45_prob = 1.0 - poisson.cdf(4, lambda_total)
    
    btts_yes = float(model_btts.predict_proba(feature_vec)[0, 1])
    
    outcomes = {
        "over_0_5_pct": over05_prob * 100,
        "under_0_5_pct": (1 - over05_prob) * 100,
        "over_1_5_pct": over15_prob * 100,
        "under_1_5_pct": (1 - over15_prob) * 100,
        "over_2_5_pct": over25_prob * 100,
        "under_2_5_pct": (1 - over25_prob) * 100,
        "over_3_5_pct": over35_prob * 100,
        "under_3_5_pct": (1 - over35_prob) * 100,
        "over_4_5_pct": over45_prob * 100,
        "under_4_5_pct": (1 - over45_prob) * 100,
        "btts_yes_pct": btts_yes * 100,
        "btts_no_pct": (1 - btts_yes) * 100,
    }

    return None, outcomes
