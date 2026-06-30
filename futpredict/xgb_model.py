"""
XGBoost market models — direct binary classification for Over/Under.

Architecture:
  - 5 × XGBClassifier (binary:logistic) for Over 0.5 … 4.5
  - 1 × XGBClassifier (binary:logistic) for BTTS

No more Tweedie regression → Poisson CDF Frankenstein.
Each classifier's .predict_proba() gives a natively calibrated probability.
"""
import os
import json
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from .config import (XGB_PARAMS, BTTS_PARAMS, OVER_PARAMS, ET_PARAMS,
                     XGB_TRAIN_START, MAX_GOALS,
                     XGB_TOTALS_MODEL, XGB_BTTS_MODEL, XGB_ET_MODEL, XGB_META_PATH,
                     XGB_OVER_THRESHOLDS, XGB_OVER_MODELS, TRAIN_END_DATE)
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
    # Variance features (new)
    "form_gf_std_a", "form_ga_std_a", "form_tot_std_a",
    "form_gf_std_b", "form_ga_std_b", "form_tot_std_b",
    "cs_rate_a", "fts_rate_a", "avg_tot_a",
    "cs_rate_b", "fts_rate_b", "avg_tot_b",
    "h2h_win_rate_a", "h2h_total_matches", "h2h_gd",
    "tournament_weight", "is_neutral", "days_since_last",
    # Tournament phase (new)
    "is_knockout",
    # Interpretable tempo features (new)
    "early_sot_ratio_a", "early_sot_ratio_b",
    "both_early_scorers",
    "tempo_variance_a", "tempo_variance_b",
    # Interaction parity features
    "abs_elo_diff", "abs_fifa_diff", "abs_rank_diff",
    "combined_cs_rate", "combined_fga",
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
    
    # Build per-team tournament match counter for is_knockout inference
    team_tournament_counter = defaultdict(lambda: defaultdict(int))

    for idx, row in matches_df.iterrows():
        team_a, team_b = row["home_team"], row["away_team"]
        match_date = row["date"]
        score_a, score_b = row["home_score"], row["away_score"]
        tournament = row.get("tournament", "")
        match_year = match_date[:4] if match_date else ""
        tourn_year_key = f"{tournament}_{match_year}"

        # Increment tournament match count for knockout detection
        team_tournament_counter[team_a][tourn_year_key] += 1
        team_tournament_counter[team_b][tourn_year_key] += 1
        match_num_a = team_tournament_counter[team_a][tourn_year_key]
        match_num_b = team_tournament_counter[team_b][tourn_year_key]
        
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

                # Variance features (new)
                gf_vals_a = [x[1] for x in hist_a]
                ga_vals_a = [x[2] for x in hist_a]
                tot_vals_a = [x[1] + x[2] for x in hist_a]
                form_gf_std_a = float(np.std(gf_vals_a)) if len(gf_vals_a) > 1 else 0.0
                form_ga_std_a = float(np.std(ga_vals_a)) if len(ga_vals_a) > 1 else 0.0
                form_tot_std_a = float(np.std(tot_vals_a)) if len(tot_vals_a) > 1 else 0.0
                
                gf_b = sum(x[1] for x in hist_b) / len(hist_b)
                ga_b = sum(x[2] for x in hist_b) / len(hist_b)
                wr_b = sum(1 if x[1]>x[2] else (0.5 if x[1]==x[2] else 0) for x in hist_b) / len(hist_b)
                cs_rate_b = sum(1 for x in hist_b if x[2] == 0) / len(hist_b)
                fts_rate_b = sum(1 for x in hist_b if x[1] == 0) / len(hist_b)
                avg_tot_b = sum(x[1] + x[2] for x in hist_b) / len(hist_b)

                # Variance features (new)
                gf_vals_b = [x[1] for x in hist_b]
                ga_vals_b = [x[2] for x in hist_b]
                tot_vals_b = [x[1] + x[2] for x in hist_b]
                form_gf_std_b = float(np.std(gf_vals_b)) if len(gf_vals_b) > 1 else 0.0
                form_ga_std_b = float(np.std(ga_vals_b)) if len(ga_vals_b) > 1 else 0.0
                form_tot_std_b = float(np.std(tot_vals_b)) if len(tot_vals_b) > 1 else 0.0


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

                # is_knockout: if either team's match count in this tournament+year >= 4
                is_ko = 1 if (match_num_a >= 4 or match_num_b >= 4) else 0
                
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

                # Interpretable tempo features (new)
                early_sot_a, tempo_var_a = _compute_tempo_features(h_sot_bins)
                early_sot_b, tempo_var_b = _compute_tempo_features(a_sot_bins)
                
                total_goals = int(score_a) + int(score_b)
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
                    # Variance features
                    "form_gf_std_a": form_gf_std_a,
                    "form_ga_std_a": form_ga_std_a,
                    "form_tot_std_a": form_tot_std_a,
                    "form_gf_std_b": form_gf_std_b,
                    "form_ga_std_b": form_ga_std_b,
                    "form_tot_std_b": form_tot_std_b,
                    "cs_rate_a": cs_rate_a, "fts_rate_a": fts_rate_a, "avg_tot_a": avg_tot_a,
                    "cs_rate_b": cs_rate_b, "fts_rate_b": fts_rate_b, "avg_tot_b": avg_tot_b,
                    "h2h_win_rate_a": h2h_wr,
                    "h2h_total_matches": h2h_total,
                    "h2h_gd": h2h_gd,
                    "tournament_weight": tourn_w,
                    "is_neutral": is_neutral,
                    "days_since_last": min(days_since, 180),
                    # Tournament phase
                    "is_knockout": is_ko,
                    # Tempo features
                    "early_sot_ratio_a": early_sot_a,
                    "early_sot_ratio_b": early_sot_b,
                    "both_early_scorers": early_sot_a * early_sot_b,
                    "tempo_variance_a": tempo_var_a,
                    "tempo_variance_b": tempo_var_b,
                    "abs_elo_diff": abs(elo_a - elo_b),
                    "abs_fifa_diff": abs((fifa_a or median) - (fifa_b or median)),
                    "abs_rank_diff": abs(rank_a - rank_b),
                    "combined_cs_rate": cs_rate_a * cs_rate_b,
                    "combined_fga": ga_a * ga_b,
                    # Targets
                    "total_goals": total_goals,
                    "btts": 1 if int(score_a) > 0 and int(score_b) > 0 else 0,
                    "goes_to_et": 1 if int(score_a) == int(score_b) else 0,
                    "advance_target": 1 if row.get("advance_team") == team_a else (0 if row.get("advance_team") == team_b else None),
                }
                # Binary Over targets for direct classification
                for thresh in XGB_OVER_THRESHOLDS:
                    key = f"over_{str(thresh).replace('.', '_')}"
                    feature_row[key] = 1 if total_goals > thresh else 0

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

        team_history[team_a].append((match_date, score_a, score_b))
        team_history[team_b].append((match_date, score_b, score_a))
        h2h_key = tuple(sorted([team_a, team_b]))
        h2h_history[h2h_key].append((team_a, score_a, score_b))
        
    return pd.DataFrame(rows)

def _compute_xg_concentration(xg_player_dict, team, match_date):
    """Compute Herfindahl-like xG concentration index for a team's most recent match."""
    # Find the most recent match for this team before match_date
    best_key = None
    for key in xg_player_dict:
        if key[0] == team and key[1] < match_date:
            if best_key is None or key[1] > best_key[1]:
                best_key = key
    if best_key is None:
        return 0.33  # default: evenly distributed
    xg_vals = xg_player_dict[best_key]
    total_xg = sum(xg_vals)
    if total_xg <= 0:
        return 0.33
    shares = sorted([v / total_xg for v in xg_vals], reverse=True)
    # Top-3 share
    return sum(shares[:3])


def _compute_tempo_features(sot_bins):
    """Extract interpretable tempo features from 6-bin SOT time series."""
    sot_arr = np.array(sot_bins, dtype=float)
    total_sot = sot_arr.sum()
    if total_sot > 0:
        early_ratio = sot_arr[:2].sum() / total_sot  # first 30 min
    else:
        early_ratio = 0.33  # default
    tempo_var = float(np.std(sot_arr))
    return early_ratio, tempo_var


def _models_exist():
    all_over_exist = all(os.path.exists(p) for p in XGB_OVER_MODELS.values())
    return (all_over_exist and
            os.path.exists(XGB_BTTS_MODEL) and
            os.path.exists(XGB_ET_MODEL) and
            os.path.exists(XGB_META_PATH))

def _load_cached():
    over_models = {}
    for thresh, path in XGB_OVER_MODELS.items():
        m = xgb.XGBClassifier()
        m.load_model(path)
        over_models[thresh] = m
    model_btts = xgb.XGBClassifier()
    model_btts.load_model(XGB_BTTS_MODEL)
    model_et = xgb.XGBClassifier()
    model_et.load_model(XGB_ET_MODEL)
    with open(XGB_META_PATH, "r") as f:
        meta = json.load(f)
    return over_models, model_btts, model_et, meta

def _save_models(over_models, model_btts, model_et, meta):
    os.makedirs(os.path.dirname(XGB_BTTS_MODEL), exist_ok=True)
    for thresh, model in over_models.items():
        model.save_model(XGB_OVER_MODELS[thresh])
    model_btts.save_model(XGB_BTTS_MODEL)
    if model_et is not None:
        model_et.save_model(XGB_ET_MODEL)
    with open(XGB_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

def train_xgb(conn, force=False):
    """Train or load cached XGBoost market models (5 Over classifiers + BTTS)."""
    if not force and _models_exist():
        try:
            over_models, model_b, model_e, meta = _load_cached()
            print(f"   ✓  Loaded cached XGBoost Markets ({meta.get('n_train', '?')} "
                  f"training samples)")
            print(f"      Over 2.5 LogLoss: {meta.get('over_2_5_logloss', 0):.3f} │ "
                  f"BTTS LogLoss: {meta.get('btts_logloss', 0):.3f}")
            return over_models, model_b, model_e, meta
        except Exception as e:
            print(f"   ⚠  Cache load failed ({e}), retraining...")

    print("   ⬇  Building features for market prediction...")
    df = build_features(conn, force_momentum=force)
    
    if len(df) == 0:
        print(f"   ✗  Feature engineering failed")
        return None, None, None, None

    if len(df) < 1000:
        print(f"   ✗  Not enough data ({len(df)} samples)")
        return None, None, None, None

    print(f"   ✓  {len(df):,} samples built ({len(FEATURE_COLUMNS)} features)")

    X = df[FEATURE_COLUMNS].values
    y_btts = df["btts"].values

    split = int(len(df) * 0.80)
    X_train, X_val = X[:split], X[split:]
    yb_train, yb_val = y_btts[:split], y_btts[split:]

    print("   ⚙  Training XGBoost Market models (5 Over, BTTS + Extra Time)...")

    # Train 5 binary classifiers for Over thresholds
    from sklearn.metrics import log_loss
    over_models = {}
    over_logloss = {}
    for thresh in XGB_OVER_THRESHOLDS:
        target_col = f"over_{str(thresh).replace('.', '_')}"
        y_over = df[target_col].values
        yt_train, yt_val = y_over[:split], y_over[split:]
        
        model = xgb.XGBClassifier(**OVER_PARAMS)
        model.fit(X_train, yt_train, eval_set=[(X_val, yt_val)], verbose=False)
        over_models[thresh] = model
        
        pred_proba = model.predict_proba(X_val)[:, 1]
        ll = float(log_loss(yt_val, pred_proba))
        over_logloss[thresh] = ll

    # Train BTTS classifier
    btts_params = BTTS_PARAMS.copy()
    model_btts = xgb.XGBClassifier(**btts_params)
    model_btts.fit(X_train, yb_train, eval_set=[(X_val, yb_val)], verbose=False)

    pred_b_proba = model_btts.predict_proba(X_val)[:, 1]
    btts_logloss = float(log_loss(yb_val, pred_b_proba))

    # Train ET classifier on Knockout games only
    df_ko = df[df["is_knockout"] == 1]
    if len(df_ko) > 100:
        X_ko = df_ko[FEATURE_COLUMNS].values
        y_et = df_ko["goes_to_et"].values
        split_ko = int(len(df_ko) * 0.80)
        Xk_train, Xk_val = X_ko[:split_ko], X_ko[split_ko:]
        ye_train, ye_val = y_et[:split_ko], y_et[split_ko:]
        
        et_params = ET_PARAMS.copy()
        
        model_et = xgb.XGBClassifier(**et_params)
        model_et.fit(Xk_train, ye_train, eval_set=[(Xk_val, ye_val)], verbose=False)
        
        pred_e_proba = model_et.predict_proba(Xk_val)[:, 1]
        et_logloss = float(log_loss(ye_val, pred_e_proba))
    else:
        model_et = None
        et_logloss = 0.0

    # Feature importance (aggregate across all models)
    all_importances = np.zeros(len(FEATURE_COLUMNS))
    for m in over_models.values():
        all_importances += m.feature_importances_
    all_importances += model_btts.feature_importances_
    if model_et is not None:
        all_importances += model_et.feature_importances_
        all_importances /= (len(over_models) + 2)
    else:
        all_importances /= (len(over_models) + 1)
        
    combined = sorted(zip(FEATURE_COLUMNS, all_importances), key=lambda x: -x[1])

    meta = {
        "n_train": len(df),
        "n_val": int(len(X_val)),
        "n_ko_train": len(df_ko),
        "over_2_5_logloss": over_logloss.get(2.5, 0),
        "over_logloss": {str(k): v for k, v in over_logloss.items()},
        "btts_logloss": btts_logloss,
        "et_logloss": et_logloss,
        "trained_at": datetime.datetime.now().isoformat(),
    }

    print(f"   ✓  Done! Over 2.5 LogLoss: {over_logloss.get(2.5, 0):.3f} │ "
          f"BTTS LogLoss: {btts_logloss:.3f} │ ET LogLoss: {et_logloss:.3f}")
    for thresh, ll in over_logloss.items():
        print(f"      Over {thresh}: LogLoss={ll:.3f}")
    print(f"      Top features: {combined[0][0]} ({combined[0][1]:.3f}), "
          f"{combined[1][0]} ({combined[1][1]:.3f})")

    _save_models(over_models, model_btts, model_et, meta)
    print(f"   ✓  Models cached")

    return over_models, model_btts, model_et, meta


def predict_xgb(over_models, model_btts, model_et, form_a, form_b,
                fifa_pts_a, fifa_pts_b, h2h, venue="neutral", conn=None,
                team_a=None, team_b=None, match_date=None,
                absent_a=None, absent_b=None, is_knockout=False):
    """
    Predict markets using XGBoost direct classification.
    
    Args:
        over_models: dict of {threshold: XGBClassifier} for Over/Under
        model_btts: XGBClassifier for BTTS
        absent_a: list of absent player names for team A (for lineup shock)
        absent_b: list of absent player names for team B (for lineup shock)
        is_knockout: whether this is a knockout-phase match
    """
    if over_models is None or model_btts is None:
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



    # Compute dynamic tournament weight and days_since_last
    tourn_w = 0.85  # Default: World Cup weight
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

    # Variance features from recent form
    form_gf_std_a = form_ga_std_a = form_tot_std_a = 0.0
    form_gf_std_b = form_ga_std_b = form_tot_std_b = 0.0
    if form_a and isinstance(form_a, dict) and form_a.get("recent_results"):
        recent = form_a["recent_results"][:5]
        gf_vals = [r["gf"] for r in recent]
        ga_vals = [r["ga"] for r in recent]
        tot_vals = [r["gf"] + r["ga"] for r in recent]
        if len(gf_vals) > 1:
            form_gf_std_a = float(np.std(gf_vals))
            form_ga_std_a = float(np.std(ga_vals))
            form_tot_std_a = float(np.std(tot_vals))
    if form_b and isinstance(form_b, dict) and form_b.get("recent_results"):
        recent = form_b["recent_results"][:5]
        gf_vals = [r["gf"] for r in recent]
        ga_vals = [r["ga"] for r in recent]
        tot_vals = [r["gf"] + r["ga"] for r in recent]
        if len(gf_vals) > 1:
            form_gf_std_b = float(np.std(gf_vals))
            form_ga_std_b = float(np.std(ga_vals))
            form_tot_std_b = float(np.std(tot_vals))

    # Tempo features
    early_sot_a, tempo_var_a = _compute_tempo_features(h_sot_bins)
    early_sot_b, tempo_var_b = _compute_tempo_features(a_sot_bins)
    
    base_feat = [
        elo_a, elo_b, elo_a - elo_b,
        fifa_pts_a or median, fifa_pts_b or median,
        (fifa_pts_a or median) - (fifa_pts_b or median),
        rank_a, rank_b, rank_a - rank_b,
        fgf_a, fga_a, fgf_b, fga_b, fgd_a, fgd_b,
        fwr_a, fwr_b,
        # Variance features
        form_gf_std_a, form_ga_std_a, form_tot_std_a,
        form_gf_std_b, form_ga_std_b, form_tot_std_b,
        cs_rate_a, fts_rate_a, avg_tot_a,
        cs_rate_b, fts_rate_b, avg_tot_b,
        h2h_wr, h2h_total, h2h_gd,
        tourn_w, is_neutral, days_since,
        # Knockout flag
        1 if is_knockout else 0,
        # Tempo features
        early_sot_a, early_sot_b,
        early_sot_a * early_sot_b,  # both_early_scorers interaction
        tempo_var_a, tempo_var_b,
        # Parity
        abs(elo_a - elo_b),
        abs((fifa_pts_a or median) - (fifa_pts_b or median)),
        abs(rank_a - rank_b),
        cs_rate_a * cs_rate_b,
        fga_a * fga_b,
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

    # Direct classification: predict_proba for each threshold
    outcomes = {}
    for thresh in XGB_OVER_THRESHOLDS:
        thresh_str = str(thresh).replace('.', '_')
        if thresh in over_models:
            prob = float(over_models[thresh].predict_proba(feature_vec)[0, 1])
        else:
            prob = 0.5  # fallback
        outcomes[f"over_{thresh_str}_pct"] = prob * 100
        outcomes[f"under_{thresh_str}_pct"] = (1 - prob) * 100
    
    btts_yes = float(model_btts.predict_proba(feature_vec)[0, 1])
    outcomes["btts_yes_pct"] = btts_yes * 100
    outcomes["btts_no_pct"] = (1 - btts_yes) * 100

    if is_knockout and model_et is not None:
        et_prob = float(model_et.predict_proba(feature_vec)[0, 1])
        outcomes["xgb_et_prob"] = et_prob

    return None, outcomes, feature_vec


def _compute_xg_concentration_live(conn, team, match_date):
    """Compute xG concentration at inference time from player_stats."""
    try:
        date_filter = f"AND match_date < '{match_date}'" if match_date else ""
        df = pd.read_sql_query(
            f"SELECT player_id, SUM(expected_goals_xg) as total_xg "
            f"FROM player_stats WHERE team = ? {date_filter} "
            f"GROUP BY player_id ORDER BY total_xg DESC",
            conn, params=(team,))
        if len(df) == 0:
            return 0.33
        total_xg = df['total_xg'].sum()
        if total_xg <= 0:
            return 0.33
        top3 = df.head(3)['total_xg'].sum()
        return top3 / total_xg
    except Exception:
        return 0.33


def _apply_lineup_shock(conn, team, team_xg, absent_players):
    """Reduce team_xg based on absent players' xG share."""
    try:
        df = pd.read_sql_query(
            "SELECT player_name, SUM(expected_goals_xg) as total_xg "
            "FROM player_stats WHERE team = ? GROUP BY player_name",
            conn, params=(team,))
        if len(df) == 0:
            return team_xg
        total_team_xg = df['total_xg'].sum()
        if total_team_xg <= 0:
            return team_xg
        
        absent_xg = 0
        for name in absent_players:
            match = df[df['player_name'].str.contains(name, case=False, na=False)]
            if len(match) > 0:
                absent_xg += match['total_xg'].iloc[0]
        
        shock_ratio = absent_xg / total_team_xg
        return team_xg * (1.0 - shock_ratio)
    except Exception:
        return team_xg
