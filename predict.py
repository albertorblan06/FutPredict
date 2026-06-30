#!/usr/bin/env python3
"""
FutPredict v2.0 — CLI Entry Point

Usage:
  python predict.py "Spain" "Germany" --venue neutral
  python predict.py --retrain-xgb --retrain-dl

If team names are omitted, the script will prompt interactively.
"""
import argparse
import sys
import numpy as np
import os
import sqlite3

# Prevent OpenMP deadlock between PyTorch and XGBoost on Apple Silicon
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from futpredict.config import DB_PATH
from futpredict.data import (get_connection, setup_database, download_all,
                             import_matches, import_historical_rankings, import_shootouts)
from futpredict.rankings import (load_rankings_from_csv, import_rankings_2026_to_db,
                                 get_fifa_points, get_fifa_rank)
from futpredict.names import find_team_in_db, suggest_teams
import datetime
from futpredict.analysis import get_recent_form, get_head_to_head
from futpredict.home_advantage import compute_home_advantage
from futpredict.statistical_model import fit_dixon_coles, predict_dc
from futpredict.xgb_model import train_xgb, predict_xgb
from futpredict.xgb_advance import train_advance_xgb, predict_advance
from futpredict.xgb_advanced import train_advanced_xgb, predict_advanced
from futpredict.deep_model import train_lstm_mdn, predict_lstm
from futpredict.blending import blend_matrices, optimize_blend_weights, matrix_to_outcomes
from futpredict.simulation import run_simulation
from futpredict.report import print_report


def main():
    parser = argparse.ArgumentParser(description="FutPredict v2.0 — Match Prediction Engine")
    parser.add_argument("team_a", nargs="?", help="Name of Team A")
    parser.add_argument("team_b", nargs="?", help="Name of Team B")
    parser.add_argument("--venue", choices=["home_a", "home_b", "neutral"],
                        default="neutral", help="Match venue (default: neutral)")
    parser.add_argument("--retrain-xgb", action="store_true",
                        help="Force retrain XGBoost model")
    parser.add_argument("--retrain-dl", action="store_true",
                        help="Force retrain LSTM+MDN model")
    parser.add_argument("--sims", type=int, default=100000,
                        help="Number of Monte Carlo simulations")
    parser.add_argument("--train-only", action="store_true",
                        help="Train models and exit without predicting")
    parser.add_argument("--update-data", action="store_true",
                        help="Force download latest matches from GitHub and rebuild DB")
    parser.add_argument("--date", type=str, default=None,
                        help="Simulate predictions as if today is this date (YYYY-MM-DD). Ignores any data after this date to prevent leakage.")
    parser.add_argument("--absent-a", type=str, default=None,
                        help="Comma-separated list of absent player names for Team A (lineup shock)")
    parser.add_argument("--absent-b", type=str, default=None,
                        help="Comma-separated list of absent player names for Team B (lineup shock)")
    parser.add_argument("--knockout", action="store_true",
                        help="Flag this match as a knockout-phase game (suppresses goal totals)")
    parser.add_argument("--odds-a", type=float, default=None,
                        help="Decimal odds for Team A (Home/Team 1) win (e.g. 2.10)")
    parser.add_argument("--odds-d", type=float, default=None,
                        help="Decimal odds for Draw (e.g. 3.40)")
    parser.add_argument("--odds-b", type=float, default=None,
                        help="Decimal odds for Team B (Away/Team 2) win (e.g. 3.80)")
    parser.add_argument("--odds-qa", type=float, default=None,
                        help="Decimal odds for Team A To Qualify (e.g. 1.80)")
    parser.add_argument("--odds-qb", type=float, default=None,
                        help="Decimal odds for Team B To Qualify (e.g. 2.00)")
    
    args = parser.parse_args()

    # Interactive input
    team_a_input = args.team_a
    team_b_input = args.team_b

    if not args.train_only and (not team_a_input or not team_b_input):
        print("\n╔════════════════════════════════════════════════════════════╗")
        print("║  FUTPREDICT v2.0 — Interactive Mode                        ║")
        print("╚════════════════════════════════════════════════════════════╝\n")
        if not team_a_input:
            team_a_input = input("Enter Team A: ").strip()
        if not team_b_input:
            team_b_input = input("Enter Team B: ").strip()
            
        if not team_a_input or not team_b_input:
            print("Error: Both teams must be provided.")
            sys.exit(1)

    print("\n┌─ Step 1/8: Initializing & Loading Data...")
    
    # Initialize DB and Rankings
    is_first_run = not os.path.exists(DB_PATH)
    conn = get_connection()
    
    if is_first_run or args.update_data:
        print("   ⚙  Setting up database (fetching latest GitHub datasets)...")
        setup_database(conn)
        import_rankings_2026_to_db(conn)
        download_all()
        import_shootouts(conn)
        import_matches(conn)
        import_historical_rankings(conn)
        
        # New: Also update advanced stats
        sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
        try:
            from build_advanced_stats import update_advanced_stats
            update_advanced_stats(conn)
            
            from build_player_db import ingest_player_stats
            ingest_player_stats()
        except ImportError as e:
            print(f"   ⚠ Could not update advanced/player stats: {e}")
            
    load_rankings_from_csv()
    
    cur = conn.execute("SELECT COUNT(*) FROM matches")
    match_count = cur.fetchone()[0]
        
    print(f"   ✓  Database ready ({match_count:,} matches)")
    
    if args.train_only:
        print("\n┌─ Step 2/3: Training Machine Learning (XGBoost)...")
        train_xgb(conn, force=args.retrain_xgb)
        train_advance_xgb(conn, force=args.retrain_xgb)
        train_advanced_xgb(conn, force=args.retrain_xgb)
        
        try:
            from futpredict.player_model import train_player_model
            train_player_model(conn, force=args.retrain_xgb)
        except Exception as e:
            print(f"   ⚠ Could not train player model: {e}")
        
        print("\n┌─ Step 3/3: Training Deep Learning (LSTM+MDN)...")
        train_lstm_mdn(conn, force=args.retrain_dl)
        
        print("\n   ✓  Training complete. Exiting (--train-only).")
        sys.exit(0)
    
    print("\n┌─ Step 2/8: Resolving Team Names...")
    db_name_a = find_team_in_db(conn, team_a_input)
    db_name_b = find_team_in_db(conn, team_b_input)
    
    for req_name, db_name, label in [(team_a_input, db_name_a, "Team A"), 
                                     (team_b_input, db_name_b, "Team B")]:
        if not db_name:
            print(f"   ✗  Error: Could not find {label} '{req_name}' in database.")
            sug = suggest_teams(conn, req_name)
            if sug:
                print(f"      Did you mean: {', '.join(sug)}?")
            sys.exit(1)
        rank = get_fifa_rank(db_name) or "?"
        print(f"   ✓  {req_name} → DB: '{db_name}' │ FIFA: #{rank}")

    print("\n┌─ Step 3/8: Analyzing Form & H2H...")
    form_a = get_recent_form(conn, db_name_a, reference_date=args.date)
    form_b = get_recent_form(conn, db_name_b, reference_date=args.date)
    h2h = get_head_to_head(conn, db_name_a, db_name_b, reference_date=args.date)
    
    if form_a:
        print(f"   ✓  {db_name_a}: {form_a['matches_analyzed']} matches analyzed")
    if form_b:
        print(f"   ✓  {db_name_b}: {form_b['matches_analyzed']} matches analyzed")
    print(f"   ✓  H2H: {h2h['total_matches']} historical matches")

    print("\n┌─ Step 4/8: Computing Dynamic Home Advantage...")
    ha_factor = compute_home_advantage(conn, team_a_db=db_name_a if args.venue == "home_a" else None)
    print(f"   ✓  HA factor: {ha_factor*100:.1f}%")
    
    print("\n┌─ Step 5/8: Statistical Model (Dixon-Coles)...")
    dc_params = fit_dixon_coles(conn)
    if dc_params:
        # Override home advantage from DC if we have a confederation-specific one
        if args.venue != "neutral":
            dc_params["home_adv"] = 1.0 + ha_factor
        
        matrix_dc, dc_details = predict_dc(dc_params, db_name_a, db_name_b, args.venue)
    else:
        print("   ✗  Failed to fit Dixon-Coles model.")
        matrix_dc, dc_details = None, None

    print("\n┌─ Step 6/8: Machine Learning (XGBoost)...")
    over_models, xgb_btts, xgb_et, xgb_meta = train_xgb(conn, force=args.retrain_xgb)
    
    if args.knockout:
        advance_model = train_advance_xgb(conn, force=args.retrain_xgb)
    else:
        advance_model = None
    fifa_pts_a, _ = get_fifa_points(db_name_a)
    fifa_pts_b, _ = get_fifa_points(db_name_b)
    
    # Parse absent players
    absent_a = [p.strip() for p in args.absent_a.split(",")] if args.absent_a else None
    absent_b = [p.strip() for p in args.absent_b.split(",")] if args.absent_b else None
    
    matrix_xgb, xgb_details, feature_vec = predict_xgb(over_models, xgb_btts, xgb_et, form_a, form_b, 
                                          fifa_pts_a, fifa_pts_b, h2h, args.venue,
                                          conn=conn, team_a=db_name_a, team_b=db_name_b,
                                          match_date=args.date,
                                          absent_a=absent_a, absent_b=absent_b,
                                          is_knockout=args.knockout)
    if xgb_details:
        if args.knockout and advance_model:
            xgb_details["advance_prob"] = predict_advance(advance_model, feature_vec)
        print("   ✓  Market metrics predicted.")

    print("   ⬇  Loading Advanced Models (Corners, Cards, SOT, Possession)...")
    adv_cor, adv_car, adv_sot, adv_pos, adv_meta = train_advanced_xgb(conn, force=args.retrain_xgb)
    
    adv_details = predict_advanced(adv_cor, adv_car, adv_sot, adv_pos, db_name_a, db_name_b, 
                                   match_date=args.date if args.date else datetime.date.today().isoformat(), conn=conn)
    
    if adv_details:
        print("   ✓  Advanced metrics predicted.")
    else:
        print("   ⚠  Could not predict advanced metrics (insufficient historical advanced stats).")
    
    print("\n┌─ Step 7/8: Deep Learning (LSTM+MDN & GoalCountNet)...")
    lstm_model_agg, lstm_model_cons, lstm_goals, lstm_idx, lstm_meta_agg, lstm_meta_cons = train_lstm_mdn(conn, force=args.retrain_dl)
    matrix_lstm, lstm_details_agg, lstm_details_cons = predict_lstm(
        lstm_model_agg, lstm_model_cons, lstm_goals, lstm_idx, conn, 
        db_name_a, db_name_b, args.venue, match_date=args.date, 
        meta_agg=lstm_meta_agg, meta_cons=lstm_meta_cons
    )

    print("\n┌─ Step 8/8: Simulating Specialized Markets...")
    
    try:
        from futpredict.player_model import train_player_model, predict_goalscorers
        player_model, _ = train_player_model(conn, force=False)
        goalscorers = predict_goalscorers(player_model, conn, db_name_a, db_name_b)
    except Exception as e:
        print(f"   ⚠ Could not predict goalscorers: {e}")
        goalscorers = []

    # Ensemble GoalCountNet with XGBoost for Totals
    # XGBoost gets 65% weight — trained as direct binary classifiers for O/U
    # LSTM GoalCountNet gets 35% — useful smoothing but has independence assumption
    W_XGB, W_DL = 0.65, 0.35
    if matrix_lstm is not None and xgb_details is not None:
        # Sanity check: compute Over 1.5 from the score matrix
        dl_over_1_5 = sum(matrix_lstm[i, j] for i in range(10) for j in range(10) if i + j > 1.5) * 100
        dl_usable = dl_over_1_5 >= 40.0  # Real football is ~80-85% Over 1.5
        
        for thresh_num, thresh_str in [(1.5, "1_5"), (2.5, "2_5"), (3.5, "3_5"), (4.5, "4_5")]:
            o_prob = 0
            for i in range(10):
                for j in range(10):
                    if i + j > thresh_num:
                        o_prob += matrix_lstm[i, j]
            
            dl_over = o_prob * 100
            dl_under = (1 - o_prob) * 100
            
            xgb_over = xgb_details.get(f"over_{thresh_str}_pct", 0)
            xgb_under = xgb_details.get(f"under_{thresh_str}_pct", 0)
            
            if dl_usable:
                xgb_details[f"over_{thresh_str}_pct"] = W_XGB * xgb_over + W_DL * dl_over
                xgb_details[f"under_{thresh_str}_pct"] = W_XGB * xgb_under + W_DL * dl_under
            # else: keep XGBoost-only values
            
        # Ensemble BTTS (50/50 — both models calibrate well for this market)
        dl_btts_prob = 0
        for i in range(1, 10):
            for j in range(1, 10):
                dl_btts_prob += matrix_lstm[i, j]
                
        dl_btts_yes = dl_btts_prob * 100
        dl_btts_no = (1 - dl_btts_prob) * 100
        
        xgb_btts_yes = xgb_details.get("btts_yes_pct", 0)
        xgb_btts_no = xgb_details.get("btts_no_pct", 0)
        
        if dl_usable:
            xgb_details["btts_yes_pct"] = (xgb_btts_yes + dl_btts_yes) / 2.0
            xgb_details["btts_no_pct"] = (xgb_btts_no + dl_btts_no) / 2.0
            
        # Ensemble Extra Time with LSTM Draw (Idea 1: 60/40)
        if xgb_details.get("xgb_et_prob") is not None:
            raw_et = xgb_details["xgb_et_prob"]
            d_agg = lstm_details_agg.get("draw_pct", 0) / 100.0 if lstm_details_agg else 0.0
            d_cons = lstm_details_cons.get("draw_pct", 0) / 100.0 if lstm_details_cons else 0.0
            if lstm_details_agg and lstm_details_cons:
                lstm_draw_prob = (d_agg + d_cons) / 2.0
            elif lstm_details_agg or lstm_details_cons:
                lstm_draw_prob = d_agg or d_cons
            else:
                lstm_draw_prob = 0.0
                
            if lstm_draw_prob > 0:
                xgb_details["xgb_et_prob"] = (raw_et * 0.60) + (lstm_draw_prob * 0.40)

    outcomes = {
        "n_sims": args.sims,
        "lstm_agg": lstm_details_agg or {},
        "lstm_cons": lstm_details_cons or {},
        "xgb": xgb_details or {}
    }
    
    if matrix_dc is not None:
        outcomes["dc"] = run_simulation(matrix_dc, n_sims=args.sims)
    
    # Compile models info
    models_info = {
        "venue": args.venue,
        "dc": dc_details,
        "xgb": xgb_details,
        "xgb_meta": xgb_meta,
        "lstm_agg": lstm_details_agg,
        "lstm_cons": lstm_details_cons,
        "lstm_meta_agg": lstm_meta_agg,
        "lstm_meta_cons": lstm_meta_cons,
        "advanced": adv_details,
        "goalscorers": goalscorers
    }
    
    home_adv_info = {
        "factor": ha_factor,
        "confederation": "Dynamic"
    }

    print_report(
        name_a=team_a_input, name_b=team_b_input,
        db_name_a=db_name_a, db_name_b=db_name_b,
        fifa_pts_a=fifa_pts_a, fifa_pts_b=fifa_pts_b,
        rank_a=get_fifa_rank(db_name_a), rank_b=get_fifa_rank(db_name_b),
        form_a=form_a, form_b=form_b, h2h=h2h,
        outcomes=outcomes, match_count=match_count,
        models_info=models_info, home_adv_info=home_adv_info,
        odds=(args.odds_a, args.odds_d, args.odds_b) if args.odds_a else None,
        is_knockout=args.knockout,
        odds_q=(args.odds_qa, args.odds_qb) if args.odds_qa else None
    )

if __name__ == "__main__":
    main()
