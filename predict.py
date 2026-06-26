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
                             import_matches, import_historical_rankings)
from futpredict.rankings import (load_rankings_from_csv, import_rankings_2026_to_db,
                                 get_fifa_points, get_fifa_rank)
from futpredict.names import find_team_in_db, suggest_teams
import datetime
from futpredict.analysis import get_recent_form, get_head_to_head
from futpredict.home_advantage import compute_home_advantage
from futpredict.statistical_model import fit_dixon_coles, predict_dc
from futpredict.xgb_model import train_xgb, predict_xgb
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
    xgb_h, xgb_a, xgb_meta = train_xgb(conn, force=args.retrain_xgb)
    fifa_pts_a, _ = get_fifa_points(db_name_a)
    fifa_pts_b, _ = get_fifa_points(db_name_b)
    
    matrix_xgb, xgb_details = predict_xgb(xgb_h, xgb_a, form_a, form_b, 
                                          fifa_pts_a, fifa_pts_b, h2h, args.venue, conn=conn, team_a=db_name_a, team_b=db_name_b, match_date=args.date)

    print("   ⬇  Loading Advanced Models (Corners, Cards, SOT, Possession)...")
    adv_cor, adv_car, adv_sot, adv_pos, adv_meta = train_advanced_xgb(conn, force=args.retrain_xgb)
    
    adv_details = predict_advanced(adv_cor, adv_car, adv_sot, adv_pos, db_name_a, db_name_b, 
                                   match_date=args.date if args.date else datetime.date.today().isoformat(), conn=conn)
    
    if adv_details:
        print("   ✓  Advanced metrics predicted.")
    else:
        print("   ⚠  Could not predict advanced metrics (insufficient historical advanced stats).")
    
    print("\n┌─ Step 7/8: Deep Learning (LSTM+MDN & GoalCountNet)...")
    lstm_model, lstm_goals, lstm_idx, lstm_meta = train_lstm_mdn(conn, force=args.retrain_dl)
    matrix_lstm, lstm_details = predict_lstm(lstm_model, lstm_goals, lstm_idx, conn, 
                                             db_name_a, db_name_b, args.venue, match_date=args.date)

    print("\n┌─ Step 8/8: Simulating Specialized Markets...")
    
    try:
        from futpredict.player_model import train_player_model, predict_goalscorers
        player_model, _ = train_player_model(conn, force=False)
        goalscorers = predict_goalscorers(player_model, conn, db_name_a, db_name_b)
    except Exception as e:
        print(f"   ⚠ Could not predict goalscorers: {e}")
        goalscorers = []

    # Ensemble GoalCountNet with XGBoost for Totals
    if matrix_lstm is not None and xgb_details is not None:
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
            
            xgb_details[f"over_{thresh_str}_pct"] = (xgb_over + dl_over) / 2.0
            xgb_details[f"under_{thresh_str}_pct"] = (xgb_under + dl_under) / 2.0
            
        # Ensemble BTTS
        dl_btts_prob = 0
        for i in range(1, 10):
            for j in range(1, 10):
                dl_btts_prob += matrix_lstm[i, j]
                
        dl_btts_yes = dl_btts_prob * 100
        dl_btts_no = (1 - dl_btts_prob) * 100
        
        xgb_btts_yes = xgb_details.get("btts_yes_pct", 0)
        xgb_btts_no = xgb_details.get("btts_no_pct", 0)
        
        xgb_details["btts_yes_pct"] = (xgb_btts_yes + dl_btts_yes) / 2.0
        xgb_details["btts_no_pct"] = (xgb_btts_no + dl_btts_no) / 2.0

    outcomes = {
        "n_sims": args.sims,
        "lstm": lstm_details or {},
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
        "lstm": lstm_details,
        "lstm_meta": lstm_meta,
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
        models_info=models_info, home_adv_info=home_adv_info
    )

if __name__ == "__main__":
    main()
