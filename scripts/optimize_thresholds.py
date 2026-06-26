import os
import sys
import sqlite3
import numpy as np
import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from futpredict.config import DB_PATH
from futpredict.data import get_connection
from futpredict.statistical_model import fit_dixon_coles
from futpredict.xgb_model import train_xgb, predict_xgb
from futpredict.deep_model import train_lstm_mdn, predict_lstm
from futpredict.analysis import get_recent_form, get_head_to_head
from futpredict.rankings import get_fifa_points

def collect_predictions(conn):
    print("Loading models...")
    xgb_h, xgb_a, _ = train_xgb(conn, force=False)
    lstm_model, lstm_goals, lstm_idx, _ = train_lstm_mdn(conn, force=False)
    dc_params = fit_dixon_coles(conn, reference_date='2026-06-01')
    
    print("Fetching matches for optimization (Last 500 matches)...")
    cur = conn.execute("""
        SELECT date, home_team, away_team, home_score, away_score, tournament, neutral
        FROM matches
        WHERE date >= '2025-01-01' AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY date DESC LIMIT 500
    """)
    matches = cur.fetchall()
    
    data = []
    
    for date_str, home, away, hs, as_, tourn, neut in tqdm(matches):
        try:
            form_a = get_recent_form(conn, home, reference_date=date_str)
            form_b = get_recent_form(conn, away, reference_date=date_str)
            h2h = get_head_to_head(conn, home, away, reference_date=date_str)
            fifa_pts_a, _ = get_fifa_points(home)
            fifa_pts_b, _ = get_fifa_points(away)
            
            _, xgb_o = predict_xgb(xgb_h, xgb_a, form_a, form_b, fifa_pts_a, fifa_pts_b, h2h, venue="neutral", conn=conn, team_a=home, team_b=away, match_date=date_str)
            matrix_lstm, lstm_o = predict_lstm(lstm_model, lstm_goals, lstm_idx, conn, home, away, venue="neutral", match_date=date_str)
            
            if not xgb_o or not lstm_o: continue
            
            w_a = lstm_o["win_a_pct"]
            w_d = lstm_o["draw_pct"]
            w_b = lstm_o["win_b_pct"]
            
            actual_1x2 = "1" if hs > as_ else ("2" if as_ > hs else "X")
            actual_goals = hs + as_
            
            # Simple ensemble for totals
            dl_over_25 = np.sum(matrix_lstm[np.triu_indices_from(matrix_lstm, k=3)]) * 100 + \
                         np.sum(matrix_lstm[np.tril_indices_from(matrix_lstm, k=-3)]) * 100 + \
                         matrix_lstm[2, 1]*100 + matrix_lstm[1, 2]*100 + matrix_lstm[2, 2]*100 + \
                         matrix_lstm[3, 0]*100 + matrix_lstm[0, 3]*100
                         
            o25 = (xgb_o.get("over_2_5_pct", 50) + dl_over_25) / 2.0
            u25 = 100.0 - o25
            
            pts_diff = abs((fifa_pts_a or 0) - (fifa_pts_b or 0))
            is_tourney = pts_diff > 200
            
            data.append({
                "w_a": w_a, "w_d": w_d, "w_b": w_b,
                "o25": o25, "u25": u25,
                "actual_1x2": actual_1x2,
                "actual_goals": actual_goals,
                "is_tourney": is_tourney
            })
        except Exception:
            continue
            
    return data

def optimize(data):
    best_hit_rate = 0
    best_params = {}
    
    # Grid search ranges
    win_thresholds = np.arange(55.0, 80.0, 1.0)
    dc_thresholds = np.arange(55.0, 80.0, 1.0)
    totals_thresholds = np.arange(55.0, 75.0, 1.0)
    
    print("Running Grid Search...")
    for w_t in win_thresholds:
        for dc_t in dc_thresholds:
            for t_t in totals_thresholds:
                bets = 0
                wins = 0
                
                for row in data:
                    # 1X2 Logic
                    w_a, w_d, w_b = row["w_a"], row["w_d"], row["w_b"]
                    dc_1x = w_a + w_d
                    dc_x2 = w_b + w_d
                    dc_12 = w_a + w_b
                    
                    bet_1x2 = None
                    if w_a > w_t: bet_1x2 = "1"
                    elif w_b > w_t: bet_1x2 = "2"
                    elif dc_1x > dc_t: bet_1x2 = "1X"
                    elif dc_x2 > dc_t: bet_1x2 = "X2"
                    elif dc_12 > dc_t: bet_1x2 = "12"
                    
                    if not bet_1x2: continue
                    
                    act = row["actual_1x2"]
                    match_win = False
                    if bet_1x2 == "1" and act == "1": match_win = True
                    elif bet_1x2 == "2" and act == "2": match_win = True
                    elif bet_1x2 == "1X" and act in ["1", "X"]: match_win = True
                    elif bet_1x2 == "X2" and act in ["2", "X"]: match_win = True
                    elif bet_1x2 == "12" and act in ["1", "2"]: match_win = True
                    
                    # Totals Logic
                    o25, u25 = row["o25"], row["u25"]
                    bet_tot = None
                    if o25 > t_t: bet_tot = "O"
                    elif u25 > t_t: bet_tot = "U"
                    
                    if not bet_tot: continue
                    
                    act_g = row["actual_goals"]
                    tot_win = False
                    if bet_tot == "O" and act_g > 2.5: tot_win = True
                    elif bet_tot == "U" and act_g < 2.5: tot_win = True
                    
                    bets += 1
                    if match_win and tot_win:
                        wins += 1
                        
                if bets > 50: # Require at least 50 bets in 500 matches (10% volume)
                    hr = wins / bets
                    if hr > best_hit_rate:
                        best_hit_rate = hr
                        best_params = {"win_t": w_t, "dc_t": dc_t, "tot_t": t_t, "volume": bets}
                        
    print(f"\nOptimization Complete:")
    print(f"Optimal Win Threshold: {best_params['win_t']}%")
    print(f"Optimal Double Chance Threshold: {best_params['dc_t']}%")
    print(f"Optimal Totals Threshold: {best_params['tot_t']}%")
    print(f"Resulting Combined Hit Rate: {best_hit_rate*100:.1f}%")
    print(f"Betting Volume: {best_params['volume']} / {len(data)} matches")

if __name__ == "__main__":
    conn = get_connection()
    d = collect_predictions(conn)
    optimize(d)
