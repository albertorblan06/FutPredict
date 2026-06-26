import re
import sqlite3
import sys
import os
import math
import numpy as np
import argparse
from scipy.stats import poisson

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from futpredict.config import DB_PATH
from futpredict.names import find_team_in_db
from futpredict.analysis import get_recent_form, get_head_to_head
from futpredict.statistical_model import fit_dixon_coles
from futpredict.xgb_model import train_xgb, predict_xgb
from futpredict.xgb_advanced import train_advanced_xgb, predict_advanced
from futpredict.deep_model import train_lstm_mdn, predict_lstm
from futpredict.rankings import get_fifa_points, load_rankings_from_csv

text = """
Mexico vs. South Africa	2 – 0
South Korea vs. Czechia	2 – 1
Canada vs. Bosnia and Herzegovina	1 – 1
USA vs. Paraguay	4 – 1
Qatar vs. Switzerland	1 – 1
Brazil vs. Morocco	1 – 1
Haiti vs. Scotland	0 – 1
Australia vs. Turkey	2 – 0
Germany vs. Curaçao	7 – 1
Netherlands vs. Japan	2 – 2
Ivory Coast vs. Ecuador	1 – 0
Sweden vs. Tunisia	5 – 1
Spain vs. Cape Verde	0 – 0
Belgium vs. Egypt	1 – 1
Saudi Arabia vs. Uruguay	1 – 1
Iran vs. New Zealand	2 – 2
France vs. Senegal	3 – 1
Iraq vs. Norway	1 – 4
Argentina vs. Algeria	3 – 0
Austria vs. Jordan	3 – 1
Portugal vs. DR Congo	1 – 1
England vs. Croatia	4 – 2
Ghana vs. Panama	1 – 0
Uzbekistan vs. Colombia	1 – 3
Czechia vs. South Africa	1 – 1
Switzerland vs. Bosnia and Herzegovina	4 – 1
Canada vs. Qatar	6 – 0
Mexico vs. South Korea	1 – 0
USA vs. Australia	2 – 0
Scotland vs. Morocco	0 – 1
Brazil vs. Haiti	3 – 0
Turkey vs. Paraguay	0 – 1
Netherlands vs. Sweden	5 – 1
Germany vs. Ivory Coast	2 – 1
Ecuador vs. Curaçao	0 – 0
Tunisia vs. Japan	0 – 4
Spain vs. Saudi Arabia	4 – 0
Belgium vs. Iran	0 – 0
Uruguay vs. Cape Verde	2 – 2
New Zealand vs. Egypt	1 – 3
Argentina vs. Austria	2 – 0
France vs. Iraq	3 – 0
Norway vs. Senegal	3 – 2
Jordan vs. Algeria	1 – 2
Portugal vs. Uzbekistan	5 – 0
England vs. Ghana	0 – 0
Panama vs. Croatia	0 – 1
Colombia vs. DR Congo	1 – 0
Switzerland vs. Canada	2 – 1
Bosnia and Herzegovina vs. Qatar	3 – 1
Morocco vs. Haiti	4 – 2
Scotland vs. Brazil	0 – 3
South Africa vs. South Korea	1 – 0
Czechia vs. Mexico	0 – 3
"""

def main():
    parser = argparse.ArgumentParser(description="FutPredict Batch Test")
    parser.add_argument("--date", type=str, default="2026-06-01",
                        help="Simulate predictions as if today is this date")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    load_rankings_from_csv()
    
    # Load models once
    dc_params = fit_dixon_coles(conn, reference_date=args.date)
    xgb_h, xgb_a, _ = train_xgb(conn, force=False)
    lstm_model, lstm_goals, lstm_idx, _ = train_lstm_mdn(conn, force=False)
    m_cor, m_car, m_sot, m_pos, _ = train_advanced_xgb(conn, force=False)
    
    matches = []
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) == 2:
            teams = parts[0].split(" vs. ")
            score = parts[1].split(" – ")
            matches.append((teams[0].strip(), teams[1].strip(), int(score[0].strip()), int(score[1].strip())))
        
    print(f"Parsed {len(matches)} matches.")
    
    match_bets = 0; match_won = 0; match_lost = 0
    totals_bets = 0; totals_won = 0; totals_lost = 0
    tourney_bets = 0; tourney_won = 0
    combined_normal_bets = 0; combined_normal_won = 0
    combined_tourney_bets = 0; combined_tourney_won = 0
    
    adv_corners_bets = 0; adv_corners_won = 0
    adv_cards_bets = 0; adv_cards_won = 0
    adv_sot_bets = 0; adv_sot_won = 0
    adv_pos_bets = 0; adv_pos_won = 0
    
    results = []
    test_date = args.date
    
    for (team_a, team_b, score_a, score_b) in matches:
        team_a_db = find_team_in_db(conn, team_a)
        team_b_db = find_team_in_db(conn, team_b)
        
        if not team_a_db or not team_b_db:
            print(f"Skipping {team_a} vs {team_b} (names not found)")
            continue
        
        form_a = get_recent_form(conn, team_a_db, reference_date=test_date)
        form_b = get_recent_form(conn, team_b_db, reference_date=test_date)
        h2h = get_head_to_head(conn, team_a_db, team_b_db, reference_date=test_date)
        fifa_pts_a, _ = get_fifa_points(team_a_db)
        fifa_pts_b, _ = get_fifa_points(team_b_db)
        
        _, xgb_details = predict_xgb(xgb_h, xgb_a, form_a, form_b, fifa_pts_a, fifa_pts_b, h2h, venue="neutral", conn=conn, team_a=team_a_db, team_b=team_b_db, match_date=test_date)
        matrix_lstm, lstm_details = predict_lstm(lstm_model, lstm_goals, lstm_idx, conn, team_a_db, team_b_db, venue="neutral", match_date=test_date)
        
        actual_1x2 = "1" if score_a > score_b else ("2" if score_b > score_a else "X")
        actual_goals = score_a + score_b
        
        # Advanced Stats Eval
        cur = conn.execute("SELECT home_corners, away_corners, home_cards, away_cards, home_sot, away_sot, home_possession FROM advanced_stats WHERE home_team = ? AND away_team = ? AND match_date >= ? ORDER BY match_date ASC LIMIT 1", (team_a_db, team_b_db, test_date))
        adv_row = cur.fetchone()
        
        if adv_row:
            act_cor = adv_row[0] + adv_row[1]
            act_car = adv_row[2] + adv_row[3]
            act_sot = adv_row[4] + adv_row[5]
            act_pos_home = adv_row[6]
            
            adv_preds = predict_advanced(m_cor, m_car, m_sot, m_pos, team_a_db, team_b_db, test_date, conn)
            if adv_preds:
                lambda_cor = adv_preds["expected_corners"]
                lambda_car = adv_preds["expected_cards"]
                lambda_sot = adv_preds["expected_sot"]
                
                for thresh in [8.5, 9.5]:
                    k = int(thresh)
                    o_cor = (1.0 - poisson.cdf(k, lambda_cor)) * 100
                    u_cor = poisson.cdf(k, lambda_cor) * 100
                    if o_cor > 60.0:
                        adv_corners_bets += 1
                        if act_cor > thresh: adv_corners_won += 1
                        break
                    elif u_cor > 60.0:
                        adv_corners_bets += 1
                        if act_cor < thresh: adv_corners_won += 1
                        break
                        
                for thresh in [3.5, 4.5]:
                    k = int(thresh)
                    o_car = (1.0 - poisson.cdf(k, lambda_car)) * 100
                    u_car = poisson.cdf(k, lambda_car) * 100
                    if o_car > 60.0:
                        adv_cards_bets += 1
                        if act_car > thresh: adv_cards_won += 1
                        break
                    elif u_car > 60.0:
                        adv_cards_bets += 1
                        if act_car < thresh: adv_cards_won += 1
                        break
                        
                for thresh in [7.5, 8.5]:
                    k = int(thresh)
                    o_sot = (1.0 - poisson.cdf(k, lambda_sot)) * 100
                    u_sot = poisson.cdf(k, lambda_sot) * 100
                    if o_sot > 60.0:
                        adv_sot_bets += 1
                        if act_sot > thresh: adv_sot_won += 1
                        break
                    elif u_sot > 60.0:
                        adv_sot_bets += 1
                        if act_sot < thresh: adv_sot_won += 1
                        break
                        
                pos_h = adv_preds.get("home_possession", 50.0)
                pred_pos_winner = "1" if pos_h > 50 else "2"
                act_pos_winner = "1" if act_pos_home > 50 else "2"
                adv_pos_bets += 1
                if pred_pos_winner == act_pos_winner: adv_pos_won += 1
        
        w_a = lstm_details["win_a_pct"]
        w_d = lstm_details["draw_pct"]
        w_b = lstm_details["win_b_pct"]
        
        dc_1x = w_a + w_d
        dc_x2 = w_b + w_d
        dc_12 = w_a + w_b
        
        bet_1x2 = None
        if w_a > 68.0: bet_1x2 = "1"
        elif w_b > 68.0: bet_1x2 = "2"
        else:
            if dc_1x > 70.0: bet_1x2 = "1X"
            elif dc_x2 > 70.0: bet_1x2 = "X2"
            elif dc_12 > 70.0: bet_1x2 = "12"
                
        match_win = False
        if bet_1x2 == "1" and actual_1x2 == "1": match_win = True
        elif bet_1x2 == "2" and actual_1x2 == "2": match_win = True
        elif bet_1x2 == "1X" and actual_1x2 in ["1", "X"]: match_win = True
        elif bet_1x2 == "X2" and actual_1x2 in ["X", "2"]: match_win = True
        elif bet_1x2 == "12" and actual_1x2 in ["1", "2"]: match_win = True
        
        if bet_1x2 is not None:
            match_bets += 1
            if match_win: match_won += 1
            else: match_lost += 1

        # Ensemble GoalCountNet with XGBoost for Totals
        dl_ou_probs = {}
        if matrix_lstm is not None:
            dl_ou_probs["over_1_5_pct"] = np.sum(matrix_lstm[np.triu_indices_from(matrix_lstm, k=2)]) * 100 + \
                                          np.sum(matrix_lstm[np.tril_indices_from(matrix_lstm, k=-2)]) * 100 + \
                                          matrix_lstm[1, 1]*100
            dl_ou_probs["over_2_5_pct"] = np.sum(matrix_lstm[np.triu_indices_from(matrix_lstm, k=3)]) * 100 + \
                                          np.sum(matrix_lstm[np.tril_indices_from(matrix_lstm, k=-3)]) * 100 + \
                                          matrix_lstm[2, 1]*100 + matrix_lstm[1, 2]*100 + matrix_lstm[2, 2]*100 + \
                                          matrix_lstm[3, 0]*100 + matrix_lstm[0, 3]*100
            dl_ou_probs["over_3_5_pct"] = np.sum(matrix_lstm[np.triu_indices_from(matrix_lstm, k=4)]) * 100 + \
                                          np.sum(matrix_lstm[np.tril_indices_from(matrix_lstm, k=-4)]) * 100
                                          
            # We can simplify discrete over/under calculation:
            total_prob = 0
            for i in range(10):
                for j in range(10):
                    total_prob += matrix_lstm[i, j]
            
            for thresh_num, thresh_str in [(1.5, "1_5"), (2.5, "2_5"), (3.5, "3_5")]:
                o_prob = 0
                for i in range(10):
                    for j in range(10):
                        if i + j > thresh_num:
                            o_prob += matrix_lstm[i, j]
                dl_ou_probs[f"over_{thresh_str}_pct"] = o_prob * 100
                dl_ou_probs[f"under_{thresh_str}_pct"] = (1 - o_prob) * 100

        best_totals_diff = 0
        normal_bet_totals = None
        for thresh in ["1_5", "2_5", "3_5"]:
            x_o = xgb_details.get(f"over_{thresh}_pct", 0)
            x_u = xgb_details.get(f"under_{thresh}_pct", 0)
            d_o = dl_ou_probs.get(f"over_{thresh}_pct", x_o)
            d_u = dl_ou_probs.get(f"under_{thresh}_pct", x_u)
            
            o_val = (x_o + d_o) / 2.0
            u_val = (x_u + d_u) / 2.0
            
            if o_val > 61.5 and (o_val - 50.0) > best_totals_diff:
                best_totals_diff = o_val - 50.0
                normal_bet_totals = f"Over {thresh.replace('_', '.')}"
            elif u_val > 58.0 and (u_val - 50.0) > best_totals_diff:
                best_totals_diff = u_val - 50.0
                normal_bet_totals = f"Under {thresh.replace('_', '.')}"
                
        tourney_bet_totals = None
        pts_a = fifa_pts_a or 0
        pts_b = fifa_pts_b or 0
        pts_diff = abs(pts_a - pts_b)
        
        is_tourney = pts_diff > 200
        if is_tourney:
            is_a_underdog = pts_a < pts_b
            underdog_form = form_a if is_a_underdog else form_b
            underdog_ga = underdog_form.get("weighted_ga", 1.5) if underdog_form else 1.5
            
            if underdog_ga < 1.2:
                x_u = xgb_details.get("under_3_5_pct", 0)
                d_u = dl_ou_probs.get("under_3_5_pct", x_u)
                u35 = (x_u + d_u) / 2.0
                
                if u35 > 65.0: tourney_bet_totals = "Under 3.5"
                else: tourney_bet_totals = "Under 2.5"
            else:
                x_o25 = xgb_details.get("over_2_5_pct", 0)
                d_o25 = dl_ou_probs.get("over_2_5_pct", x_o25)
                o25 = (x_o25 + d_o25) / 2.0
                
                x_o35 = xgb_details.get("over_3_5_pct", 0)
                d_o35 = dl_ou_probs.get("over_3_5_pct", x_o35)
                o35 = (x_o35 + d_o35) / 2.0
                
                ratio = o35 / o25 if o25 > 0 else 0
                if ratio > 0.65: tourney_bet_totals = "Over 3.5"
                else: tourney_bet_totals = "Over 2.5"
                    
        bet_totals = tourney_bet_totals if is_tourney else normal_bet_totals
                 
        normal_totals_win = False
        if normal_bet_totals is not None:
            parts = normal_bet_totals.split()
            threshold = float(parts[1])
            if parts[0] == "Over" and actual_goals > threshold: normal_totals_win = True
            elif parts[0] == "Under" and actual_goals < threshold: normal_totals_win = True
                
        tourney_totals_win = False
        if is_tourney and tourney_bet_totals is not None:
            parts = tourney_bet_totals.split()
            threshold = float(parts[1])
            if parts[0] == "Over" and actual_goals > threshold: tourney_totals_win = True
            elif parts[0] == "Under" and actual_goals < threshold: tourney_totals_win = True
                
        totals_win = tourney_totals_win if is_tourney else normal_totals_win
        
        if bet_totals is not None:
            totals_bets += 1
            if totals_win: totals_won += 1
            else: totals_lost += 1
            if is_tourney:
                tourney_bets += 1
                if totals_win: tourney_won += 1
                
        if bet_1x2 is not None and normal_bet_totals is not None:
            combined_normal_bets += 1
            if match_win and normal_totals_win: combined_normal_won += 1
                
        if bet_1x2 is not None and is_tourney and tourney_bet_totals is not None:
            combined_tourney_bets += 1
            if match_win and tourney_totals_win: combined_tourney_won += 1

        results.append({
            "match": f"{team_a} vs {team_b}",
            "score": f"{score_a}-{score_b}",
            "bet_1x2": bet_1x2 or "No Bet",
            "actual_1x2": actual_1x2,
            "match_result": "✅" if match_win else ("❌" if bet_1x2 else "➖"),
            "normal_totals": normal_bet_totals or "No Bet",
            "tourney_totals": tourney_bet_totals or "N/A",
            "totals_result": "✅" if totals_win else ("❌" if bet_totals else "➖"),
            "combined_normal_result": "✅" if (match_win and normal_totals_win) else ("❌" if (bet_1x2 and normal_bet_totals) else "➖"),
            "combined_tourney_result": "✅" if (match_win and tourney_totals_win) else ("❌" if (bet_1x2 and tourney_bet_totals) else "➖")
        })

    print("="*50)
    print("\n### Match Breakdown")
    print("| Match | Score | 1X2 Bet | 1X2 Act | 1X2 Result | Normal Totals | Tourney Override | Totals Result | Comb Normal | Comb Tourney |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r["match"]} | {r["score"]} | {r["bet_1x2"]} | {r["actual_1x2"]} | {r["match_result"]} | {r["normal_totals"]} | {r["tourney_totals"]} | {r["totals_result"]} | {r["combined_normal_result"]} | {r["combined_tourney_result"]} |")
              
    print(f"\n### 💰 Betting Simulator Results")
    print(f"**1X2 / Double Chance Market:**")
    print(f"- Total Bets Placed: {match_bets} (Skipped {len(results)-match_bets} No Bets)")
    if match_bets > 0:
        print(f"- **Win Rate**: {match_won}/{match_bets} ({match_won/match_bets*100:.1f}%)")
    
    print(f"\n**Totals (O/U) Market:**")
    print(f"- Total Bets Placed: {totals_bets} (Skipped {len(results)-totals_bets} No Bets)")
    if totals_bets > 0:
        print(f"- **Overall Win Rate**: {totals_won}/{totals_bets} ({totals_won/totals_bets*100:.1f}%)")
    if tourney_bets > 0:
        print(f"- **Tournament Mode Override Win Rate**: {tourney_won}/{tourney_bets} ({tourney_won/tourney_bets*100:.1f}%)")
        
    print(f"\n**Combined (1X2 + Totals) Market:**")
    print(f"- **Combined (Normal) Hit Rate**: {combined_normal_won}/{combined_normal_bets} "
          f"({combined_normal_won/combined_normal_bets*100:.1f}% if placed)")
    if combined_tourney_bets > 0:
        print(f"- **Combined (Tourney) Hit Rate**: {combined_tourney_won}/{combined_tourney_bets} "
              f"({combined_tourney_won/combined_tourney_bets*100:.1f}% if placed)")
              
    print(f"\n**Advanced Stats Models:**")
    print(f"- **Corners**: {adv_corners_won}/{adv_corners_bets} ({(adv_corners_won/adv_corners_bets*100) if adv_corners_bets>0 else 0:.1f}%)")
    print(f"- **Cards**: {adv_cards_won}/{adv_cards_bets} ({(adv_cards_won/adv_cards_bets*100) if adv_cards_bets>0 else 0:.1f}%)")
    print(f"- **Shots on Target**: {adv_sot_won}/{adv_sot_bets} ({(adv_sot_won/adv_sot_bets*100) if adv_sot_bets>0 else 0:.1f}%)")
    print(f"- **Possession Winner**: {adv_pos_won}/{adv_pos_bets} ({(adv_pos_won/adv_pos_bets*100) if adv_pos_bets>0 else 0:.1f}%)")
        
    print("-" * 30)

if __name__ == "__main__":
    main()
