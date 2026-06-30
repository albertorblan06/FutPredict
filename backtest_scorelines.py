import re
import os
import subprocess
import traceback
import math

# We define the true 90-MINUTE regular-time scores for the 2022 World Cup Knockout
# format: (date, team_a, team_b, true_score_a, true_score_b)
MATCHES_2022_90MIN = [
    ("2022-12-03", "Netherlands", "USA", 3, 1),
    ("2022-12-03", "Argentina", "Australia", 2, 1),
    ("2022-12-04", "France", "Poland", 3, 1),
    ("2022-12-04", "England", "Senegal", 3, 0),
    ("2022-12-05", "Japan", "Croatia", 1, 1),
    ("2022-12-05", "Brazil", "South Korea", 4, 1),
    ("2022-12-06", "Morocco", "Spain", 0, 0),
    ("2022-12-06", "Portugal", "Switzerland", 6, 1),
    ("2022-12-09", "Croatia", "Brazil", 0, 0),
    ("2022-12-09", "Netherlands", "Argentina", 2, 2),
    ("2022-12-10", "Morocco", "Portugal", 1, 0),
    ("2022-12-10", "England", "France", 1, 2),
    ("2022-12-13", "Argentina", "Croatia", 3, 0),
    ("2022-12-14", "France", "Morocco", 2, 0),
    ("2022-12-18", "Argentina", "France", 2, 2)
]

# 2024 Euros Knockout (90-minute regular-time scores)
MATCHES_2024_90MIN = [
    ("2024-06-29", "Switzerland", "Italy", 2, 0),
    ("2024-06-29", "Germany", "Denmark", 2, 0),
    ("2024-06-30", "England", "Slovakia", 1, 1),
    ("2024-06-30", "Spain", "Georgia", 4, 1),
    ("2024-07-01", "France", "Belgium", 1, 0),
    ("2024-07-01", "Portugal", "Slovenia", 0, 0),
    ("2024-07-02", "Romania", "Netherlands", 0, 3),
    ("2024-07-02", "Austria", "Turkey", 1, 2),
    ("2024-07-05", "Spain", "Germany", 1, 1),
    ("2024-07-05", "Portugal", "France", 0, 0),
    ("2024-07-06", "England", "Switzerland", 1, 1),
    ("2024-07-06", "Netherlands", "Turkey", 2, 1),
    ("2024-07-09", "Spain", "France", 2, 1),
    ("2024-07-10", "Netherlands", "England", 1, 2),
    ("2024-07-14", "Spain", "England", 2, 1)
]

CONFIG_FILE = "futpredict/config.py"

def set_training_cutoff(date_str):
    with open(CONFIG_FILE, "r") as f:
        content = f.read()
    new_content = re.sub(r'TRAIN_END_DATE\s*=\s*".*?"', f'TRAIN_END_DATE = "{date_str}"', content)
    with open(CONFIG_FILE, "w") as f:
        f.write(new_content)
    print(f"\n[*] Updated TRAIN_END_DATE to {date_str}")

def force_retrain():
    print("[*] Forcing total retrain to flush any future data...")
    script = """
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sqlite3
from futpredict.config import DB_PATH
from futpredict.xgb_model import train_xgb
from futpredict.xgb_advanced import train_advanced_xgb
from futpredict.deep_model import train_lstm_mdn
from futpredict.player_model import train_player_model
conn = sqlite3.connect(DB_PATH)
print("  - Training XGB Models")
train_xgb(conn, force=True)
print("  - Training Advanced XGB Models")
train_advanced_xgb(conn, force=True)
print("  - Training LSTM Models")
train_lstm_mdn(conn, force=True)
print("  - Training Player Models")
train_player_model(conn, force=True)
print("  - Retrain successful.")
"""
    with open("temp_retrain.py", "w") as f:
        f.write(script)
    subprocess.run(["python3", "temp_retrain.py"], check=True)
    os.remove("temp_retrain.py")

def evaluate_matches(matches, tag):
    print(f"\n================ RUNNING {tag} TEST SUITE ================")
    
    total = len(matches)
    top1_exact_hits = 0
    top2_exact_hits = 0
    top1_1x2_hits = 0
    top2_1x2_hits = 0

    for date_str, team_a, team_b, true_score_a, true_score_b in matches:
        cmd = ["python3", "predict.py", team_a, team_b, "--knockout", "--date", date_str]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        out = result.stdout
        
        top_scores = []
        parsing_scores = False
        
        for line in out.split("\n"):
            if "TOP SCORELINES" in line:
                parsing_scores = True
                continue
                
            if parsing_scores:
                # Break parsing if we hit another section header or blank lines after scores
                if "MARKET PROBABILITIES" in line:
                    break
                    
                match = re.search(r'^\s*(\d+)\.\s+(\d+)-(\d+)\s+\((.*?)\)', line)
                if match:
                    rank = int(match.group(1))
                    pred_ga = int(match.group(2))
                    pred_gb = int(match.group(3))
                    label = match.group(4)
                    top_scores.append({
                        "rank": rank,
                        "pred_ga": pred_ga,
                        "pred_gb": pred_gb
                    })
        
        if len(top_scores) < 2:
            print(f"[!] Warning: Could not parse top 2 scorelines for {team_a} vs {team_b}")
            continue
            
        top1 = top_scores[0]
        top2 = top_scores[1]
        
        # Calculate 1X2 for truth
        if true_score_a > true_score_b: true_1x2 = '1'
        elif true_score_a < true_score_b: true_1x2 = '2'
        else: true_1x2 = 'X'
        
        # Calculate 1X2 for Top 1
        if top1["pred_ga"] > top1["pred_gb"]: top1_1x2 = '1'
        elif top1["pred_ga"] < top1["pred_gb"]: top1_1x2 = '2'
        else: top1_1x2 = 'X'
        
        # Calculate 1X2 for Top 2
        if top2["pred_ga"] > top2["pred_gb"]: top2_1x2 = '1'
        elif top2["pred_ga"] < top2["pred_gb"]: top2_1x2 = '2'
        else: top2_1x2 = 'X'
        
        # Exact Hit Logic
        top1_exact = (top1["pred_ga"] == true_score_a and top1["pred_gb"] == true_score_b)
        top2_exact = (top2["pred_ga"] == true_score_a and top2["pred_gb"] == true_score_b)
        
        if top1_exact: top1_exact_hits += 1
        if top1_exact or top2_exact: top2_exact_hits += 1
        
        # 1X2 Hit Logic
        top1_1x2_hit = (top1_1x2 == true_1x2)
        top2_1x2_hit = (top2_1x2 == true_1x2)
        
        if top1_1x2_hit: top1_1x2_hits += 1
        if top1_1x2_hit or top2_1x2_hit: top2_1x2_hits += 1
        
        print(f"[{date_str}] {team_a} vs {team_b} | True Score: {true_score_a}-{true_score_b} (1X2: {true_1x2})")
        print(f"   - Top 1 Pred: {top1['pred_ga']}-{top1['pred_gb']} (1X2: {top1_1x2}) | Exact: {'✅' if top1_exact else '❌'} | 1X2: {'✅' if top1_1x2_hit else '❌'}")
        print(f"   - Top 2 Pred: {top2['pred_ga']}-{top2['pred_gb']} (1X2: {top2_1x2}) | Exact: {'✅' if top2_exact else '❌'} | 1X2: {'✅' if top2_1x2_hit else '❌'}")
        
    print(f"\n--- {tag} RESULTS ---")
    print(f"Top 1 Exact Scoreline Hit Rate:  {(top1_exact_hits / total) * 100:.1f}% ({top1_exact_hits}/{total})")
    print(f"Top 1 OR 2 Exact Scoreline Hit Rate: {(top2_exact_hits / total) * 100:.1f}% ({top2_exact_hits}/{total})")
    print(f"Top 1 1X2 (Win/Draw/Loss) Hit Rate:  {(top1_1x2_hits / total) * 100:.1f}% ({top1_1x2_hits}/{total})")
    print(f"Top 1 OR 2 1X2 (W/D/L) Hit Rate: {(top2_1x2_hits / total) * 100:.1f}% ({top2_1x2_hits}/{total})")
    
    return total, top1_exact_hits, top2_exact_hits, top1_1x2_hits, top2_1x2_hits

def main():
    try:
        # Phase 1: 2022 World Cup
        set_training_cutoff("2022-11-15")
        force_retrain()
        t1, e1_1, e1_2, r1_1, r1_2 = evaluate_matches(MATCHES_2022_90MIN, "2022 WORLD CUP")
        
        # Phase 2: 2024 Euros
        set_training_cutoff("2024-06-01")
        force_retrain()
        t2, e2_1, e2_2, r2_1, r2_2 = evaluate_matches(MATCHES_2024_90MIN, "2024 EUROS")
        
        # Final Report
        ttot = t1 + t2
        etot_1 = e1_1 + e2_1
        etot_2 = e1_2 + e2_2
        rtot_1 = r1_1 + r2_1
        rtot_2 = r1_2 + r2_2
        
        print(f"\n================ STRICT OOS BACKTEST RESULTS ================")
        print(f"Matches Evaluated: {ttot}")
        print(f"Top 1 Exact Scoreline Hit Rate:  {(etot_1 / ttot) * 100:.1f}% ({etot_1}/{ttot})")
        print(f"Top 1 OR 2 Exact Scoreline Hit Rate: {(etot_2 / ttot) * 100:.1f}% ({etot_2}/{ttot})")
        print(f"Top 1 1X2 (Win/Draw/Loss) Hit Rate:  {(rtot_1 / ttot) * 100:.1f}% ({rtot_1}/{ttot})")
        print(f"Top 1 OR 2 1X2 (W/D/L) Hit Rate: {(rtot_2 / ttot) * 100:.1f}% ({rtot_2}/{ttot})")
        
    except Exception as e:
        print(f"\n[!] An error occurred: {e}")
        traceback.print_exc()
    finally:
        print("\n================ RESTORING SYSTEM ================")
        print("[*] Reverting to most recent data (2030-01-01) and restoring codebase...")
        set_training_cutoff("2030-01-01")
        force_retrain()
        print("[*] System perfectly restored.")

if __name__ == "__main__":
    main()
