import re
import os
import subprocess
import traceback
import math

MATCHES_2022 = [
    # date, team A, team B, true_advance, true_et
    ("2022-12-03", "Netherlands", "USA", "Netherlands", False),
    ("2022-12-03", "Argentina", "Australia", "Argentina", False),
    ("2022-12-04", "France", "Poland", "France", False),
    ("2022-12-04", "England", "Senegal", "England", False),
    ("2022-12-05", "Japan", "Croatia", "Croatia", True),
    ("2022-12-05", "Brazil", "South Korea", "Brazil", False),
    ("2022-12-06", "Morocco", "Spain", "Morocco", True),
    ("2022-12-06", "Portugal", "Switzerland", "Portugal", False),
    ("2022-12-09", "Croatia", "Brazil", "Croatia", True),
    ("2022-12-09", "Netherlands", "Argentina", "Argentina", True),
    ("2022-12-10", "Morocco", "Portugal", "Morocco", False),
    ("2022-12-10", "England", "France", "France", False),
    ("2022-12-13", "Argentina", "Croatia", "Argentina", False),
    ("2022-12-14", "France", "Morocco", "France", False),
    ("2022-12-18", "Argentina", "France", "Argentina", True)
]

MATCHES_2024 = [
    ("2024-06-29", "Switzerland", "Italy", "Switzerland", False),
    ("2024-06-29", "Germany", "Denmark", "Germany", False),
    ("2024-06-30", "England", "Slovakia", "England", True),
    ("2024-06-30", "Spain", "Georgia", "Spain", False),
    ("2024-07-01", "France", "Belgium", "France", False),
    ("2024-07-01", "Portugal", "Slovenia", "Portugal", True),
    ("2024-07-02", "Romania", "Netherlands", "Netherlands", False),
    ("2024-07-02", "Austria", "Turkey", "Turkey", False),
    ("2024-07-05", "Spain", "Germany", "Spain", True),
    ("2024-07-05", "Portugal", "France", "France", True),
    ("2024-07-06", "England", "Switzerland", "England", True),
    ("2024-07-06", "Netherlands", "Turkey", "Netherlands", False),
    ("2024-07-09", "Spain", "France", "Spain", False),
    ("2024-07-10", "Netherlands", "England", "England", False),
    ("2024-07-14", "Spain", "England", "Spain", False)
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
    hits = 0
    total = len(matches)
    brier_score = 0.0
    log_loss = 0.0
    
    for date_str, team_a, team_b, true_advance, true_et in matches:
        cmd = ["python3", "predict.py", team_a, team_b, "--knockout", "--date", date_str]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        out = result.stdout
        
        pred_a_pct = 0.0
        pred_b_pct = 0.0
        pred_et_pct = 0.0
        
        for line in out.split("\n"):
            if "Advances" in line and "%" in line:
                if team_a in line:
                    match = re.search(r'([\d\.]+)%', line)
                    if match: pred_a_pct = float(match.group(1))
                elif team_b in line:
                    match = re.search(r'([\d\.]+)%', line)
                    if match: pred_b_pct = float(match.group(1))
            if "Goes to Extra Time" in line and "%" in line:
                match = re.search(r'([\d\.]+)%', line)
                if match: pred_et_pct = float(match.group(1))
                
        pred_advance = team_a if pred_a_pct >= pred_b_pct else team_b
        is_hit = pred_advance == true_advance
        if is_hit: hits += 1
        
        et_true_val = 1.0 if true_et else 0.0
        et_pred_val = pred_et_pct / 100.0
        
        brier = (et_pred_val - et_true_val) ** 2
        brier_score += brier
        
        p_clipped = max(min(et_pred_val, 0.999), 0.001)
        ll = -(et_true_val * math.log(p_clipped) + (1 - et_true_val) * math.log(1 - p_clipped))
        log_loss += ll
        
        print(f"[{date_str}] {team_a} vs {team_b}")
        print(f"   - Pred Advance: {pred_advance} ({max(pred_a_pct, pred_b_pct)}%) | True: {true_advance} {'✅ (HIT)' if is_hit else '❌ (MISS)'}")
        print(f"   - Pred ET Prob: {pred_et_pct}% | True: {'Yes (ET)' if true_et else 'No (90m)'}")
        
    avg_brier = brier_score / total
    avg_ll = log_loss / total
    acc = (hits / total) * 100
    print(f"\n--- {tag} RESULTS ---")
    print(f"To Qualify Accuracy: {acc:.1f}% ({hits}/{total})")
    print(f"Extra Time Brier Score: {avg_brier:.3f}")
    print(f"Extra Time LogLoss: {avg_ll:.3f}")
    
    return hits, total, brier_score, log_loss

def main():
    try:
        # Phase 1: 2022 World Cup
        set_training_cutoff("2022-11-15")
        force_retrain()
        h1, t1, b1, l1 = evaluate_matches(MATCHES_2022, "2022 WORLD CUP")
        
        # Phase 2: 2024 Euros
        set_training_cutoff("2024-06-01")
        force_retrain()
        h2, t2, b2, l2 = evaluate_matches(MATCHES_2024, "2024 EUROS")
        
        # Final Report
        htot = h1 + h2
        ttot = t1 + t2
        print(f"\n================ STRICT OOS BACKTEST RESULTS ================")
        print(f"Matches Evaluated: {ttot}")
        print(f"To Qualify Accuracy: {(htot/ttot)*100:.1f}% ({htot}/{ttot})")
        print(f"Extra Time Brier Score: {(b1+b2)/ttot:.3f}")
        print(f"Extra Time LogLoss: {(l1+l2)/ttot:.3f}")
        
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
