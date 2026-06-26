import sqlite3
import numpy as np
from futpredict.config import DB_PATH
from futpredict.deep_model import train_lstm_mdn, predict_lstm

def analyze():
    conn = sqlite3.connect(DB_PATH)
    print("Loading model...")
    lstm_model, lstm_goals, lstm_idx, _ = train_lstm_mdn(conn, force=False)
    
    cur = conn.execute("SELECT home_team, away_team FROM matches WHERE date > '2026-01-01' LIMIT 500")
    matches = cur.fetchall()
    
    print(f"Testing {len(matches)} matches...")
    max_probs = []
    
    for h, a in matches:
        _, details = predict_lstm(lstm_model, lstm_goals, lstm_idx, conn, h, a, "neutral")
        if details:
            p_a = details["win_a_pct"] / 100.0
            p_b = details["win_b_pct"] / 100.0
            max_probs.append(max(p_a, p_b))
            
    if not max_probs:
        return
        
    avg_max = np.mean(max_probs) * 100
    over_65 = sum(1 for p in max_probs if p > 0.65)
    pct_over_65 = over_65 / len(max_probs) * 100
    
    print(f"\nCurrent live output (T=1.05) | Avg Max Prob: {avg_max:.1f}% | Matches >65%: {pct_over_65:.1f}%")

if __name__ == "__main__":
    analyze()
