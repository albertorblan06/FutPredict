import argparse
import sqlite3
import os
import sys

# Prevent OpenMP deadlock
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from futpredict.config import DB_PATH
from futpredict.data import get_connection
from futpredict.xgb_advanced import train_advanced_xgb, predict_advanced

def main():
    parser = argparse.ArgumentParser(description="Test Advanced Models against Kaggle 2026 data without data leakage.")
    parser.add_argument("--date", type=str, default="2026-06-01", help="Date constraint to prevent data leakage (YYYY-MM-DD).")
    args = parser.parse_args()
    
    conn = get_connection()
    
    print(f"Loading/Training Advanced Models (Pre-Kaggle 2026 data)...")
    m_cor, m_car, m_sot, m_pos, meta = train_advanced_xgb(conn, force=False)
    
    # Fetch test matches from Kaggle dataset (2026 World Cup)
    # The Kaggle matches are identifiable because their date is >= 2026-06-11
    query = f"""
        SELECT m.date, m.home_team, m.away_team, 
               a.home_possession, a.home_corners, a.away_corners,
               a.home_cards, a.away_cards, a.home_sot, a.away_sot
        FROM matches m
        JOIN advanced_stats a ON m.date = a.match_date AND m.home_team = a.home_team AND m.away_team = a.away_team
        WHERE m.date >= '2026-06-11'
        ORDER BY m.date
    """
    df = pd.read_sql_query(query, conn)
    print(f"Found {len(df)} test matches from 2026 World Cup.")
    
    if len(df) == 0:
        print("No test matches found.")
        return
        
    mse_pos = 0
    err_cor = 0
    err_car = 0
    err_sot = 0
    
    count = 0
    for _, row in df.iterrows():
        team_a = row["home_team"]
        team_b = row["away_team"]
        
        # Predict using constraints
        preds = predict_advanced(m_cor, m_car, m_sot, m_pos, team_a, team_b, args.date, conn)
        if not preds:
            continue
            
        actual_pos = row["home_possession"]
        actual_cor = row["home_corners"] + row["away_corners"]
        actual_car = row["home_cards"] + row["away_cards"]
        actual_sot = row["home_sot"] + row["away_sot"]
        
        mse_pos += (preds["home_possession"] - actual_pos)**2
        err_cor += abs(preds["expected_corners"] - actual_cor)
        err_car += abs(preds["expected_cards"] - actual_car)
        err_sot += abs(preds["expected_sot"] - actual_sot)
        count += 1
        
    if count > 0:
        print(f"\nResults over {count} matches:")
        print(f"Possession RMSE:     {math.sqrt(mse_pos/count):.1f}%")
        print(f"Corners MAE:         {err_cor/count:.2f}")
        print(f"Cards MAE:           {err_car/count:.2f}")
        print(f"Shots on Target MAE: {err_sot/count:.2f}")
        
if __name__ == "__main__":
    import pandas as pd
    import math
    main()
