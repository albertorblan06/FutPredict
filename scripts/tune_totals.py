import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

sys.path.append("/Users/albertorblan/WorldCupPredict")
from futpredict.data import get_db_connection
from futpredict.xgb_model import build_features
from futpredict.config import XGB_PARAMS

def main():
    conn = get_db_connection()
    df = build_features(conn)
    if len(df) == 0:
        print("Error: No data available for tuning.")
        return
        
    X = df.drop(columns=["home_score", "away_score", "date", "home_team", "away_team", "total_goals", "btts"], errors="ignore")
    # For totals, we predict home_score + away_score
    y = df["home_score"] + df["away_score"]
    
    print(f"Dataset shape: {X.shape}")
    print("Running GridSearchCV for Totals model (tweedie_variance_power)...")
    
    powers = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    best_power = None
    best_mae = float('inf')
    
    for p in powers:
        maes = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            params = XGB_PARAMS.copy()
            params["tweedie_variance_power"] = p
            params["objective"] = "reg:tweedie"
            params["eval_metric"] = "mae"
            
            # Use early stopping for fast tuning
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
            
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            maes.append(mae)
            
        avg_mae = np.mean(maes)
        print(f"tweedie_variance_power={p:.1f} -> MAE: {avg_mae:.4f}")
        
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_power = p
            
    print(f"\nBest tweedie_variance_power for Totals: {best_power} (MAE: {best_mae:.4f})")

if __name__ == "__main__":
    main()
