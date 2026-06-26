import os
import sys
import sqlite3
import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futpredict.config import DB_PATH
from futpredict.xgb_model import build_features, FEATURE_COLUMNS

def objective(trial, X, y_totals):
    params = {
        "objective": "multi:softprob",
        "num_class": 5,
        "eval_metric": "mlogloss",
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "random_state": 42
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    log_losses = []
    
    for train_idx, val_idx in cv.split(X, y_totals):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_totals[train_idx], y_totals[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_val)
        loss = log_loss(y_val, preds)
        log_losses.append(loss)
        
    return np.mean(log_losses)

if __name__ == "__main__":
    print("Loading data and building features...")
    conn = sqlite3.connect(DB_PATH)
    df = build_features(conn)
    X = df[FEATURE_COLUMNS].values
    y_totals = df["total_goals_class"].values
    conn.close()
    
    print(f"Dataset shape: {X.shape}. Starting Optuna optimization...")
    
    # Enable pruned logging
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y_totals), n_trials=30)
    
    print("\n[OPTIMIZATION COMPLETE]")
    print(f"Best Trial: {study.best_trial.value}")
    print("Best Params:")
    for key, value in study.best_trial.params.items():
        print(f"  '{key}': {value},")
