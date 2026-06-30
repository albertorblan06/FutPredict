"""
XGBoost Knockout Advance model.
Trains on matches with a definitive winner (including penalty shootouts)
to predict P(Home Advances) vs P(Away Advances).
"""
import os
import xgboost as xgb
import pandas as pd
import numpy as np
from .config import XGB_ADVANCE_MODEL, XGB_PARAMS
from .xgb_model import build_features, FEATURE_COLUMNS

def train_advance_xgb(conn, force=False):
    """Train the XGBoost model to predict which team advances in a tie."""
    if not force and os.path.exists(XGB_ADVANCE_MODEL):
        model = xgb.XGBClassifier()
        model.load_model(XGB_ADVANCE_MODEL)
        return model

    print("   ⬇  Building features for Knockout Advance model...")
    df = build_features(conn)
    
    # Filter for matches that have a definitive advance_target
    df_adv = df[df["advance_target"].notnull()].copy()
    
    if len(df_adv) < 100:
        print("   ⚠  Not enough knockout data to train advance model.")
        return None
        
    X = df_adv[FEATURE_COLUMNS]
    y = df_adv["advance_target"].astype(int)
    
    print(f"   ⚙  Training XGBoost Advance model ({len(X)} samples)...")
    
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)
    
    os.makedirs(os.path.dirname(XGB_ADVANCE_MODEL), exist_ok=True)
    model.save_model(XGB_ADVANCE_MODEL)
    print("   ✓  Advance model trained and cached.")
    return model

def predict_advance(model, feature_vec):
    """Predict advance probability given a feature vector (numpy array)."""
    if not model:
        return 0.5
    X = pd.DataFrame(feature_vec, columns=FEATURE_COLUMNS)
    probs = model.predict_proba(X)[0]
    return float(probs[1]) # Probability of Class 1 (Home Advances)
