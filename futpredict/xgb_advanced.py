"""
Advanced XGBoost models for predicting advanced stats.
Predicts Corners, Cards, Shots on Target, and Possession.
"""
import os
import json
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from .config import (XGB_PARAMS, XGB_CORNERS_MODEL, XGB_CARDS_MODEL, 
                     XGB_SOT_MODEL, XGB_POSSESSION_MODEL, DATA_DIR, TRAIN_END_DATE)


ADVANCED_META_PATH = os.path.join(DATA_DIR, "xgb_advanced_meta.json")

def _get_time_bins(bins_json, default_val=0):
    try:
        bins = json.loads(bins_json)
        return [float(b) for b in bins]
    except:
        return [default_val] * 6

def build_advanced_features(conn):
    """Build feature matrix for Advanced Stats using time-binning."""
    query = f"""
        SELECT m.date, m.home_team, m.away_team, m.home_score, m.away_score,
               a.home_possession, a.away_possession, a.home_corners, a.away_corners,
               a.home_cards, a.away_cards, a.home_sot, a.away_sot,
               a.home_possession_bins, a.away_possession_bins,
               a.home_corners_bins, a.away_corners_bins,
               a.home_cards_bins, a.away_cards_bins,
               a.home_sot_bins, a.away_sot_bins
        FROM matches m
        JOIN advanced_stats a ON m.date = a.match_date AND m.home_team = a.home_team AND m.away_team = a.away_team
        WHERE m.date >= '2018-01-01' AND m.date < '{TRAIN_END_DATE}'
        ORDER BY m.date
    """
    df = pd.read_sql_query(query, conn)
    
    rows = []
    
    # We need a rolling window. 
    # For predicting match at index `idx`, we look at previous matches of home and away.
    for idx in range(len(df)):
        row = df.iloc[idx]
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]
        
        # Get last 5 matches for home team
        past_home = df[((df["home_team"] == home) | (df["away_team"] == home)) & (df["date"] < date)].tail(5)
        # Get last 5 matches for away team
        past_away = df[((df["home_team"] == away) | (df["away_team"] == away)) & (df["date"] < date)].tail(5)
        
        if len(past_home) == 0 or len(past_away) == 0:
            continue
            
        # Aggregate time bins for home team
        h_pos_bins, h_cor_bins, h_car_bins, h_sot_bins = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
        for _, p in past_home.iterrows():
            is_home = (p["home_team"] == home)
            h_pos_bins += _get_time_bins(p["home_possession_bins"] if is_home else p["away_possession_bins"], 50)
            h_cor_bins += _get_time_bins(p["home_corners_bins"] if is_home else p["away_corners_bins"], 0)
            h_car_bins += _get_time_bins(p["home_cards_bins"] if is_home else p["away_cards_bins"], 0)
            h_sot_bins += _get_time_bins(p["home_sot_bins"] if is_home else p["away_sot_bins"], 0)
            
        h_pos_bins /= len(past_home)
        h_cor_bins /= len(past_home)
        h_car_bins /= len(past_home)
        h_sot_bins /= len(past_home)
        
        # Aggregate time bins for away team
        a_pos_bins, a_cor_bins, a_car_bins, a_sot_bins = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
        for _, p in past_away.iterrows():
            is_home = (p["home_team"] == away)
            a_pos_bins += _get_time_bins(p["home_possession_bins"] if is_home else p["away_possession_bins"], 50)
            a_cor_bins += _get_time_bins(p["home_corners_bins"] if is_home else p["away_corners_bins"], 0)
            a_car_bins += _get_time_bins(p["home_cards_bins"] if is_home else p["away_cards_bins"], 0)
            a_sot_bins += _get_time_bins(p["home_sot_bins"] if is_home else p["away_sot_bins"], 0)
            
        a_pos_bins /= len(past_away)
        a_cor_bins /= len(past_away)
        a_car_bins /= len(past_away)
        a_sot_bins /= len(past_away)
        
        feat = {}
        for i in range(6):
            feat[f"h_pos_{i}"] = h_pos_bins[i]
            feat[f"h_cor_{i}"] = h_cor_bins[i]
            feat[f"h_car_{i}"] = h_car_bins[i]
            feat[f"h_sot_{i}"] = h_sot_bins[i]
            
            feat[f"a_pos_{i}"] = a_pos_bins[i]
            feat[f"a_cor_{i}"] = a_cor_bins[i]
            feat[f"a_car_{i}"] = a_car_bins[i]
            feat[f"a_sot_{i}"] = a_sot_bins[i]
            
        # Targets
        # Only add valid targets (not null)
        if pd.notna(row["home_corners"]) and pd.notna(row["away_corners"]):
            feat["target_corners"] = row["home_corners"] + row["away_corners"]
            feat["target_cards"] = row["home_cards"] + row["away_cards"]
            feat["target_sot"] = row["home_sot"] + row["away_sot"]
            feat["target_possession"] = row["home_possession"]
            rows.append(feat)
            
    if not rows:
        return pd.DataFrame()
        
    return pd.DataFrame(rows)


def train_advanced_xgb(conn, force=False):
    """Train XGBoost models for Corners, Cards, SOT, Possession."""
    
    if not force and os.path.exists(XGB_CORNERS_MODEL):
        try:
            m_cor = xgb.XGBRegressor()
            m_cor.load_model(XGB_CORNERS_MODEL)
            m_car = xgb.XGBRegressor()
            m_car.load_model(XGB_CARDS_MODEL)
            m_sot = xgb.XGBRegressor()
            m_sot.load_model(XGB_SOT_MODEL)
            m_pos = xgb.XGBRegressor()
            m_pos.load_model(XGB_POSSESSION_MODEL)
            with open(ADVANCED_META_PATH, "r") as f:
                meta = json.load(f)
            print("   ✓  Loaded cached Advanced Models")
            return m_cor, m_car, m_sot, m_pos, meta
        except:
            print("   ⚠  Failed to load cached advanced models, retraining...")

    print("   ⬇  Building Advanced Features (Time Bins)...")
    df = build_advanced_features(conn)
    
    if len(df) < 20:
        print(f"   ✗  Not enough advanced data ({len(df)} samples). Models won't be trained.")
        return None, None, None, None, None
        
    print(f"   ✓  {len(df)} advanced samples built")
    
    feature_cols = [c for c in df.columns if c.startswith("h_") or c.startswith("a_")]
    X = df[feature_cols].values
    
    y_cor = df["target_corners"].values
    y_car = df["target_cards"].values
    y_sot = df["target_sot"].values
    y_pos = df["target_possession"].values
    
    split = int(len(df) * 0.8)
    if split == 0:
        split = len(df)
    X_train, X_val = X[:split], X[split:] if split < len(X) else X
    
    print("   ⚙  Training Advanced Regressors...")
    
    # Corners (Poisson count regression)
    params_count = XGB_PARAMS.copy()
    params_count["objective"] = "count:poisson"
    if "tweedie_variance_power" in params_count:
        del params_count["tweedie_variance_power"]
        
    m_cor = xgb.XGBRegressor(**params_count)
    m_cor.fit(X_train, y_cor[:split], eval_set=[(X_val, y_cor[split:])] if len(X_val) > 0 else None, verbose=False)
    
    m_car = xgb.XGBRegressor(**params_count)
    m_car.fit(X_train, y_car[:split], eval_set=[(X_val, y_car[split:])] if len(X_val) > 0 else None, verbose=False)
    
    m_sot = xgb.XGBRegressor(**params_count)
    m_sot.fit(X_train, y_sot[:split], eval_set=[(X_val, y_sot[split:])] if len(X_val) > 0 else None, verbose=False)
    
    # Possession (Linear Regression)
    params_lin = XGB_PARAMS.copy()
    params_lin["objective"] = "reg:squarederror"
    if "tweedie_variance_power" in params_lin:
        del params_lin["tweedie_variance_power"]
        
    m_pos = xgb.XGBRegressor(**params_lin)
    m_pos.fit(X_train, y_pos[:split], eval_set=[(X_val, y_pos[split:])] if len(X_val) > 0 else None, verbose=False)
    
    m_cor.save_model(XGB_CORNERS_MODEL)
    m_car.save_model(XGB_CARDS_MODEL)
    m_sot.save_model(XGB_SOT_MODEL)
    m_pos.save_model(XGB_POSSESSION_MODEL)
    
    meta = {"n_train": len(X_train), "feature_cols": feature_cols}
    with open(ADVANCED_META_PATH, "w") as f:
        json.dump(meta, f)
        
    print("   ✓  Advanced Models trained and cached!")
    return m_cor, m_car, m_sot, m_pos, meta


def predict_advanced(m_cor, m_car, m_sot, m_pos, team_a, team_b, match_date, conn):
    """Predict advanced stats for a match."""
    if m_cor is None:
        return None
        
    # We need to construct the feature vector for these teams
    # Same logic as build_advanced_features but for just these two teams right before match_date
    query = f"""
        SELECT m.date, m.home_team, m.away_team, 
               a.home_possession_bins, a.away_possession_bins,
               a.home_corners_bins, a.away_corners_bins,
               a.home_cards_bins, a.away_cards_bins,
               a.home_sot_bins, a.away_sot_bins
        FROM matches m
        JOIN advanced_stats a ON m.date = a.match_date AND m.home_team = a.home_team AND m.away_team = a.away_team
        WHERE (m.home_team = '{team_a}' OR m.away_team = '{team_a}' OR m.home_team = '{team_b}' OR m.away_team = '{team_b}')
          AND m.date < '{match_date}'
        ORDER BY m.date
    """
    df = pd.read_sql_query(query, conn)
    past_home = df[(df["home_team"] == team_a) | (df["away_team"] == team_a)].tail(5)
    past_away = df[(df["home_team"] == team_b) | (df["away_team"] == team_b)].tail(5)
    
    if len(past_home) == 0 or len(past_away) == 0:
        return None
        
    # Aggregate home
    h_pos_bins, h_cor_bins, h_car_bins, h_sot_bins = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
    for _, p in past_home.iterrows():
        is_home = (p["home_team"] == team_a)
        h_pos_bins += _get_time_bins(p["home_possession_bins"] if is_home else p["away_possession_bins"], 50)
        h_cor_bins += _get_time_bins(p["home_corners_bins"] if is_home else p["away_corners_bins"], 0)
        h_car_bins += _get_time_bins(p["home_cards_bins"] if is_home else p["away_cards_bins"], 0)
        h_sot_bins += _get_time_bins(p["home_sot_bins"] if is_home else p["away_sot_bins"], 0)
        
    h_pos_bins /= len(past_home)
    h_cor_bins /= len(past_home)
    h_car_bins /= len(past_home)
    h_sot_bins /= len(past_home)
    
    # Aggregate away
    a_pos_bins, a_cor_bins, a_car_bins, a_sot_bins = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
    for _, p in past_away.iterrows():
        is_home = (p["home_team"] == team_b)
        a_pos_bins += _get_time_bins(p["home_possession_bins"] if is_home else p["away_possession_bins"], 50)
        a_cor_bins += _get_time_bins(p["home_corners_bins"] if is_home else p["away_corners_bins"], 0)
        a_car_bins += _get_time_bins(p["home_cards_bins"] if is_home else p["away_cards_bins"], 0)
        a_sot_bins += _get_time_bins(p["home_sot_bins"] if is_home else p["away_sot_bins"], 0)
        
    a_pos_bins /= len(past_away)
    a_cor_bins /= len(past_away)
    a_car_bins /= len(past_away)
    a_sot_bins /= len(past_away)
    
    feat = []
    for i in range(6):
        feat.extend([h_pos_bins[i], h_cor_bins[i], h_car_bins[i], h_sot_bins[i]])
        feat.extend([a_pos_bins[i], a_cor_bins[i], a_car_bins[i], a_sot_bins[i]])
        
    X = np.array([feat])
    
    return {
        "expected_corners": float(m_cor.predict(X)[0]),
        "expected_cards": float(m_car.predict(X)[0]),
        "expected_sot": float(m_sot.predict(X)[0]),
        "home_possession": float(m_pos.predict(X)[0])
    }
