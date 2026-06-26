"""
Player-level XGBoost models to predict anytime goalscorers.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from futpredict.config import XGB_PARAMS, TRAIN_END_DATE

PLAYER_MODEL_PATH = "cache/player_model.json"
PLAYER_META_PATH = "cache/player_meta.json"

FEATURE_COLUMNS = [
    "rolling_xg", "rolling_xa", "rolling_shots",
    "rolling_goals", "rolling_assists",
    "rolling_off_contrib", "rolling_def_contrib", "rolling_rating"
]

def build_player_features(conn):
    """
    Builds the player feature matrix from player_stats.
    Uses rolling averages over the last 5 matches for each player.
    """
    query = f"SELECT * FROM player_stats WHERE match_date < '{TRAIN_END_DATE}' ORDER BY match_date ASC"
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        return pd.DataFrame()

    if len(df) == 0:
        return pd.DataFrame()

    # Sort by player and date
    df = df.sort_values(by=['player_id', 'match_date'])

    # Calculate rolling features
    df['rolling_xg'] = df.groupby('player_id')['expected_goals_xg'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['rolling_xa'] = df.groupby('player_id')['expected_assists_xa'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['rolling_shots'] = df.groupby('player_id')['shots_on_target'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['rolling_goals'] = df.groupby('player_id')['goals'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['rolling_assists'] = df.groupby('player_id')['assists'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['rolling_off_contrib'] = df.groupby('player_id')['offensive_contribution'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['rolling_def_contrib'] = df.groupby('player_id')['defensive_contribution'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['rolling_rating'] = df.groupby('player_id')['player_rating'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # Drop NA (first match for each player)
    df = df.dropna(subset=FEATURE_COLUMNS)

    # Target: Scored at least 1 goal
    df['scored_anytime'] = (df['goals'] > 0).astype(int)

    return df

def train_player_model(conn, force=False):
    """Train the XGBoost goalscorer model."""
    if not force and os.path.exists(PLAYER_MODEL_PATH):
        return _load_cached()

    print("   ⬇  Building features for Goalscorer prediction...")
    df = build_player_features(conn)
    
    if len(df) < 100:
        print("   ✗  Not enough player data.")
        return None, None

    print(f"   ✓  {len(df):,} player samples built.")
    print("   ⚙  Training XGBoost Goalscorer model...")

    X = df[FEATURE_COLUMNS].values
    y = df['scored_anytime'].values

    params = {k: v for k, v in XGB_PARAMS.items() if k not in ["objective", "tweedie_variance_power"]}
    model = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss"
    )
    model.fit(X, y)

    # Save
    os.makedirs("cache", exist_ok=True)
    model.save_model(PLAYER_MODEL_PATH)
    
    meta = {"features": FEATURE_COLUMNS, "samples": len(df)}
    with open(PLAYER_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("   ✓  Goalscorer model cached.")
    return model, meta

def _load_cached():
    model = xgb.XGBClassifier()
    try:
        model.load_model(PLAYER_MODEL_PATH)
        with open(PLAYER_META_PATH, "r") as f:
            meta = json.load(f)
        return model, meta
    except Exception:
        return None, None

def get_team_top_11(conn, team_name):
    """
    Heuristic to get the likely starting 11 for a team,
    based on most total minutes played historically.
    """
    query = """
        SELECT player_id, player_name, position, SUM(minutes_played) as total_mins
        FROM player_stats
        WHERE team = ?
        GROUP BY player_id
        ORDER BY total_mins DESC
        LIMIT 11
    """
    return pd.read_sql_query(query, conn, params=(team_name,))

def get_player_current_features(conn, player_id):
    """Get the latest rolling stats for a specific player."""
    query = """
        SELECT * FROM player_stats 
        WHERE player_id = ? 
        ORDER BY match_date DESC 
        LIMIT 5
    """
    df = pd.read_sql_query(query, conn, params=(player_id,))
    if len(df) == 0:
        return None
        
    # We aggregate the last 5 matches
    features = {
        "rolling_xg": df['expected_goals_xg'].mean(),
        "rolling_xa": df['expected_assists_xa'].mean(),
        "rolling_shots": df['shots_on_target'].mean(),
        "rolling_goals": df['goals'].mean(),
        "rolling_assists": df['assists'].mean(),
        "rolling_off_contrib": df['offensive_contribution'].mean(),
        "rolling_def_contrib": df['defensive_contribution'].mean(),
        "rolling_rating": df['player_rating'].mean(),
    }
    return features

def predict_goalscorers(model, conn, team_a, team_b):
    """Predict goalscoring probabilities for the top 11 of each team."""
    if model is None:
        return []

    squad_a = get_team_top_11(conn, team_a)
    squad_b = get_team_top_11(conn, team_b)

    squad_a['team'] = team_a
    squad_b['team'] = team_b
    all_players = pd.concat([squad_a, squad_b])
    
    results = []
    
    for _, row in all_players.iterrows():
        feats_dict = get_player_current_features(conn, row['player_id'])
        if not feats_dict:
            continue
            
        vec = np.array([[feats_dict[c] for c in FEATURE_COLUMNS]])
        prob = float(model.predict_proba(vec)[0, 1])
        
        results.append({
            "player_name": row['player_name'],
            "team": row['team'],
            "position": row['position'],
            "prob": prob * 100,
            "xg_avg": feats_dict["rolling_xg"]
        })
        
    # Sort by probability descending
    results.sort(key=lambda x: x["prob"], reverse=True)
    return results
