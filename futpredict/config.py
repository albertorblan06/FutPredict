"""
Configuration constants, hyperparameters, and file paths.
No more editing source code to change teams — use the CLI.
"""
import os

# ── PATHS ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "futpredict.db")
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
SHOOTOUTS_CSV = os.path.join(DATA_DIR, "shootouts.csv")
RANKINGS_HIST_CSV = os.path.join(DATA_DIR, "rankings_historical.csv")
RANKINGS_2026_CSV = os.path.join(DATA_DIR, "fifa_rankings_2026.csv")
XGB_TOTALS_MODEL = os.path.join(DATA_DIR, "xgb_totals.json")  # Legacy — kept for cache cleanup
XGB_BTTS_MODEL = os.path.join(DATA_DIR, "xgb_btts.json")
XGB_ET_MODEL = os.path.join(DATA_DIR, "xgb_et.json")
XGB_ADVANCE_MODEL = os.path.join(DATA_DIR, "xgb_advance.json")

# ── OVER/UNDER DIRECT CLASSIFIERS ──
XGB_OVER_THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
XGB_OVER_MODELS = {
    t: os.path.join(DATA_DIR, f"xgb_over_{str(t).replace('.', '_')}.json")
    for t in XGB_OVER_THRESHOLDS
}
XGB_CORNERS_MODEL = os.path.join(DATA_DIR, "xgb_corners.json")
XGB_CARDS_MODEL = os.path.join(DATA_DIR, "xgb_cards.json")
XGB_SOT_MODEL = os.path.join(DATA_DIR, "xgb_sot.json")
XGB_POSSESSION_MODEL = os.path.join(DATA_DIR, "xgb_possession.json")
XGB_META_PATH = os.path.join(DATA_DIR, "xgb_meta.json")
LSTM_MODEL_PATH = os.path.join(DATA_DIR, "lstm_mdn.pt")
LSTM_META_PATH = os.path.join(DATA_DIR, "lstm_meta.json")
LSTM_CONS_MODEL_PATH = os.path.join(DATA_DIR, "lstm_mdn_cons.pt")
LSTM_CONS_META_PATH = os.path.join(DATA_DIR, "lstm_cons_meta.json")

# ── DATA SOURCES ──
RESULTS_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")
SHOOTOUTS_URL = ("https://raw.githubusercontent.com/martj42/"
                 "international_results/master/shootouts.csv")
RANKINGS_URL = ("https://raw.githubusercontent.com/Dato-Futbol/"
                "fifa-ranking/master/ranking_fifa_historical.csv")

# ── ANALYSIS PARAMS ──
NUM_SIMULATIONS = 100_000
LOOKBACK_MATCHES = 30
DECAY_HALF_LIFE_DAYS = 365
H2H_DECAY_HALF_LIFE = 1095
MAX_GOALS = 9          # Score matrix dimension (0..MAX_GOALS-1)

# ── STATISTICAL MODEL ──
DC_TRAIN_YEARS = 4     # Dixon-Coles: train on last N years

# ── XGBOOST COUNT REGRESSION ──
XGB_TRAIN_START = 2018
TRAIN_END_DATE = "2030-01-01"
XGB_PARAMS = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.5,
    "max_depth": 6,
    "learning_rate": 0.015021154371452704,
    "n_estimators": 318,
    "tree_method": "hist",
    "subsample": 0.9064590974891047,
    "colsample_bytree": 0.6471670717778785,
    "min_child_weight": 4,
    "gamma": 0.46856765686432866,
    "alpha": 5.0, # L1 regularization for feature pruning
    "verbosity": 0,
    "random_state": 42,
}

BTTS_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "tree_method": "hist",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "gamma": 0.1,
    "alpha": 0.1,
    "verbosity": 0,
    "random_state": 42,
}

OVER_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "tree_method": "hist",
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "gamma": 0.2,
    "alpha": 0.5,
    "verbosity": 0,
    "random_state": 42,
}

ET_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.015,
    "n_estimators": 250,
    "tree_method": "hist",
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "min_child_weight": 4,
    "gamma": 0.3,
    "scale_pos_weight": 1.0,
    "alpha": 1.0,
    "verbosity": 0,
    "random_state": 42,
}


LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_SEQ_LEN = 15      # Last N matches per team
LSTM_EMBED_DIM = 32     # Team embedding dimension
LSTM_LR = 2e-3
LSTM_EPOCHS = 120
LSTM_BATCH = 64
LSTM_PATIENCE = 15
LSTM_TRAIN_START = 2018
LSTM_DROPOUT = 0.50

# ── FOCAL LOSS (1X2 head) ──
FOCAL_ALPHA = [0.25, 0.50, 0.25]  # Up-weight draws (class 1) by 2×
FOCAL_GAMMA = 2.0                  # Focusing parameter
