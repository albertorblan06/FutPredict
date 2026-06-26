"""
LSTM + Mixture Density Network for football score prediction.

Architecture:
  - Siamese LSTM encoder (shared weights for both teams)
  - Team embeddings (16d per team)
  - MDN output head: predicts (μ_h, μ_a, α_h, α_a, ρ) for a
    bivariate Negative Binomial distribution
  - Loss: negative log-likelihood of actual scores under predicted distribution

This is Phase 2 — treats team history as a pure time series.
"""
from futpredict.config import LSTM_DROPOUT
import os
import json
import datetime
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import nbinom
from .config import (LSTM_HIDDEN, LSTM_LAYERS, LSTM_SEQ_LEN, LSTM_EMBED_DIM,
                     LSTM_LR, LSTM_EPOCHS, LSTM_BATCH, LSTM_PATIENCE,
                     LSTM_TRAIN_START, LSTM_MODEL_PATH, LSTM_META_PATH,
                     MAX_GOALS, TRAIN_END_DATE)
from .rankings import get_fifa_rank, get_fifa_points, get_median_fifa
from .analysis import get_tournament_weight
from .statistical_model import build_score_matrix

# ═══════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class FootballClassifier(nn.Module):
    """
    Siamese LSTM for 3-class match outcome classification.

    Input: two sequences (team A history, team B history) + context features
    Output: logits for [Home Win, Draw, Away Win]
    """

    def __init__(self, seq_feature_dim, context_dim, n_teams,
                 hidden_size=128, num_layers=2, embed_dim=16, dropout=0.3):
        super().__init__()

        # Team embeddings
        self.team_embed = nn.Embedding(n_teams + 1, embed_dim, padding_idx=0)

        # Shared LSTM encoder (Siamese — same weights for both teams)
        self.lstm = nn.LSTM(
            input_size=seq_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # MLP head: concat team states + context → classification logits
        mlp_input = hidden_size * 2 + embed_dim * 2 + context_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output head for 3 classes
        self.outcome_head = nn.Linear(64, 3)

    def forward(self, seq_a, seq_b, team_a_idx, team_b_idx, context):
        # Encode both teams through shared LSTM
        _, (h_a, _) = self.lstm(seq_a)
        _, (h_b, _) = self.lstm(seq_b)

        state_a = h_a[-1]  # (batch, hidden)
        state_b = h_b[-1]

        # Team embeddings
        emb_a = self.team_embed(team_a_idx)
        emb_b = self.team_embed(team_b_idx)

        # Concatenate everything
        combined = torch.cat([state_a, state_b, emb_a, emb_b, context], dim=1)
        h = self.mlp(combined)

        # Output logits
        logits = self.outcome_head(h)
        return logits

class GoalCountNet(nn.Module):
    """
    Siamese LSTM for 10-class exact goal prediction for both home and away.
    Predicts 0-9 goals as discrete classes.
    """
    def __init__(self, seq_feature_dim, context_dim, n_teams,
                 hidden_size=128, num_layers=2, embed_dim=16, dropout=0.3):
        super().__init__()

        self.team_embed = nn.Embedding(n_teams + 1, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=seq_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        mlp_input = hidden_size * 2 + embed_dim * 2 + context_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads for 10 classes (0 to 9 goals)
        self.home_goals_head = nn.Linear(64, 10)
        self.away_goals_head = nn.Linear(64, 10)

    def forward(self, seq_a, seq_b, team_a_idx, team_b_idx, context):
        _, (h_a, _) = self.lstm(seq_a)
        _, (h_b, _) = self.lstm(seq_b)

        state_a = h_a[-1]
        state_b = h_b[-1]

        emb_a = self.team_embed(team_a_idx)
        emb_b = self.team_embed(team_b_idx)

        combined = torch.cat([state_a, state_b, emb_a, emb_b, context], dim=1)
        h = self.mlp(combined)

        home_logits = self.home_goals_head(h)
        away_logits = self.away_goals_head(h)
        return home_logits, away_logits

# ═══════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════


SEQ_FEATURES = [
    "gf", "ga", "opp_rank", "tournament_w", "is_home",
    "days_gap", "result_pts", "cumulative_form",
    "possession", "corners", "cards", "sot",
]


def _build_team_sequences(conn, seq_len=None):
    """
    Build per-team match sequences for LSTM input.
    Returns: dict[team_name] → list of feature vectors (chronological)
    """
    seq_len = seq_len or LSTM_SEQ_LEN

    cur = conn.execute(f"""
        SELECT m.date, m.home_team, m.away_team, m.home_score, m.away_score,
               m.tournament, m.neutral,
               a.home_possession, a.away_possession, a.home_corners, a.away_corners,
               a.home_cards, a.away_cards, a.home_sot, a.away_sot
        FROM matches m
        LEFT JOIN advanced_stats a ON m.date = a.match_date AND m.home_team = a.home_team AND m.away_team = a.away_team
        WHERE m.date >= '{LSTM_TRAIN_START}-01-01' AND m.date < '{TRAIN_END_DATE}'
        ORDER BY m.date
    """)
    rows = cur.fetchall()

    team_seqs = {}  # team → list of feature dicts
    last_date = {}  # team → last match date

    for date_str, home, away, hs, as_, tourn, neutral, h_pos, a_pos, h_cor, a_cor, h_car, a_car, h_sot, a_sot in rows:
        hs, as_ = int(hs), int(as_)
        tourn_w = get_tournament_weight(tourn)

        for team, is_home in [(home, True), (away, False)]:
            gf = hs if is_home else as_
            ga = as_ if is_home else hs
            opp = away if is_home else home
            opp_rank = get_fifa_rank(opp) or 100
            
            pos = float(h_pos if is_home else a_pos) if h_pos is not None else 50.0
            cor = float(h_cor if is_home else a_cor) if h_cor is not None else 0.0
            car = float(h_car if is_home else a_car) if h_car is not None else 0.0
            sot = float(h_sot if is_home else a_sot) if h_sot is not None else 0.0

            # Days since last match
            if team in last_date:
                try:
                    d1 = datetime.date.fromisoformat(last_date[team])
                    d2 = datetime.date.fromisoformat(date_str)
                    days_gap = min((d2 - d1).days, 180)
                except Exception:
                    days_gap = 30
            else:
                days_gap = 60

            # Result points
            result_pts = 1.0 if gf > ga else (0.5 if gf == ga else 0.0)

            # Cumulative form (rolling 5 match average)
            if team not in team_seqs:
                team_seqs[team] = []
            recent = team_seqs[team][-5:]
            cum_form = np.mean([r["result_pts"] for r in recent]) if recent else 0.5

            feature = {
                "gf": gf, "ga": ga,
                "opp_rank": min(opp_rank, 211) / 211.0,  # normalize
                "tournament_w": tourn_w,
                "is_home": 1.0 if is_home else 0.0,
                "days_gap": min(days_gap, 180) / 180.0,
                "result_pts": result_pts,
                "cumulative_form": cum_form,
                "possession": pos / 100.0,
                "corners": min(cor, 20.0) / 20.0,
                "cards": min(car, 10.0) / 10.0,
                "sot": min(sot, 20.0) / 20.0,
                "date": date_str,
                "actual_gf": gf,
                "actual_ga": ga,
            }
            team_seqs.setdefault(team, []).append(feature)
            last_date[team] = date_str

    return team_seqs


def _build_training_data(conn, seq_len=None):
    """
    Build training dataset: for each match, extract sequences for both teams.

    Returns:
        list of dicts with keys: seq_a, seq_b, team_a, team_b,
        context, home_goals, away_goals, date
    """
    seq_len = seq_len or LSTM_SEQ_LEN
    team_seqs = _build_team_sequences(conn, seq_len)

    # Build team index
    all_teams = sorted(team_seqs.keys())
    team_idx = {t: i + 1 for i, t in enumerate(all_teams)}  # 0 = padding

    # Build match dataset
    cur = conn.execute(f"""
        SELECT date, home_team, away_team, home_score, away_score,
               tournament, neutral
        FROM matches
        WHERE date >= '{LSTM_TRAIN_START + 2}-01-01'
        ORDER BY date
    """)

    samples = []
    for date_str, home, away, hs, as_, tourn, neutral in cur.fetchall():
        # Get team sequences BEFORE this match
        seq_a_full = team_seqs.get(home, [])
        seq_b_full = team_seqs.get(away, [])

        # Filter to before this match
        seq_a = [s for s in seq_a_full if s["date"] < date_str][-seq_len:]
        seq_b = [s for s in seq_b_full if s["date"] < date_str][-seq_len:]

        if len(seq_a) < 3 or len(seq_b) < 3:
            continue

        # Convert to feature vectors
        feat_keys = ["gf", "ga", "opp_rank", "tournament_w", "is_home",
                      "days_gap", "result_pts", "cumulative_form",
                      "possession", "corners", "cards", "sot"]

        vec_a = [[s[k] for k in feat_keys] for s in seq_a]
        vec_b = [[s[k] for k in feat_keys] for s in seq_b]

        # Pad sequences
        while len(vec_a) < seq_len:
            vec_a.insert(0, [0.0] * len(feat_keys))
        while len(vec_b) < seq_len:
            vec_b.insert(0, [0.0] * len(feat_keys))

        # Context features
        fifa_a, _ = get_fifa_points(home)
        fifa_b, _ = get_fifa_points(away)
        median = get_median_fifa()
        rank_a = get_fifa_rank(home) or 100
        rank_b = get_fifa_rank(away) or 100
        is_neutral = 1.0 if str(neutral).upper() == "TRUE" else 0.0

        context = [
            (fifa_a or median) / 2000.0,  # normalize
            (fifa_b or median) / 2000.0,
            ((fifa_a or median) - (fifa_b or median)) / 500.0,
            rank_a / 211.0,
            rank_b / 211.0,
            is_neutral,
        ]

        samples.append({
            "seq_a": vec_a,
            "seq_b": vec_b,
            "team_a_idx": team_idx.get(home, 0),
            "team_b_idx": team_idx.get(away, 0),
            "context": context,
            "home_goals": int(hs),
            "away_goals": int(as_),
            "date": date_str,
        })

    return samples, team_idx


class MatchDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "seq_a": torch.FloatTensor(s["seq_a"]),
            "seq_b": torch.FloatTensor(s["seq_b"]),
            "team_a_idx": torch.LongTensor([s["team_a_idx"]]),
            "team_b_idx": torch.LongTensor([s["team_b_idx"]]),
            "context": torch.FloatTensor(s["context"]),
            "home_goals": torch.FloatTensor([s["home_goals"]]),
            "away_goals": torch.FloatTensor([s["away_goals"]]),
        }


# ═══════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

def train_lstm_mdn(conn, force=False):
    """
    Train or load cached LSTM+MDN model.

    Returns: (model, team_idx, meta) or (None, None, None) on failure
    """
    # Ensure deterministic training
    torch.manual_seed(42)
    np.random.seed(42)

    if not force and os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_META_PATH):
        try:
            with open(LSTM_META_PATH, "r") as f:
                meta = json.load(f)
            team_idx = meta.get("team_idx", {})
            n_teams = meta.get("n_teams", 500)
            print("   [DEBUG] Initializing FootballClassifier and GoalCountNet...")
            model = FootballClassifier(
                seq_feature_dim=len(SEQ_FEATURES),
                context_dim=6,
                n_teams=n_teams,
                hidden_size=LSTM_HIDDEN,
                num_layers=LSTM_LAYERS,
                embed_dim=LSTM_EMBED_DIM,
                dropout=LSTM_DROPOUT,
            )
            model_goals = GoalCountNet(
                seq_feature_dim=len(SEQ_FEATURES),
                context_dim=6,
                n_teams=n_teams,
                hidden_size=LSTM_HIDDEN,
                num_layers=LSTM_LAYERS,
                embed_dim=LSTM_EMBED_DIM,
                dropout=LSTM_DROPOUT,
            )
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            model.to(device)
            model_goals.to(device)
            print("   [DEBUG] Loading state dict from disk...")
            model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
            
            goals_path = LSTM_MODEL_PATH.replace(".pt", "_goals.pt")
            if os.path.exists(goals_path):
                model_goals.load_state_dict(torch.load(goals_path, map_location=device))
                
            print("   [DEBUG] State dict loaded. Setting eval mode...")
            model.eval()
            model_goals.eval()
            print(f"   ✓  Loaded cached LSTM+MDN ({meta.get('n_train', '?')} "
                  f"training samples)\n"
                  f"      Val NLL: {meta.get('val_nll', 0):.4f}")
            return model, model_goals, team_idx, meta
        except Exception as e:
            print(f"   ⚠  Cache load failed ({e}), retraining...")

    print("   ⬇  Building LSTM sequences from historical data...")
    try:
        samples, team_idx = _build_training_data(conn)
    except Exception as e:
        print(f"   ✗  Sequence building failed: {e}")
        return None, None, None

    if len(samples) < 2000:
        print(f"   ✗  Not enough data ({len(samples)} samples)")
        return None, None, None

    n_teams = max(team_idx.values()) + 1
    print(f"   ✓  {len(samples):,} match sequences built "
          f"({n_teams} teams, {LSTM_SEQ_LEN} steps)")

    # Walk-forward split
    split = int(len(samples) * 0.85)
    train_data = MatchDataset(samples[:split])
    val_data = MatchDataset(samples[split:])

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=LSTM_BATCH, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=LSTM_BATCH, shuffle=False)

    # Define device for hardware acceleration (Apple Silicon MPS or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = FootballClassifier(
        seq_feature_dim=len(SEQ_FEATURES),
        context_dim=6,
        n_teams=n_teams,
        hidden_size=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
        embed_dim=LSTM_EMBED_DIM,
        dropout=LSTM_DROPOUT,
    )
    model_goals = GoalCountNet(
        seq_feature_dim=len(SEQ_FEATURES),
        context_dim=6,
        n_teams=n_teams,
        hidden_size=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
        embed_dim=LSTM_EMBED_DIM,
        dropout=LSTM_DROPOUT,
    )
    model.to(device)
    model_goals.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LSTM_LR,
                                  weight_decay=1e-4)
    optimizer_goals = torch.optim.AdamW(model_goals.parameters(), lr=LSTM_LR,
                                        weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    scheduler_goals = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_goals, T_0=10, T_mult=2, eta_min=1e-5)

    print(f"   ⚙  Training LSTM+MDN & GoalCountNet ({LSTM_EPOCHS} epochs, "
          f"batch={LSTM_BATCH})...")

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_state_goals = None

    for epoch in range(LSTM_EPOCHS):
        # Training
        model.train()
        model_goals.train()
        train_loss = 0.0
        train_loss_goals = 0.0
        n_batches = 0
        for batch in train_loader:
            seq_a = batch["seq_a"].to(device)
            seq_b = batch["seq_b"].to(device)
            t_a_idx = batch["team_a_idx"].squeeze(1).to(device)
            t_b_idx = batch["team_b_idx"].squeeze(1).to(device)
            ctx = batch["context"].to(device)
            h_g = batch["home_goals"].squeeze(1).to(device)
            a_g = batch["away_goals"].squeeze(1).to(device)

            outcomes = torch.where(h_g > a_g, 0, torch.where(h_g == a_g, 1, 2)).long()
            
            # Clamp goals to max 9 for discrete classification
            h_g_class = torch.clamp(h_g.long(), 0, 9)
            a_g_class = torch.clamp(a_g.long(), 0, 9)

            logits = model(seq_a, seq_b, t_a_idx, t_b_idx, ctx)
            h_logits, a_logits = model_goals(seq_a, seq_b, t_a_idx, t_b_idx, ctx)
            
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, outcomes)
            loss_goals = criterion(h_logits, h_g_class) + criterion(a_logits, a_g_class)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            optimizer_goals.zero_grad()
            loss_goals.backward()
            torch.nn.utils.clip_grad_norm_(model_goals.parameters(), 1.0)
            optimizer_goals.step()
            
            train_loss += loss.item()
            train_loss_goals += loss_goals.item()
            n_batches += 1
            
            if n_batches % 50 == 0:
                print(".", end="", flush=True)
        print()

        avg_train = train_loss / max(n_batches, 1)
        avg_train_goals = train_loss_goals / max(n_batches, 1)

        # Validation
        model.eval()
        model_goals.eval()
        val_loss = 0.0
        val_loss_goals = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_a = batch["seq_a"].to(device)
                seq_b = batch["seq_b"].to(device)
                t_a_idx = batch["team_a_idx"].squeeze(1).to(device)
                t_b_idx = batch["team_b_idx"].squeeze(1).to(device)
                ctx = batch["context"].to(device)
                h_g = batch["home_goals"].squeeze(1).to(device)
                a_g = batch["away_goals"].squeeze(1).to(device)
                outcomes = torch.where(h_g > a_g, 0, torch.where(h_g == a_g, 1, 2)).long()
                
                h_g_class = torch.clamp(h_g.long(), 0, 9)
                a_g_class = torch.clamp(a_g.long(), 0, 9)

                logits = model(seq_a, seq_b, t_a_idx, t_b_idx, ctx)
                h_logits, a_logits = model_goals(seq_a, seq_b, t_a_idx, t_b_idx, ctx)
                
                loss = criterion(logits, outcomes)
                loss_goals = criterion(h_logits, h_g_class) + criterion(a_logits, a_g_class)
                
                val_loss += loss.item()
                val_loss_goals += loss_goals.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        avg_val_goals = val_loss_goals / max(n_val, 1)
        scheduler.step()
        scheduler_goals.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:3d}/{LSTM_EPOCHS}: "
                  f"train={avg_train:.4f}, val={avg_val:.4f} "
                  f"| goals_train={avg_train_goals:.4f}, goals_val={avg_val_goals:.4f}")

        # Use 1X2 val_loss for early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_state_goals = {k: v.clone() for k, v in model_goals.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= LSTM_PATIENCE:
                print(f"      Early stopping at epoch {epoch+1}")
                break

    # Load best state
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    
    if best_state_goals:
        model_goals.load_state_dict(best_state_goals)
    model_goals.eval()

    # Save
    os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), LSTM_MODEL_PATH)
    
    goals_path = LSTM_MODEL_PATH.replace(".pt", "_goals.pt")
    torch.save(model_goals.state_dict(), goals_path)

    meta = {
        "n_train": split,
        "n_val": len(samples) - split,
        "n_teams": n_teams,
        "val_nll": float(best_val_loss),
        "team_idx": team_idx,
        "trained_at": datetime.datetime.now().isoformat(),
        "seq_len": LSTM_SEQ_LEN,
        "hidden_size": LSTM_HIDDEN,
    }
    with open(LSTM_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   ✓  LSTM+MDN trained! Val NLL: {best_val_loss:.4f}")
    print(f"   ✓  GoalCountNet saved to {os.path.basename(goals_path)}")
    print(f"   ✓  Model cached to {os.path.basename(LSTM_MODEL_PATH)}")

    return model, model_goals, team_idx, meta


# ═══════════════════════════════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════════════════════════════

def _get_single_team_sequence(conn, team_name, seq_len=LSTM_SEQ_LEN, match_date=None):
    """Fetch historical matches for a specific team."""
    date_filter = f"AND m.date < '{match_date}'" if match_date else ""
    cur = conn.execute(f"""
        SELECT m.date, m.home_team, m.away_team, m.home_score, m.away_score,
               m.tournament, m.neutral,
               a.home_possession, a.away_possession, a.home_corners, a.away_corners,
               a.home_cards, a.away_cards, a.home_sot, a.away_sot
        FROM matches m
        LEFT JOIN advanced_stats a ON m.date = a.match_date AND m.home_team = a.home_team AND m.away_team = a.away_team
        WHERE (m.home_team = '{team_name}' OR m.away_team = '{team_name}')
        {date_filter}
        ORDER BY m.date
    """)
    rows = cur.fetchall()
    
    seq = []
    last_date = None
    
    for date_str, home, away, hs, as_, tourn, neutral, h_pos, a_pos, h_cor, a_cor, h_car, a_car, h_sot, a_sot in rows:
        if hs is None or as_ is None:
            continue
            
        hs, as_ = int(hs), int(as_)
        tourn_w = get_tournament_weight(tourn)
        is_home = (home == team_name)
        
        gf = hs if is_home else as_
        ga = as_ if is_home else hs
        opp = away if is_home else home
        opp_rank = get_fifa_rank(opp) or 100
        
        pos = float(h_pos if is_home else a_pos) if h_pos is not None else 50.0
        cor = float(h_cor if is_home else a_cor) if h_cor is not None else 0.0
        car = float(h_car if is_home else a_car) if h_car is not None else 0.0
        sot = float(h_sot if is_home else a_sot) if h_sot is not None else 0.0
        
        if last_date:
            try:
                d1 = datetime.date.fromisoformat(last_date)
                d2 = datetime.date.fromisoformat(date_str)
                days_gap = min((d2 - d1).days, 180)
            except Exception:
                days_gap = 30
        else:
            days_gap = 60
            
        result_pts = 1.0 if gf > ga else (0.5 if gf == ga else 0.0)
        
        recent = seq[-5:]
        cum_form = np.mean([r["result_pts"] for r in recent]) if recent else 0.5
        
        feature = {
            "gf": gf, "ga": ga,
            "opp_rank": min(opp_rank, 211) / 211.0,
            "tournament_w": tourn_w,
            "is_home": 1.0 if is_home else 0.0,
            "days_gap": min(days_gap, 180) / 180.0,
            "result_pts": result_pts,
            "cumulative_form": cum_form,
            "possession": pos / 100.0,
            "corners": min(cor, 20.0) / 20.0,
            "cards": min(car, 10.0) / 10.0,
            "sot": min(sot, 20.0) / 20.0,
        }
        seq.append(feature)
        last_date = date_str
        
    return seq[-seq_len:]


def predict_lstm(model, model_goals, team_idx, conn, team_a_db, team_b_db, venue="neutral", match_date=None):
    """
    Generate a score probability matrix using the LSTM+MDN and GoalCountNet.

    Returns: (score_matrix, details) or (None, None)
    """
    if model is None or model_goals is None:
        return None, None

    try:
        seq_a = _get_single_team_sequence(conn, team_a_db, LSTM_SEQ_LEN, match_date)
        seq_b = _get_single_team_sequence(conn, team_b_db, LSTM_SEQ_LEN, match_date)
    except Exception:
        return None, None

    feat_keys = ["gf", "ga", "opp_rank", "tournament_w", "is_home",
                  "days_gap", "result_pts", "cumulative_form",
                  "possession", "corners", "cards", "sot"]

    if len(seq_a) < 3 or len(seq_b) < 3:
        return None, None

    vec_a = [[s[k] for k in feat_keys] for s in seq_a]
    vec_b = [[s[k] for k in feat_keys] for s in seq_b]

    while len(vec_a) < LSTM_SEQ_LEN:
        vec_a.insert(0, [0.0] * len(feat_keys))
    while len(vec_b) < LSTM_SEQ_LEN:
        vec_b.insert(0, [0.0] * len(feat_keys))

    # Context
    fifa_a, _ = get_fifa_points(team_a_db)
    fifa_b, _ = get_fifa_points(team_b_db)
    median = get_median_fifa()
    rank_a = get_fifa_rank(team_a_db) or 100
    rank_b = get_fifa_rank(team_b_db) or 100
    is_neutral = 1.0 if venue == "neutral" else 0.0

    context = [
        (fifa_a or median) / 2000.0,
        (fifa_b or median) / 2000.0,
        ((fifa_a or median) - (fifa_b or median)) / 500.0,
        rank_a / 211.0,
        rank_b / 211.0,
        is_neutral,
    ]

    t_a_idx = team_idx.get(team_a_db, 0)
    t_b_idx = team_idx.get(team_b_db, 0)

    cpu_device = torch.device('cpu')
    model.to(cpu_device)
    model.eval()
    model_goals.to(cpu_device)
    model_goals.eval()
    
    with torch.no_grad():
        logits = model(
            torch.FloatTensor([vec_a]).to(cpu_device),
            torch.FloatTensor([vec_b]).to(cpu_device),
            torch.LongTensor([t_a_idx]).to(cpu_device),
            torch.LongTensor([t_b_idx]).to(cpu_device),
            torch.FloatTensor([context]).to(cpu_device),
        )
        h_goals_logits, a_goals_logits = model_goals(
            torch.FloatTensor([vec_a]).to(cpu_device),
            torch.FloatTensor([vec_b]).to(cpu_device),
            torch.LongTensor([t_a_idx]).to(cpu_device),
            torch.LongTensor([t_b_idx]).to(cpu_device),
            torch.FloatTensor([context]).to(cpu_device),
        )
        
        # Apply Temperature Scaling to soften extreme confidence
        T = 1.20
        probs = torch.nn.functional.softmax(logits / T, dim=1)[0].cpu().numpy()
        h_goals_probs = torch.nn.functional.softmax(h_goals_logits, dim=1)[0].cpu().numpy()
        a_goals_probs = torch.nn.functional.softmax(a_goals_logits, dim=1)[0].cpu().numpy()

    # Generate 10x10 score matrix for GoalCountNet
    score_matrix = np.outer(h_goals_probs, a_goals_probs)

    # Logits match the target outcomes: 0=Home Win, 1=Draw, 2=Away Win
    outcomes = {
        "win_a_pct": float(probs[0]) * 100,
        "draw_pct": float(probs[1]) * 100,
        "win_b_pct": float(probs[2]) * 100,
        "mu_h": float(probs[0] * 2), # Pseudo-lambda for stability
        "mu_a": float(probs[2] * 2),
        "logits": logits.cpu().numpy()[0].tolist()
    }

    return score_matrix, outcomes
