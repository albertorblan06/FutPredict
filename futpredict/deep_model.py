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
                     LSTM_CONS_MODEL_PATH, LSTM_CONS_META_PATH,
                     MAX_GOALS, TRAIN_END_DATE,
                     FOCAL_ALPHA, FOCAL_GAMMA)
from .focal_loss import FocalLoss
from .rankings import get_fifa_rank, get_fifa_points, get_median_fifa
from .analysis import get_tournament_weight
from .statistical_model import build_score_matrix


def _to_ordinal_target(goal_class, n_classes=10):
    """Convert integer goal count to cumulative ordinal vector.
    
    If a team scores 3 goals, the target is [1, 1, 1, 0, 0, 0, 0, 0, 0, 0].
    This encodes the ordinal structure: to score 3, you first score 1 and 2.
    """
    target = torch.zeros(n_classes)
    target[:goal_class] = 1.0
    return target


def _ordinal_targets_batch(goal_classes, n_classes=10):
    """Convert a batch of integer goal counts to ordinal target matrix."""
    batch_size = goal_classes.shape[0]
    targets = torch.zeros(batch_size, n_classes)
    for i in range(batch_size):
        g = int(goal_classes[i].item())
        targets[i, :g] = 1.0
    return targets


def _ordinal_logits_to_pmf(logits):
    """Convert ordinal cumulative logits to discrete goal PMF.
    
    P(goals >= k) = sigmoid(logit_k)
    P(goals = k) = P(goals >= k) - P(goals >= k+1)
    """
    probs_ge = torch.sigmoid(logits)  # P(goals >= k) for k=0..9
    # P(goals = k) = P(goals >= k) - P(goals >= k+1)
    # For the last class: P(goals = 9) = P(goals >= 9)
    pmf = torch.zeros_like(probs_ge)
    for k in range(probs_ge.shape[-1] - 1):
        pmf[..., k] = probs_ge[..., k] - probs_ge[..., k + 1]
    pmf[..., -1] = probs_ge[..., -1]
    # Clamp to avoid negative probabilities from numerical imprecision
    pmf = torch.clamp(pmf, min=1e-8)
    # Normalize
    pmf = pmf / pmf.sum(dim=-1, keepdim=True)
    return pmf

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

        # Output heads for Poisson regression (log lambda)
        self.home_goals_head = nn.Linear(64, 1)
        self.away_goals_head = nn.Linear(64, 1)

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
    Train or load cached dual LSTM+MDN models (Aggressive and Conservative).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    if not force and os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_META_PATH) and os.path.exists(LSTM_CONS_MODEL_PATH) and os.path.exists(LSTM_CONS_META_PATH):
        try:
            with open(LSTM_META_PATH, "r") as f:
                meta_agg = json.load(f)
            with open(LSTM_CONS_META_PATH, "r") as f:
                meta_cons = json.load(f)
                
            team_idx = meta_agg.get("team_idx", {})
            n_teams = meta_agg.get("n_teams", 500)
            
            print("   [DEBUG] Initializing Dual FootballClassifiers and GoalCountNet...")
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
            model_agg = FootballClassifier(
                seq_feature_dim=len(SEQ_FEATURES), context_dim=6, n_teams=n_teams,
                hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                embed_dim=LSTM_EMBED_DIM, dropout=LSTM_DROPOUT,
            )
            model_cons = FootballClassifier(
                seq_feature_dim=len(SEQ_FEATURES), context_dim=6, n_teams=n_teams,
                hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                embed_dim=LSTM_EMBED_DIM, dropout=LSTM_DROPOUT,
            )
            model_goals = GoalCountNet(
                seq_feature_dim=len(SEQ_FEATURES), context_dim=6, n_teams=n_teams,
                hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                embed_dim=LSTM_EMBED_DIM, dropout=LSTM_DROPOUT,
            )
            
            model_agg.to(device)
            model_cons.to(device)
            model_goals.to(device)
            
            model_agg.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
            model_cons.load_state_dict(torch.load(LSTM_CONS_MODEL_PATH, map_location=device))
            
            goals_path = LSTM_MODEL_PATH.replace(".pt", "_goals.pt")
            if os.path.exists(goals_path):
                model_goals.load_state_dict(torch.load(goals_path, map_location=device))
                
            model_agg.eval()
            model_cons.eval()
            model_goals.eval()
            
            print(f"   ✓  Loaded dual LSTM+MDN models ({meta_agg.get('n_train', '?')} training samples)")
            return model_agg, model_cons, model_goals, team_idx, meta_agg, meta_cons
        except Exception as e:
            print(f"   ⚠  Cache load failed ({e}), retraining...")

    print("   ⬇  Building LSTM sequences from historical data...")
    try:
        samples, team_idx = _build_training_data(conn)
    except Exception as e:
        print(f"   ✗  Sequence building failed: {e}")
        return None, None, None, None, None, None

    if len(samples) < 2000:
        print(f"   ✗  Not enough data ({len(samples)} samples)")
        return None, None, None, None, None, None

    n_teams = max(team_idx.values()) + 1
    print(f"   ✓  {len(samples):,} match sequences built ({n_teams} teams, {LSTM_SEQ_LEN} steps)")

    split = int(len(samples) * 0.85)
    train_data = MatchDataset(samples[:split])
    val_data = MatchDataset(samples[split:])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=LSTM_BATCH, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=LSTM_BATCH, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def _train_single_model(is_aggressive):
        model = FootballClassifier(
            seq_feature_dim=len(SEQ_FEATURES), context_dim=6, n_teams=n_teams,
            hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
            embed_dim=LSTM_EMBED_DIM, dropout=LSTM_DROPOUT,
        )
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
        
        if is_aggressive:
            model_goals = GoalCountNet(
                seq_feature_dim=len(SEQ_FEATURES), context_dim=6, n_teams=n_teams,
                hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                embed_dim=LSTM_EMBED_DIM, dropout=LSTM_DROPOUT,
            )
            model_goals.to(device)
            optimizer_goals = torch.optim.AdamW(model_goals.parameters(), lr=LSTM_LR, weight_decay=1e-3)
            scheduler_goals = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_goals, T_0=10, T_mult=2, eta_min=1e-5)
        else:
            model_goals = None
            
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        best_state_goals = None
        
        for epoch in range(LSTM_EPOCHS):
            model.train()
            if model_goals: model_goals.train()
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
                
                logits = model(seq_a, seq_b, t_a_idx, t_b_idx, ctx)
                
                if is_aggressive:
                    criterion_1x2 = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
                    loss = criterion_1x2(logits, outcomes)
                else:
                    criterion_1x2 = nn.CrossEntropyLoss()
                    ce_loss = criterion_1x2(logits, outcomes)
                    # Entropy Minimization to force sharpness
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                    loss = ce_loss + 0.25 * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                
                if model_goals:
                    h_g_target = torch.clamp(h_g.float(), 0, 9).unsqueeze(1)
                    a_g_target = torch.clamp(a_g.float(), 0, 9).unsqueeze(1)
                    h_logits, a_logits = model_goals(seq_a, seq_b, t_a_idx, t_b_idx, ctx)
                    criterion_goals = nn.PoissonNLLLoss(log_input=True)
                    loss_goals = criterion_goals(h_logits, h_g_target) + criterion_goals(a_logits, a_g_target)
                    
                    optimizer_goals.zero_grad()
                    loss_goals.backward()
                    torch.nn.utils.clip_grad_norm_(model_goals.parameters(), 1.0)
                    optimizer_goals.step()
                    train_loss_goals += loss_goals.item()
                    
                n_batches += 1
                if n_batches % 50 == 0:
                    print(".", end="", flush=True)
            print()
            
            avg_train = train_loss / max(n_batches, 1)
            avg_train_goals = train_loss_goals / max(n_batches, 1)
            
            # Validation
            model.eval()
            if model_goals: model_goals.eval()
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
                    
                    logits = model(seq_a, seq_b, t_a_idx, t_b_idx, ctx)
                    
                    if is_aggressive:
                        criterion_1x2 = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
                        loss = criterion_1x2(logits, outcomes)
                    else:
                        criterion_1x2 = nn.CrossEntropyLoss()
                        ce_loss = criterion_1x2(logits, outcomes)
                        probs = torch.softmax(logits, dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                        loss = ce_loss + 0.25 * entropy
                        
                    val_loss += loss.item()
                    
                    if model_goals:
                        h_g_target = torch.clamp(h_g.float(), 0, 9).unsqueeze(1)
                        a_g_target = torch.clamp(a_g.float(), 0, 9).unsqueeze(1)
                        h_logits, a_logits = model_goals(seq_a, seq_b, t_a_idx, t_b_idx, ctx)
                        criterion_goals = nn.PoissonNLLLoss(log_input=True)
                        loss_goals = criterion_goals(h_logits, h_g_target) + criterion_goals(a_logits, a_g_target)
                        val_loss_goals += loss_goals.item()
                        
                    n_val += 1
            
            avg_val = val_loss / max(n_val, 1)
            avg_val_goals = val_loss_goals / max(n_val, 1)
            
            scheduler.step()
            if model_goals: scheduler_goals.step()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"      Epoch {epoch+1:3d}/{LSTM_EPOCHS}: train={avg_train:.4f}, val={avg_val:.4f}" + 
                      (f" | goals_train={avg_train_goals:.4f}, goals_val={avg_val_goals:.4f}" if model_goals else ""))
                
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if model_goals:
                    best_state_goals = {k: v.clone() for k, v in model_goals.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= LSTM_PATIENCE:
                    print(f"      Early stopping at epoch {epoch+1}")
                    break
                    
        # Load best state
        if best_state: model.load_state_dict(best_state)
        model.eval()
        if model_goals and best_state_goals:
            model_goals.load_state_dict(best_state_goals)
            model_goals.eval()
            
        if is_aggressive:
            learned_temp = _calibrate_temperature(model, val_loader, device)
            print(f"   ✓  Platt Scaling: learned T* = {learned_temp:.4f}")
        else:
            learned_temp = 1.0
            print("   ✓  Platt Scaling: Skipped (T* = 1.0) for sharp Conservative predictions")
        
        meta = {
            "n_train": split,
            "n_val": len(samples) - split,
            "n_teams": n_teams,
            "val_nll": float(best_val_loss),
            "team_idx": team_idx,
            "trained_at": datetime.datetime.now().isoformat(),
            "seq_len": LSTM_SEQ_LEN,
            "hidden_size": LSTM_HIDDEN,
            "learned_temperature": learned_temp,
        }
        
        return model, model_goals, meta
        
    print(f"   ⚙  Training Aggressive LSTM+MDN & GoalCountNet ({LSTM_EPOCHS} epochs, batch={LSTM_BATCH})...")
    model_agg, model_goals, meta_agg = _train_single_model(is_aggressive=True)
    
    os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
    torch.save(model_agg.state_dict(), LSTM_MODEL_PATH)
    goals_path = LSTM_MODEL_PATH.replace(".pt", "_goals.pt")
    torch.save(model_goals.state_dict(), goals_path)
    with open(LSTM_META_PATH, "w") as f:
        json.dump(meta_agg, f, indent=2)
        
    del model_agg
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    
    print(f"   ⚙  Training Conservative LSTM+MDN ({LSTM_EPOCHS} epochs, batch={LSTM_BATCH})...")
    model_cons, _, meta_cons = _train_single_model(is_aggressive=False)
    
    torch.save(model_cons.state_dict(), LSTM_CONS_MODEL_PATH)
    with open(LSTM_CONS_META_PATH, "w") as f:
        json.dump(meta_cons, f, indent=2)
        
    # Reload model_agg for return
    model_agg = FootballClassifier(
        seq_feature_dim=len(SEQ_FEATURES), context_dim=6, n_teams=n_teams,
        hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
        embed_dim=LSTM_EMBED_DIM, dropout=LSTM_DROPOUT,
    )
    model_agg.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    model_agg.to(device)
    model_agg.eval()
        
    print(f"   ✓  LSTM+MDN Dual Models trained!")
    return model_agg, model_cons, model_goals, team_idx, meta_agg, meta_cons

class TemperatureScaler(nn.Module):
    """Learns a single temperature parameter to calibrate logits.
    
    After training, the model's logits are divided by T* before softmax.
    T* is optimized to minimize NLL on the validation set via L-BFGS.
    This preserves top-1 accuracy while correcting probability magnitudes.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature


def _calibrate_temperature(model, val_loader, device):
    """Learn optimal temperature from validation set logits.
    
    Returns the learned temperature as a float.
    """
    model.eval()
    scaler = TemperatureScaler().to(device)
    
    # Collect all validation logits and targets
    all_logits = []
    all_targets = []
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
            
            logits = model(seq_a, seq_b, t_a_idx, t_b_idx, ctx)
            all_logits.append(logits)
            all_targets.append(outcomes)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Optimize temperature via L-BFGS
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    
    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(all_logits)
        loss = criterion(scaled_logits, all_targets)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    learned_temp = float(scaler.temperature.item())
    # Sanity clamp: temperature should be between 0.5 and 5.0
    learned_temp = max(0.5, min(5.0, learned_temp))
    
    return learned_temp


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



def predict_lstm(model_agg, model_cons, model_goals, team_idx, conn, team_a_db, team_b_db, venue="neutral", match_date=None, meta_agg=None, meta_cons=None):
    """
    Generate a score probability matrix using the LSTM+MDN and GoalCountNet.

    Returns: (score_matrix, details_agg, details_cons) or (None, None, None)
    """
    if model_agg is None or model_cons is None or model_goals is None:
        return None, None, None

    try:
        seq_a = _get_single_team_sequence(conn, team_a_db, LSTM_SEQ_LEN, match_date)
        seq_b = _get_single_team_sequence(conn, team_b_db, LSTM_SEQ_LEN, match_date)
    except Exception:
        return None, None, None

    feat_keys = ["gf", "ga", "opp_rank", "tournament_w", "is_home",
                  "days_gap", "result_pts", "cumulative_form",
                  "possession", "corners", "cards", "sot"]

    if len(seq_a) < 3 or len(seq_b) < 3:
        return None, None, None

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
    model_agg.to(cpu_device)
    model_agg.eval()
    model_cons.to(cpu_device)
    model_cons.eval()
    model_goals.to(cpu_device)
    model_goals.eval()
    
    with torch.no_grad():
        logits_agg = model_agg(
            torch.FloatTensor([vec_a]).to(cpu_device),
            torch.FloatTensor([vec_b]).to(cpu_device),
            torch.LongTensor([t_a_idx]).to(cpu_device),
            torch.LongTensor([t_b_idx]).to(cpu_device),
            torch.FloatTensor([context]).to(cpu_device),
        )
        logits_cons = model_cons(
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
        
        T_agg = meta_agg.get("learned_temperature", 1.20) if meta_agg else 1.20
        probs_agg = torch.nn.functional.softmax(logits_agg / T_agg, dim=1)[0].cpu().numpy()
        
        T_cons = meta_cons.get("learned_temperature", 1.20) if meta_cons else 1.20
        probs_cons = torch.nn.functional.softmax(logits_cons / T_cons, dim=1)[0].cpu().numpy()
        
        from scipy.stats import poisson
        lambda_h = torch.exp(h_goals_logits).item()
        lambda_a = torch.exp(a_goals_logits).item()
        
        h_goals_pmf = poisson.pmf(np.arange(10), lambda_h)
        a_goals_pmf = poisson.pmf(np.arange(10), lambda_a)

    score_matrix = np.outer(h_goals_pmf, a_goals_pmf)
    
    # Apply Dixon-Coles ρ correction to fix independence assumption
    # ρ = -0.04 is the empirical value for international football
    # This corrects P(0-0), P(1-0), P(0-1), P(1-1) for goal correlation
    rho_dc = -0.04
    mu_h = float(np.sum(np.arange(len(h_goals_pmf)) * h_goals_pmf))
    mu_a = float(np.sum(np.arange(len(a_goals_pmf)) * a_goals_pmf))
    if mu_h > 0.01 and mu_a > 0.01:  # guard against degenerate PMFs
        score_matrix[0, 0] *= max(0.01, 1 - mu_h * mu_a * rho_dc)
        score_matrix[0, 1] *= max(0.01, 1 + mu_h * rho_dc)
        score_matrix[1, 0] *= max(0.01, 1 + mu_a * rho_dc)
        score_matrix[1, 1] *= max(0.01, 1 - rho_dc)
        # Re-normalize
        score_matrix = np.clip(score_matrix, 1e-10, None)
        score_matrix /= score_matrix.sum()

    # Calculate exact totals probabilities from the 10x10 matrix
    o25_prob = 0.0
    u35_prob = 0.0
    for h in range(10):
        for a in range(10):
            if h + a > 2.5: o25_prob += score_matrix[h, a]
            if h + a < 3.5: u35_prob += score_matrix[h, a]

    total_expected_goals = mu_h + mu_a

    outcomes_agg = {
        "win_a_pct": float(probs_agg[0]) * 100,
        "draw_pct": float(probs_agg[1]) * 100,
        "win_b_pct": float(probs_agg[2]) * 100,
        "mu_h": float(mu_h),
        "mu_a": float(mu_a),
        "expected_goals": float(total_expected_goals),
        "over_2_5_pct": float(o25_prob * 100),
        "under_3_5_pct": float(u35_prob * 100),
        "logits": logits_agg.cpu().numpy()[0].tolist()
    }
    
    outcomes_cons = {
        "win_a_pct": float(probs_cons[0]) * 100,
        "draw_pct": float(probs_cons[1]) * 100,
        "win_b_pct": float(probs_cons[2]) * 100,
        "mu_h": float(mu_h),
        "mu_a": float(mu_a),
        "expected_goals": float(total_expected_goals),
        "over_2_5_pct": float(o25_prob * 100),
        "under_3_5_pct": float(u35_prob * 100),
        "logits": logits_cons.cpu().numpy()[0].tolist()
    }
    
    return score_matrix, outcomes_agg, outcomes_cons
