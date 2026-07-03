"""
Learned Weights — Optuna-based Bayesian optimization for tournament weights,
time-decay half-lives, and Elo K-factors.

Replaces all hardcoded weighting constants with values learned from data via
multi-objective optimization (Brier Score + Over/Under LogLoss) using
expanding-window temporal cross-validation to prevent data leakage.
"""
import os
import json
import math
import datetime
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional

from .config import DATA_DIR, MAX_GOALS

# ═══════════════════════════════════════════════════════════════
#  LEARNED WEIGHTS DATACLASS
# ═══════════════════════════════════════════════════════════════

LEARNED_WEIGHTS_PATH = os.path.join(DATA_DIR, "learned_weights.json")

@dataclass
class LearnedWeights:
    """Single source of truth for all learnable weighting parameters."""

    # Tournament weights (0.0–1.0 scale)
    w_world_cup: float = 1.0
    w_world_cup_qual: float = 0.85
    w_continental_final: float = 0.90
    w_continental_qual: float = 0.80
    w_nations_league: float = 0.80
    w_friendly: float = 0.60
    w_other: float = 0.70

    # Time decay half-lives (in days)
    decay_half_life: float = 365.0
    h2h_decay_half_life: float = 1095.0
    dc_decay_half_life: float = 730.0  # Dixon-Coles (was hardcoded 365*2)

    # Elo K-factors
    k_world_cup: float = 60.0
    k_continental: float = 50.0
    k_continental_qual: float = 40.0
    k_qualifier: float = 30.0
    k_nations_league: float = 30.0
    k_friendly: float = 15.0
    k_other: float = 20.0

    # Metadata (not optimized)
    brier_score: float = -1.0
    ou_logloss: float = -1.0
    optimized_at: str = ""
    n_trials: int = 0

    def save(self, path: Optional[str] = None):
        path = path or LEARNED_WEIGHTS_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "LearnedWeights":
        path = path or LEARNED_WEIGHTS_PATH
        with open(path, "r") as f:
            data = json.load(f)
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# Cached singleton
_cached_weights: Optional[LearnedWeights] = None


def get_weights() -> LearnedWeights:
    """Load learned weights from disk, or return defaults if not calibrated."""
    global _cached_weights
    if _cached_weights is not None:
        return _cached_weights

    if os.path.exists(LEARNED_WEIGHTS_PATH):
        try:
            _cached_weights = LearnedWeights.load()
            return _cached_weights
        except Exception:
            pass

    _cached_weights = LearnedWeights()  # defaults
    return _cached_weights


def reset_weights_cache():
    """Force reload on next get_weights() call."""
    global _cached_weights
    _cached_weights = None


# ═══════════════════════════════════════════════════════════════
#  TOURNAMENT WEIGHT LOOKUP (used by analysis.py, etc.)
# ═══════════════════════════════════════════════════════════════

def tournament_weight_from_learned(tournament_name: str,
                                   weights: Optional[LearnedWeights] = None) -> float:
    """Classify tournament and return weight from learned parameters."""
    weights = weights or get_weights()
    t = (tournament_name or "").lower()

    if "world cup" in t:
        return weights.w_world_cup_qual if "qualif" in t else weights.w_world_cup
    if any(kw in t for kw in ("copa amér", "copa amer", "uefa euro",
                               "european championship")):
        return weights.w_continental_qual if "qualif" in t else weights.w_continental_final
    if any(kw in t for kw in ("african cup", "africa cup", "afcon")):
        return weights.w_continental_qual if "qualif" in t else weights.w_continental_final
    if "asian cup" in t:
        return weights.w_continental_qual if "qualif" in t else weights.w_continental_final
    if "gold cup" in t or "concacaf nations" in t:
        return weights.w_continental_final
    if "nations league" in t:
        return weights.w_nations_league
    if "confederations cup" in t:
        return weights.w_continental_final
    if "friendly" in t:
        return weights.w_friendly
    return weights.w_other


def k_factor_from_learned(tournament_name: str,
                          weights: Optional[LearnedWeights] = None) -> float:
    """Determine the Elo K-factor from learned parameters."""
    weights = weights or get_weights()
    t = (tournament_name or "").lower()

    if "world cup" in t and "qualif" not in t:
        return weights.k_world_cup
    if any(kw in t for kw in ("copa amér", "copa amer", "uefa euro",
                               "european championship", "african cup", "asian cup")):
        return weights.k_continental if "qualif" not in t else weights.k_continental_qual
    if "qualif" in t:
        return weights.k_qualifier
    if "nations league" in t:
        return weights.k_nations_league
    if "gold cup" in t or "concacaf" in t:
        return weights.k_continental_qual
    if "friendly" in t:
        return weights.k_friendly
    return weights.k_other


# ═══════════════════════════════════════════════════════════════
#  OPTUNA OPTIMIZATION
# ═══════════════════════════════════════════════════════════════

# Temporal CV folds: (train_start, train_end, val_start, val_end)
CV_FOLDS = [
    ("2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]


def _suggest_weights(trial) -> LearnedWeights:
    """Suggest a full set of weights from an Optuna trial."""
    return LearnedWeights(
        # Tournament weights
        w_world_cup=trial.suggest_float("w_world_cup", 0.85, 1.0),
        w_world_cup_qual=trial.suggest_float("w_world_cup_qual", 0.65, 0.95),
        w_continental_final=trial.suggest_float("w_continental_final", 0.70, 1.0),
        w_continental_qual=trial.suggest_float("w_continental_qual", 0.55, 0.90),
        w_nations_league=trial.suggest_float("w_nations_league", 0.50, 0.90),
        w_friendly=trial.suggest_float("w_friendly", 0.30, 0.80),
        w_other=trial.suggest_float("w_other", 0.40, 0.85),

        # Decay half-lives
        decay_half_life=trial.suggest_float("decay_half_life", 120, 900),
        h2h_decay_half_life=trial.suggest_float("h2h_decay_half_life", 500, 2500),
        dc_decay_half_life=trial.suggest_float("dc_decay_half_life", 300, 1500),

        # Elo K-factors
        k_world_cup=trial.suggest_float("k_world_cup", 30, 80),
        k_continental=trial.suggest_float("k_continental", 25, 70),
        k_continental_qual=trial.suggest_float("k_continental_qual", 15, 55),
        k_qualifier=trial.suggest_float("k_qualifier", 15, 50),
        k_nations_league=trial.suggest_float("k_nations_league", 15, 50),
        k_friendly=trial.suggest_float("k_friendly", 5, 35),
        k_other=trial.suggest_float("k_other", 10, 40),
    )


def _compute_brier_score(probs_1x2, actual_1x2):
    """Compute Brier Score for 1X2 predictions.

    Args:
        probs_1x2: list of (p_home, p_draw, p_away) tuples
        actual_1x2: list of actual outcomes ("1", "X", "2")
    """
    brier = 0.0
    for (p1, px, p2), actual in zip(probs_1x2, actual_1x2):
        vec = [0.0, 0.0, 0.0]
        if actual == "1":
            vec[0] = 1.0
        elif actual == "X":
            vec[1] = 1.0
        else:
            vec[2] = 1.0
        brier += (p1 - vec[0]) ** 2 + (px - vec[1]) ** 2 + (p2 - vec[2]) ** 2
    return brier / max(len(probs_1x2), 1)


def _compute_ou_logloss(ou_probs, actual_totals, threshold=2.5):
    """Compute LogLoss for Over/Under predictions.

    Args:
        ou_probs: list of P(Over threshold) values
        actual_totals: list of actual total goals
    """
    eps = 1e-15
    ll = 0.0
    for p_over, total in zip(ou_probs, actual_totals):
        p_over = max(eps, min(1 - eps, p_over))
        actual_over = 1.0 if total > threshold else 0.0
        ll -= actual_over * math.log(p_over) + (1 - actual_over) * math.log(1 - p_over)
    return ll / max(len(ou_probs), 1)


def _evaluate_fold(conn, weights, train_end, val_start, val_end):
    """Evaluate a single temporal fold using Dixon-Coles with given weights.

    Returns: (brier_score, ou_logloss) on the validation set.
    """
    from .statistical_model import fit_dixon_coles, predict_dc

    # Fit Dixon-Coles on training period with trial weights
    dc_params = fit_dixon_coles(
        conn, force=True,
        reference_date=val_start,
        weights=weights,
    )

    if dc_params is None:
        return 1.0, 1.0  # worst case

    # Get validation matches
    cur = conn.execute("""
        SELECT date, home_team, away_team, home_score, away_score, tournament
        FROM matches
        WHERE date >= ? AND date < ?
          AND home_score IS NOT NULL
        ORDER BY date
    """, (val_start, val_end))
    val_matches = cur.fetchall()

    if len(val_matches) < 50:
        return 1.0, 1.0  # not enough validation data

    probs_1x2 = []
    actual_1x2 = []
    ou_probs = []
    actual_totals = []

    for date_str, home, away, hs, as_, tourn in val_matches:
        hs, as_ = int(hs), int(as_)

        try:
            matrix, _ = predict_dc(dc_params, home, away, venue="neutral")
        except Exception:
            continue

        if matrix is None:
            continue

        mg = matrix.shape[0]

        # 1X2 probabilities from score matrix
        p_home = sum(matrix[h, a] for h in range(mg) for a in range(mg) if h > a)
        p_draw = sum(matrix[h, a] for h in range(mg) for a in range(mg) if h == a)
        p_away = sum(matrix[h, a] for h in range(mg) for a in range(mg) if h < a)
        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total
        else:
            p_home = p_draw = p_away = 1.0 / 3.0

        probs_1x2.append((p_home, p_draw, p_away))
        actual = "1" if hs > as_ else ("X" if hs == as_ else "2")
        actual_1x2.append(actual)

        # Over/Under 2.5
        p_over = sum(matrix[h, a] for h in range(mg) for a in range(mg) if h + a > 2)
        ou_probs.append(p_over)
        actual_totals.append(hs + as_)

    if len(probs_1x2) < 30:
        return 1.0, 1.0

    brier = _compute_brier_score(probs_1x2, actual_1x2)
    logloss = _compute_ou_logloss(ou_probs, actual_totals)

    return brier, logloss


def _objective(trial, conn):
    """Optuna multi-objective: minimize (Brier Score, O/U LogLoss)."""
    weights = _suggest_weights(trial)

    brier_scores = []
    ou_loglosses = []

    for train_start, train_end, val_start, val_end in CV_FOLDS:
        brier, logloss = _evaluate_fold(conn, weights, train_end, val_start, val_end)
        brier_scores.append(brier)
        ou_loglosses.append(logloss)

    mean_brier = np.mean(brier_scores)
    mean_logloss = np.mean(ou_loglosses)

    return mean_brier, mean_logloss


def run_optimization(conn, n_trials: int = 150, show_progress: bool = True):
    """Run the Optuna multi-objective Bayesian optimization study.

    Args:
        conn: SQLite connection to the futpredict database
        n_trials: number of Optuna trials to run
        show_progress: whether to print progress

    Returns:
        LearnedWeights with the best parameters
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for weight calibration. "
            "Install it with: pip install optuna"
        )

    # Suppress Optuna's internal logging unless debug
    if not show_progress:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("╔════════════════════════════════════════════════════════════╗")
    print("║  WEIGHT CALIBRATION — Optuna Multi-Objective Optimization ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"   Trials: {n_trials}")
    print(f"   Objectives: Brier Score (1X2) + O/U LogLoss")
    print(f"   CV Folds: {len(CV_FOLDS)} expanding-window temporal folds")
    print(f"   Parameters: 17 (7 tournament weights + 3 decays + 7 Elo K-factors)")
    print()

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name="futpredict_weights",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    )

    def objective_wrapper(trial):
        return _objective(trial, conn)

    study.optimize(
        objective_wrapper,
        n_trials=n_trials,
        show_progress_bar=show_progress,
    )

    # Select the best trial from the Pareto front
    # Use a weighted combination: 0.6 * Brier + 0.4 * LogLoss (normalized)
    pareto_trials = study.best_trials
    if not pareto_trials:
        print("   ⚠  No successful trials. Using defaults.")
        return LearnedWeights()

    # Normalize objectives across Pareto front for fair comparison
    briers = [t.values[0] for t in pareto_trials]
    loglosses = [t.values[1] for t in pareto_trials]
    b_min, b_max = min(briers), max(briers)
    l_min, l_max = min(loglosses), max(loglosses)

    best_score = float("inf")
    best_trial = pareto_trials[0]

    for t in pareto_trials:
        b_norm = (t.values[0] - b_min) / max(b_max - b_min, 1e-8)
        l_norm = (t.values[1] - l_min) / max(l_max - l_min, 1e-8)
        combined = 0.6 * b_norm + 0.4 * l_norm
        if combined < best_score:
            best_score = combined
            best_trial = t

    # Build LearnedWeights from best trial
    p = best_trial.params
    best_weights = LearnedWeights(
        w_world_cup=p["w_world_cup"],
        w_world_cup_qual=p["w_world_cup_qual"],
        w_continental_final=p["w_continental_final"],
        w_continental_qual=p["w_continental_qual"],
        w_nations_league=p["w_nations_league"],
        w_friendly=p["w_friendly"],
        w_other=p["w_other"],
        decay_half_life=p["decay_half_life"],
        h2h_decay_half_life=p["h2h_decay_half_life"],
        dc_decay_half_life=p["dc_decay_half_life"],
        k_world_cup=p["k_world_cup"],
        k_continental=p["k_continental"],
        k_continental_qual=p["k_continental_qual"],
        k_qualifier=p["k_qualifier"],
        k_nations_league=p["k_nations_league"],
        k_friendly=p["k_friendly"],
        k_other=p["k_other"],
        brier_score=best_trial.values[0],
        ou_logloss=best_trial.values[1],
        optimized_at=datetime.datetime.now().isoformat(),
        n_trials=n_trials,
    )

    # Save to disk
    best_weights.save()
    reset_weights_cache()

    print(f"\n   ✓  Optimization complete!")
    print(f"      Brier Score (1X2): {best_weights.brier_score:.4f}")
    print(f"      O/U LogLoss:      {best_weights.ou_logloss:.4f}")
    print(f"      Pareto front:     {len(pareto_trials)} solutions")
    print(f"      Saved to:         {LEARNED_WEIGHTS_PATH}")

    # Print comparison table
    defaults = LearnedWeights()
    print(f"\n   ┌{'─'*56}┐")
    print(f"   │ {'Parameter':<24} │ {'Default':>8} │ {'Learned':>8} │ {'Δ':>6} │")
    print(f"   ├{'─'*56}┤")

    params_to_show = [
        ("w_friendly", "w_friendly"),
        ("w_world_cup", "w_world_cup"),
        ("w_continental_final", "w_continental_final"),
        ("w_nations_league", "w_nations_league"),
        ("w_other", "w_other"),
        ("decay_half_life", "decay_half_life"),
        ("h2h_decay_half_life", "h2h_decay_half_life"),
        ("dc_decay_half_life", "dc_decay_half_life"),
        ("k_world_cup", "k_world_cup"),
        ("k_friendly", "k_friendly"),
        ("k_qualifier", "k_qualifier"),
    ]

    for label, attr in params_to_show:
        d = getattr(defaults, attr)
        l = getattr(best_weights, attr)
        pct = ((l - d) / d * 100) if d != 0 else 0
        sign = "+" if pct > 0 else ""
        print(f"   │ {label:<24} │ {d:>8.1f} │ {l:>8.1f} │ {sign}{pct:>4.1f}% │")

    print(f"   └{'─'*56}┘")

    return best_weights
