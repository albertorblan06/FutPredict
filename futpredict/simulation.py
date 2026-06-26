"""
Monte Carlo simulation engine.
Works with score probability matrices (9x9) from any model.
"""
import random
import collections
import numpy as np
from .config import NUM_SIMULATIONS, MAX_GOALS


def sample_from_score_matrix(score_matrix):
    """Sample a single (home_goals, away_goals) from a score probability matrix."""
    flat = score_matrix.ravel()
    flat = flat / flat.sum()  # ensure normalization
    idx = np.random.choice(len(flat), p=flat)
    return divmod(idx, score_matrix.shape[1])


def run_simulation(score_matrix, n_sims=None):
    """
    Run Monte Carlo simulation using a score probability matrix.

    Args:
        score_matrix: np.array of shape (MAX_GOALS, MAX_GOALS) with P(h,a)
        n_sims: number of simulations

    Returns: dict with percentages and top scorelines
    """
    n_sims = n_sims or NUM_SIMULATIONS
    flat = score_matrix.ravel()
    flat = flat / flat.sum()

    # Vectorized sampling
    indices = np.random.choice(len(flat), size=n_sims, p=flat)
    home_goals = indices // score_matrix.shape[1]
    away_goals = indices % score_matrix.shape[1]

    win_a = np.sum(home_goals > away_goals)
    draw = np.sum(home_goals == away_goals)
    win_b = np.sum(home_goals < away_goals)
    total_goals = home_goals + away_goals

    # Score frequencies
    score_counts = collections.Counter(zip(home_goals.tolist(),
                                           away_goals.tolist()))
    top_scores = score_counts.most_common(10)

    over_2_5 = np.sum(total_goals > 2)
    btts = np.sum((home_goals > 0) & (away_goals > 0))

    raw_over_pct = over_2_5 / n_sims * 100
    expected_total = float(np.mean(total_goals))
    
    # Empirical recalibration for Over 2.5
    # Poisson/NegBin matrices notoriously underestimate variance in football,
    # leading to muted Over 2.5 probabilities. We stretch the probability to 
    # make it sharper, and add a small bump near the 2.5 threshold.
    centered = raw_over_pct - 50.0
    sharpened = 50.0 + (centered * 1.35)  # 35% more decisive
    bump = 5.0 * np.exp(-0.5 * ((expected_total - 2.5) / 0.5)**2)
    calibrated_over = max(5.0, min(95.0, sharpened + bump))

    return {
        "n_sims": n_sims,
        "win_a_pct": win_a / n_sims * 100,
        "draw_pct": draw / n_sims * 100,
        "win_b_pct": win_b / n_sims * 100,
        "top_scores": top_scores,
        "avg_goals_a": float(np.mean(home_goals)),
        "avg_goals_b": float(np.mean(away_goals)),
        "over_2_5_pct": calibrated_over,
        "under_2_5_pct": 100.0 - calibrated_over,
        "btts_yes_pct": btts / n_sims * 100,
        "btts_no_pct": (n_sims - btts) / n_sims * 100,
    }
