"""
Optimal blending of score probability matrices.
Finds mathematically optimal weights via log-likelihood maximization,
instead of arbitrary ALPHA_FORM, ALPHA_RANK, etc.
"""
import numpy as np
from scipy.optimize import minimize_scalar, minimize


def blend_matrices(matrices, weights=None):
    """
    Blend multiple score probability matrices with given weights.

    Args:
        matrices: list of np.arrays (score matrices from different models)
        weights: list of floats (must sum to 1.0). If None, equal weights.

    Returns: blended score matrix (normalized)
    """
    if not matrices:
        raise ValueError("Need at least one matrix")
    if len(matrices) == 1:
        return matrices[0]

    if weights is None:
        weights = [1.0 / len(matrices)] * len(matrices)

    blended = np.zeros_like(matrices[0])
    for w, m in zip(weights, matrices):
        blended += w * m

    total = blended.sum()
    if total > 0:
        blended /= total
    return blended


from scipy.optimize import minimize

def optimize_blend_weights(matrices, actual_scores):
    """
    Find optimal blend weights by maximizing log-likelihood on validation data.

    Args:
        matrices: list of score matrices (from different models)
        actual_scores: list of (home_goals, away_goals) tuples

    Returns: list of optimal weights
    """
    if not matrices:
        return []
    if len(matrices) == 1:
        return [1.0]
    if not actual_scores:
        # Fallback to equal weights if no validation data is available
        return [1.0 / len(matrices)] * len(matrices)

    def neg_ll(w):
        blended = np.zeros_like(matrices[0])
        for weight, m in zip(w, matrices):
            blended += weight * m
            
        blended = np.clip(blended, 1e-15, None)
        blended /= blended.sum()
        
        ll = 0.0
        for h, a in actual_scores:
            if h < blended.shape[0] and a < blended.shape[1]:
                ll += np.log(blended[h, a])
            else:
                ll += np.log(1e-15)
        return -ll

    # Constraints: weights must sum to 1, and be between 0 and 1
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 1.0) for _ in matrices]
    init_w = np.array([1.0 / len(matrices)] * len(matrices))

    result = minimize(neg_ll, init_w, method='SLSQP', bounds=bounds, constraints=cons)
    return result.x.tolist()


def matrix_to_outcomes(score_matrix):
    """
    Extract win/draw/loss percentages and other stats from a score matrix.

    Returns dict with win_a_pct, draw_pct, win_b_pct, etc.
    """
    mg = score_matrix.shape[0]
    win_a = draw = win_b = 0.0

    for h in range(mg):
        for a in range(mg):
            p = score_matrix[h, a]
            if h > a:
                win_a += p
            elif h == a:
                draw += p
            else:
                win_b += p

    total = win_a + draw + win_b
    if total > 0:
        win_a /= total
        draw /= total
        win_b /= total

    # Top scorelines
    flat = []
    for h in range(mg):
        for a in range(mg):
            flat.append(((h, a), score_matrix[h, a]))
    flat.sort(key=lambda x: -x[1])
    top_scores = flat[:10]

    # Expected goals
    avg_h = sum(h * score_matrix[h, :].sum() for h in range(mg))
    avg_a = sum(a * score_matrix[:, a].sum() for a in range(mg))

    # Over/under 2.5
    over_2_5 = sum(score_matrix[h, a] for h in range(mg) for a in range(mg) if h + a > 2)
    btts = sum(score_matrix[h, a] for h in range(1, mg) for a in range(1, mg))

    return {
        "win_a_pct": win_a * 100,
        "draw_pct": draw * 100,
        "win_b_pct": win_b * 100,
        "top_scores": [(s, p * 100) for s, p in top_scores],
        "avg_goals_a": avg_h,
        "avg_goals_b": avg_a,
        "over_2_5_pct": over_2_5 * 100,
        "under_2_5_pct": (1 - over_2_5) * 100,
        "btts_yes_pct": btts * 100,
        "btts_no_pct": (1 - btts) * 100,
    }
