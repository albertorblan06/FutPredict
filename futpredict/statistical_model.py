"""
Dixon-Coles model with Negative Binomial marginals.

The proper statistical model for football score prediction:
- Attack/defense parameters per team (fitted via MLE)
- ρ correlation parameter for low-scoring outcomes
- Negative Binomial instead of Poisson (handles overdispersion)
- Outputs a full 9×9 score probability matrix
"""
import os
import json
import datetime
import math
import numpy as np
from scipy.stats import nbinom
from scipy.special import gammaln
from scipy.optimize import minimize
from .config import DC_TRAIN_YEARS, MAX_GOALS
from .analysis import get_tournament_weight
from .weight_optimizer import get_weights


def _dc_tau(h, a, lambda_h, lambda_a, rho):
    """
    Dixon-Coles correction factor for low-scoring outcomes.
    Adjusts P(0-0), P(1-0), P(0-1), P(1-1) to match observed frequencies.
    """
    if h == 0 and a == 0:
        return 1 - lambda_h * lambda_a * rho
    elif h == 0 and a == 1:
        return 1 + lambda_h * rho
    elif h == 1 and a == 0:
        return 1 + lambda_a * rho
    elif h == 1 and a == 1:
        return 1 - rho
    return 1.0


def _negbin_pmf(k, mu, alpha):
    """
    Negative Binomial PMF parameterized by mean (mu) and dispersion (alpha).

    Variance = mu + alpha * mu^2
    When alpha → 0, this reduces to Poisson.

    Maps to scipy's nbinom(n, p) where:
        n = 1/alpha
        p = 1/(1 + alpha*mu)
    """
    if alpha < 1e-6:
        # Reduce to Poisson for near-zero dispersion
        return np.exp(-mu) * mu**k / math.factorial(int(k)) if mu > 0 else (1.0 if k == 0 else 0.0)

    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return nbinom.pmf(int(k), n, p)


def _fast_negbin_log_pmf(k, mu, alpha):
    """Vectorized Negative Binomial log-PMF."""
    if alpha < 1e-6:
        # Poisson log-pmf: k * log(mu) - mu - log(k!)
        return k * np.log(mu) - mu - gammaln(k + 1)
        
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return gammaln(k + n) - gammaln(n) - gammaln(k + 1) + n * np.log(p) + k * np.log(1 - p)


def _dc_tau_vectorized(h, a, lambda_h, lambda_a, rho):
    """Vectorized Dixon-Coles correction factor."""
    tau = np.ones_like(lambda_h)
    m00 = (h == 0) & (a == 0)
    m01 = (h == 0) & (a == 1)
    m10 = (h == 1) & (a == 0)
    m11 = (h == 1) & (a == 1)
    
    tau[m00] = 1 - lambda_h[m00] * lambda_a[m00] * rho
    tau[m01] = 1 + lambda_h[m01] * rho
    tau[m10] = 1 + lambda_a[m10] * rho
    tau[m11] = 1 - rho
    
    return np.clip(tau, 1e-8, None)


def build_score_matrix(lambda_h, lambda_a, alpha_h, alpha_a, rho,
                       max_goals=None):
    """
    Build a score probability matrix using bivariate Negative Binomial
    with Dixon-Coles correction.

    Returns: np.array of shape (max_goals, max_goals)
    """
    mg = max_goals or MAX_GOALS
    matrix = np.zeros((mg, mg))

    for h in range(mg):
        for a in range(mg):
            p_h = _negbin_pmf(h, lambda_h, alpha_h)
            p_a = _negbin_pmf(a, lambda_a, alpha_a)
            tau = _dc_tau(h, a, lambda_h, lambda_a, rho)
            matrix[h, a] = max(0, tau * p_h * p_a)

    # Normalize
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


def _neg_log_likelihood(params, home_idx, away_idx, home_goals, away_goals, weights, n_teams):
    """
    Vectorized negative log-likelihood for the Dixon-Coles model.
    """
    attack = np.exp(params[:n_teams])
    defense = np.exp(params[n_teams:2*n_teams])
    home_adv = np.exp(params[2*n_teams])
    rho = np.tanh(params[2*n_teams+1]) * 0.1  # constrain to [-0.1, 0.1]
    alpha = np.exp(params[2*n_teams+2])

    lambda_h = attack[home_idx] * defense[away_idx] * home_adv
    lambda_a = attack[away_idx] * defense[home_idx]

    lambda_h = np.clip(lambda_h, 0.1, 5.0)
    lambda_a = np.clip(lambda_a, 0.1, 5.0)

    log_p_h = _fast_negbin_log_pmf(home_goals, lambda_h, alpha)
    log_p_a = _fast_negbin_log_pmf(away_goals, lambda_a, alpha)
    tau = _dc_tau_vectorized(home_goals, away_goals, lambda_h, lambda_a, rho)

    log_prob = log_p_h + log_p_a + np.log(tau)
    ll = np.sum(weights * log_prob)

    # Regularization: pull attack/defense toward 1.0
    reg = 0.001 * (np.sum((np.log(attack))**2) + np.sum((np.log(defense))**2))
    return -ll + reg


def fit_dixon_coles(conn, force=False, years_back=None, reference_date=None,
                    weights=None):
    """
    Fit the Dixon-Coles Negative Binomial model to recent match data.
    
    Args:
        reference_date: If provided, only use matches before this date and
                        compute time-decay relative to it (prevents data leakage).
        weights: Optional LearnedWeights for tournament weights and decay.
    """
    cache_path = "data/dc_model.json"
    if not force and reference_date is None and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            print(f"   ✓  Loaded cached Dixon-Coles model (NLL={data.get('nll', 0):.1f})")
            return data
        except Exception as e:
            print(f"   ⚠  Cache load failed ({e}), retraining...")

    ref = datetime.date.fromisoformat(reference_date) if reference_date else datetime.date.today()
    years_back = years_back or DC_TRAIN_YEARS
    cutoff = (ref - datetime.timedelta(days=years_back * 365)).isoformat()

    date_upper = reference_date or '9999-12-31'
    cur = conn.execute("""
        SELECT home_team, away_team, home_score, away_score,
               tournament, date, neutral
        FROM matches
        WHERE date >= ? AND date < ? AND home_score IS NOT NULL
        ORDER BY date
    """, (cutoff, date_upper))

    rows = cur.fetchall()
    if len(rows) < 500:
        return None

    # Build team index
    teams = set()
    for home, away, *_ in rows:
        teams.add(home)
        teams.add(away)
    teams = sorted(teams)
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # Prepare match data with time-decay weighting
    weights = weights or get_weights()
    decay_lambda = math.log(2) / weights.dc_decay_half_life

    matches = []
    for home, away, hs, as_, tournament, date_str, neutral in rows:
        try:
            match_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue
        days_ago = (ref - match_date).days
        time_w = math.exp(-decay_lambda * days_ago)
        tourn_w = get_tournament_weight(tournament, weights=weights)
        weight = time_w * tourn_w
        matches.append((home, away, int(hs), int(as_), weight))

    if len(matches) < 500:
        return None

    print(f"   ⚙  Fitting Dixon-Coles NegBin on {len(matches):,} matches "
          f"({n_teams} teams)...")

    # Vectorize inputs
    home_idx = np.array([team_idx[m[0]] for m in matches])
    away_idx = np.array([team_idx[m[1]] for m in matches])
    home_goals = np.array([m[2] for m in matches])
    away_goals = np.array([m[3] for m in matches])
    weights_arr = np.array([m[4] for m in matches])

    # Initial parameters
    init_attack = np.zeros(n_teams)      # log(1.0) = 0
    init_defense = np.zeros(n_teams)
    init_home = np.log(1.1)              # ~10% home advantage
    init_rho = 0.0
    init_alpha = np.log(0.05)            # small dispersion (close to Poisson)

    x0 = np.concatenate([init_attack, init_defense,
                         [init_home, init_rho, init_alpha]])

    # Constraint: sum of attacks = n_teams (identifiability)
    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(home_idx, away_idx, home_goals, away_goals, weights_arr, n_teams),
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-6},
    )

    if not result.success:
        print(f"   ⚠  Optimizer warning: {result.message}")

    # Extract fitted parameters
    attack = np.exp(result.x[:n_teams])
    defense = np.exp(result.x[n_teams:2*n_teams])
    home_adv = np.exp(result.x[2*n_teams])
    rho = np.tanh(result.x[2*n_teams+1]) * 0.1
    alpha = np.exp(result.x[2*n_teams+2])

    # Normalize: mean attack = 1.0
    mean_att = attack.mean()
    attack /= mean_att
    defense *= mean_att

    nll = result.fun
    print(f"   ✓  Fitted! NLL={nll:.1f}, ρ={rho:.4f}, α={alpha:.4f}, "
          f"HA={home_adv:.3f}")

    res_dict = {
        "attack": {t: float(attack[i]) for t, i in team_idx.items()},
        "defense": {t: float(defense[i]) for t, i in team_idx.items()},
        "home_adv": float(home_adv),
        "rho": float(rho),
        "alpha": float(alpha),
        "n_teams": n_teams,
        "n_matches": len(matches),
        "nll": float(nll),
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(res_dict, f, indent=2)

    return res_dict


def predict_dc(dc_params, team_a, team_b, venue="neutral"):
    """
    Generate a score probability matrix for a specific matchup
    using fitted Dixon-Coles parameters.

    Returns: np.array of shape (MAX_GOALS, MAX_GOALS)
    """
    attack = dc_params["attack"]
    defense = dc_params["defense"]
    ha = dc_params["home_adv"]
    rho = dc_params["rho"]
    alpha = dc_params["alpha"]

    # Default values for unknown teams
    default_att = np.median(list(attack.values()))
    default_def = np.median(list(defense.values()))

    att_a = attack.get(team_a, default_att)
    def_a = defense.get(team_a, default_def)
    att_b = attack.get(team_b, default_att)
    def_b = defense.get(team_b, default_def)

    if venue == "home_a":
        lambda_h = att_a * def_b * ha
        lambda_a = att_b * def_a
    elif venue == "home_b":
        lambda_h = att_a * def_b
        lambda_a = att_b * def_a * ha
    else:
        lambda_h = att_a * def_b
        lambda_a = att_b * def_a

    lambda_h = max(0.2, min(5.0, lambda_h))
    lambda_a = max(0.2, min(5.0, lambda_a))

    return build_score_matrix(lambda_h, lambda_a, alpha, alpha, rho), {
        "lambda_h": lambda_h,
        "lambda_a": lambda_a,
        "att_a": att_a, "def_a": def_a,
        "att_b": att_b, "def_b": def_b,
        "rho": float(rho),
        "alpha": float(alpha),
    }
