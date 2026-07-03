"""
Calibration Report — Compare default vs learned weights.

Usage:
    python -m futpredict.calibration_report

Outputs a comparison table showing parameter differences and
Brier Score / O/U LogLoss improvements.
"""
import os
import sys
import sqlite3
import numpy as np
from dataclasses import asdict

# Prevent OpenMP deadlock
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from futpredict.config import DB_PATH
from futpredict.weight_optimizer import (
    LearnedWeights, get_weights, reset_weights_cache,
    _evaluate_fold, CV_FOLDS, LEARNED_WEIGHTS_PATH,
)


def run_report():
    """Generate a before/after comparison report."""
    conn = sqlite3.connect(DB_PATH)

    print("╔════════════════════════════════════════════════════════════╗")
    print("║          CALIBRATION REPORT — Default vs Learned          ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    # Load learned weights (or defaults)
    if os.path.exists(LEARNED_WEIGHTS_PATH):
        learned = LearnedWeights.load()
        has_learned = True
        print(f"   ✓  Found learned weights (calibrated at {learned.optimized_at})")
    else:
        learned = LearnedWeights()
        has_learned = False
        print("   ⚠  No learned weights found — comparing defaults only")
        print("      Run `python predict.py --calibrate` first.\n")

    defaults = LearnedWeights()

    # Evaluate both weight sets on the same CV folds
    print("\n   ⚙  Evaluating Default weights on temporal CV folds...")
    default_briers = []
    default_loglosses = []
    for _, _, val_start, val_end in CV_FOLDS:
        b, l = _evaluate_fold(conn, defaults, None, val_start, val_end)
        default_briers.append(b)
        default_loglosses.append(l)
        fold_label = f"{val_start[:4]}"
        print(f"      Fold {fold_label}: Brier={b:.4f}, LogLoss={l:.4f}")

    default_mean_brier = np.mean(default_briers)
    default_mean_ll = np.mean(default_loglosses)
    print(f"      Mean:    Brier={default_mean_brier:.4f}, LogLoss={default_mean_ll:.4f}")

    if has_learned:
        print("\n   ⚙  Evaluating Learned weights on temporal CV folds...")
        learned_briers = []
        learned_loglosses = []
        for _, _, val_start, val_end in CV_FOLDS:
            b, l = _evaluate_fold(conn, learned, None, val_start, val_end)
            learned_briers.append(b)
            learned_loglosses.append(l)
            fold_label = f"{val_start[:4]}"
            print(f"      Fold {fold_label}: Brier={b:.4f}, LogLoss={l:.4f}")

        learned_mean_brier = np.mean(learned_briers)
        learned_mean_ll = np.mean(learned_loglosses)
        print(f"      Mean:    Brier={learned_mean_brier:.4f}, LogLoss={learned_mean_ll:.4f}")
    else:
        learned_mean_brier = default_mean_brier
        learned_mean_ll = default_mean_ll

    # Print parameter comparison
    skip_keys = {"brier_score", "ou_logloss", "optimized_at", "n_trials"}
    d_dict = asdict(defaults)
    l_dict = asdict(learned)

    print(f"\n   ┌{'─'*58}┐")
    print(f"   │ {'Parameter':<24} │ {'Default':>9} │ {'Learned':>9} │ {'Δ':>7} │")
    print(f"   ├{'─'*58}┤")

    for key in d_dict:
        if key in skip_keys:
            continue
        d_val = d_dict[key]
        l_val = l_dict[key]
        if isinstance(d_val, (int, float)):
            pct = ((l_val - d_val) / d_val * 100) if d_val != 0 else 0
            sign = "+" if pct > 0 else ""
            print(f"   │ {key:<24} │ {d_val:>9.2f} │ {l_val:>9.2f} │ {sign}{pct:>5.1f}% │")

    print(f"   ├{'─'*58}┤")

    # Metrics comparison
    b_pct = ((learned_mean_brier - default_mean_brier) / default_mean_brier * 100)
    l_pct = ((learned_mean_ll - default_mean_ll) / default_mean_ll * 100)
    b_sign = "+" if b_pct > 0 else ""
    l_sign = "+" if l_pct > 0 else ""

    print(f"   │ {'Brier Score (1X2)':<24} │ {default_mean_brier:>9.4f} │ {learned_mean_brier:>9.4f} │ {b_sign}{b_pct:>5.1f}% │")
    print(f"   │ {'O/U LogLoss':<24} │ {default_mean_ll:>9.4f} │ {learned_mean_ll:>9.4f} │ {l_sign}{l_pct:>5.1f}% │")
    print(f"   └{'─'*58}┘")

    if has_learned:
        if learned_mean_brier < default_mean_brier:
            print(f"\n   ✓  Learned weights IMPROVED Brier Score by {abs(b_pct):.1f}%")
        elif learned_mean_brier > default_mean_brier:
            print(f"\n   ⚠  Learned weights REGRESSED Brier Score by {abs(b_pct):.1f}%")
        else:
            print(f"\n   ─  Brier Score unchanged")

        if learned_mean_ll < default_mean_ll:
            print(f"   ✓  Learned weights IMPROVED O/U LogLoss by {abs(l_pct):.1f}%")
        elif learned_mean_ll > default_mean_ll:
            print(f"   ⚠  Learned weights REGRESSED O/U LogLoss by {abs(l_pct):.1f}%")
        else:
            print(f"   ─  O/U LogLoss unchanged")

    conn.close()


if __name__ == "__main__":
    run_report()
