#!/usr/bin/env python3
"""Spectral vs Kish plug-in n_eff for the 32-judge panel on alphaNLI (binary labels).

ESTIMAND NOTE: the n_eff values computed here are JUDGE-PANEL REDUNDANCY measures
— the effective number of independent judges — computed from the Pearson correlation
matrix of binary (0/1) vote vectors across items.  They are NOT:
  - D = inverse-Simpson label diversity (a within-item spread measure)
  - nu_H = human-equivalence from the paper's MSE calibration curve
  - the paper's n_eff (which uses Pearson error correlations against gold labels)
  - the paper's PR (which is computed on residual one-hot vectors minus human probs)

This script computes three forms of judge-panel n_eff from the RAW BINARY VOTE
correlation matrix R_{ij} = Pearson(judge_i_binary_votes, judge_j_binary_votes):
  - n_eff_Kish = k / (1 + (k-1) * rho_bar)  [equicorrelation plug-in]
  - n_eff_PR   = (sum lambda_i)^2 / sum(lambda_i^2)  [spectral participation ratio]
  - n_eff_CN   = 1 + (k-1) * (1 - Var(lambda) / k)  [Cheverud/Nyholt form]

The gap n_eff_PR vs n_eff_Kish reveals whether the panel violates compound symmetry.
Compound symmetry (all pairwise correlations equal) would make them coincide.

Dataset: alphaNLI only (binary labels {1, 2}).
Arm: baseline (votes/baseline/).
Parse-fail handling: COMPLETE CASE — items where ANY judge has parse_fail=True
are excluded. Only items with ALL 32 valid votes are used.

Usage:
    .venv/bin/python spectral_neff_alphanli.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent
REPO = PROJECT / "repo"
sys.path.insert(0, str(REPO / "src"))

import layout  # noqa: E402  (repo src)

DATASET = "alphanli"
ARM = "baseline"
LABELS = ("1", "2")


# ---------------------------------------------------------------------------
# Step 1: Load vote matrix — 32 judges × N_items, binary (0=label1, 1=label2)
# ---------------------------------------------------------------------------

def load_binary_vote_matrix() -> tuple[np.ndarray, list[str], list[str]]:
    """
    Returns:
        vote_matrix: shape (32, N_complete) — float 0.0 or 1.0
            row i = binary indicator for judge i (0 if vote=="1", 1 if vote=="2")
        judge_keys: list of 32 judge names (in file-sort order)
        item_uids: list of N_complete UID strings (complete-case only)

    Construction:
        - iterate over SORTED glob of *.jsonl files (one per judge)
        - for each file, read all 1000 lines into a dict uid -> (label, parse_fail)
        - after collecting all judges, drop any uid where ANY judge has parse_fail=True
        - encode: 0.0 for label "1", 1.0 for label "2"
    """
    votes_dir = layout.votes_dir(REPO, DATASET, ARM)
    uids_ordered = layout.items_file(REPO, DATASET).read_text(encoding="utf-8").split()
    uid_to_pos = {uid: i for i, uid in enumerate(uids_ordered)}
    n_total = len(uids_ordered)

    judge_files = sorted(votes_dir.glob("*.jsonl"))
    assert len(judge_files) == 32, f"Expected 32 judge files, found {len(judge_files)}"

    # raw_votes[judge_idx, item_idx] = label index (0 or 1) or -1 if parse_fail
    raw_votes = np.full((32, n_total), fill_value=-1, dtype=np.int8)
    judge_keys: list[str] = []

    label_to_idx = {"1": 0, "2": 1}

    for j_idx, jpath in enumerate(judge_files):
        judge_keys.append(jpath.stem)
        with open(jpath, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                uid = str(r["uid"])
                pos = uid_to_pos.get(uid)
                if pos is None:
                    continue
                if r.get("parse_fail", False):
                    # leave as -1
                    pass
                else:
                    lbl = str(r["label"])
                    if lbl in label_to_idx:
                        raw_votes[j_idx, pos] = label_to_idx[lbl]
                    # else leave as -1 (unknown label — treated as fail)

    # Complete-case: keep only items where ALL judges have a valid vote
    # A valid vote is 0 or 1 (i.e., raw_votes[:, pos] >= 0 for all judges)
    complete_mask = np.all(raw_votes >= 0, axis=0)  # shape (n_total,)
    n_complete = int(complete_mask.sum())

    vote_matrix = raw_votes[:, complete_mask].astype(float)  # (32, n_complete)
    item_uids_complete = [uids_ordered[i] for i, ok in enumerate(complete_mask) if ok]

    n_dropped = n_total - n_complete
    print(f"Total items in roster: {n_total}")
    print(f"Items with any parse_fail: {n_dropped}")
    print(f"Complete-case items (all 32 valid): {n_complete}")

    return vote_matrix, judge_keys, item_uids_complete


# ---------------------------------------------------------------------------
# Step 2: Pearson correlation matrix R (32 × 32)
# ---------------------------------------------------------------------------

def pearson_corr_matrix(X: np.ndarray) -> np.ndarray:
    """
    X: shape (k, n) — k judge binary-vote vectors over n items.
    Returns R: shape (k, k) — Pearson correlation matrix.

    np.corrcoef treats rows as variables. We want corr(judge_i, judge_j)
    over items, so we pass X directly (rows = judges).
    """
    R = np.corrcoef(X)  # (k, k)
    # Force exact 1.0 on diagonal (floating-point noise)
    np.fill_diagonal(R, 1.0)
    return R


# ---------------------------------------------------------------------------
# Step 3+4: Compute all n_eff quantities and diagnostics
# ---------------------------------------------------------------------------

def compute_neff(R: np.ndarray) -> dict:
    k = R.shape[0]
    assert R.shape == (k, k), "R must be square"

    # Off-diagonal indices
    triu_idx = np.triu_indices(k, k=1)

    off_diag = R[triu_idx]  # upper triangle only (symmetric)
    rho_bar = float(off_diag.mean())
    rho_std = float(off_diag.std())
    rho_min = float(off_diag.min())
    rho_max = float(off_diag.max())

    # --- (b) Kish equicorrelation plug-in ---
    n_eff_kish = float(k / (1.0 + (k - 1) * rho_bar))

    # --- (c) Spectral participation ratio ---
    # eigenvalues of a PSD matrix via eigvalsh (symmetric solver, more stable)
    eigenvalues = np.linalg.eigvalsh(R)  # ascending order
    # clip tiny negative numerical noise
    eigenvalues_clipped = np.clip(eigenvalues, 0.0, None)

    trace_R = float(np.trace(R))  # should be exactly k (unit diagonal)
    sum_lambda = float(eigenvalues_clipped.sum())
    sum_lambda2 = float(np.sum(eigenvalues_clipped ** 2))

    # Method 1: (sum lambda)^2 / sum(lambda^2)
    n_eff_pr_m1 = float(sum_lambda ** 2 / sum_lambda2)

    # Method 2: k^2 / ||R||_F^2   (because trace(R) = k and ||R||_F^2 = sum lambda_i^2)
    frob_sq = float(np.sum(R ** 2))  # Frobenius norm squared
    n_eff_pr_m2 = float(k ** 2 / frob_sq)

    # Sanity: the two should agree closely
    pr_methods_agree = abs(n_eff_pr_m1 - n_eff_pr_m2) < 1e-6

    # Use method 1 as the reported value (directly from eigenvalues)
    n_eff_pr = n_eff_pr_m1

    # --- (d) Cheverud/Nyholt ---
    # n_eff_CN = 1 + (k-1) * (1 - Var(lambda) / k)
    # Var(lambda) = population variance (divide by k, not k-1)
    var_lambda = float(np.var(eigenvalues_clipped))  # population variance
    n_eff_cn = float(1.0 + (k - 1) * (1.0 - var_lambda / k))

    lambda_max = float(eigenvalues_clipped.max())
    lambda_min = float(eigenvalues_clipped.min())

    return {
        "k": k,
        "rho_bar": rho_bar,
        "rho_std_offdiag": rho_std,
        "rho_min_offdiag": rho_min,
        "rho_max_offdiag": rho_max,
        "n_eff_kish": n_eff_kish,
        "n_eff_pr": n_eff_pr,
        "n_eff_pr_method2_frob": n_eff_pr_m2,
        "pr_methods_agree": pr_methods_agree,
        "n_eff_cn": n_eff_cn,
        "trace_R": trace_R,
        "sum_lambda": sum_lambda,
        "sum_lambda2_from_eig": sum_lambda2,
        "frob_sq": frob_sq,
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "var_lambda": var_lambda,
        "eigenvalues": eigenvalues_clipped.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Spectral vs Kish n_eff — alphaNLI 32-judge panel")
    print("Estimand: JUDGE-PANEL REDUNDANCY (not label diversity)")
    print("=" * 60)
    print()

    # Step 1: Load
    vote_matrix, judge_keys, item_uids = load_binary_vote_matrix()
    n_items = vote_matrix.shape[1]
    print(f"\nVote matrix shape: {vote_matrix.shape}  (judges × items)")
    print(f"Judges: {judge_keys}")

    # Step 2: Pearson correlation
    R = pearson_corr_matrix(vote_matrix)
    print(f"\nCorrelation matrix shape: {R.shape}")
    print(f"Diagonal check (all 1.0): {np.allclose(np.diag(R), 1.0)}")
    print(f"Symmetry check: {np.allclose(R, R.T)}")

    # Step 3+4: n_eff quantities
    m = compute_neff(R)

    print()
    print("--- Compound-symmetry diagnostics ---")
    print(f"  rho_bar (mean off-diagonal)   = {m['rho_bar']:.4f}")
    print(f"  std(off-diagonal entries)      = {m['rho_std_offdiag']:.4f}")
    print(f"  min(off-diagonal)             = {m['rho_min_offdiag']:.4f}")
    print(f"  max(off-diagonal)             = {m['rho_max_offdiag']:.4f}")
    print(f"  lambda_max                    = {m['lambda_max']:.4f}")
    print(f"  lambda_min                    = {m['lambda_min']:.4f}")
    print(f"  Var(lambda) [population]      = {m['var_lambda']:.4f}")
    print()
    print("--- Effective-number estimates ---")
    print(f"  n_eff_Kish  = {m['n_eff_kish']:.4f}   [k / (1 + (k-1)*rho_bar)]")
    print(f"  n_eff_PR    = {m['n_eff_pr']:.4f}   [spectral participation ratio, from eigenvalues]")
    print(f"  n_eff_PR    = {m['n_eff_pr_method2_frob']:.4f}   [same, via k^2/||R||_F^2 — sanity check]")
    print(f"  n_eff_CN    = {m['n_eff_cn']:.4f}   [Cheverud/Nyholt: 1 + (k-1)*(1-Var(lam)/k)]")
    print(f"  PR methods agree (<1e-6): {m['pr_methods_agree']}")
    print()
    gap = m['n_eff_pr'] - m['n_eff_kish']
    direction = "ABOVE" if gap > 0 else "BELOW"
    print(f"  n_eff_PR is {direction} n_eff_Kish by {abs(gap):.4f} ({abs(gap)/m['n_eff_kish']*100:.2f}%)")
    print()
    print("--- Trace/Frobenius sanity ---")
    print(f"  trace(R)                       = {m['trace_R']:.6f}  (should be {m['k']}.0)")
    print(f"  sum(eigenvalues)               = {m['sum_lambda']:.6f}  (should equal trace)")
    print(f"  sum(lambda_i^2) from eig       = {m['sum_lambda2_from_eig']:.6f}")
    print(f"  ||R||_F^2                      = {m['frob_sq']:.6f}  (should match above)")
    print()

    # -------------------------------------------------------------------
    # Reference: the paper's n_eff and PR for alphaNLI (different objects)
    # -------------------------------------------------------------------
    ref_path = REPO / "reference" / "reported_values.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    paper_neff = ref["datasets"]["alphanli"]["n_eff"]
    paper_pr = ref["datasets"]["alphanli"]["PR"]
    print("--- Paper's reported values for alphaNLI (DIFFERENT OBJECTS — for context only) ---")
    print(f"  paper n_eff (Kish on ERROR corr vs gold)  = {paper_neff:.4f}")
    print(f"  paper PR    (spectral of residual Gram)   = {paper_pr:.4f}")
    print("  NOTE: these are NOT the same estimand as the above; do not compare directly.")
    print()

    # -------------------------------------------------------------------
    # Save results JSON
    # -------------------------------------------------------------------
    results = {
        "dataset": DATASET,
        "arm": ARM,
        "labels": list(LABELS),
        "n_items_in_roster": 1000,
        "n_items_complete_case": n_items,
        "n_items_dropped_parse_fail": 1000 - n_items,
        "parse_fail_handling": "complete-case: items where ANY of the 32 judges has parse_fail=True are excluded",
        "vote_encoding": "0 if label=='1', 1 if label=='2'",
        "correlation_matrix_construction": (
            "R = np.corrcoef(vote_matrix) where vote_matrix is shape (32, n_items), "
            "rows are judges, columns are items; each row is the binary vote vector "
            "(0=label1, 1=label2) for that judge over the complete-case items"
        ),
        "judge_keys": judge_keys,
        "estimand_guard": (
            "n_eff here = JUDGE-PANEL REDUNDANCY (effective independent judges) "
            "computed from raw binary vote Pearson correlations. "
            "NOT D (label diversity), NOT nu_H (human equivalence), "
            "NOT the paper's n_eff (which uses error correlations against gold labels), "
            "NOT the paper's PR (which uses residual one-hot Gram)."
        ),
        "n_eff_kish": round(m["n_eff_kish"], 4),
        "n_eff_pr": round(m["n_eff_pr"], 4),
        "n_eff_pr_method2_frob": round(m["n_eff_pr_method2_frob"], 4),
        "n_eff_cn": round(m["n_eff_cn"], 4),
        "rho_bar": round(m["rho_bar"], 4),
        "rho_std_offdiag": round(m["rho_std_offdiag"], 4),
        "rho_min_offdiag": round(m["rho_min_offdiag"], 4),
        "rho_max_offdiag": round(m["rho_max_offdiag"], 4),
        "lambda_max": round(m["lambda_max"], 4),
        "lambda_min": round(m["lambda_min"], 4),
        "var_lambda": round(m["var_lambda"], 4),
        "n_eff_pr_vs_kish_gap": round(m["n_eff_pr"] - m["n_eff_kish"], 4),
        "n_eff_pr_vs_kish_direction": "ABOVE" if m["n_eff_pr"] >= m["n_eff_kish"] else "BELOW",
        "pr_methods_agree": m["pr_methods_agree"],
        "eigenvalues_sorted_ascending": [round(x, 6) for x in m["eigenvalues"]],
        "paper_reported_neff_alphanli": paper_neff,
        "paper_reported_pr_alphanli": paper_pr,
        "paper_estimand_note": (
            "paper n_eff = Kish(phi_bar) where phi_bar = mean Pearson error-correlation "
            "vs gold labels; paper PR = spectral PR of residual Gram (one-hot minus human probs). "
            "Both differ from this script's R which is built from raw binary votes only."
        ),
    }

    results_path = PROJECT / "results-spectral-neff.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(f"Results saved to: {results_path}")
    print("Done.")


if __name__ == "__main__":
    main()
