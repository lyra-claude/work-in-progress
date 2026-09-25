#!/usr/bin/env python3
"""Error-correlation n_eff for the 32-judge panel on the two 3-label ChaosNLI
datasets (MNLI-m and SNLI).

Direct extension of error_corr_neff_alphanli.py. Construction is IDENTICAL: the
per-judge error object is the BINARY indicator e_ij = 1 if judge i's vote != gold_j
else 0 (works identically for 3-label datasets -- a 3-way label is still reduced to
correct/incorrect vs gold). One unit-diagonal Pearson ERROR-correlation matrix C_err
is built from the mean-centered binary error vectors, and the same estimators are
computed on it.

ESTIMAND (same object as formulas.panel_metrics / the alphaNLI script):
  e_ij  = 1 if vote_ij != gold_j else 0        (binary error vs gold)
  C_err = k x k Pearson correlation of the k mean-centered binary error vectors,
          diagonal forced to exactly 1.0 (trace = k).

Gold = majority_label (via votes_io.load_panel, gold_field="majority_label"),
identical to the alphaNLI script.

Complete-case: items where ANY covered judge has parse_fail=True are dropped
(failure_policy="drop-items").

Usage:
    .venv/bin/python error_corr_neff_3label.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT / "repo"
sys.path.insert(0, str(REPO / "src"))

import votes_io  # noqa: E402  (repo src)

DATASETS = {
    "mnli_m": PROJECT / "chaosNLI_v1.0" / "chaosNLI_mnli_m.jsonl",
    "snli": PROJECT / "chaosNLI_v1.0" / "chaosNLI_snli.jsonl",
}


def build_error_correlation(dataset, human_path):
    """Return (C_err, n_items, k, panel). Identical construction to the alphaNLI script."""
    panel = votes_io.load_panel(REPO, dataset, human_path, failure_policy="drop-items")
    idx = panel.idx          # (k, n) integer label indices
    gold = panel.gold        # (n,) integer gold labels
    k, n = idx.shape

    # Binary error (residual vs gold), mean-centered per judge -- identical object.
    binary = (idx != gold[None, :]).astype(float)
    binary -= binary.mean(axis=1, keepdims=True)
    variance = np.sum(binary ** 2, axis=1)
    if np.any(variance <= 0.0):
        raise ValueError("a judge has zero error variance; Pearson correlation undefined")

    C_err = (binary @ binary.T) / np.sqrt(np.outer(variance, variance))
    np.fill_diagonal(C_err, 1.0)  # force exact unit diagonal (trace = k)
    return C_err, n, k, panel


def compute(C_err):
    k = C_err.shape[0]
    triu = np.triu_indices(k, 1)
    off = C_err[triu]
    rho_bar = float(off.mean())
    rho_std = float(off.std())

    eig = np.linalg.eigvalsh(C_err)
    eig = np.clip(eig, 0.0, None)
    sum_l = float(eig.sum())
    sum_l2 = float(np.sum(eig ** 2))

    n_eff_kish = k / (1.0 + (k - 1) * rho_bar)
    n_eff_pr = sum_l ** 2 / sum_l2

    # Analytic compound symmetry with the actual k.
    lam_top = 1.0 + (k - 1) * rho_bar
    lam_rest = 1.0 - rho_bar
    n_eff_pr_cs = k ** 2 / (lam_top ** 2 + (k - 1) * lam_rest ** 2)

    total_gap = n_eff_pr - n_eff_kish
    formula_part = n_eff_pr_cs - n_eff_kish
    cs_departure = n_eff_pr - n_eff_pr_cs

    def pct(x):
        return float(100.0 * x / total_gap) if total_gap != 0 else None

    return {
        "k": int(k),
        "rho_bar": rho_bar,
        "std_offdiag": rho_std,
        "min_offdiag": float(off.min()),
        "max_offdiag": float(off.max()),
        "trace": float(np.trace(C_err)),
        "sum_lambda": sum_l,
        "sum_lambda2": sum_l2,
        "lambda_max": float(eig.max()),
        "lambda_min": float(eig.min()),
        "n_eff_Kish": float(n_eff_kish),
        "n_eff_PR": float(n_eff_pr),
        "n_eff_PR_CS": float(n_eff_pr_cs),
        "total_gap": float(total_gap),
        "formula_part": float(formula_part),
        "cs_departure": float(cs_departure),
        "total_gap_pct": 100.0,
        "formula_part_pct": pct(formula_part),
        "cs_departure_pct": pct(cs_departure),
    }


def main():
    results = {
        "matrix_definition": (
            "C_err = k x k Pearson correlation of per-judge BINARY error vectors "
            "e_ij = (vote_ij != gold_j), mean-centered per judge, unit diagonal (trace=k). "
            "Identical error object to error_corr_neff_alphanli.py / formulas.panel_metrics; "
            "3-way labels reduced to binary correct/incorrect vs gold."
        ),
        "failure_policy": "drop-items (complete-case: drop any item with any judge parse_fail)",
        "gold_source": "majority_label (via votes_io.load_panel, gold_field=majority_label)",
        "datasets": {},
    }

    for dataset, human_path in DATASETS.items():
        C_err, n, k, panel = build_error_correlation(dataset, human_path)
        m = compute(C_err)

        print("=" * 66)
        print(f"ERROR-correlation n_eff  --  {dataset} {k}-judge panel")
        print("=" * 66)
        print(f"[coverage] n (complete-case items)   = {n}")
        print(f"           n_original_items          = {panel.provenance['n_original_items']}")
        print(f"           dropped_items (parse_fail)= {panel.provenance['dropped_items']}")
        print(f"           k (judges with votes)     = {k}")
        print(f"           rho_bar (mean off-diag)   = {m['rho_bar']:.4f}")
        print(f"           std(off-diag)             = {m['std_offdiag']:.4f}")
        print(f"           n_eff_Kish                = {m['n_eff_Kish']:.4f}")
        print(f"           n_eff_PR                  = {m['n_eff_PR']:.4f}")
        print(f"           n_eff_PR_CS               = {m['n_eff_PR_CS']:.4f}")
        print(f"           total_gap  (PR - Kish)    = {m['total_gap']:.4f}  (100.00%)")
        print(f"           formula_part (PR_CS-Kish) = {m['formula_part']:.4f}  ({m['formula_part_pct']:.2f}%)")
        print(f"           cs_departure (PR - PR_CS) = {m['cs_departure']:.4f}  ({m['cs_departure_pct']:.2f}%)")
        print()

        results["datasets"][dataset] = {
            "n_complete_case_items": int(n),
            "n_original_items": panel.provenance["n_original_items"],
            "dropped_items": panel.provenance["dropped_items"],
            "parse_fail_cells": panel.provenance["parse_fail_cells"],
            "parse_fail_items": panel.provenance["parse_fail_items"],
            "k": int(k),
            "rho_bar": round(m["rho_bar"], 4),
            "std_offdiag": round(m["std_offdiag"], 4),
            "min_offdiag": round(m["min_offdiag"], 4),
            "max_offdiag": round(m["max_offdiag"], 4),
            "trace": round(m["trace"], 4),
            "sum_lambda": round(m["sum_lambda"], 4),
            "sum_lambda2": round(m["sum_lambda2"], 4),
            "lambda_max": round(m["lambda_max"], 4),
            "lambda_min": round(m["lambda_min"], 4),
            "n_eff_Kish": round(m["n_eff_Kish"], 4),
            "n_eff_PR": round(m["n_eff_PR"], 4),
            "n_eff_PR_CS": round(m["n_eff_PR_CS"], 4),
            "total_gap": round(m["total_gap"], 4),
            "formula_part": round(m["formula_part"], 4),
            "cs_departure": round(m["cs_departure"], 4),
            "total_gap_pct": 100.0,
            "formula_part_pct": round(m["formula_part_pct"], 2),
            "cs_departure_pct": round(m["cs_departure_pct"], 2),
        }

    out = PROJECT / "results-error-corr-neff-3label.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
