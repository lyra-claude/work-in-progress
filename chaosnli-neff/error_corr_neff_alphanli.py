#!/usr/bin/env python3
"""Error-correlation n_eff for the 32-judge alphaNLI panel (Tasks A-E).

Builds ONE unit-diagonal Pearson ERROR-correlation matrix C_err from the per-judge
binary error (residual-vs-gold) vectors, then computes two effective-sample-size
estimators on it (Kish plug-in, participation ratio), an analytic compound-symmetric
control, and a decomposition of the Kish->PR gap. Finally cross-checks against the
paper's own alphaNLI values and states the normalization relationship.

ESTIMAND DEFINITIONS (all Tasks A-D use the SAME matrix C_err):
  error vector for judge i, item j:  e_ij = 1 if vote_ij != gold_j else 0
      (this is the paper's own error object: binary = (idx != gold), as in
       formulas.panel_metrics)
  C_err = 32x32 Pearson CORRELATION matrix of the 32 mean-centered error vectors,
          with the diagonal forced to exactly 1.0 (trace = 32).

Complete-case: items where ANY of the 32 judges has parse_fail=True are dropped
(votes_io.load_panel failure_policy="drop-items").

Usage:
    .venv/bin/python error_corr_neff_alphanli.py
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

DATASET = "alphanli"
CHAOSNLI_FILE = PROJECT / "chaosNLI_v1.0" / "chaosNLI_alphanli.jsonl"
K = 32


def build_error_correlation():
    """Return (C_err, n_items, gold, idx, panel).

    C_err is the 32x32 unit-diagonal Pearson correlation matrix of the per-judge
    binary error vectors e_ij = (vote_ij != gold_j), built exactly as
    formulas.panel_metrics builds its n_eff object.
    """
    panel = votes_io.load_panel(REPO, DATASET, CHAOSNLI_FILE, failure_policy="drop-items")
    idx = panel.idx          # (32, n) integer label indices
    gold = panel.gold        # (n,) integer gold labels
    k, n = idx.shape
    assert k == K, f"expected {K} judges, got {k}"

    # Binary error (residual vs gold), mean-centered per judge -- identical to
    # formulas.panel_metrics: binary = (idx != gold); binary -= mean; phi via inner products.
    binary = (idx != gold[None, :]).astype(float)
    binary -= binary.mean(axis=1, keepdims=True)
    variance = np.sum(binary ** 2, axis=1)
    if np.any(variance <= 0.0):
        raise ValueError("a judge has zero error variance; Pearson correlation undefined")

    C_err = (binary @ binary.T) / np.sqrt(np.outer(variance, variance))
    np.fill_diagonal(C_err, 1.0)  # force exact unit diagonal (trace = 32)
    return C_err, n, gold, idx, panel


def task_a(C_err):
    k = C_err.shape[0]
    triu = np.triu_indices(k, 1)
    off = C_err[triu]
    rho_bar = float(off.mean())
    rho_std = float(off.std())
    eig = np.linalg.eigvalsh(C_err)          # ascending
    eig = np.clip(eig, 0.0, None)            # kill tiny negative numerical noise
    return {
        "rho_bar_err": rho_bar,
        "std_offdiag": rho_std,
        "min_offdiag": float(off.min()),
        "max_offdiag": float(off.max()),
        "eigenvalues_ascending": eig.tolist(),
        "sum_lambda": float(eig.sum()),
        "sum_lambda2": float(np.sum(eig ** 2)),
        "trace": float(np.trace(C_err)),
    }


def task_b(C_err, rho_bar, eig):
    k = C_err.shape[0]
    n_eff_kish = k / (1.0 + (k - 1) * rho_bar)
    sum_l = np.sum(eig)
    sum_l2 = np.sum(eig ** 2)
    n_eff_pr = float(sum_l ** 2 / sum_l2)
    return {"n_eff_Kish": float(n_eff_kish), "n_eff_PR": n_eff_pr}


def task_c(rho_bar, k=K):
    # Exact compound-symmetric eigenvalues: one = 1+(k-1)rho, (k-1) = 1-rho.
    lam_top = 1.0 + (k - 1) * rho_bar
    lam_rest = 1.0 - rho_bar
    n_eff_pr_cs = k ** 2 / (lam_top ** 2 + (k - 1) * lam_rest ** 2)
    # Kish under CS: k/(1+(k-1)rho) -- identical to Task B Kish by construction.
    n_eff_kish_cs = k / (1.0 + (k - 1) * rho_bar)
    return {"n_eff_PR_CS": float(n_eff_pr_cs), "n_eff_Kish_CS": float(n_eff_kish_cs)}


def task_d(n_eff_pr, n_eff_kish, n_eff_pr_cs):
    total_gap = n_eff_pr - n_eff_kish
    formula_part = n_eff_pr_cs - n_eff_kish
    cs_departure = n_eff_pr - n_eff_pr_cs
    def pct(x):
        return float(100.0 * x / total_gap) if total_gap != 0 else None
    return {
        "total_gap": float(total_gap),
        "formula_part": float(formula_part),
        "cs_departure": float(cs_departure),
        "total_gap_pct": pct(total_gap),
        "formula_part_pct": pct(formula_part),
        "cs_departure_pct": pct(cs_departure),
    }


def task_e():
    ref = json.loads((REPO / "reference" / "reported_values.json").read_text(encoding="utf-8"))
    a = ref["datasets"]["alphanli"]
    return {
        "paper_n_eff": a["n_eff"],
        "paper_nu_H": a["nu_H"],
        "paper_PR": a["PR"],
        "paper_n_items": a["n_items"],
    }


def main():
    C_err, n_items, gold, idx, panel = build_error_correlation()

    A = task_a(C_err)
    eig = np.asarray(A["eigenvalues_ascending"])
    B = task_b(C_err, A["rho_bar_err"], eig)
    C = task_c(A["rho_bar_err"])
    D = task_d(B["n_eff_PR"], B["n_eff_Kish"], C["n_eff_PR_CS"])
    E = task_e()

    print("=" * 66)
    print("ERROR-correlation n_eff  --  alphaNLI 32-judge panel")
    print("Matrix: C_err = Pearson correlation of binary error vectors")
    print("        e_ij = (vote_ij != gold_j), unit diagonal, trace = 32")
    print("=" * 66)
    print(f"\n[A] n (complete-case items)          = {n_items}")
    print(f"    n_original_items                 = {panel.provenance['n_original_items']}")
    print(f"    dropped_items (parse_fail)       = {panel.provenance['dropped_items']}")
    print(f"    rho_bar_err (mean off-diagonal)  = {A['rho_bar_err']:.6f}")
    print(f"    std(off-diagonal)                = {A['std_offdiag']:.6f}")
    print(f"    off-diagonal range               = [{A['min_offdiag']:.6f}, {A['max_offdiag']:.6f}]")
    print(f"    trace(C_err)                     = {A['trace']:.6f}")
    print(f"    sum(lambda)                      = {A['sum_lambda']:.6f}")
    print(f"    sum(lambda^2)                    = {A['sum_lambda2']:.6f}")
    print(f"    lambda_max / lambda_min          = {eig.max():.6f} / {eig.min():.6f}")

    print(f"\n[B] n_eff_Kish = 32/(1+31*rho_bar)   = {B['n_eff_Kish']:.4f}")
    print(f"    n_eff_PR   = (Sum l)^2/Sum l^2   = {B['n_eff_PR']:.4f}")

    print(f"\n[C] n_eff_PR_CS (compound-sym)       = {C['n_eff_PR_CS']:.4f}")
    print(f"    n_eff_Kish_CS (== Task B Kish)   = {C['n_eff_Kish_CS']:.4f}")

    print(f"\n[D] total_gap  (PR - Kish)           = {D['total_gap']:.4f}  (100.00%)")
    print(f"    formula_part (PR_CS - Kish)      = {D['formula_part']:.4f}  ({D['formula_part_pct']:.2f}%)")
    print(f"    cs_departure (PR - PR_CS)        = {D['cs_departure']:.4f}  ({D['cs_departure_pct']:.2f}%)")

    print(f"\n[E] paper n_eff (alphaNLI)           = {E['paper_n_eff']:.6f}")
    print(f"    paper nu_H (alphaNLI)            = {E['paper_nu_H']:.6f}")
    print(f"    paper PR (alphaNLI)              = {E['paper_PR']:.6f}")

    results = {
        "dataset": DATASET,
        "matrix_definition": (
            "C_err = 32x32 Pearson correlation of per-judge binary error vectors "
            "e_ij = (vote_ij != gold_j), mean-centered per judge, unit diagonal (trace=32). "
            "Identical error object to formulas.panel_metrics n_eff."
        ),
        "failure_policy": "drop-items (complete-case: drop any item with any judge parse_fail)",
        "gold_source": "chaosNLI_alphanli.jsonl majority_label (via votes_io.load_panel)",
        "task_A": {
            "n_complete_case_items": n_items,
            "n_original_items": panel.provenance["n_original_items"],
            "dropped_items": panel.provenance["dropped_items"],
            **{k: A[k] for k in ("rho_bar_err", "std_offdiag", "min_offdiag",
                                 "max_offdiag", "trace", "sum_lambda", "sum_lambda2")},
            "lambda_max": float(eig.max()),
            "lambda_min": float(eig.min()),
            "eigenvalues_ascending": [round(x, 8) for x in A["eigenvalues_ascending"]],
        },
        "task_B": {
            "n_eff_Kish": round(B["n_eff_Kish"], 4),
            "n_eff_PR": round(B["n_eff_PR"], 4),
        },
        "task_C": {
            "n_eff_PR_CS": round(C["n_eff_PR_CS"], 4),
            "n_eff_Kish_CS": round(C["n_eff_Kish_CS"], 4),
            "note": "n_eff_Kish is identical under CS by construction (same formula, same rho_bar).",
        },
        "task_D": {
            "total_gap": round(D["total_gap"], 4),
            "formula_part": round(D["formula_part"], 4),
            "cs_departure": round(D["cs_departure"], 4),
            "total_gap_pct": 100.0,
            "formula_part_pct": round(D["formula_part_pct"], 2),
            "cs_departure_pct": round(D["cs_departure_pct"], 2),
        },
        "task_E": {
            "paper_n_eff": E["paper_n_eff"],
            "paper_nu_H": E["paper_nu_H"],
            "paper_PR": E["paper_PR"],
            "paper_n_items": E["paper_n_items"],
            "normalization_note": (
                "paper n_eff USES THE SAME error-correlation object as C_err: it is Kish on "
                "phi_bar = mean Pearson correlation of binary error vectors (idx != gold), "
                "unit diagonal. paper nu_H / PR do NOT: they are the spectral participation "
                "ratio of the RESIDUAL GRAM correlation (one-hot votes minus the human "
                "probability distribution, not error vs gold), a differently-constructed "
                "unit-diagonal matrix. So my n_eff_Kish is directly comparable to paper n_eff; "
                "my n_eff_PR is NOT the same object as paper nu_H/PR."
            ),
        },
    }
    out = PROJECT / "results-error-corr-neff.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
