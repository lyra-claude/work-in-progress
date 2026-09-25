#!/usr/bin/env python3
"""Blind independent verification: error-correlation n_eff for 3-label ChaosNLI (MNLI-m, SNLI).
Recomputed from scratch. Does not read any prior-agent answer files."""
import json
import os
import numpy as np

BASE = "/home/lyra/projects/chaosnli-neff"
RAW = os.path.join(BASE, "chaosNLI_v1.0")
PANEL = os.path.join(BASE, "repo/panel/panel-chaosnli.json")

DATASETS = {
    "MNLI-m": {
        "raw": os.path.join(RAW, "chaosNLI_mnli_m.jsonl"),
        "votes": os.path.join(BASE, "repo/datasets/chaosnli-mnli-m-1000/votes/baseline"),
    },
    "SNLI": {
        "raw": os.path.join(RAW, "chaosNLI_snli.jsonl"),
        "votes": os.path.join(BASE, "repo/datasets/chaosnli-snli-1000/votes/baseline"),
    },
}


def load_judges():
    with open(PANEL) as f:
        p = json.load(f)
    return [j["judge_key"] for j in p["judges"]]


def load_gold(raw_path):
    gold = {}
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gold[r["uid"]] = r["majority_label"]  # string "e"/"n"/"c"
    return gold


def load_votes(votes_dir, judge):
    """Return {uid: label or None (parse_fail / missing)}."""
    path = os.path.join(votes_dir, judge + ".jsonl")
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("parse_fail", False):
                out[r["uid"]] = None
            else:
                out[r["uid"]] = r.get("label", None)
    return out


def analyze(name, cfg, judges):
    gold = load_gold(cfg["raw"])
    votes = {j: load_votes(cfg["votes"], j) for j in judges}
    k = len(judges)

    # Universe of items: those present in gold (all raw uids)
    uids = list(gold.keys())

    # Complete-case: keep item only if every judge has a non-None vote for it
    kept = []
    for u in uids:
        ok = True
        for j in judges:
            v = votes[j].get(u, None)
            if v is None:
                ok = False
                break
        if ok:
            kept.append(u)

    n = len(kept)
    # Build error matrix E: n x k, e=1 if vote != gold
    E = np.zeros((n, k), dtype=float)
    for a, u in enumerate(kept):
        g = gold[u]
        for b, j in enumerate(judges):
            E[a, b] = 0.0 if votes[j][u] == g else 1.0

    # Pearson correlation matrix of columns (binary error vectors)
    C = np.corrcoef(E, rowvar=False)  # k x k, unit diagonal
    # guard: any column with zero variance -> corrcoef gives nan; report if so
    nan_cols = np.where(np.isnan(np.diag(C)))[0]

    # off-diagonal
    off_mask = ~np.eye(k, dtype=bool)
    off = C[off_mask]
    rho_bar = float(np.nanmean(off))
    off_std = float(np.nanstd(off))

    # eigenvalues
    lam = np.linalg.eigvalsh(C)

    n_eff_kish = k / (1 + (k - 1) * rho_bar)
    n_eff_pr = (np.sum(lam) ** 2) / np.sum(lam ** 2)
    n_eff_pr_cs = k ** 2 / ((1 + (k - 1) * rho_bar) ** 2 + (k - 1) * (1 - rho_bar) ** 2)

    total = n_eff_pr - n_eff_kish
    formula = n_eff_pr_cs - n_eff_kish
    cs_dep = n_eff_pr - n_eff_pr_cs

    print(f"\n===== {name} =====")
    print(f"n (complete-case items) = {n}")
    print(f"k (judges) = {k}")
    if len(nan_cols):
        print(f"WARNING zero-variance judge columns (nan corr): {[judges[i] for i in nan_cols]}")
    print(f"rho_bar (mean off-diag)  = {rho_bar:.4f}")
    print(f"off-diag std             = {off_std:.4f}")
    print(f"n_eff_Kish   = {n_eff_kish:.4f}")
    print(f"n_eff_PR     = {n_eff_pr:.4f}")
    print(f"n_eff_PR_CS  = {n_eff_pr_cs:.4f}")
    print(f"--- decomposition (total = PR - Kish) ---")
    print(f"total   (PR - Kish)    = {total:.4f}")
    print(f"formula (PR_CS - Kish) = {formula:.4f}  ({100*formula/total:.1f}%)")
    print(f"cs_dep  (PR - PR_CS)   = {cs_dep:.4f}  ({100*cs_dep/total:.1f}%)")
    print(f"total pct check        = 100.0% ({100*total/total:.1f}%)")

    return {
        "n": n, "k": k, "rho_bar": rho_bar, "off_std": off_std,
        "n_eff_kish": n_eff_kish, "n_eff_pr": n_eff_pr, "n_eff_pr_cs": n_eff_pr_cs,
        "total": total, "formula": formula, "cs_dep": cs_dep,
    }


if __name__ == "__main__":
    judges = load_judges()
    print(f"Loaded {len(judges)} judges from panel.")
    for name, cfg in DATASETS.items():
        analyze(name, cfg, judges)
