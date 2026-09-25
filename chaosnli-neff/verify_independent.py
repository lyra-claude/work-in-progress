"""
Independent verification script for chaosnli-neff analysis.
Computes from raw data only; does NOT read or reuse any strat*.py outputs.

Definitions:
  p_human[k] = count_k / sum(counts)  for the ~100-human label counts
  p_llm[k]   = votes_k / 32           for the 32-LLM vote tallies
  H_i        = -sum_k p_human[k] * log(p_human[k])  (natural log, 0*log0=0)
  D_human_i  = 1 / sum_k p_human[k]^2
  D_llm_i    = 1 / sum_k p_llm[k]^2
  g_i        = D_human_i - D_llm_i
"""

import json
import os
import math
import numpy as np
from scipy import stats

# ──────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────

ROOT = "/home/lyra/projects/chaosnli-neff"

DATASETS = {
    "MNLI-m": {
        "human_jsonl": f"{ROOT}/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl",
        "repo_dir": f"{ROOT}/repo/datasets/chaosnli-mnli-m-1000",
    },
    "SNLI": {
        "human_jsonl": f"{ROOT}/chaosNLI_v1.0/chaosNLI_snli.jsonl",
        "repo_dir": f"{ROOT}/repo/datasets/chaosnli-snli-1000",
    },
    "alphaNLI": {
        "human_jsonl": f"{ROOT}/chaosNLI_v1.0/chaosNLI_alphanli.jsonl",
        "repo_dir": f"{ROOT}/repo/datasets/chaosnli-alphanli-1000",
    },
}


def load_human(path):
    """Load ChaosNLI JSONL → dict uid -> label_counter dict."""
    result = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            result[item["uid"]] = item["label_counter"]
    return result


def load_llm_votes(repo_dir, label_order):
    """
    Load all 32 judge JSONL files from votes/baseline/.
    Returns dict uid -> {label: count}.
    Assumption: we use the 'baseline' arm only (not 'swap').
    parse_fail items are excluded from counts (treated as missing vote).
    """
    vote_dir = os.path.join(repo_dir, "votes", "baseline")
    judge_files = sorted(os.listdir(vote_dir))
    assert len(judge_files) == 32, f"Expected 32 judge files, got {len(judge_files)}"

    # uid -> {label: count}
    vote_counts = {}

    for fname in judge_files:
        fpath = os.path.join(vote_dir, fname)
        with open(fpath) as f:
            for line in f:
                item = json.loads(line)
                uid = item["uid"]
                if uid not in vote_counts:
                    vote_counts[uid] = {lbl: 0 for lbl in label_order}
                if not item["parse_fail"] and item["label"] in vote_counts[uid]:
                    vote_counts[uid][item["label"]] += 1

    return vote_counts


def load_manifest(repo_dir):
    with open(os.path.join(repo_dir, "manifest.json")) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# Stat helpers
# ──────────────────────────────────────────────────────────────

def shannon_entropy(counts_dict):
    """Natural-log Shannon entropy from a dict of non-negative counts."""
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    h = 0.0
    for v in counts_dict.values():
        if v > 0:
            p = v / total
            h -= p * math.log(p)
    return h


def inv_simpson(counts_dict):
    """Inverse-Simpson D = (sum c)^2 / sum c^2 from a counts dict."""
    total = sum(counts_dict.values())
    if total == 0:
        return float("nan")
    sum_sq = sum(v * v for v in counts_dict.values())
    if sum_sq == 0:
        return float("nan")
    return (total ** 2) / sum_sq


def ols_slope(x, y):
    """OLS slope of y on x."""
    x = np.array(x)
    y = np.array(y)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (xm * xm).sum()
    if denom == 0:
        return float("nan")
    return (xm * ym).sum() / denom


def partial_corr_residuals(h, g, d_human):
    """
    Partial Pearson correlation of H and g controlling for D_human.
    Method: regress H on D_human, take residuals;
            regress g on D_human, take residuals;
            Pearson of the two residual series.
    """
    h = np.array(h)
    g = np.array(g)
    d = np.array(d_human)

    # residuals of H ~ D_human
    slope_h = ols_slope(d, h)
    intercept_h = h.mean() - slope_h * d.mean()
    resid_h = h - (slope_h * d + intercept_h)

    # residuals of g ~ D_human
    slope_g = ols_slope(d, g)
    intercept_g = g.mean() - slope_g * d.mean()
    resid_g = g - (slope_g * d + intercept_g)

    r, _ = stats.pearsonr(resid_h, resid_g)
    return r


# ──────────────────────────────────────────────────────────────
# Main computation
# ──────────────────────────────────────────────────────────────

all_rows = []   # list of dicts: subset, uid, H, D_human, D_llm, g

for subset_name, cfg in DATASETS.items():
    manifest = load_manifest(cfg["repo_dir"])
    label_order = manifest["task"]["labels"]  # e.g. ['e','n','c'] or ['1','2']

    human_db = load_human(cfg["human_jsonl"])
    llm_votes = load_llm_votes(cfg["repo_dir"], label_order)

    # Get ordered UIDs from repo (the 1000 selected items)
    with open(os.path.join(cfg["repo_dir"], "items", "uids.txt")) as f:
        repo_uids = [l.strip() for l in f]

    n_matched = 0
    for uid in repo_uids:
        if uid not in human_db:
            print(f"WARNING: uid {uid} not found in human data for {subset_name}")
            continue
        if uid not in llm_votes:
            print(f"WARNING: uid {uid} not found in LLM votes for {subset_name}")
            continue

        h_counts = human_db[uid]  # label -> int (~100 total)
        l_counts = llm_votes[uid]  # label -> int (0..32 total)

        # Normalize both to the same label set
        # For human: restrict to labels in label_order
        h_counts_ordered = {lbl: h_counts.get(lbl, 0) for lbl in label_order}
        # llm_votes already keyed to label_order

        H_i = shannon_entropy(h_counts_ordered)
        D_human_i = inv_simpson(h_counts_ordered)
        D_llm_i = inv_simpson(l_counts)
        g_i = D_human_i - D_llm_i

        all_rows.append({
            "subset": subset_name,
            "uid": uid,
            "H": H_i,
            "D_human": D_human_i,
            "D_llm": D_llm_i,
            "g": g_i,
        })
        n_matched += 1

    print(f"{subset_name}: {n_matched} items matched and computed")

print(f"\nTotal rows: {len(all_rows)}")

# ──────────────────────────────────────────────────────────────
# Compute statistics per subset and pooled
# ──────────────────────────────────────────────────────────────

def compute_stats(rows, label="Pooled"):
    H = np.array([r["H"] for r in rows])
    D_h = np.array([r["D_human"] for r in rows])
    D_l = np.array([r["D_llm"] for r in rows])
    g = np.array([r["g"] for r in rows])

    n = len(rows)

    # 1. OLS slopes
    slope_human_on_H = ols_slope(H, D_h)
    slope_llm_on_H = ols_slope(H, D_l)
    slope_ratio = slope_llm_on_H / slope_human_on_H if slope_human_on_H != 0 else float("nan")

    # 2. Pearson and Spearman between H and D_human
    pearson_H_Dhuman, _ = stats.pearsonr(H, D_h)
    spearman_H_Dhuman, _ = stats.spearmanr(H, D_h)

    # 3. Spearman between H and g
    spearman_H_g, _ = stats.spearmanr(H, g)

    # 4. Partial correlation H vs g controlling for D_human
    pcorr = partial_corr_residuals(H, g, D_h)

    # 5. Quintile table
    quintile_bins = np.percentile(H, [0, 20, 40, 60, 80, 100])
    quintile_bins[0] -= 1e-10  # ensure first bin captures minimum

    print(f"\n{'='*60}")
    print(f"  {label}  (n={n})")
    print(f"{'='*60}")
    print(f"\n1. OLS slopes:")
    print(f"   slope(D_human ~ H) = {slope_human_on_H:.4f}")
    print(f"   slope(D_llm   ~ H) = {slope_llm_on_H:.4f}")
    print(f"   ratio (llm/human)  = {slope_ratio:.4f}")
    print(f"\n2. Pearson(H, D_human)  = {pearson_H_Dhuman:.4f}")
    print(f"   Spearman(H, D_human) = {spearman_H_Dhuman:.4f}")
    print(f"\n3. Spearman(H, g)       = {spearman_H_g:.4f}")
    print(f"\n4. Partial corr(H, g | D_human) [Pearson of residuals] = {pcorr:.4f}")
    print(f"\n5. Quintile table (by H_i):")
    print(f"   {'Q':>3}  {'n':>5}  {'mean_H':>8}  {'mean_D_human':>13}  {'mean_D_llm':>11}  {'mean_gap':>9}")
    for q in range(5):
        mask = (H > quintile_bins[q]) & (H <= quintile_bins[q+1])
        qn = mask.sum()
        qH = H[mask].mean()
        qDh = D_h[mask].mean()
        qDl = D_l[mask].mean()
        qg = g[mask].mean()
        print(f"   {q+1:>3}  {qn:>5}  {qH:>8.4f}  {qDh:>13.4f}  {qDl:>11.4f}  {qg:>9.4f}")

    return {
        "label": label,
        "n": n,
        "slope_human_on_H": slope_human_on_H,
        "slope_llm_on_H": slope_llm_on_H,
        "slope_ratio": slope_ratio,
        "pearson_H_Dhuman": pearson_H_Dhuman,
        "spearman_H_Dhuman": spearman_H_Dhuman,
        "spearman_H_g": spearman_H_g,
        "partial_corr_H_g_given_Dhuman": pcorr,
    }


# Per-subset
for subset_name in DATASETS:
    subset_rows = [r for r in all_rows if r["subset"] == subset_name]
    compute_stats(subset_rows, label=subset_name)

# Pooled
compute_stats(all_rows, label="Pooled (all 3000)")
