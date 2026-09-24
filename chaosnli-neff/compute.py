#!/usr/bin/env python3
"""ChaosNLI distributional-diversity analysis.

Measures D = inverse-Simpson (effective number of active labels) per item,
computed separately from the 100-human vote distribution and the 32-LLM vote
distribution. These are purely within-item spread measures, NOT 'n_eff of
judges' or 'effective sample size'. See results.md Estimands section for
the three distinct quantities and why they must not be conflated.

Usage:
    python3 compute.py

Requires:
    - repo/  (clone of Chao1208/32judges-votes)
    - chaosNLI_v1.0/  (ChaosNLI v1.0 JSONL files, CC BY-NC 4.0)
    - .venv/  (numpy, pandas, matplotlib)

Outputs:
    - results.md
    - figure.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent
REPO = PROJECT / "repo"
CHAOS_DIR = PROJECT / "chaosNLI_v1.0"
sys.path.insert(0, str(REPO / "src"))

import layout  # noqa: E402  (repo src)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASETS = {
    "mnli_m":   {"chaos_file": "chaosNLI_mnli_m.jsonl",   "labels": ("e", "n", "c")},
    "snli":     {"chaos_file": "chaosNLI_snli.jsonl",      "labels": ("e", "n", "c")},
    "alphanli": {"chaos_file": "chaosNLI_alphanli.jsonl",  "labels": ("1", "2")},
}

# ---------------------------------------------------------------------------
# Core computation: inverse-Simpson diversity
# ---------------------------------------------------------------------------

def inv_simpson(counts: np.ndarray) -> float:
    """D = (Σ c_l)^2 / Σ c_l^2. Returns NaN if total == 0."""
    total = counts.sum()
    if total == 0.0:
        return float("nan")
    return float(total ** 2 / np.sum(counts ** 2))


# ---------------------------------------------------------------------------
# Load human label_counter for a given ChaosNLI split
# ---------------------------------------------------------------------------

def load_human_counts(chaos_path: Path, uids_wanted: set, labels: tuple) -> dict[str, np.ndarray]:
    """Map uid -> count vector (in label order) from ChaosNLI JSONL."""
    result: dict[str, np.ndarray] = {}
    with open(chaos_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            uid = str(r["uid"])
            if uid not in uids_wanted:
                continue
            counter = r["label_counter"]
            counts = np.array([counter.get(lbl, 0) for lbl in labels], dtype=float)
            result[uid] = counts
    return result


# ---------------------------------------------------------------------------
# Load 32-LLM votes (baseline arm) for a given split
# ---------------------------------------------------------------------------

def load_llm_votes(repo_root: Path, dataset: str, labels: tuple) -> dict[str, dict]:
    """
    Returns:
        votes_by_uid: uid -> {"counts": np.ndarray, "n_valid": int, "n_failed": int}
    """
    label_idx = {lbl: i for i, lbl in enumerate(labels)}
    votes_dir = layout.votes_dir(repo_root, dataset, "baseline")
    uids_file = layout.items_file(repo_root, dataset)
    uids_ordered = uids_file.read_text(encoding="utf-8").split()

    # Initialize: uid -> list of valid label indices
    vote_lists: dict[str, list[int]] = {uid: [] for uid in uids_ordered}
    fail_counts: dict[str, int] = {uid: 0 for uid in uids_ordered}

    for jpath in sorted(votes_dir.glob("*.jsonl")):
        with open(jpath, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                uid = str(r["uid"])
                if uid not in vote_lists:
                    continue
                if r.get("parse_fail", False):
                    fail_counts[uid] += 1
                else:
                    lbl = str(r["label"])
                    if lbl in label_idx:
                        vote_lists[uid].append(label_idx[lbl])

    result: dict[str, dict] = {}
    for uid in uids_ordered:
        counts = np.zeros(len(labels), dtype=float)
        for idx in vote_lists[uid]:
            counts[idx] += 1
        result[uid] = {
            "counts": counts,
            "n_valid": int(counts.sum()),
            "n_failed": fail_counts[uid],
        }
    return result


# ---------------------------------------------------------------------------
# Bootstrap CI for mean difference (no scipy needed)
# ---------------------------------------------------------------------------

def bootstrap_ci(diffs: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                 rng_seed: int = 42) -> tuple[float, float, float]:
    """
    Returns (mean_diff, ci_lo, ci_hi) using percentile bootstrap.
    diffs = D_human - D_llm per item.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(diffs)
    means = np.array([
        rng.choice(diffs, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_lo = float(np.percentile(means, 100 * alpha / 2))
    ci_hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(diffs.mean()), ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Per-split analysis
# ---------------------------------------------------------------------------

def analyze_split(dataset: str) -> dict:
    meta = DATASETS[dataset]
    labels = meta["labels"]
    chaos_path = CHAOS_DIR / meta["chaos_file"]

    print(f"\n--- {dataset} ---")

    # Load UIDs for this split
    uids_ordered = layout.items_file(REPO, dataset).read_text(encoding="utf-8").split()
    uids_wanted = set(uids_ordered)
    print(f"  UIDs in split: {len(uids_ordered)}")

    # Load human counts
    human_counts = load_human_counts(chaos_path, uids_wanted, labels)
    n_human_found = len(human_counts)
    n_missing_human = len(uids_wanted) - n_human_found
    print(f"  Human records found: {n_human_found}  missing: {n_missing_human}")
    if n_missing_human > 0:
        missing = uids_wanted - set(human_counts)
        print(f"  Missing UIDs (first 3): {sorted(missing)[:3]}")

    # Load LLM votes
    llm_votes = load_llm_votes(REPO, dataset, labels)
    total_parse_fails = sum(v["n_failed"] for v in llm_votes.values())
    total_valid = sum(v["n_valid"] for v in llm_votes.values())
    print(f"  Total LLM vote cells: valid={total_valid}, parse_fail={total_parse_fails}")

    # Compute per-item D values (only for items where both human and LLM are available)
    records = []
    for uid in uids_ordered:
        if uid not in human_counts:
            continue
        h_counts = human_counts[uid]
        l_data = llm_votes[uid]
        l_counts = l_data["counts"]

        d_human = inv_simpson(h_counts)
        d_llm = inv_simpson(l_counts)

        if np.isnan(d_human) or np.isnan(d_llm):
            continue

        records.append({
            "uid": uid,
            "D_human": d_human,
            "D_llm": d_llm,
            "n_human": int(h_counts.sum()),
            "n_llm_valid": l_data["n_valid"],
            "n_llm_failed": l_data["n_failed"],
        })

    df = pd.DataFrame(records)
    n_joined = len(df)
    print(f"  Items joined and analyzed: {n_joined}")

    diffs = df["D_human"].values - df["D_llm"].values
    mean_diff, ci_lo, ci_hi = bootstrap_ci(diffs)

    result = {
        "dataset": dataset,
        "n_joined": n_joined,
        "total_parse_fails": total_parse_fails,
        "D_human_mean": float(df["D_human"].mean()),
        "D_human_median": float(df["D_human"].median()),
        "D_human_std": float(df["D_human"].std()),
        "D_llm_mean": float(df["D_llm"].mean()),
        "D_llm_median": float(df["D_llm"].median()),
        "D_llm_std": float(df["D_llm"].std()),
        "mean_diff": mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "frac_human_gt_llm": float((df["D_human"] > df["D_llm"]).mean()),
        "df": df,
    }
    print(f"  D_human mean={result['D_human_mean']:.4f}  D_llm mean={result['D_llm_mean']:.4f}")
    print(f"  mean diff (D_human - D_llm) = {mean_diff:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Fraction D_human > D_llm = {result['frac_human_gt_llm']:.3f}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    split_results = {}
    for ds in DATASETS:
        split_results[ds] = analyze_split(ds)

    # Pooled analysis
    print("\n--- POOLED ---")
    all_dfs = [r["df"] for r in split_results.values()]
    pooled_df = pd.concat(all_dfs, ignore_index=True)
    diffs_pooled = pooled_df["D_human"].values - pooled_df["D_llm"].values
    mean_diff_p, ci_lo_p, ci_hi_p = bootstrap_ci(diffs_pooled)
    total_fails = sum(r["total_parse_fails"] for r in split_results.values())

    pooled = {
        "dataset": "pooled",
        "n_joined": len(pooled_df),
        "total_parse_fails": total_fails,
        "D_human_mean": float(pooled_df["D_human"].mean()),
        "D_human_median": float(pooled_df["D_human"].median()),
        "D_human_std": float(pooled_df["D_human"].std()),
        "D_llm_mean": float(pooled_df["D_llm"].mean()),
        "D_llm_median": float(pooled_df["D_llm"].median()),
        "D_llm_std": float(pooled_df["D_llm"].std()),
        "mean_diff": mean_diff_p,
        "ci_lo": ci_lo_p,
        "ci_hi": ci_hi_p,
        "frac_human_gt_llm": float((pooled_df["D_human"] > pooled_df["D_llm"]).mean()),
    }
    print(f"  n_joined: {pooled['n_joined']}, parse_fails: {total_fails}")
    print(f"  D_human mean={pooled['D_human_mean']:.4f}  D_llm mean={pooled['D_llm_mean']:.4f}")
    print(f"  mean diff = {mean_diff_p:.4f}  95% CI [{ci_lo_p:.4f}, {ci_hi_p:.4f}]")
    print(f"  Fraction D_human > D_llm = {pooled['frac_human_gt_llm']:.3f}")

    # Panel n_eff from repo's saved reference values (reported SEPARATELY)
    ref_path = REPO / "reference" / "reported_values.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    print("\n--- Panel n_eff (from repo reported_values.json, SEPARATE estimand) ---")
    for ds_key, vals in ref["datasets"].items():
        print(f"  {ds_key}: n_eff_panel = {vals['n_eff']:.4f}  nu_H = {vals['nu_H']:.4f}")

    # -----------------------------------------------------------------------
    # Produce figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    datasets_order = ["mnli_m", "snli", "alphanli"]
    titles = ["MNLI-m", "SNLI", "alphaNLI", "Pooled"]
    dfs_to_plot = [split_results[ds]["df"] for ds in datasets_order] + [pooled_df]

    for ax, df_plot, title in zip(axes, dfs_to_plot, titles):
        data_h = df_plot["D_human"].values
        data_l = df_plot["D_llm"].values

        # Violin plot approximated with two KDE-based distributions
        # Use violinplot with matplotlib (no scipy dependency needed — matplotlib
        # computes the KDE internally via gaussian_kde from numpy)
        parts = ax.violinplot([data_h, data_l], positions=[0, 1],
                              showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], ["#4878CF", "#6ACC65"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Human\n(100)", "LLM\n(32)"], fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Source", fontsize=9)

        # Annotate with means
        ax.annotate(f"μ={data_h.mean():.2f}", xy=(0, data_h.mean()),
                    xytext=(0.05, data_h.mean()), textcoords="data",
                    fontsize=7, color="#4878CF")
        ax.annotate(f"μ={data_l.mean():.2f}", xy=(1, data_l.mean()),
                    xytext=(1.05, data_l.mean()), textcoords="data",
                    fontsize=7, color="#6ACC65")

    axes[0].set_ylabel("D = Inverse-Simpson (eff. labels)", fontsize=10)
    fig.suptitle(
        "Per-item distributional diversity D = Inv-Simpson(votes)\n"
        "Human (100 annotators) vs 32-LLM panel — ChaosNLI",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    fig_path = PROJECT / "figure.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    fig_size = fig_path.stat().st_size
    print(f"\nFigure saved: figure.png ({fig_size/1024:.1f} KB)")
    if fig_size > 300 * 1024:
        print("WARNING: figure exceeds 300KB target")

    # -----------------------------------------------------------------------
    # Produce results.md
    # -----------------------------------------------------------------------
    lines = []
    lines.append("# ChaosNLI Distributional Diversity Analysis")
    lines.append("")
    lines.append("**Paper:** Li, Yu, Li — *How Many Humans Is a Judge Panel Worth?* (2609.21277)")
    lines.append("**Data:** ChaosNLI v1.0 (CC BY-NC 4.0) × 32judges-votes (CC BY 4.0)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Estimands")
    lines.append("")
    lines.append(
        "Three distinct quantities are computed here. **They are NOT interchangeable "
        "and must never be averaged together.**"
    )
    lines.append("")
    lines.append(
        "- **D = distributional diversity (inverse-Simpson), per item.**  "
        "For a vote-count vector c = (c_1, ..., c_L) over the L labels, "
        "D = (Σ_l c_l)² / Σ_l c_l². This is the *effective number of active labels* "
        "— a within-item spread measure. Computed identically from the 100-human counts "
        "and the 32-LLM counts. This is the apples-to-apples axis. "
        "Named `D_human` and `D_llm` in code. "
        "**NEVER call this 'effective number of raters' or 'n_eff of judges'.**"
    )
    lines.append("")
    lines.append(
        "- **n_eff_panel = k / (1 + (k−1)·φ̄), LLM-panel only.**  "
        "k=32, φ̄ = mean pairwise inter-judge Pearson error correlation across items. "
        "Requires fixed judge identity across all items; NOT computable for the anonymous "
        "human crowd. Loaded from `reference/reported_values.json` (repo-computed). "
        "This is a between-judge correlation measure, not a label-spread measure."
    )
    lines.append("")
    lines.append(
        "- **ν_H = human-equivalence (paper's calibration curve).**  "
        "How many humans a 32-judge panel is 'worth' under the paper's MSE calibration. "
        "Also loaded from `reference/reported_values.json`. Distinct from both D and n_eff_panel."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## D (inverse-Simpson) Results")
    lines.append("")
    lines.append("### Per-Split and Pooled")
    lines.append("")
    lines.append("| Split | N joined | Parse fails | D_human mean | D_human median | D_human std | D_llm mean | D_llm median | D_llm std | mean(D_H−D_L) | 95% CI | frac D_H>D_L |")
    lines.append("|-------|----------|-------------|--------------|----------------|-------------|------------|--------------|-----------|---------------|--------|--------------|")

    all_results = {ds: split_results[ds] for ds in datasets_order}
    all_results["pooled"] = pooled

    for ds, r in all_results.items():
        label = {"mnli_m": "MNLI-m", "snli": "SNLI", "alphanli": "alphaNLI", "pooled": "**Pooled**"}[ds]
        lines.append(
            f"| {label} | {r['n_joined']} | {r['total_parse_fails']} | "
            f"{r['D_human_mean']:.4f} | {r['D_human_median']:.4f} | {r['D_human_std']:.4f} | "
            f"{r['D_llm_mean']:.4f} | {r['D_llm_median']:.4f} | {r['D_llm_std']:.4f} | "
            f"{r['mean_diff']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | "
            f"{r['frac_human_gt_llm']:.3f} |"
        )

    lines.append("")
    lines.append("Bootstrap CI: 5000 resamples, percentile method, seed=42.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## n_eff_panel and ν_H (separate estimands, from repo)")
    lines.append("")
    lines.append("These numbers are loaded directly from `reference/reported_values.json` "
                 "as computed by the 32judges-votes reproduction pipeline. "
                 "They are SEPARATE from D above and must not be interpreted as label-spread.")
    lines.append("")
    lines.append("| Split | n_eff_panel | ν_H (human-equiv) | PR (participation ratio) |")
    lines.append("|-------|-------------|-------------------|--------------------------|")
    for ds_key in ["mnli_m", "snli", "alphanli"]:
        vals = ref["datasets"][ds_key]
        label = {"mnli_m": "MNLI-m", "snli": "SNLI", "alphanli": "alphaNLI"}[ds_key]
        lines.append(
            f"| {label} | {vals['n_eff']:.4f} | {vals['nu_H']:.4f} | {vals['PR']:.4f} |"
        )
    lines.append("")
    lines.append("**n_eff_panel** = k / (1 + (k−1)·φ̄) where φ̄ is the mean Pearson inter-judge "
                 "error correlation (k=32). **ν_H** = human-equivalence from the paper's MSE "
                 "calibration curve. Both measure panel correlation structure, NOT label spread.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Figure")
    lines.append("")
    lines.append("See `figure.png` — violin plots of per-item D_human vs D_llm distributions "
                 "for each split and pooled.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Data Provenance")
    lines.append("")
    lines.append("- **ChaosNLI v1.0**: Nie, Zhou, Bansal (EMNLP 2020). "
                 "License: CC BY-NC 4.0. Not redistributed here — download from "
                 "https://github.com/easonnie/ChaosNLI")
    lines.append("- **32-judge votes**: Li, Yu, Li (2609.21277). "
                 "License: CC BY 4.0. https://github.com/Chao1208/32judges-votes")
    lines.append("- Human data source used for download: "
                 "https://github.com/TheGuy-26/chaosnli-deberta-modernbert "
                 "(ChaosNLI content re-licensed CC BY-NC 4.0 from original).")

    results_path = PROJECT / "results.md"
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"results.md written ({results_path.stat().st_size} bytes)")

    print("\nDone.")


if __name__ == "__main__":
    main()
