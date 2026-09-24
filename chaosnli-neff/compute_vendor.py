#!/usr/bin/env python3
"""Vendor-family co-failure analysis for the 32-judge ChaosNLI panel.

Tests whether judges from the SAME vendor family agree with each other MORE
than judges from DIFFERENT families, using:
  - Raw pairwise agreement (a_ij)
  - Cohen's kappa (κ_ij): chance-corrected agreement

IMPORTANT ESTIMAND NOTE
-----------------------
This measures inter-judge label AGREEMENT / correlation — the co-failure-relevant
quantity. It is NOT an error rate. No gold labels are used anywhere in this script.
A high within-family κ means same-vendor judges tend to emit the same label, not that
they are necessarily correct.

Usage:
    python3 compute_vendor.py

Requires:
    - repo/  (clone of Chao1208/32judges-votes)
    - .venv/  (numpy, pandas, matplotlib)

Outputs:
    - results-vendor.md
    - figure-vendor.png
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
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
sys.path.insert(0, str(REPO / "src"))

import layout  # noqa: E402  (repo src)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASETS = {
    "mnli_m":   {"labels": ("e", "n", "c")},
    "snli":     {"labels": ("e", "n", "c")},
    "alphanli": {"labels": ("1", "2")},
}

PANEL_FILE = REPO / "panel" / "panel-chaosnli.json"

# ---------------------------------------------------------------------------
# Load panel metadata (judge_key → family)
# ---------------------------------------------------------------------------

def load_panel() -> dict[str, str]:
    """Return dict: judge_key -> family (from the repo's authoritative panel file)."""
    doc = json.loads(PANEL_FILE.read_text(encoding="utf-8"))
    return {j["judge_key"]: j["family"] for j in doc["judges"]}


# ---------------------------------------------------------------------------
# Load per-judge vote arrays for one split
# ---------------------------------------------------------------------------

def load_judge_votes(dataset: str, labels: tuple[str, ...]) -> dict[str, dict[str, str | None]]:
    """
    Returns: {judge_key: {uid: label_str_or_None}}
    label is None if parse_fail=True.
    """
    votes_dir = layout.votes_dir(REPO, dataset, "baseline")
    uids_ordered = layout.items_file(REPO, dataset).read_text(encoding="utf-8").split()
    uid_set = set(uids_ordered)
    valid_labels = set(labels)

    result: dict[str, dict[str, str | None]] = {}
    for jpath in sorted(votes_dir.glob("*.jsonl")):
        judge_key = jpath.stem
        judge_votes: dict[str, str | None] = {}
        with open(jpath, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                uid = str(r["uid"])
                if uid not in uid_set:
                    continue
                if r.get("parse_fail", False):
                    judge_votes[uid] = None
                else:
                    lbl = str(r["label"])
                    judge_votes[uid] = lbl if lbl in valid_labels else None
        result[judge_key] = judge_votes
    return result


# ---------------------------------------------------------------------------
# Compute pairwise agreement and Cohen's kappa
# ---------------------------------------------------------------------------

def pairwise_kappa(
    votes_i: dict[str, str | None],
    votes_j: dict[str, str | None],
    labels: tuple[str, ...],
) -> tuple[float, float, int]:
    """
    Compute raw agreement and Cohen's kappa for two judges over their shared items.

    Returns:
        (raw_agreement, kappa, n_shared)
    where n_shared is the number of items where BOTH judges have a valid label.
    Returns (nan, nan, 0) if there are fewer than 2 shared items.
    """
    shared_uids = [
        uid for uid in votes_i
        if uid in votes_j and votes_i[uid] is not None and votes_j[uid] is not None
    ]
    n = len(shared_uids)
    if n < 2:
        return float("nan"), float("nan"), n

    labels_i = [votes_i[uid] for uid in shared_uids]
    labels_j = [votes_j[uid] for uid in shared_uids]

    # Raw agreement
    agree = sum(li == lj for li, lj in zip(labels_i, labels_j))
    a_ij = agree / n

    # Marginal frequencies for each judge over shared items
    freq_i = {lbl: labels_i.count(lbl) / n for lbl in labels}
    freq_j = {lbl: labels_j.count(lbl) / n for lbl in labels}

    # Chance agreement p_e = Σ_label freq_i(label) * freq_j(label)
    p_e = sum(freq_i[lbl] * freq_j[lbl] for lbl in labels)

    if p_e >= 1.0:
        # Degenerate: both judges always pick the same single label
        return a_ij, float("nan"), n

    kappa = (a_ij - p_e) / (1.0 - p_e)
    return a_ij, kappa, n


# ---------------------------------------------------------------------------
# Per-split analysis
# ---------------------------------------------------------------------------

def analyze_split(
    dataset: str,
    judge_family: dict[str, str],
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per unordered judge pair:
        judge_i, judge_j, family_i, family_j, same_family,
        raw_agree, kappa, n_shared
    """
    labels = DATASETS[dataset]["labels"]
    print(f"\n--- Loading {dataset} ---")
    judge_votes = load_judge_votes(dataset, labels)

    judge_keys = sorted(judge_votes.keys())
    print(f"  Judges found: {len(judge_keys)}")

    rows = []
    for ki, kj in combinations(judge_keys, 2):
        fi = judge_family.get(ki, "unknown")
        fj = judge_family.get(kj, "unknown")
        same = fi == fj
        a_ij, kap, n_shared = pairwise_kappa(judge_votes[ki], judge_votes[kj], labels)
        rows.append({
            "split": dataset,
            "judge_i": ki,
            "judge_j": kj,
            "family_i": fi,
            "family_j": fj,
            "same_family": same,
            "raw_agree": a_ij,
            "kappa": kap,
            "n_shared": n_shared,
        })

    df = pd.DataFrame(rows)
    n_within = df["same_family"].sum()
    n_cross = (~df["same_family"]).sum()
    print(f"  Pairs: within-family={n_within}, cross-family={n_cross}, total={len(df)}")
    return df


# ---------------------------------------------------------------------------
# Permutation test on κ gap (within - cross)
# ---------------------------------------------------------------------------

def permutation_test_kappa_gap(
    df: pd.DataFrame,
    judge_family: dict[str, str],
    n_perm: int = 2000,
) -> tuple[float, float]:
    """
    Observed gap: mean(within-κ) - mean(cross-κ) on pooled df.
    Permutation null: shuffle family labels among the 32 judges, recompute gap.
    Returns (observed_gap, p_value_one_sided).
    Each permutation uses np.random.default_rng(seed=perm_index).
    """
    judges = sorted(set(df["judge_i"].tolist() + df["judge_j"].tolist()))
    families_orig = [judge_family[j] for j in judges]

    def compute_gap_from_assignment(fam_assign: dict[str, str]) -> float:
        within_k = []
        cross_k = []
        for _, row in df.iterrows():
            if pd.isna(row["kappa"]):
                continue
            fi = fam_assign[row["judge_i"]]
            fj = fam_assign[row["judge_j"]]
            if fi == fj:
                within_k.append(row["kappa"])
            else:
                cross_k.append(row["kappa"])
        if not within_k or not cross_k:
            return float("nan")
        return float(np.mean(within_k) - np.mean(cross_k))

    # Observed gap (using original family assignment from df columns)
    within_kappas = df.loc[df["same_family"] & df["kappa"].notna(), "kappa"].values
    cross_kappas = df.loc[~df["same_family"] & df["kappa"].notna(), "kappa"].values
    observed_gap = float(np.mean(within_kappas) - np.mean(cross_kappas))

    print(f"\n  Permutation test: observed gap = {observed_gap:.6f}")
    print(f"  Running {n_perm} permutations (deterministic seeds 0..{n_perm-1})...")

    perm_gaps = []
    for perm_idx in range(n_perm):
        rng = np.random.default_rng(perm_idx)
        shuffled = list(families_orig)
        rng.shuffle(shuffled)
        fam_assign = dict(zip(judges, shuffled))
        gap = compute_gap_from_assignment(fam_assign)
        if not np.isnan(gap):
            perm_gaps.append(gap)

    perm_gaps_arr = np.array(perm_gaps)
    p_value = float(np.mean(perm_gaps_arr >= observed_gap))
    print(f"  p-value (one-sided, gap >= observed): {p_value:.4f}")
    return observed_gap, p_value


# ---------------------------------------------------------------------------
# Report summary stats
# ---------------------------------------------------------------------------

def summary_stats(df: pd.DataFrame, label: str) -> dict:
    within = df[df["same_family"] & df["kappa"].notna()]
    cross = df[~df["same_family"] & df["kappa"].notna()]
    within_a = df[df["same_family"] & df["raw_agree"].notna()]
    cross_a = df[~df["same_family"] & df["raw_agree"].notna()]

    return {
        "label": label,
        "n_within_pairs": int(df["same_family"].sum()),
        "n_cross_pairs": int((~df["same_family"]).sum()),
        "within_raw": float(within_a["raw_agree"].mean()) if len(within_a) > 0 else float("nan"),
        "cross_raw": float(cross_a["raw_agree"].mean()) if len(cross_a) > 0 else float("nan"),
        "gap_raw": float(within_a["raw_agree"].mean() - cross_a["raw_agree"].mean())
                   if len(within_a) > 0 and len(cross_a) > 0 else float("nan"),
        "within_kappa": float(within["kappa"].mean()) if len(within) > 0 else float("nan"),
        "cross_kappa": float(cross["kappa"].mean()) if len(cross) > 0 else float("nan"),
        "gap_kappa": float(within["kappa"].mean() - cross["kappa"].mean())
                     if len(within) > 0 and len(cross) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Per-family within-κ breakdown
# ---------------------------------------------------------------------------

def per_family_within_kappa(pooled_df: pd.DataFrame, judge_family: dict[str, str]) -> pd.DataFrame:
    """
    For each family, compute the mean within-family kappa using only within-family pairs.
    Families with <2 judges have 0 within-family pairs and get NaN.
    """
    rows = []
    within_df = pooled_df[pooled_df["same_family"] & pooled_df["kappa"].notna()]
    for fam, grp in within_df.groupby("family_i"):
        n_pairs = len(grp)
        mean_k = float(grp["kappa"].mean())
        # Count judges in this family
        n_judges = sum(1 for f in judge_family.values() if f == fam)
        rows.append({"family": fam, "n_judges": n_judges, "n_within_pairs": n_pairs, "mean_within_kappa": mean_k})

    # Add families with 1 judge (0 within pairs)
    families_seen = {r["family"] for r in rows}
    family_counts = {}
    for f in judge_family.values():
        family_counts[f] = family_counts.get(f, 0) + 1
    for fam, cnt in family_counts.items():
        if fam not in families_seen:
            rows.append({"family": fam, "n_judges": cnt, "n_within_pairs": 0, "mean_within_kappa": float("nan")})

    result = pd.DataFrame(rows).sort_values("mean_within_kappa", ascending=False, na_position="last")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    pooled_df: pd.DataFrame,
    family_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: violin / boxplot of kappa within vs cross
    ax = axes[0]
    within_k = pooled_df.loc[pooled_df["same_family"] & pooled_df["kappa"].notna(), "kappa"].values
    cross_k = pooled_df.loc[~pooled_df["same_family"] & pooled_df["kappa"].notna(), "kappa"].values

    vp = ax.violinplot([within_k, cross_k], positions=[0, 1],
                       showmedians=True, showextrema=False)
    for pc, color in zip(vp["bodies"], ["#E84B3A", "#4878CF"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within-family", "Cross-family"], fontsize=11)
    ax.set_ylabel("Cohen's κ (pairwise)", fontsize=10)
    ax.set_title("Pairwise κ: Within- vs Cross-family\n(pooled, 3 splits)", fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    # Mean annotations
    ax.annotate(f"μ={within_k.mean():.4f}", xy=(0, within_k.mean()),
                xytext=(0.05, within_k.mean() + 0.01), fontsize=9, color="#E84B3A")
    ax.annotate(f"μ={cross_k.mean():.4f}", xy=(1, cross_k.mean()),
                xytext=(1.05, cross_k.mean() + 0.01), fontsize=9, color="#4878CF")

    # Right: per-family mean within-κ bar chart
    ax2 = axes[1]
    valid_fam = family_df[family_df["mean_within_kappa"].notna()].copy()
    invalid_fam = family_df[family_df["mean_within_kappa"].isna()].copy()

    families = list(valid_fam["family"]) + list(invalid_fam["family"])
    values = list(valid_fam["mean_within_kappa"]) + [0] * len(invalid_fam)
    colors = ["#E84B3A"] * len(valid_fam) + ["#AAAAAA"] * len(invalid_fam)

    bars = ax2.barh(families, values, color=colors, alpha=0.8)
    ax2.set_xlabel("Mean within-family κ", fontsize=10)
    ax2.set_title("Per-family within-κ\n(sorted desc; gray = 1 judge, 0 pairs)", fontsize=11, fontweight="bold")
    ax2.axvline(0, color="black", linewidth=0.8)

    # Annotate values
    for bar, v in zip(bars, values):
        ax2.text(max(v + 0.005, 0.005), bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}" if v != 0 else "n/a",
                 va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    file_size = out_path.stat().st_size
    print(f"\nFigure saved: {out_path.name} ({file_size / 1024:.1f} KB)")
    if file_size > 300 * 1024:
        print("WARNING: figure exceeds 300KB target — try lower dpi")


# ---------------------------------------------------------------------------
# Write results-vendor.md
# ---------------------------------------------------------------------------

def write_results(
    split_stats: list[dict],
    pooled_stat: dict,
    observed_gap: float,
    p_value: float,
    family_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = []
    lines.append("# Vendor-Family Co-Failure Analysis — 32-Judge ChaosNLI Panel")
    lines.append("")
    lines.append("**Paper:** Li, Yu, Li — *How Many Humans Is a Judge Panel Worth?* (2609.21277)")
    lines.append("**Data:** 32judges-votes (CC BY 4.0), 3 splits × 1000 items × 32 judges")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Estimands")
    lines.append("")
    lines.append(
        "This analysis measures **inter-judge label agreement**, not error rate. "
        "No gold labels are used. Two estimands:"
    )
    lines.append("")
    lines.append(
        "- **Raw pairwise agreement** a_ij = fraction of shared (non-parse_fail) items "
        "where judge i and judge j emit the same label. **Not chance-corrected.** "
        "alphaNLI is binary (2 labels), so random agreement baseline ≈ 50%; MNLI/SNLI ≈ 33%. "
        "Raw agreement is systematically inflated for alphaNLI and mixes two effects."
    )
    lines.append("")
    lines.append(
        "- **Cohen's κ_ij** = (a_ij − p_e) / (1 − p_e), where "
        "p_e = Σ_label freq_i(label) × freq_j(label) is the chance-agreement computed "
        "from each judge's *own* marginal label frequencies over the shared items. "
        "This controls for label imbalance and random-agreement floors. "
        "**Preferred estimand for comparing across splits.**"
    )
    lines.append("")
    lines.append(
        "**Kappa-paradox caveat:** κ can be lower than raw agreement when prevalence is "
        "extreme (one label dominates). If raw agreement and κ tell different stories below, "
        "we flag it explicitly."
    )
    lines.append("")
    lines.append("**Family-label source:** repo's authoritative `panel/panel-chaosnli.json` "
                 "(not inferred from model names).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Pair Counts")
    lines.append("")
    lines.append("32 judges → 32×31/2 = 496 unordered pairs per split. "
                 "Family sizes: OpenAI 5, Alibaba 4, Google 4, Moonshot 4, Zhipu 4, "
                 "Anthropic 3, DeepSeek 3, ByteDance 2, xAI 2, MiniMax 1.")
    lines.append("")

    # Compute expected within/cross pairs from family sizes
    # Within pairs: Σ C(n_k, 2)
    family_sizes = {"openai": 5, "alibaba": 4, "google": 4, "moonshot": 4, "zhipu": 4,
                    "anthropic": 3, "deepseek": 3, "bytedance": 2, "xai": 2, "minimax": 1}
    n_within = sum(n * (n - 1) // 2 for n in family_sizes.values())
    n_cross = 496 - n_within
    lines.append(f"- Within-family pairs: {n_within} ({n_within/496*100:.1f}% of total)")
    lines.append(f"- Cross-family pairs: {n_cross} ({n_cross/496*100:.1f}% of total)")
    lines.append(f"- Families with only 1 judge (0 within-pairs): MiniMax")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Pooled Results (all 3 splits)")
    lines.append("")
    p = pooled_stat
    lines.append(f"| Metric | Within-family | Cross-family | Gap (within − cross) |")
    lines.append(f"|--------|---------------|--------------|----------------------|")
    lines.append(f"| Raw agreement | {p['within_raw']:.4f} | {p['cross_raw']:.4f} | {p['gap_raw']:+.4f} |")
    lines.append(f"| Cohen's κ | {p['within_kappa']:.4f} | {p['cross_kappa']:.4f} | {p['gap_kappa']:+.4f} |")
    lines.append("")
    lines.append(f"**Permutation test (κ gap, 2000 permutations, seeds 0–1999):**")
    lines.append(f"- Observed κ gap: {observed_gap:+.6f}")
    lines.append(f"- One-sided p-value (fraction of permutations ≥ observed): **{p_value:.4f}**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-Split Results (κ and raw agreement)")
    lines.append("")
    lines.append("| Split | N pairs (within) | N pairs (cross) | within κ | cross κ | κ gap | within raw | cross raw | raw gap |")
    lines.append("|-------|-----------------|-----------------|----------|---------|-------|------------|-----------|---------|")
    for s in split_stats:
        lines.append(
            f"| {s['label']} | {s['n_within_pairs']} | {s['n_cross_pairs']} | "
            f"{s['within_kappa']:.4f} | {s['cross_kappa']:.4f} | {s['gap_kappa']:+.4f} | "
            f"{s['within_raw']:.4f} | {s['cross_raw']:.4f} | {s['gap_raw']:+.4f} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-Family Within-κ Ranking")
    lines.append("")
    lines.append("Sorted by mean within-family κ (descending). "
                 "Computed pooled across all 3 splits.")
    lines.append("")
    lines.append("| Rank | Family | N judges | N within-pairs | Mean within-κ |")
    lines.append("|------|--------|----------|----------------|---------------|")
    for rank, (_, row) in enumerate(family_df.iterrows(), 1):
        kappa_str = f"{row['mean_within_kappa']:.4f}" if not pd.isna(row["mean_within_kappa"]) else "n/a (1 judge)"
        lines.append(
            f"| {rank} | {row['family']} | {int(row['n_judges'])} | "
            f"{int(row['n_within_pairs'])} | {kappa_str} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Kappa-Paradox Check")
    lines.append("")

    # Check if raw and kappa tell the same directional story
    raw_gap = pooled_stat["gap_raw"]
    kap_gap = pooled_stat["gap_kappa"]
    if (raw_gap > 0) == (kap_gap > 0):
        lines.append(
            f"Raw agreement gap ({raw_gap:+.4f}) and κ gap ({kap_gap:+.4f}) point in the **same direction**. "
            "No kappa paradox."
        )
    else:
        lines.append(
            f"**DIVERGENCE:** Raw agreement gap ({raw_gap:+.4f}) and κ gap ({kap_gap:+.4f}) point in "
            "**opposite directions**. This is a kappa-paradox instance: one or both groups have "
            "extreme label prevalence that inflates or deflates raw agreement. "
            "Prefer κ as the estimand for within/cross comparisons."
        )
    lines.append("")
    lines.append("Note: alphaNLI is binary (2 labels), so its chance-agreement floor is ~0.5, "
                 "substantially higher than MNLI/SNLI (~0.33). κ corrects for this; raw agreement does not. "
                 "This is the main source of inflation in per-split raw agreement differences.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "A positive κ gap (within > cross) would indicate that judges from the same vendor "
        "tend to agree with each other more than with judges from other vendors, net of "
        "label-imbalance effects. This is the co-failure-relevant signal: if vendor identity "
        "predicts shared label choices, a vendor-homogeneous panel is less informative than "
        "its size suggests."
    )
    lines.append("")
    lines.append(
        "**The permutation p-value tests whether the observed gap is larger than expected by "
        "chance under the null that family labels are uninformative for pairwise agreement.** "
        "The permutation procedure shuffles the family assignment across the 32 fixed judge keys "
        "while holding all vote data constant; this controls for unequal family sizes."
    )
    lines.append("")
    lines.append(
        "**Caution:** Even a statistically significant within-family κ advantage could reflect "
        "shared training data, shared task framing, or shared post-training procedures — "
        "'vendor' is a coarse proxy for any of these. Likewise, the absence of a gap would not "
        "rule out co-failure at a finer level (e.g., shared fine-tuning). This analysis tests "
        "the vendor-as-axis hypothesis, not co-failure in general."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Figure")
    lines.append("")
    lines.append("See `figure-vendor.png` — left: violin plot of pairwise κ distributions "
                 "(within vs cross, pooled); right: per-family mean within-κ bar chart.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nresults-vendor.md written ({out_path.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load panel
    judge_family = load_panel()
    print(f"Panel loaded: {len(judge_family)} judges")
    family_counts: dict[str, int] = {}
    for f in judge_family.values():
        family_counts[f] = family_counts.get(f, 0) + 1
    print("  Family sizes:", {k: v for k, v in sorted(family_counts.items())})

    # Per-split analysis
    split_dfs: list[pd.DataFrame] = []
    split_stats: list[dict] = []
    split_labels = {"mnli_m": "MNLI-m", "snli": "SNLI", "alphanli": "alphaNLI"}

    for ds in DATASETS:
        df_split = analyze_split(ds, judge_family)
        split_dfs.append(df_split)
        stat = summary_stats(df_split, split_labels[ds])
        split_stats.append(stat)
        print(
            f"  {split_labels[ds]}: within κ={stat['within_kappa']:.4f}, "
            f"cross κ={stat['cross_kappa']:.4f}, gap={stat['gap_kappa']:+.4f}"
        )

    # Pooled
    print("\n--- POOLED ---")
    pooled_df = pd.concat(split_dfs, ignore_index=True)
    pooled_stat = summary_stats(pooled_df, "Pooled")
    print(
        f"  Within κ={pooled_stat['within_kappa']:.4f}, "
        f"cross κ={pooled_stat['cross_kappa']:.4f}, "
        f"gap={pooled_stat['gap_kappa']:+.4f}"
    )
    print(
        f"  Within raw={pooled_stat['within_raw']:.4f}, "
        f"cross raw={pooled_stat['cross_raw']:.4f}, "
        f"gap={pooled_stat['gap_raw']:+.4f}"
    )

    # Per-family within-κ breakdown
    family_df = per_family_within_kappa(pooled_df, judge_family)
    print("\n--- Per-family within-κ ---")
    for _, row in family_df.iterrows():
        k_str = f"{row['mean_within_kappa']:.4f}" if not pd.isna(row["mean_within_kappa"]) else "n/a"
        print(f"  {row['family']:12s}  n_judges={int(row['n_judges'])}  "
              f"n_within_pairs={int(row['n_within_pairs'])}  mean_within_κ={k_str}")

    # Permutation test
    observed_gap, p_value = permutation_test_kappa_gap(pooled_df, judge_family, n_perm=2000)

    # Figure
    fig_path = PROJECT / "figure-vendor.png"
    make_figure(pooled_df, family_df, fig_path)

    # Results markdown
    results_path = PROJECT / "results-vendor.md"
    write_results(
        split_stats, pooled_stat, observed_gap, p_value,
        family_df, pooled_df, results_path
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
