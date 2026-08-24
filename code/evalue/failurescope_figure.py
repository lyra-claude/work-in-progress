"""
failurescope_figure.py
======================
Generate a two-panel publication-quality figure from the FailureScope dataset
showing: (LEFT) pairwise co-failure correlation φ matrix for 6 frontier judges;
(RIGHT) raw vs. difficulty-conditional excess co-failure (anti-artifact result).

Saves: code/evalue/failurescope_real.png (≥150 dpi, tight layout)

Run from repo root:
    python3 code/evalue/failurescope_figure.py
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})

DATA_PATH = "/home/lyra/data-scratch/failurescope/failurescope-single-turn-v1.0.0/failures.json"
OUT_PATH = "code/evalue/failurescope_real.png"

FRONTIER_MODELS = [
    "claude-haiku",
    "claude-sonnet",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5.4",
    "gpt-5.4-nano",
]

# Shorter display names for axes
MODEL_LABELS = [
    "Haiku",
    "Sonnet",
    "GPT-4o",
    "4o-mini",
    "GPT-5.4",
    "5.4-nano",
]

OPEN_MODELS = [
    "falcon-7b",
    "llama2-13b",
    "llama2-70b",
    "llama2-7b",
    "llama3-70b-instruct",
    "mistral-7b-v0.1",
    "mixtral-8x22b",
    "mixtral-8x7b",
    "phi-1.5",
    "qwen1.5-72b",
    "yi-34b",
    "zephyr-7b",
]

# ── Load data ──────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    failures = json.load(f)

# Build full-set matrix (1253 × 6)
mat_all = []
for r in failures:
    ev = {e["model_id"]: e["passed"] for e in r["eval_results"]}
    mat_all.append([0 if ev[m] else 1 for m in FRONTIER_MODELS])
mat_all = np.array(mat_all, dtype=float)
N_all = len(mat_all)

# Build held-out set (218 × 6) with open-model difficulty proxy
rows_held = []
for r in failures:
    ev = {e["model_id"]: e["passed"] for e in r["eval_results"]}
    open_fails = [0 if ev[m] else 1 for m in OPEN_MODELS if m in ev]
    if len(open_fails) >= 10:
        rows_held.append({
            "frontier_fail": np.array([0 if ev[m] else 1 for m in FRONTIER_MODELS]),
            "diff_open": np.mean(open_fails),
        })

mat_held = np.array([r["frontier_fail"] for r in rows_held], dtype=float)
diff_held = np.array([r["diff_open"] for r in rows_held])
N_held = len(mat_held)

k = len(FRONTIER_MODELS)
pairs = list(combinations(range(k), 2))

# ── Compute φ matrix (full 1253-item set) ─────────────────────────────────────
phi_matrix = np.eye(k)
excess_raw_per_pair = {}
for i, j in pairs:
    phi = np.corrcoef(mat_all[:, i], mat_all[:, j])[0, 1]
    phi_matrix[i, j] = phi
    phi_matrix[j, i] = phi
    p1, p2 = mat_all[:, i].mean(), mat_all[:, j].mean()
    excess_raw_per_pair[(i, j)] = (mat_all[:, i] * mat_all[:, j]).mean() - p1 * p2

mean_phi = np.mean([phi_matrix[i, j] for i, j in pairs])
kish_neff = k / (1 + (k - 1) * mean_phi)
mean_excess_raw = np.mean(list(excess_raw_per_pair.values()))

# ── Compute conditioned excess on 218-item held-out set ───────────────────────
N_BINS = 5
bin_edges = np.percentile(diff_held, np.linspace(0, 100, N_BINS + 1))
bin_edges[-1] += 1e-9
bins_assigned = np.digitize(diff_held, bin_edges[1:])

# Raw excess on 218-item subset
raw_218 = []
for i, j in pairs:
    p1, p2 = mat_held[:, i].mean(), mat_held[:, j].mean()
    excess = (mat_held[:, i] * mat_held[:, j]).mean() - p1 * p2
    raw_218.append(excess)
mean_excess_218 = np.mean(raw_218)

# Difficulty-conditional excess on 218-item held-out set
cond_held_per_pair = []
for i, j in pairs:
    ws, tw = 0.0, 0
    for b in range(N_BINS):
        mask = bins_assigned == b
        n_b = mask.sum()
        if n_b < 2:
            continue
        x1 = mat_held[mask, i]
        x2 = mat_held[mask, j]
        p1, p2 = x1.mean(), x2.mean()
        excess = (x1 * x2).mean() - p1 * p2
        ws += excess * n_b
        tw += n_b
    cond_held_per_pair.append(ws / tw if tw > 0 else np.nan)
mean_cond_held = np.nanmean(cond_held_per_pair)

pct_attr = (mean_excess_218 - mean_cond_held) / mean_excess_218 * 100
pct_survival = 100.0 - pct_attr

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

# Colorblind-safe palette: use the "RdYlGn" diverging map restricted to
# [0, 1] range (phi goes 0–1). We clip diagonal at 1.0 as expected.
# Use a perceptually uniform colorblind-safe ramp (viridis-adjacent).
cmap = plt.colormaps["YlOrRd"]  # yellow→red, readable under most CVD

# ── PANEL A: φ heatmap ────────────────────────────────────────────────────────
ax = axes[0]
im = ax.imshow(phi_matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

# Annotate each cell
for i in range(k):
    for j in range(k):
        val = phi_matrix[i, j]
        # Use dark text on light cells, light text on dark cells
        text_color = "black" if val < 0.65 else "white"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight="bold")

ax.set_xticks(range(k))
ax.set_yticks(range(k))
ax.set_xticklabels(MODEL_LABELS, rotation=35, ha="right", fontsize=8)
ax.set_yticklabels(MODEL_LABELS, fontsize=8)

ax.set_title(
    f"Pairwise co-failure correlation φ\n"
    f"N=1,253 items · φ̄={mean_phi:.3f} · n_eff={kish_neff:.3f}",
    fontsize=9.5, pad=8,
)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("φ (Pearson corr. of fail indicators)", fontsize=7.5)
cbar.ax.tick_params(labelsize=7.5)

# ── PANEL B: Raw vs. Conditional excess bar chart ─────────────────────────────
ax2 = axes[1]

# Colorblind-safe: use blue (raw) and orange (conditional) — safe for
# protanopia/deuteranopia. Chosen from Wong (2011) colorblind palette.
COLOR_RAW = "#0072B2"       # blue
COLOR_COND = "#E69F00"      # amber/orange

bar_labels = ["Raw excess\n(N=218)", "Difficulty-conditional\nexcess (N=218)"]
bar_values = [mean_excess_218, mean_cond_held]
bar_colors = [COLOR_RAW, COLOR_COND]

bars = ax2.bar(bar_labels, bar_values, color=bar_colors, width=0.45,
               edgecolor="black", linewidth=0.7, zorder=3)

# Annotate bars with value
for bar, val in zip(bars, bar_values):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             val + 0.001,
             f"{val:.4f}",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Add a survival bracket
y_top = max(bar_values) * 1.18
ax2.annotate(
    "",
    xy=(1, bar_values[1] + 0.001),
    xytext=(1, y_top),
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
)
ax2.annotate(
    "",
    xy=(0, bar_values[0] + 0.001),
    xytext=(0, y_top),
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
)
ax2.annotate(
    "",
    xy=(1, y_top),
    xytext=(0, y_top),
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
)
ax2.text(0.5, y_top + 0.0015,
         f"{pct_survival:.1f}% survives\ndifficulty conditioning",
         ha="center", va="bottom", fontsize=8, color="darkgreen", fontweight="bold")

ax2.set_ylim(0, y_top + 0.012)
ax2.set_ylabel("Mean pairwise excess co-failure\nP(both fail) − P(fail₁)·P(fail₂)", fontsize=8.5)
ax2.set_title(
    "Excess co-failure: raw vs. difficulty-conditional\n"
    "Open-source difficulty proxy (12 models, 5 quintile bins)",
    fontsize=9.5, pad=8,
)
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
ax2.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Legend patches
raw_patch = mpatches.Patch(color=COLOR_RAW, label="Raw (marginal-independence baseline)")
cond_patch = mpatches.Patch(color=COLOR_COND,
                             label="Conditioned on held-out difficulty\n(unbiased — independent proxy)")
ax2.legend(handles=[raw_patch, cond_patch], fontsize=7.5, loc="upper right",
           framealpha=0.9)

fig.suptitle(
    "FailureScope real-data validation: co-failure excess among 6 frontier LLM judges\n"
    "Dataset: FailureScope arXiv 2606.09878 (Saban et al.), CC-BY-4.0/MIT",
    fontsize=9, y=1.01,
)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
print(f"Figure saved to {OUT_PATH}")
print(f"\nKey numbers used in figure:")
print(f"  φ̄ (full 1253-item set):         {mean_phi:.4f}")
print(f"  Kish n_eff (k=6):               {kish_neff:.3f}")
print(f"  Raw excess (218-item subset):   {mean_excess_218:.4f}")
print(f"  Cond. excess (held-out diff):   {mean_cond_held:.4f}")
print(f"  % surviving conditioning:       {pct_survival:.1f}%")
