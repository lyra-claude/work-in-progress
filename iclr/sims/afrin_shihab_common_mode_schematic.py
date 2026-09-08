"""
afrin_shihab_common_mode_schematic.py

Motivation schematic for the ICLR paper introduction.

Shows the orthogonal decomposition of the judge-panel error vector e in R^J
into a common-mode component along 1=(1,...,1) and a disagreement component
in the orthogonal complement. Illustrates that reward hacking / common-mode
failure shifts e along the all-ones direction while leaving disagreement
unchanged — so a monitor watching only disagreement is blind to it.

Grounded in:
  Afrin & Shihab (2026), arXiv 2608.08002,
  "Evaluator Ensembles Under Reward Hacking: Covariance Geometry and
  Finite-Search Guarantees", Prop 3 (exact projector decomposition)
  and Prop 4 (identification limit).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 200,
})

fig, ax = plt.subplots(figsize=(6.5, 5.5))

# ── Coordinates ────────────────────────────────────────────────────────────
# e0: initial error vector (projected onto the 2-D plane)
e0 = np.array([1.4, 0.9])   # (disagreement, common-mode)
# e1: after reward hacking — same disagreement, higher common-mode
e1 = np.array([1.4, 2.2])

# ── Dashed horizontal guide (shared disagreement coordinate) ───────────────
ax.plot([0.0, e0[0]], [e0[1], e0[1]],
        color="steelblue", lw=1.0, ls="--", zorder=1, alpha=0.6)
ax.plot([0.0, e1[0]], [e1[1], e1[1]],
        color="#c0392b", lw=1.0, ls="--", zorder=1, alpha=0.45)

# horizontal guide showing same x-coordinate for both points
ax.annotate("",
            xy=(e1[0] + 0.55, e0[1]),
            xytext=(e1[0] + 0.55, e1[1]),
            arrowprops=dict(arrowstyle="<->", color="dimgray",
                            lw=0.8, mutation_scale=10))
ax.text(e1[0] + 0.65, (e0[1] + e1[1]) / 2,
        "disagreement\nunchanged",
        fontsize=9, color="dimgray", va="center", ha="left")

# dashed vertical guide at shared disagreement x
ax.axvline(x=e0[0], color="dimgray", lw=0.7, ls=":", alpha=0.5, zorder=0)

# ── Initial error vector e0 ────────────────────────────────────────────────
ax.annotate("",
            xy=e0, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>",
                            color="steelblue", lw=1.5,
                            mutation_scale=14))
ax.scatter(*e0, color="steelblue", s=55, zorder=5)
ax.text(e0[0] - 0.22, e0[1] - 0.16,
        r"$e_0$", fontsize=13, color="steelblue", va="top")

# ── Reward-hacking arrow (bold, vertical) ─────────────────────────────────
ax.annotate("",
            xy=e1, xytext=e0,
            arrowprops=dict(arrowstyle="-|>",
                            color="#c0392b", lw=2.8,
                            mutation_scale=18))
ax.scatter(*e1, color="#c0392b", s=70, zorder=5)
ax.text(e1[0] - 0.22, e1[1] + 0.12,
        r"$e_1$", fontsize=13, color="#c0392b", va="bottom")

# label on the reward-hacking arrow
ax.text(e0[0] - 0.62, (e0[1] + e1[1]) / 2,
        "reward\nhacking\n(+common-mode)",
        fontsize=9, color="#c0392b", va="center", ha="center",
        style="italic")

# ── Annotation box ────────────────────────────────────────────────────────
box_text = (
    "A disagreement monitor reads only\n"
    "the horizontal axis\n"
    r"$\Rightarrow$ blind to motion along $\mathbf{1}$"
    "\n(Afrin & Shihab 2026, Prop. 3–4)"
)
ax.text(0.03, 0.97, box_text,
        transform=ax.transAxes,
        fontsize=9.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                  ec="goldenrod", lw=1.0, alpha=0.92))

# ── Axes cosmetics ────────────────────────────────────────────────────────
ax.set_xlim(-0.15, 3.0)
ax.set_ylim(-0.15, 3.0)
ax.set_xlabel(
    r"Disagreement  (orthogonal complement,  $Pe$)",
    labelpad=6)
ax.set_ylabel(
    "Common-mode error  (all-ones direction,  " + r"$\bar{e}\cdot\mathbf{1}$)" +
    "\n[drives ensemble risk]",
    labelpad=6)
ax.set_title(
    "Common-mode vs. disagreement decomposition\n"
    r"of the judge-panel error vector $e \in \mathbb{R}^J$",
    pad=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# origin label
ax.text(-0.09, -0.09, "0", fontsize=11, color="dimgray")
ax.plot(0, 0, "k+", ms=8, mew=1.2, zorder=6)

fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────
import pathlib

outdir = pathlib.Path(__file__).parent.parent / "notes"
outdir.mkdir(exist_ok=True)

png_path = outdir / "afrin_shihab_common_mode_schematic.png"
pdf_path = outdir / "afrin_shihab_common_mode_schematic.pdf"

fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print(f"PNG saved: {png_path}")
print(f"PDF saved: {pdf_path}")
