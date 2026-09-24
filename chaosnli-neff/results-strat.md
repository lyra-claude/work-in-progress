# ChaosNLI — Difficulty Stratification of the Diversity Gap

**Question:** do LLM judge panels reproduce the *label-spread* (disagreement) that human annotators show, and does any shortfall grow on harder / more-contested items?

**Difficulty axis:** `H_i` = normalized Shannon entropy of the 100-human label distribution, divided by `log(#labels)` so `H_i ∈ [0,1]` and is label-count-invariant. Higher `H_i` = more contested among humans.

**Spread measure:** `D` = inverse-Simpson `(Σc)²/Σc²` = the **effective number of active labels** per item, computed identically from the 100 human votes (`D_human`) and the ≤32 LLM votes (`D_llm`). `g_i = D_human_i − D_llm_i` is the gap. `D` is NOT 'effective number of raters' and NOT `n_eff_panel`; `H` is the difficulty axis, `g` is the gap — three distinct objects, never conflated.

---

## Headline finding: LLM spread under-tracks human difficulty

LLM panels **do** track human difficulty — `D_llm` rises with `H` — but they track it at only a fraction of the human rate, so the gap widens monotonically as items get harder. The honest effect measure is the **slope comparison**, not any single correlation.

### OLS slopes of D on H (pooled and per split)

| Split | slope D_human~H | slope D_llm~H | LLM % of human rate | gap slope (H) |
|-------|-----------------|---------------|---------------------|---------------|
| MNLI-m | +2.3250 | +0.5532 | 23.8% | +1.7718 |
| SNLI | +1.8027 | +0.8852 | 49.1% | +0.9176 |
| alphaNLI | +0.9151 | +0.4679 | 51.1% | +0.4472 |
| **Pooled** | +1.5536 | +0.6556 | 42.2% | +0.8980 |

**Pooled reading:** human label-spread rises `+1.55` effective labels per unit `H`; the LLM panel rises only `+0.66` (**42% of the human rate**). The gap slope `+0.90` is robustly positive: the shortfall grows with difficulty. OLS = simple least squares, no scipy.

### Tercile table (pooled, equal-count bins of H, easy → hard)

| Tercile | n | H range | H mean | mean D_human | mean D_llm | mean gap g |
|---------|---|---------|--------|--------------|------------|------------|
| 1 | 1001 | [-0.000, 0.403] | 0.208 | 1.1088 | 1.0603 | 0.0484 |
| 2 | 997 | [0.408, 0.665] | 0.558 | 1.5970 | 1.3245 | 0.2725 |
| 3 | 1002 | [0.665, 1.000] | 0.810 | 2.0804 | 1.4598 | 0.6206 |

Easy → hard, pooled: `D_human` climbs 1.11 → 1.60 → 2.08 while `D_llm` climbs only 1.06 → 1.32 → 1.46; the gap widens 0.048 → 0.273 → 0.621. Per-split tercile tables are in the appendix below.

---

## Robustness (the effect survives both stress tests)

- **Equal-n (sample-size is not the cause).** Subsampling the 100 human votes down to each item's LLM vote count (20 reps, `numpy.default_rng(seed=0)`) retains **96.1%** of the pooled gap (mean gap 0.314 → 0.302); ρ(H, gap) on the equal-n comparison is +0.578. The gap is real, not a small-sample artifact of inverse-Simpson bias.
- **Drop-ceiling (not a geometry artifact).** Only 3.1% of items sit at `H=0` (unanimous). Dropping them moves pooled ρ(H, gap) +0.606 → +0.593. The effect lives in the contested-item interior, not at the unanimous floor.

---

## Why ρ(H, gap) = 0.61 is NOT the effect size

The pooled Spearman `ρ(H, gap) ≈ +0.61` is a **headline correlation only** — it is very largely mechanical and must not be presented as the size of the effect. `gap = D_human − D_llm` contains `D_human`, and `H` and `D_human` are two summaries of the *same* human vote distribution (ρ(H, D_human) ≈ 0.93–1.0). Once you partial out `D_human`, the correlation collapses:

> **Partial ρ(H, gap | D_human) ≈ +0.027 pooled.** Almost the entire `+0.61` was the `D_human` term inside `gap` correlating with itself through `H`.

So `0.61` is a mechanical near-tautology. The genuine, non-mechanical signal is the **under-tracking slope** above (LLM D rises at ~40% of the human rate) — that quantity subtracts the LLM side and cannot be produced by the shared-distribution confound.

### Decoupling correlations (which split actually shows it)

| Split | ρ(H, D_human) | ρ(H, D_llm) | ρ(H, gap) | partial ρ(H, gap \| D_human) |
|-------|---------------|-------------|-----------|------------------------------|
| MNLI-m | +0.9637 | +0.2096 | +0.6104 | +0.0897 |
| SNLI | +0.9774 | +0.5495 | +0.5405 | +0.1501 |
| alphaNLI | +1.0000 | +0.6066 | +0.6064 | NaN |
| **Pooled** | +0.9325 | +0.5457 | +0.6061 | +0.0268 |

- **alphaNLI is EXCLUDED from the decoupling / confound claim.** It has only 2 labels, so `H` and `D_human` are exact monotone transforms of each other (ρ = 1.000). There `ρ(H, D_llm) = ρ(H, gap)` is a **tautology carrying zero decoupling information**, and the partial correlation is degenerate (reported **NaN**). alphaNLI is retained only for the raw gap (appendix), clearly marked.
- **MNLI-m is the CLEAN case.** 0% of its items are unanimous (`H=0`), and its `ρ(H, D_llm) ≈ +0.21` is far below `ρ(H, D_human) ≈ +0.96` — the largest genuine decoupling of the three splits, i.e. the strongest evidence of LLM under-tracking.

Spearman ρ = Pearson correlation of average-ranks. Partial ρ via the rank-residual method: rank-transform `H`, `gap`, `D_human`; least-squares-regress rank(H) on rank(D_human) and rank(gap) on rank(D_human); Pearson-correlate the residuals (no scipy).

---

## Bottom line

The effect is **real** — it survives equal-n subsampling (~96% of the gap retained) and drop-ceiling. But its **honest measure is the slope / tercile under-tracking** (LLM label-spread grows at ~40% of the human rate, so the gap widens monotonically with difficulty), **not** the mechanical `ρ(H, gap) = 0.61`.

---

## Figure

See `figure-strat.png` — pooled `D_human` and `D_llm` decile means against difficulty `H`, with the two OLS regression lines overlaid. The steep human line vs the shallow LLM line (≈40% of the slope), and the widening vertical gap between them, is the under-tracking effect.

---

## Appendix — full Spearman table and per-split terciles

### Spearman rank correlations (all splits)

| Split | n | ρ(H, gap) | p | ρ(H, D_llm) | p | ρ(H, D_human) | p |
|-------|---|-----------|---|-------------|---|---------------|---|
| MNLI-m | 1000 | +0.6104 | 3.91e-103 | +0.2096 | 2.19e-11 | +0.9637 | 0.00e+00 |
| SNLI | 1000 | +0.5405 | 6.24e-77 | +0.5495 | 5.58e-80 | +0.9774 | 0.00e+00 |
| alphaNLI | 1000 | +0.6064 | 1.87e-101 | +0.6066 | 1.59e-101 | +1.0000 | 0.00e+00 |
| **Pooled** | 3000 | +0.6061 | 2.17e-300 | +0.5457 | 1.75e-232 | +0.9325 | 0.00e+00 |

Two-sided p from the t-approximation `t = ρ·sqrt((n−2)/(1−ρ²))`, df = n−2, via the regularized incomplete beta (no scipy).

### Per-split tercile tables (equal-count bins of H, easy → hard)

alphaNLI raw gap shown for completeness; it is excluded from the decoupling claim (2-label tautology).

#### MNLI-m

| Tercile | n | H range | H mean | mean D_human | mean D_llm | mean gap g |
|---------|---|---------|--------|--------------|------------|------------|
| 1 | 333 | [0.051, 0.624] | 0.485 | 1.4815 | 1.3202 | 0.1613 |
| 2 | 334 | [0.624, 0.749] | 0.686 | 1.9257 | 1.4088 | 0.5168 |
| 3 | 333 | [0.749, 1.000] | 0.855 | 2.3567 | 1.5004 | 0.8563 |

#### SNLI

| Tercile | n | H range | H mean | mean D_human | mean D_llm | mean gap g |
|---------|---|---------|--------|--------------|------------|------------|
| 1 | 333 | [-0.000, 0.421] | 0.262 | 1.1837 | 1.0832 | 0.1005 |
| 2 | 334 | [0.425, 0.613] | 0.533 | 1.5958 | 1.2953 | 0.3006 |
| 3 | 333 | [0.614, 0.998] | 0.720 | 2.0468 | 1.5186 | 0.5282 |

#### alphaNLI

| Tercile | n | H range | H mean | mean D_human | mean D_llm | mean gap g |
|---------|---|---------|--------|--------------|------------|------------|
| 1 | 334 | [-0.000, 0.141] | 0.082 | 1.0225 | 1.0117 | 0.0109 |
| 2 | 333 | [0.194, 0.500] | 0.310 | 1.1229 | 1.0700 | 0.0529 |
| 3 | 333 | [0.529, 1.000] | 0.796 | 1.6250 | 1.3259 | 0.2991 |

---

## Remaining caveats

1. **Difficulty is human-defined.** Using human entropy as the difficulty axis builds in the human side. An LLM-agnostic or item-intrinsic difficulty (e.g. third-source gold-label ambiguity) would be a cleaner axis but is not available in-repo. The slope framing partly mitigates this by comparing the two sides' response to the *same* axis.
2. **Discreteness / ties.** With ~100 human votes and 2–3 labels, `H` and `D` take few distinct values; rank methods handle ties, and the equal-n test above confirms the effect is not a small-sample D bias.
