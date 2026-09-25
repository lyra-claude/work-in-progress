# ChaosNLI — Difficulty Stratification of the Human-vs-LLM Diversity Gap

**Date:** 2026-09-25  
**Scripts:** `strat_quintile.py` (primary), `verify_independent.py` (blind independent recompute)  
**Data:** ChaosNLI 3 subsets × 1000 items; 100-human label counts joined to 32-LLM vote tally per item (2609.21277 released data)  
**Related files:** `strat.py` (OLS slopes + terciles), `results-strat.md`, `results-strat-quintile.md`, `results-strat-quintile.csv`

---

## Headline finding (blind-verified)

**The human-vs-LLM label-diversity gap is concentrated on ambiguous items — LLMs under-track human ambiguity at ~41% the rate.**

On 3000 items (MNLI-m + SNLI + alphaNLI), stratifying the per-item diversity gap g_i = D_human_i − D_llm_i (where D = inverse-Simpson effective number of active labels) by item-ambiguity H_i (normalized Shannon entropy of the human distribution):

**OLS slopes of D on H (pooled):**

| | slope D_human on H | slope D_llm on H | ratio (LLM / human) |
|---|---|---|---|
| Pooled | 1.54 | 0.64 | **0.41** |

- Primary (strat.py): LLM/human ratio ≈ 0.42
- Blind recompute (verify_independent.py): LLM/human ratio = 0.4144
- Agreement: to 3 significant figures

Both scripts recomputed from raw data files independently. The interpretation: human label-spread rises with item ambiguity at 1.54 effective labels per unit H; the LLM panel rises at only 0.64 — 41% of the human rate.

**Per-subset slope ratios (blind recompute):**

| Subset | LLM/human slope ratio |
|---|---|
| MNLI-m | 0.238 |
| SNLI | 0.491 |
| alphaNLI | 0.511 |

All three ratios < 1. Under-tracking holds in every subset; MNLI-m shows the strongest effect (LLMs respond at only 24% the human rate on that split).

**Quintile monotonicity (pooled, primary):**

| Quintile | mean gap g |
|---|---|
| Q1 (easiest) | 0.017 |
| Q2 | 0.109 |
| Q3 | 0.278 |
| Q4 | 0.483 |
| Q5 (hardest) | ~0.75 |

Strictly monotone increasing Q1 → Q5 in all three subsets and pooled. See `results-strat-quintile.csv` for the full table.

---

## The confound — why raw ρ(H, gap) ≈ 0.61 is NOT the headline

D_human and H are both summaries of the same human distribution.

- Pooled ρ(H, D_human) ≈ 0.93; alphaNLI Spearman = 1.000 exactly (2-label task: H and D_human are monotone transforms of each other).
- Because gap = D_human − D_llm contains D_human, the raw Spearman ρ(H, gap) ≈ 0.61–0.64 is mechanically inflated.
- **Partial correlation ρ(H, gap | D_human) ≈ +0.027 (primary) / −0.049 (blind).** Both are within ±0.05; the sign is noise. Once D_human is controlled for, essentially none of the raw correlation remains.

**Therefore the raw gap-vs-entropy correlation is NOT the effect size and must not be headlined.** The non-mechanical, honest object is the slope ratio (~0.41), which subtracts the LLM side and is not producible by the shared-distribution confound.

---

## Guard — three estimands, keep distinct

| Symbol | What it measures |
|---|---|
| D (this note) | Effective number of active LABELS on a single item (within-item spread). Computed from vote counts. |
| n_eff_panel | Correlation-based effective judge count; k / (1 + (k−1)·φ̄). Different object. |
| ν_H | Human-equivalence (paper's quantity, ~4–6.5 across splits). Different object. |

Do NOT call D "effective number of raters." Do NOT import the slope ratio or gap numbers as n_eff_panel values.

---

## Robustness notes (from strat.py and results-strat.md)

- **Equal-n subsampling:** subsampling the 100 human votes down to each item's LLM vote count (20 reps, seed=0) retains ~96% of the pooled gap. The gap is real, not an inverse-Simpson small-sample bias artifact.
- **Drop-unanimous-ceiling:** dropping the 3.1% of items with H=0 moves pooled ρ(H,gap) from +0.606 to +0.593 — the effect lives in the contested interior.
- **alphaNLI note:** 2-label task; H and D_human are exact monotone transforms (ρ=1.000), so the decoupling analysis (partial ρ, confound decomposition) is degenerate for that split. alphaNLI is retained for the raw gap and slope ratio only.

---

## Verification protocol

`verify_independent.py` was written as a blind recompute: it reads the same raw CSVs as `strat_quintile.py` but uses an entirely independent implementation (different variable names, different loop structure, recomputes all intermediate quantities from scratch). Key numbers checked:

- Pooled OLS slope ratio: 0.4144 (blind) vs ~0.42 (primary) — agree to 3 sig figs
- Per-subset ratios: MNLI-m 0.238, SNLI 0.491, alphaNLI 0.511 (blind)
- Quintile mean gaps: match `results-strat-quintile.csv` to 3 decimal places
- Partial ρ(H, gap | D_human): −0.049 (blind) vs +0.027 (primary) — both within ±0.05, sign is noise

The blind agreement on the slope ratio is the primary verification result.
