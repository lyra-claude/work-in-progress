# Vendor-Family Co-Failure Analysis — 32-Judge ChaosNLI Panel

**Paper:** Li, Yu, Li — *How Many Humans Is a Judge Panel Worth?* (2609.21277)
**Data:** 32judges-votes (CC BY 4.0), 3 splits × 1000 items × 32 judges

---

## Estimands

This analysis measures **inter-judge label agreement**, not error rate. No gold labels are used. Two estimands:

- **Raw pairwise agreement** a_ij = fraction of shared (non-parse_fail) items where judge i and judge j emit the same label. **Not chance-corrected.** alphaNLI is binary (2 labels), so random agreement baseline ≈ 50%; MNLI/SNLI ≈ 33%. Raw agreement is systematically inflated for alphaNLI and mixes two effects.

- **Cohen's κ_ij** = (a_ij − p_e) / (1 − p_e), where p_e = Σ_label freq_i(label) × freq_j(label) is the chance-agreement computed from each judge's *own* marginal label frequencies over the shared items. This controls for label imbalance and random-agreement floors. **Preferred estimand for comparing across splits.**

**Kappa-paradox caveat:** κ can be lower than raw agreement when prevalence is extreme (one label dominates). If raw agreement and κ tell different stories below, we flag it explicitly.

**Family-label source:** repo's authoritative `panel/panel-chaosnli.json` (not inferred from model names).

---

## Pair Counts

32 judges → 32×31/2 = 496 unordered pairs per split. Family sizes: OpenAI 5, Alibaba 4, Google 4, Moonshot 4, Zhipu 4, Anthropic 3, DeepSeek 3, ByteDance 2, xAI 2, MiniMax 1.

- Within-family pairs: 42 (8.5% of total)
- Cross-family pairs: 454 (91.5% of total)
- Families with only 1 judge (0 within-pairs): MiniMax

---

## Pooled Results (all 3 splits)

| Metric | Within-family | Cross-family | Gap (within − cross) |
|--------|---------------|--------------|----------------------|
| Raw agreement | 0.8459 | 0.8261 | +0.0198 |
| Cohen's κ | 0.7386 | 0.7049 | +0.0337 |

**Permutation test (κ gap, 2000 permutations, seeds 0–1999):**
- Observed κ gap: +0.033652
- One-sided p-value (fraction of permutations ≥ observed): **0.0000**

---

## Per-Split Results (κ and raw agreement)

| Split | N pairs (within) | N pairs (cross) | within κ | cross κ | κ gap | within raw | cross raw | raw gap |
|-------|-----------------|-----------------|----------|---------|-------|------------|-----------|---------|
| MNLI-m | 42 | 454 | 0.6499 | 0.6092 | +0.0407 | 0.7792 | 0.7543 | +0.0248 |
| SNLI | 42 | 454 | 0.7280 | 0.6858 | +0.0422 | 0.8394 | 0.8140 | +0.0254 |
| alphaNLI | 42 | 454 | 0.8380 | 0.8199 | +0.0181 | 0.9191 | 0.9100 | +0.0091 |

---

## Per-Family Within-κ Ranking

Sorted by mean within-family κ (descending). Computed pooled across all 3 splits.

| Rank | Family | N judges | N within-pairs | Mean within-κ |
|------|--------|----------|----------------|---------------|
| 1 | xai | 2 | 3 | 0.8614 |
| 2 | google | 4 | 18 | 0.8100 |
| 3 | bytedance | 2 | 3 | 0.7945 |
| 4 | zhipu | 4 | 18 | 0.7699 |
| 5 | alibaba | 4 | 18 | 0.7610 |
| 6 | moonshot | 4 | 18 | 0.7416 |
| 7 | openai | 5 | 30 | 0.6938 |
| 8 | anthropic | 3 | 9 | 0.6647 |
| 9 | deepseek | 3 | 9 | 0.6461 |
| 10 | minimax | 1 | 0 | n/a (1 judge) |

---

## Kappa-Paradox Check

Raw agreement gap (+0.0198) and κ gap (+0.0337) point in the **same direction**. No kappa paradox.

Note: alphaNLI is binary (2 labels), so its chance-agreement floor is ~0.5, substantially higher than MNLI/SNLI (~0.33). κ corrects for this; raw agreement does not. This is the main source of inflation in per-split raw agreement differences.

---

## Interpretation

A positive κ gap (within > cross) would indicate that judges from the same vendor tend to agree with each other more than with judges from other vendors, net of label-imbalance effects. This is the co-failure-relevant signal: if vendor identity predicts shared label choices, a vendor-homogeneous panel is less informative than its size suggests.

**The permutation p-value tests whether the observed gap is larger than expected by chance under the null that family labels are uninformative for pairwise agreement.** The permutation procedure shuffles the family assignment across the 32 fixed judge keys while holding all vote data constant; this controls for unequal family sizes.

**Caution:** Even a statistically significant within-family κ advantage could reflect shared training data, shared task framing, or shared post-training procedures — 'vendor' is a coarse proxy for any of these. Likewise, the absence of a gap would not rule out co-failure at a finer level (e.g., shared fine-tuning). This analysis tests the vendor-as-axis hypothesis, not co-failure in general.

---

## Figure

See `figure-vendor.png` — left: violin plot of pairwise κ distributions (within vs cross, pooled); right: per-family mean within-κ bar chart.
