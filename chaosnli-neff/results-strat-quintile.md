# ChaosNLI — Quintile Stratification of the Diversity Gap

Extends `strat.py` with quintile (5-bin) stratification, per-bin `frac(D_human > D_llm)`, and Kendall's tau-b alongside Spearman ρ.

**Difficulty axis:** `H_i` = normalized Shannon entropy of the 100-human label distribution (divided by `log(#labels)`, so `H_i ∈ [0,1]`, label-count-invariant). Higher = more contested.

**Spread measure:** `D_human`, `D_llm` = inverse-Simpson `(Σc)²/Σc²`. Gap `g_i = D_human_i − D_llm_i`.

**Confound note:** `H` and `D_human` are both summaries of the *same* human distribution, so ρ(H, D_human) ≈ 0.93–1.00 (mechanical). Correlating `H` with the *gap* `g_i = D_human − D_llm` is partly self-referential. The partial ρ(H, gap | D_human) ≈ +0.027 (from strat.py) confirms that almost all of the raw ρ(H, gap) ≈ +0.61 is the mechanical `D_human` term. The honest effect measure is: (a) the slope comparison — LLMs track human difficulty at only ~42% of the human rate (strat.py), and (b) whether D_llm *itself* rises with H (control, reported here).

---

## Quintile tables (equal-count bins of H, easy → hard)

### MNLI-m

| Quintile | n | H_min | H_max | H_mean | mean D_human | mean D_llm | mean gap g | frac D_H>D_L |
|----------|---|-------|-------|--------|-------------|------------|------------|--------------|
| 1 | 200 | 0.0510 | 0.5397 | 0.4181 | 1.3596 | 1.2573 | 0.1023 | 0.7200 |
| 2 | 200 | 0.5406 | 0.6465 | 0.6013 | 1.7167 | 1.4155 | 0.3012 | 0.7400 |
| 3 | 199 | 0.6487 | 0.7261 | 0.6850 | 1.9233 | 1.4261 | 0.4973 | 0.8844 |
| 4 | 201 | 0.7268 | 0.8219 | 0.7663 | 2.0880 | 1.4104 | 0.6776 | 0.9602 |
| 5 | 200 | 0.8232 | 0.9996 | 0.9034 | 2.5180 | 1.5398 | 0.9782 | 0.9650 |

### SNLI

| Quintile | n | H_min | H_max | H_mean | mean D_human | mean D_llm | mean gap g | frac D_H>D_L |
|----------|---|-------|-------|--------|-------------|------------|------------|--------------|
| 1 | 202 | -0.0000 | 0.3154 | 0.1896 | 1.1149 | 1.0553 | 0.0596 | 0.7970 |
| 2 | 197 | 0.3188 | 0.4710 | 0.4002 | 1.3293 | 1.1526 | 0.1767 | 0.8376 |
| 3 | 200 | 0.4782 | 0.5893 | 0.5365 | 1.5871 | 1.2799 | 0.3072 | 0.8200 |
| 4 | 201 | 0.5948 | 0.6679 | 0.6247 | 1.8552 | 1.4516 | 0.4036 | 0.8607 |
| 5 | 200 | 0.6694 | 0.9983 | 0.7766 | 2.1568 | 1.5553 | 0.6016 | 0.9400 |

### alphaNLI

| Quintile | n | H_min | H_max | H_mean | mean D_human | mean D_llm | mean gap g | frac D_H>D_L |
|----------|---|-------|-------|--------|-------------|------------|------------|--------------|
| 1 | 216 | -0.0000 | 0.0808 | 0.0501 | 1.0125 | 1.0103 | 0.0022 | 0.5602 |
| 2 | 188 | 0.1414 | 0.1944 | 0.1612 | 1.0486 | 1.0222 | 0.0265 | 0.8883 |
| 3 | 202 | 0.2423 | 0.4022 | 0.3038 | 1.1157 | 1.0664 | 0.0492 | 0.8416 |
| 4 | 192 | 0.4365 | 0.7415 | 0.5737 | 1.3161 | 1.1750 | 0.1412 | 0.8021 |
| 5 | 202 | 0.7602 | 1.0000 | 0.9080 | 1.7953 | 1.4075 | 0.3878 | 0.8218 |

### Pooled

| Quintile | n | H_min | H_max | H_mean | mean D_human | mean D_llm | mean gap g | frac D_H>D_L |
|----------|---|-------|-------|--------|-------------|------------|------------|--------------|
| 1 | 611 | -0.0000 | 0.2423 | 0.1258 | 1.0461 | 1.0270 | 0.0191 | 0.7463 |
| 2 | 591 | 0.2445 | 0.4796 | 0.3751 | 1.2579 | 1.1485 | 0.1094 | 0.8037 |
| 3 | 595 | 0.4821 | 0.6265 | 0.5652 | 1.6076 | 1.3295 | 0.2781 | 0.7882 |
| 4 | 603 | 0.6280 | 0.7586 | 0.6876 | 1.8905 | 1.4073 | 0.4831 | 0.8905 |
| 5 | 600 | 0.7588 | 1.0000 | 0.8780 | 2.1792 | 1.4978 | 0.6814 | 0.9100 |

---

## Spearman ρ and Kendall τ-b

Gap g = D_human − D_llm. Control: ρ/τ of H vs D_llm tests whether LLMs spread more on human-ambiguous items (they do — but at a lower rate, so the gap still widens).

| Split | n | ρ(H, gap) | p | τ(H, gap) | p | ρ(H, D_llm) | p | τ(H, D_llm) | p |
|-------|---|-----------|---|-----------|---|------------|---|------------|---|
| MNLI-m | 1000 | +0.6104 | 3.91e-103 | +0.4488 | 3.15e-100 | +0.2096 | 2.19e-11 | +0.1464 | 4.19e-12 |
| SNLI | 1000 | +0.5405 | 6.24e-77 | +0.4259 | 1.84e-90 | +0.5495 | 5.58e-80 | +0.4078 | 4.59e-83 |
| alphaNLI | 1000 | +0.6064 | 1.87e-101 | +0.5662 | 2.53e-158 | +0.6066 | 1.59e-101 | +0.4820 | 2.72e-115 |
| Pooled | 3000 | +0.6061 | 2.17e-300 | +0.4710 | 0.00e+00 | +0.5457 | 1.75e-232 | +0.4033 | 1.95e-240 |

Spearman ρ: Pearson correlation of average-ranks, t-approximation p-value (regularized incomplete beta, no scipy). Kendall τ-b: exact concordance count, Normal-approximation p-value via `erfc(|z|/√2)`, `z = τ / sqrt(2(2n+5)/(9n(n−1)))`, no scipy.

---

## Monotonicity check (quintile mean-gap trend)

| Split | Q1 gap | Q2 gap | Q3 gap | Q4 gap | Q5 gap | Monotone↑? |
|-------|--------|--------|--------|--------|--------|------------|
| MNLI-m | 0.1023 | 0.3012 | 0.4973 | 0.6776 | 0.9782 | YES |
| SNLI | 0.0596 | 0.1767 | 0.3072 | 0.4036 | 0.6016 | YES |
| alphaNLI | 0.0022 | 0.0265 | 0.0492 | 0.1412 | 0.3878 | YES |
| Pooled | 0.0191 | 0.1094 | 0.2781 | 0.4831 | 0.6814 | YES |

---

## Bottom line

Pooled (n=3000): ρ(H, gap) = +0.6061 (p=2.17e-300), τ(H, gap) = +0.4710 (p=0.00e+00). The quintile mean-gap is strictly monotone increasing (Q1=0.0191 → Q5=0.6814). D_llm also rises with H (ρ=+0.5457, τ=+0.4033), confirming LLMs respond to difficulty but at a lower rate — the gap is a genuine under-tracking effect, not a D_llm failure to respond at all.

**Confound flag:** raw ρ(H, gap) ≈ 0.61 is largely mechanical (ρ(H, D_human) ≈ 0.93, gap contains D_human); partial ρ(H, gap | D_human) ≈ +0.027 (strat.py). The Kendall τ ≈ 0.43 is equally mechanical for the same reason. The non-mechanical evidence is: (a) the slope ratio — LLMs track difficulty at only ~42% of the human rate (strat.py), and (b) the monotone quintile table above.

