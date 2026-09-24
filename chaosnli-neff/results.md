# ChaosNLI Distributional Diversity Analysis

**Paper:** Li, Yu, Li — *How Many Humans Is a Judge Panel Worth?* (2609.21277)
**Data:** ChaosNLI v1.0 (CC BY-NC 4.0) × 32judges-votes (CC BY 4.0)

---

## Estimands

Three distinct quantities are computed here. **They are NOT interchangeable and must never be averaged together.**

- **D = distributional diversity (inverse-Simpson), per item.**  For a vote-count vector c = (c_1, ..., c_L) over the L labels, D = (Σ_l c_l)² / Σ_l c_l². This is the *effective number of active labels* — a within-item spread measure. Computed identically from the 100-human counts and the 32-LLM counts. This is the apples-to-apples axis. Named `D_human` and `D_llm` in code. **NEVER call this 'effective number of raters' or 'n_eff of judges'.**

- **n_eff_panel = k / (1 + (k−1)·φ̄), LLM-panel only.**  k=32, φ̄ = mean pairwise inter-judge Pearson error correlation across items. Requires fixed judge identity across all items; NOT computable for the anonymous human crowd. Loaded from `reference/reported_values.json` (repo-computed). This is a between-judge correlation measure, not a label-spread measure.

- **ν_H = human-equivalence (paper's calibration curve).**  How many humans a 32-judge panel is 'worth' under the paper's MSE calibration. Also loaded from `reference/reported_values.json`. Distinct from both D and n_eff_panel.

---

## D (inverse-Simpson) Results

### Per-Split and Pooled

| Split | N joined | Parse fails | D_human mean | D_human median | D_human std | D_llm mean | D_llm median | D_llm std | mean(D_H−D_L) | 95% CI | frac D_H>D_L |
|-------|----------|-------------|--------------|----------------|-------------|------------|--------------|-----------|---------------|--------|--------------|
| MNLI-m | 1000 | 1 | 1.9213 | 1.9501 | 0.4134 | 1.4098 | 1.2800 | 0.4116 | 0.5115 | [0.4795, 0.5419] | 0.854 |
| SNLI | 1000 | 0 | 1.6088 | 1.5591 | 0.3970 | 1.2990 | 1.1327 | 0.3704 | 0.3097 | [0.2870, 0.3315] | 0.851 |
| alphaNLI | 1000 | 5 | 1.2566 | 1.1050 | 0.3019 | 1.1357 | 1.0000 | 0.2591 | 0.1208 | [0.1049, 0.1365] | 0.778 |
| **Pooled** | 3000 | 6 | 1.5955 | 1.5432 | 0.4621 | 1.2815 | 1.0644 | 0.3704 | 0.3140 | [0.2995, 0.3290] | 0.828 |

Bootstrap CI: 5000 resamples, percentile method, seed=42.

---

## n_eff_panel and ν_H (separate estimands, from repo)

These numbers are loaded directly from `reference/reported_values.json` as computed by the 32judges-votes reproduction pipeline. They are SEPARATE from D above and must not be interpreted as label-spread.

| Split | n_eff_panel | ν_H (human-equiv) | PR (participation ratio) |
|-------|-------------|-------------------|--------------------------|
| MNLI-m | 1.9708 | 4.2421 | 4.2282 |
| SNLI | 2.2266 | 6.4592 | 6.4218 |
| alphaNLI | 1.9990 | 6.4990 | 6.4272 |

**n_eff_panel** = k / (1 + (k−1)·φ̄) where φ̄ is the mean Pearson inter-judge error correlation (k=32). **ν_H** = human-equivalence from the paper's MSE calibration curve. Both measure panel correlation structure, NOT label spread.

---

## Figure

See `figure.png` — violin plots of per-item D_human vs D_llm distributions for each split and pooled.

---

## Data Provenance

- **ChaosNLI v1.0**: Nie, Zhou, Bansal (EMNLP 2020). License: CC BY-NC 4.0. Not redistributed here — download from https://github.com/easonnie/ChaosNLI
- **32-judge votes**: Li, Yu, Li (2609.21277). License: CC BY 4.0. https://github.com/Chao1208/32judges-votes
- Human data source used for download: https://github.com/TheGuy-26/chaosnli-deberta-modernbert (ChaosNLI content re-licensed CC BY-NC 4.0 from original).
