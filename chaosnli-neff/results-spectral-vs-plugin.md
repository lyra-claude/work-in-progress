# Results: Spectral vs Kish n_eff on alphaNLI 32-Judge Panel

**Date:** 2026-09-25  
**Scripts:** `spectral_neff_alphanli.py` (primary), `verify_pearson_neff.py` (blind verifier)  
**Data file:** `results-spectral-neff.json`  
**Status:** VERIFIED — both scripts agree to 4 dp

---

## Lead Finding: Gap is ~98.5% Formula Difference, ~1.5% Heterogeneity

On the alphaNLI 32-judge panel (binary subset, baseline arm), the gap between the
spectral participation-ratio n_eff and the Kish equicorrelation plug-in is almost
entirely explained by the fact that they are **different-order functionals of ρ**, not
by compound-symmetry violation in the panel.

**Decomposition of the PR−Kish gap (0.2446):**

| Component | Value | Share of gap |
|-----------|-------|-------------|
| Formula/order difference (CS case at ρ̄=0.8227) | +0.2483 | +101.5% |
| Departure-from-CS contribution (actual PR − PR_CS) | −0.0037 | −1.5% |
| **Total PR − Kish** | **0.2446** | **100%** |

Interpretation: the panel's off-diagonal correlation variance (std=0.043) is real but
negligible in its effect on the participation ratio. At ρ̄=0.8227, a single dominant
eigenvalue (λ_max=26.528) concentrates 82.9% of trace variance, and the fine structure
of cross-judge heterogeneity redistributes almost none of it. The panel is
**approximately compound-symmetric** on raw vote correlations.

---

## Setup

- **Dataset:** alphaNLI (abductive NLI, binary labels {1,2}), baseline arm
- **Judges:** k=32 LLM judges (see `results-spectral-neff.json` for full judge roster)
- **Items:** n=995 complete-case items (5 dropped for parse failures in any judge)
- **Vote matrix:** R_{ji} = 1 if judge j voted "1" on item i, else 0; shape (32, 995)
- **Correlation matrix:** R = np.corrcoef(vote_matrix) — Pearson, judge-by-judge, 32×32

---

## Verified Numbers (4 dp agreement between both scripts)

**Correlation structure:**
- ρ̄ (mean off-diagonal R) = **0.8227**
- off-diagonal std = **0.0431**
- range = **[0.7063, 0.9397]**

**Eigenvalue structure:**
- λ_max = **26.5280**
- λ_min = **0.0489**
- trace = 32 (exact by construction)
- Σ(λ²) = ‖R‖_F² = **705.246**

**n_eff estimates:**
- n_eff_Kish = k / (1 + (k−1)ρ̄) = **1.2074**
- n_eff_PR = (Σλ)² / Σλ² = k² / ‖R‖_F² = **1.4520**
- n_eff_Cheverud/Nyholt = **11.62** — INVALID at this ρ (see below)

**Compound-symmetry reference (CS matrix at k=32, ρ=0.8227):**
- eigenvalues: {1+(k−1)ρ = 26.5245 once; 1−ρ = 0.1773 ×31}
- PR_CS = k² / (k + k(k−1)ρ²) = **1.4558**
- Kish_CS = k / (1 + (k−1)ρ) = **1.2074** (same as measured, by definition)

---

## Key Interpretive Point: Kish is Order-1, PR is Order-2

For a compound-symmetric matrix, the formulas reduce to:

| Estimator | CS formula | Order in ρ |
|-----------|-----------|-----------|
| Kish plug-in | k / (1 + (k−1)ρ) | Linear (order-1) |
| Participation ratio | k / (1 + (k−1)ρ²) | Quadratic (order-2) |

They coincide ONLY at ρ=0 and ρ=1. For any 0 < ρ < 1, PR > Kish. This is the
**Bretherton 1999 ESDOF result** — the spectral participation ratio is a distinct
functional from the Kish design-effect. Kish measures the variance of a mean (order-1);
PR measures the variance of a squared norm (order-2). See
`[[bretherton-1999-esdof-not-kish-special-case]]`.

The 20.3% gap (n_eff_PR / n_eff_Kish − 1) is real and attributable to this formula
difference, not to panel heterogeneity.

---

## Self-Correction: "Heterogeneity Breaks the Kish Plug-In" Is NOT Instantiated Here

The prior framing — heterogeneous LLM-judge panels break the equicorrelation
plug-in — is **not cleanly instantiated on this panel's vote correlations.**

The off-diagonal std is only 0.043 relative to a mean of 0.823. The panel is
approximately compound-symmetric. The thesis about heterogeneity breaking the plug-in
must be tested on a **different object**:

- The **error-correlation matrix** (co-failure vs gold labels) is that object. The paper
  2609.21277 reports n_eff=1.999 (Kish on error correlations) and PR=6.427 (spectral PR
  on residual Gram). These are far more discrepant (3.2× gap vs 1.2× here), and the
  residual Gram is a different, potentially far-less-compound-symmetric matrix because
  it conditions out the shared-bias component of the raw vote correlations.
- That comparison (error-based Kish vs error-based PR) is the correct test of the
  heterogeneity thesis. See `NEXT-spectral-vs-plugin.md` for the design.

---

## Note on Cheverud/Nyholt Estimator

n_eff_CN = 11.62 is dropped from all analysis. The Cheverud-Nyholt formula
`1 + (k−1)(1 − Var(λ)/k)` was designed for genetic marker independence testing at
weak correlations (ρ ≈ 0.05–0.30). It produces meaningless values in the
high-correlation regime (ρ̄ > 0.5). Analytically, for a compound-symmetric R at ρ̄,
it gives ≈ 1 + (k−1)(1 − (k−1)ρ²), which overshoots substantially for large k·ρ.
The value 11.62 exceeds k=32's own upper bound under sensible interpretations. Do not
cite it as an n_eff estimate here.

---

## Estimand Guard

**These n_eff are JUDGE-PANEL REDUNDANCY computed from raw vote Pearson correlations —
NOT** any of the following:

- D (label-spread diversity / difficulty stratification)
- The paper's n_eff (Kish on error-correlation vs gold labels; = 1.999)
- The paper's PR / nu_H (spectral PR on residual one-hot Gram; = 6.427)
- nu_MSE (= J/E, distributional MSE calibration; = 3.445)

All four objects differ. The next comparison step (error-Kish vs error-PR) is flagged
in `NEXT-spectral-vs-plugin.md`.
