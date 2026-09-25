# Design Note: Extending Spectral vs Kish n_eff to 3-Way Tasks and Paper Comparison

**Status:** Planning note only. Implementation deferred.  
**Companion script:** `spectral_neff_alphanli.py` (alphaNLI binary subset, complete).

---

## 1. Extension to 3-Way Tasks (MNLI-m, SNLI)

For binary labels, building the judge vote matrix is unambiguous: each judge's vote
vector is a binary {0, 1} sequence over items, and Pearson correlation is well-defined.

For 3-label tasks (MNLI-m, SNLI; labels e/n/c), the vote for each item is categorical,
not ordinal. The Pearson correlation between two integer-coded sequences (e.g., 0/1/2)
would impose an ordering that doesn't exist. A defensible construction is:

**Proposed construction — flattened one-hot representation:**

Represent each judge as a vector of length `n_items * 3`:
- for item i, the judge's vote is encoded as a one-hot triplet (1 in the voted slot,
  0 in the others)
- concatenate over all n_items items to get a judge vector of length 3000

Compute R = Pearson correlation between these flattened one-hot vectors over the 32
judges. The resulting 32×32 R is positive semidefinite, the diagonal entries are 1.0
(after normalization), and off-diagonal entries reflect agreement structure.

**Modeling choice note:** this is a specific, defensible choice, not the unique correct
one. Alternatives include (a) Goodman-Kruskal gamma, (b) polychoric correlation, or
(c) the paper's own residual-Gram construction (one-hot minus human distribution).
The flattened one-hot Pearson is the cleanest PSD construction that admits direct
comparison to the binary case.

**Practical note:** the existing `formulas.py::onehot_panel` and `residual_gram`
in the repo already operate on one-hot tensors; the one-hot vote matrix for 3-label
tasks can be constructed via `formulas.onehot_panel(idx, n_labels=3)` where `idx`
is the (32, n_items) integer label array. Flatten to (32, 3*n_items) before corrcoef.

---

## 2. Relation to the Paper's Own n_eff Values — CRUX TO RESOLVE

The paper 2609.21277 reports two types of effective numbers per dataset:

| Symbol  | Description                          | alphaNLI value |
|---------|--------------------------------------|----------------|
| nu_H    | Spectral PR on residual Gram         | 6.4272         |
| nu_MSE  | Human-equivalence from calibration   | 3.4445         |
| n_eff   | Kish plug-in on ERROR correlations   | 1.9990         |

**Open verify-first question (DO NOT assume the answer):**

Is the paper's `nu_MSE` (= 3.4445 for alphaNLI) the Kish equicorrelation plug-in
computed on the vote-agreement correlation matrix — or is it a different distributional
form (specifically: the MSE calibration form nu_MSE = J / E, where J = human variance
and E = panel distributional error)?

From reading `formulas.py::panel_metrics`:
- `n_eff` = Kish on phi_bar (mean ERROR Pearson correlation vs gold labels)
- `nu_MSE` = J / E (ratio of human variance to panel MSE; a different quantity)
- `PR` (= nu_H context) = spectral participation ratio on the RESIDUAL GRAM

So `nu_MSE` is **NOT** the Kish plug-in on the raw vote-agreement correlation matrix.
It is instead J/E — a ratio of distributional divergence quantities. This means the
paper does NOT already provide a Kish plug-in on raw vote correlations, and our
`n_eff_Kish` from `spectral_neff_alphanli.py` is a distinct, novel measurement.

**Consequence if confirmed:**
- The paper's nu_H (6.43) ≈ PR on residual Gram — measures residual-space concentration
- Our n_eff_PR (1.45) = PR on raw binary vote correlations — measures vote-agreement concentration  
- Our n_eff_Kish (1.21) = Kish on raw vote correlations — the equicorrelation approximation
- The paper's n_eff (2.00) = Kish on ERROR correlations vs gold labels

None of these four are the same object. The gap n_eff_PR (1.45) vs n_eff_Kish (1.21)
showing 20% compound-symmetry violation is a thesis-instantiating measurement that
does NOT duplicate anything in the paper.

**Action before publishing the comparison:** read `formulas.py::panel_metrics` and
`verify_paper.py` to confirm `nu_MSE` is definitively J/E and not a Kish form.
Check `reference/reported_values.json` keys: `E`, `J`, `nu_MSE` — verify nu_MSE = J/E
numerically. Then write the four-way comparison table clearly labeling each estimand.

---

## 3. Summary of n_eff Landscape (alphaNLI, for orientation)

| n_eff form | Value | Object | Comment |
|------------|-------|--------|---------|
| n_eff_Kish (raw vote corr) | 1.21 | k/(1+(k-1)*rho_bar) on raw binary votes | our computation |
| n_eff_PR (raw vote corr)   | 1.45 | spectral PR on raw binary votes | our computation; 20% above Kish |
| n_eff_CN (raw vote corr)   | 11.62 | Cheverud/Nyholt on raw binary votes | UNRELIABLE — see note below |
| paper n_eff                | 2.00 | Kish on ERROR corr vs gold labels | paper computation |
| paper PR / nu_H            | 6.43 | spectral PR on residual Gram | paper computation |
| paper nu_MSE               | 3.44 | J/E (distributional MSE calibration) | paper computation |

**Note on n_eff_CN:** the Cheverud/Nyholt formula 1 + (k-1)*(1 - Var(lambda)/k)
gives 11.62, far above the other estimates and above k=32's theoretical maximum.
Crucially, it also gives ~11.67 for the COMPOUND-SYMMETRIC matrix at rho_bar=0.8227
— so it's not a heterogeneity artifact; the formula itself is incorrect at high
correlation. Algebraically: n_eff_CN → 1 + (k-1) - (k-1)^2*rho^2 for CS, which
goes negative for large k and rho, then the formula overshoots for intermediate rho.
The CN formula was designed for genetics p-value correction at weak correlations
(rho ~ 0.05–0.30). It is unreliable at rho > 0.5 and produces meaningless values
in the high-correlation regime here. Drop it from any published comparison.

---

## 4. Compound-Symmetry Finding (binary alphaNLI) — Corrected Interpretation

- rho_bar = 0.8227, std(off-diagonal) = 0.0431
- range: [0.7063, 0.9397] — visible spread, but small relative to mean
- lambda_max = 26.53 (single dominant mode)

**The PR-Kish gap is NOT a compound-symmetry violation diagnostic in this case.**

Critical algebraic fact: spectral PR and Kish plug-in are DIFFERENT functional forms
and they agree ONLY at rho=0 and rho=1. For any intermediate rho in the compound-
symmetric case, PR > Kish. At rho_bar=0.8227, even a perfectly compound-symmetric
matrix gives PR_CS=1.4557 vs Kish=1.2074 — a gap of 0.2483.

Decomposition of the actual PR-Kish gap (0.2446):
- Formula difference (CS case at same rho_bar): +0.2483 (101.5% of gap)
- Compound-symmetry violation (actual PR - CS-predicted PR): -0.0037 (-1.5% of gap)

The panel is **approximately compound-symmetric** on the raw vote correlation. The off-
diagonal heterogeneity (std=0.0431, range 0.71–0.94) is real but has negligible effect
on PR because the dominant eigenvalue at 26.53 so thoroughly concentrates variance that
the fine structure of off-diagonal variation doesn't redistribute it meaningfully.

**For the thesis:** the 20% gap between n_eff_PR (1.452) and n_eff_Kish (1.207) is
real but needs to be attributed correctly — it's a formula difference between two
estimators, not evidence that the panel violates compound symmetry on raw vote
correlations. Both numbers are correct; they measure slightly different things.
The stronger compound-symmetry violation claim requires checking whether the
RESIDUAL Gram (one-hot minus human probs, as in the paper's PR computation) is
compound-symmetric — that's a different matrix and likely shows more heterogeneity
because it conditions out the shared-bias component.
