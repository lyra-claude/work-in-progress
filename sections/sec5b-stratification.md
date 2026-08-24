# §5b. Calibration Failure and the Adaptive Mixture Fix

## Narrative arc (reviewer-facing)

The section builds a single causal argument across three beats:

1. **Miscalibration failure** — fixed λ=0.5 is miscalibrated for small-γ geometry
2. **Mixture as adaptive calibration** — the mixture achieves near-log-optimal growth
   uniformly over unknown γ, without foreknowledge of the true edge
3. **Efficiency cost** — logarithmic in grid size, negligible against the power gain

The γ=0.05 caveat closes the section: an honest acknowledgment of practical range.

---

## §5b Draft

### §5b. Why Fixed λ=0.5 Fails, and What the Mixture Fixes

**Difficulty stratification as the validity axis.** Apparent co-failure correlation is
substantially attenuated by shared item difficulty (Jo, Garg & Raghavan, 2602.24086);
their parametric IRT-1 null removes the confound via model fitting. Our construction
removes it structurally: the cross-item stratified baseline ensures the partner item is
drawn from the same difficulty band, bounding shared-difficulty leakage by δ=2ε rather
than estimating it parametrically. The cost of skipping stratification is direct in the
§4 contrast: Method B (unstratified cross-item, δ=0) inflates to Type-I 0.226–0.438;
Method A (stratified) holds ≤ 0.022 across the full Q-sweep. Per-stratum e-processes
combine into the panel via a plain convex e-value mixture (w_k ≥ 0, Σ w_k = 1) — a
convex combination of e-values is Ville-valid with no independence assumption. This
stratum-mixture is distinct from the λ-mixture below, which mixes over betting fractions
within a stratum; the two should not be conflated.

**The miscalibration failure.** By §3, a fixed λ is log-optimal for exactly one shock
magnitude — suboptimal elsewhere. Under small shocks (γ=0.10), λ=0.5 over-bets: the
log-optimal λ* is substantially smaller than 0.5 (§3 constrains λ ∈ [0, 1/(1+2ε)];
the realized increment edge at γ=0.10 keeps λ* ≪ 0.5). The consequence is visible in
the §4 simulation output: wealth decreases across the initial accumulation window before
the signal overcomes the miscalibration tax, and power at Q=80 collapses to 0.428 —
less than half what the mixture achieves.

The mixture prior M(λ; Λ) over a grid Λ ⊂ (0,1) fixes this by concentrating
weight where the data are informative. Under γ=0.10, posterior weight rapidly
concentrates on small λ values that track the actual edge geometry. The key
property is *uniformity*: the mixture achieves near-log-optimal wealth growth
for any γ in the testable range, not by being tuned to a specific γ but by
adapting after observing data. Formally, the e-process under the mixture is a
valid e-process by Ville's inequality (the martingale property is preserved
under the sheaf gluing; see §3), and its growth rate approaches the oracle
rate (λ=λ*(γ)) at a cost of O(log |Λ|) per unit time — logarithmic in the
grid size, which is dominated by the power gain even for modest Q.

**The γ=0.05 boundary.** We note explicitly that γ=0.05 shocks are not
reachable at 0.80 power within Q ≤ 80 under the current parameterization. This
is a genuine constraint of the method, not a presentation choice: very small
effects require either larger sample sizes or a more informative grid prior. We
flag it because leaving it implicit would constitute the wrong kind of optimism
for a submission six days before deadline — and because a practitioner who
needs γ=0.05 sensitivity should know upfront that Q ≤ 80 is insufficient.

---

## Notes for integration into scaffold

- This section follows §5 (OR/Q main result) directly.
- The Bhardwaj Thm 4.2 attribution should be verified against the full text
  before the integration pass — confirm it's Thm 4.2 and not Prop 4.2 or an
  appendix result.
- Citations to pull from references.bib (staged 2026-08-22):
  - Ville (1939) for the martingale property
  - The SKCI paper (2606.18993, He & Sutherland) as the λ-betting competitor
  - Kohli (2605.29800, solo author — not "et al.") if cited in §5b context
- Word target: ~400-500 words in the final paper. Current draft is on the long
  side for a workshop paper; trim if needed.
