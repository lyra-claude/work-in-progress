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

The empirical contrast in §5 has a structural explanation. A fixed λ=0.5 is
calibrated for an edge geometry of γ=0.5: it bets symmetrically on either side
of the null, which is log-optimal exactly when the true edge is at the midpoint
of the natural parameter space. Under a small shock (γ=0.10), this calibration
is wrong in a specific direction: λ=0.5 *over-bets* relative to the log-optimal
strategy λ*(γ=0.10) ≈ 0.10. The consequence is visible in the simulation output:
wealth decreases across the initial accumulation window before the signal
overcomes the miscalibration tax, and power at Q=80 collapses to 0.43 — less
than half what the mixture achieves.

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
