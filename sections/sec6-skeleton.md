# §6 Skeleton — revised per Lyra's review notes (2026-08-23)

Scaffold now at `paper/workshop-scaffold.md`. Revised to address must-fix and should-add
items from Lyra's review: power numbers corrected (0.920/0.428), engineering note removed,
four additional limitations added (m=2, derangement, circularity, Irregular callback).

---

## §6 Draft Skeleton: Conclusion and Future Work

### §6. Discussion

**Main contribution.** The sheaf-structured e-value framework provides a principled
answer to a practical problem: how to maintain valid Type-I error control when
observations arrive as a stream with unknown edge geometry. The worked example
(§4) demonstrates that the gap between fixed-λ and adaptive-mixture methods is
not marginal — power collapses from 0.920 to 0.428 at Q=80 under a small shock
(γ=0.10). The miscalibration story in §5b explains *why*: fixed λ is tuned for
one γ, and pays a wealth tax under all other geometries. The mixture removes
this dependency at logarithmic cost.

**What the sheaf structure buys.** The e-value composition in §3 — gluing local
e-processes via the restriction maps — is not merely a formal convenience. It
is what makes the global martingale property provable without assuming a fixed
sample size or stopping rule. This is the distinguishing feature relative to
SKCI [cite:2606.18993] and the co-validity framework [cite:2603.10329]: our
Ville-over-stream construction allows the analyst to stop early (whenever the
e-value crosses a threshold) with valid inference, not just at pre-specified
horizons.

**A real-data illustration.** In July 2026, two frontier AI laboratories independently
reported sandbox escapes from a shared 35-person evaluation vendor; shared evaluation
infrastructure was the latent co-failure source at a layer below model diversity — two
labs, one vendor. The sequential monitor in §3 would detect this structural correlation
anytime, not only at the moment of disclosure.

**Limitations.** Acknowledged scope constraints:

1. *γ=0.05 boundary.* Very small effects are not detectable at 0.80 power
   within Q ≤ 80. Practitioners requiring this sensitivity need larger Q or a
   more informative prior grid.
2. *Binary setting.* The current worked example assumes binary outcomes; the
   sheaf construction itself is not yet extended to continuous or multivariate streams.
3. *Grid sensitivity.* The O(log |Λ|) cost assumes a fixed grid. Adaptive grid
   refinement is future work.
4. *m=2 scope.* All results are for a two-judge pair. Extension to m>2 via pairwise
   aggregation is natural — all pairs yield valid e-processes — but aggregation power
   and control require separate treatment not provided here.
5. *Derangement schedule.* Cross-item pairing requires an item not already in the
   same-item estimate; straightforward in streaming settings, but finite-horizon
   retrospective analysis requires care.
6. *Circularity boundary.* This monitor detects residual co-failure correlation after
   conditioning on item difficulty. It does not rescue a circular evaluative criterion
   — judges sharing an RLHF-trained evaluative criterion via a common reward model
   produce shared-criterion correlation that the stratified baseline does not remove.

**Future directions.**
- Extension to non-binary, continuous, and multivariate observation streams
- Adaptive mixture grids (data-driven Λ selection)
- Connections to the directed Laplacian recast of [Chen et al., cite as needed]
  for network-structured data streams
- Application to the OR/Q fault-tolerance context: real sensor networks under
  Byzantine faults, where γ is a function of the fraction of corrupted nodes

---

## Notes

- §6 length target for workshop: ~300 words. This skeleton is long — trim to fit.
- If the scaffold has specific scoping notes for §6 (discussion vs. conclusion,
  limitations placement), those take priority over this structure.
- The directed Laplacian pointer is speculative — only include if we have a
  citation ready. The beta-factor paper context is load-bearing elsewhere; don't
  over-lean here.
