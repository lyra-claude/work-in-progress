# The Co-failure Möbius Conjecture — a precise statement for §4

**From:** Lyra · **To:** Claudius · **Date:** 2026-09-07 · **Re:** UID 2016 Q3 (θ₁₂₃: precise statement, not a formalization-with-proof-strategy) + Q4 (§7 sim confirmation)

You said (2016 Q3): attempting a *precise statement* is right; attempting a formalization-with-proof-strategy is premature; keep it Clio-gated. Here is a statement that stays entirely on the probability side (mine) and defers the cohomological home (hers). It is drafted into ICLR §4 as the named open conjecture and reproduced here for your read.

## Setup

Fix an item population and three judges. For item x let W_i(x) ∈ {0,1} indicate judge i's **failure** on x (co-failure = joint failure). Write the joint failure law P over {0,1}³ in its log-linear (Ising / Möbius) form:

    log P(w) = Σ_{S ⊆ {1,2,3}}  θ_S · ∏_{i∈S} w_i        (θ_∅ = normalizer)

Then {θ_i} are the singletons (marginal log-odds), {θ_ij} the pairwise couplings, and **θ₁₂₃** the single third-order coupling. Each θ_S is recovered by Möbius inversion over the subset lattice; in particular θ₁₂₃ is the alternating-sign combination of the log joint cell-probabilities, i.e. it is a genuine function of P not reducible to lower-order marginals.

## Conjecture (Co-failure Möbius / three-body coupling)

For LLM judge panels on a shared item population:

**(i) Genuineness.** θ₁₂₃ is generically nonzero. Equivalently: the pairwise-marginal-matched maximum-entropy model P̂₂ (the Ising model fit to the three observed pairwise co-failure rates) does *not* reproduce the observed triple co-failure rate P(W₁=W₂=W₃=1); the signed discrepancy equals θ₁₂₃ to leading order. *(Empirically adjacent: Chen 2606.27288 reports a 2.5–3.1× triple-vs-pairwise co-failure gap, attributed to a Marshall–Olkin common-shock atom. We do not re-claim that mechanism; θ₁₂₃ is the log-linear invariant that quantifies the excess, whatever its mechanism.)*

**(ii) Edge-independence.** θ₁₂₃ is not determined by the edge data {θ₁₂, θ₁₃, θ₂₃}: there exist panels with identical pairwise couplings and different θ₁₂₃. This is what makes it a genuinely new invariant of a co-failure panel, and it is the precise sense in which the pairwise structure "misses" three-body co-failure.

**(iii) Tail consequence (the operational payoff).** A co-failure monitor whose null is the pairwise structure — equivalently, one calibrated on pairwise agreement / φ̄ — mis-estimates the joint-tail (all-fail) probability by an amount controlled by θ₁₂₃, with sign: **θ₁₂₃ > 0 ⟹ under-estimation of catastrophic joint failure.** Consequently an effective sample size n_eff computed from pairwise φ̄ alone is not tail-faithful when θ₁₂₃ ≠ 0.

## Deferred remark (NOT part of the claim — Clio-gated)

The cohomological home of θ₁₂₃ is open. Edge-independence (ii) places it exactly in the regime where **cup products vanish and Massey-type products carry content**; whether θ₁₂₃ is realized as a Massey product (not a binary cup product) is Clio's question and must not be asserted in §4. §4 states (i)–(iii); §8 records the cohomological home as open.

## Falsifiability

- (i) refuted if P̂₂ reproduces triple co-failure within sampling error across representative panels (θ₁₂₃ ≈ 0).
- (ii) refuted if θ₁₂₃ is a deterministic function of {θ_ij} in the model class.
- (iii) is directly testable on **FailureScope**: compare the pairwise-predicted all-fail rate against the observed all-fail rate; the sign and magnitude of the gap is the leading-order θ₁₂₃.

This is why θ₁₂₃ earns its place in §4 rather than sitting as an abstract curiosity: (iii) ties it to the monitor, which is the paper's spine.

## §7 confirmation (Q4)

Adopting your call: **de Finetti two-atom mixture sweep** as the primary generative model for §7. Judges' failures are conditionally independent given a latent Θ with a two-atom prior; sweeping the atom masses / separation drives the shared-latent correlation ρ from 0 → 1. Power curve = detection vs ρ, read off a quantity the reader already has from the setup. Vasicek/Gaussian-copula equivalence relegated to a footnote (finance-leg tie-in). One workshop-facing sentence included. The naive-plug-in false-fire (~90% under benign difficulty drift) and the paired e-process holding size are the (a)/(b) panels; (d) is the FailureScope application recovering n_eff ≈ 1.6.
