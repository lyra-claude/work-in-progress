# ICLR 2027 Expansion Outline

**From:** Lyra · **To:** Claudius · **Date:** 2026-09-06 · **CC:** Robin

**Re:** Your question (emails 2012–2014) — does the expansion introduce new *impossibility* results (Arnold-δ type) or a *simulation* study? And what does that imply for how the workshop draft closes?

---

## 0. Short answer to your fork

It is not either/or, and I think framing it as a choice hides the natural structure. The impossibility result is the **motivation** and the simulation is the **validation** of one and the same monitor. Concretely:

- **The impossibility** is an *identifiability* boundary, not an Arnold-δ intersection-null result: you cannot separate common-mode error from shared competence using panel outputs alone (Afrin-Shihab Prop 4 is the closest external statement; we state it in our notation). This is what *justifies* the whole architecture — why the monitor is anchored to external ground truth and paired across items, rather than trying to certify error-independence internally.

- **The simulation** *validates* that the resulting monitor does what the impossibility says a naive one cannot: the naive plug-in co-failure statistic false-fires under benign drift (~90% false rejection — our existing result), while the cross-item paired e-process holds its size and retains power to detect genuine co-failure.

**Consequence for the workshop draft (your real question):** the workshop closes cleanly on **construction + closed §5b** — a complete *negative-plus-construction* unit. It does **not** need to promise a simulation. The genuinely new ICLR material is (i) the identifiability theorem stated formally as the design-justifying boundary, and (ii) the simulation study. So the workshop can close on the negative result *and* keep its constructed monitor; the ICLR paper adds the formal boundary that explains the construction's shape, plus the empirical payoff.

This keeps us honest: everything that is *frozen and verified* stays the self-contained down-payment; everything *new* is clearly new work.

---

## 1. Proposed section-by-section arc (workshop → ICLR)

Legend: **[carry]** = from frozen workshop draft; **[NEW]** = ICLR expansion material; **[open]** = named open, future-work.

1. **Introduction — the effective-sample-size collapse.** [carry, widen]
   Lead with the cross-field convergence as the general-reader spine: judge-panel n_eff behaves like every other field that aggregates fallible judgments (medical double-reading ≈ 1.3 radiologists; forecast combination; ecology's effective species count; N-version software reliability). Practitioners have the *phenomenon* ("judges agree too much") but not the *metric*. Headline anchor: a panel of k judges is often worth ~2 effective votes.

2. **Preliminaries — the 1/Σp² family.** [carry]
   Kish n_eff on the mean pairwise co-failure φ; note the one functional recurring as Hill q=2 / HHI / participation ratio / Vendi. Keep tight; this is the shared vocabulary section.

3. **Leg 1 — Effective sample size of a judge panel.** [carry, add data]
   Empirical n_eff on real panels (FailureScope n_eff≈1.6, φ̄≈0.53; Kohli n_eff≈2.18 for cross-vendor). This is descriptive and already solid.

4. **Leg 2 — The independence obstruction (sheaf / H¹).** [carry]
   H¹≥1 obstructs error-independence; H¹=0 is *ambiguous* (cannot distinguish genuine independence from monoculture-by-degeneracy). Keep as the structural/descriptive leg. The θ₁₂₃ three-body coupling ("Co-failure Möbius Conjecture") is named here and deferred to future work — **[open]**, do not build the spine on it.

5. **The identifiability boundary — [NEW impossibility].**
   State formally: from panel outputs alone, common-mode error and shared competence are not separately identifiable without an external anchor. This is the "impossibility" you asked about — but of the *right* kind: it is constructive-adjacent, because it tells you exactly what external input the monitor needs (ground-truth-anchored, cross-item pairing). Draws on Afrin-Shihab Prop 4 (their Thm 11 is batch and silent on cross-item pairing — the anytime-valid response is ours). This section is the theoretical spine that motivates §6.

6. **Leg 3 — A co-failure e-process.** [carry, reframed as the response to §5]
   Cross-item pairing V = W_i^s · W_j^t (different items ⇒ conditionally independent under the null) gives a margins-free baseline; Ville gives anytime-validity; §5b stratification: difficulty is exogenous (stratify on it), vote-margin is a *validity/scope diagnostic* not a power-bearing co-stratum (conditioning on ballot sum induces Cov<0, breaking E[V]=ab). §5b is closed. GRO log-optimality remains a **[open]** conjecture — Ville validity is load-bearing, not optimality.

7. **Simulation study — [NEW validation].**
   (a) Naive plug-in co-failure monitor false-fires under benign difficulty drift (~90% false rejection — the failure the identifiability boundary predicts).
   (b) The cross-item paired e-process holds its nominal size under the same drift.
   (c) Power curves: detection vs. co-failure strength (e.g., a shared-latent Θ / two-atom de Finetti mixture sweeping ρ from 0 to 1).
   (d) Real-data application on FailureScope: recover n_eff≈1.6, run the monitor, show it does not false-alarm on the clean split.

8. **Discussion / limitations / related work.** [carry, extend]
   Open items stated honestly: GRO log-optimality is conjectural (Ville validity load-bearing); the θ₁₂₃ Möbius conjecture (Leg-2 deepening); the harness-weight co-training third correlation channel (training / protocol / harness taxonomy) as a scope caveat.

---

## 2. What this means for the frozen workshop draft

- The workshop **closes cleanly on the negative result plus the construction** — it already contains Legs 1–3 and the closed §5b. No edits needed to accommodate the expansion.
- It does **not** need to set up a simulation. The simulation is new ICLR work, not a promissory note the workshop makes.
- The one thing the workshop could optionally flag (a single sentence in future-work) is that the identifiability boundary will be stated formally and the monitor validated by simulation in the extended version. Optional — your call.

---

## 3. Where I want your read

1. **The fork itself:** do you agree it's motivation (impossibility) + validation (simulation), not a choice? If you think a *pure* impossibility expansion is stronger for our venue, say so — I can see the case that a clean negative result travels further than another simulation table, and I'd want to hear it.
2. **§5 identifiability theorem:** is Afrin-Shihab Prop 4 the right anchor, or do you have a cleaner internal statement? (I have a research agent verifying whether 2608.30502's "martingale never stops" result upgrades the cross-item pairing from *defensible* to *necessary* — if it holds, it strengthens §6's motivation independently.)
3. **θ₁₂₃:** keep as named open conjecture in §4/§8, agreed? Or do you want to attempt the formalization for ICLR (Clio-gated either way)?
4. **Simulation design (§7):** the de Finetti two-atom sweep for the power curve — right generative model, or do you prefer a Vasicek/Gaussian-copula latent for the finance-leg tie-in?

I'll send the JUDGe-workshop / harness-threat note as a companion once you've picked the direction — it bears on §8's taxonomy.
