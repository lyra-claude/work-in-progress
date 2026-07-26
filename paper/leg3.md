# Leg 3 — Operational: Certify It Live

> **Status.** The per-pair anytime-valid SLA (Ville's inequality) and the
> average-of-e-values dependence-robust aggregation are proven and are the
> backbone of this leg. The quantitative H¹ growth-rate theorem was refuted and
> withdrawn (2026-07-23); the graded-cohomology link is retained only
> qualitatively — `dim H¹ ≥ 1` signals detectable coupling, nothing stronger.

<!-- The per-pair supermartingale / Ville certificate (§1) and the
     average-of-e-values aggregation (§3) are drafted in
     notes/leg3-post-refutation.md and belong above this subsection. -->

## The blind-spot atom (Ω_blind)

**Naming convention (fixed here, before any formula).** Throughout this
subsection, `π` denotes the *non-blind* (coverage) weight: the mass of the risk
space the labelling oracle — or, in the batch twin, the verifier cascade — can
actually see. Consequently `(1 − π)` is the *blind-spot mass*: the errors the
checking apparatus is structurally unable to catch. The tempting inversion —
reading `π` as the blind-spot rate — is a clean-seeming label error, and we
exclude it up front rather than downstream, so that no formula below can smuggle
it back in. Coverage is `π`; blindness is `1 − π`.

**Decay form (rate, not immunity).** Increasing the oracle's capacity does shrink
the blind mass, but only at a decaying rate. Under partial correlation between
the checked system and its checker, the per-item blind-spot coverage improves
merely *polynomially*, `1 − r_k ~ k^(−b)` (Han 2607.13918), rather than
exponentially. These are marginal returns: each added unit of monitoring effort
buys less coverage than the last, and no finite amount of scale drives the blind
mass to zero.

**The ceiling (an engineering constraint).** Because the blind-spot atom sits at
a mass `(1 − π)` that no verifier in the cascade can reach, it imposes a hard
ceiling on what monitoring can achieve: the total information / detection gain
available against the blind mass is bounded by `−ln(1 − π)`. This is not a
negative result about the e-process — it is the operating budget. It states the
best that *any* amount of monitoring effort can do against the blind mass, in
exactly the way a thermodynamic limit states the best any engine can do.

**Decorrelation is the cure (and it targets a component, not all of π).** The one
lever that loosens the ceiling is decorrelation between the oracle and the system
it checks. Crucially, it acts on the *architectural-coupling* component of `π`,
not on `π` wholesale: the ceiling `−ln(1 − π)` is tightest exactly when the
oracle and the checked system share substrate, training pipeline, or
architecture, because their blind spots then coincide. Making the checking
apparatus structurally different from what it checks — oracle-vs-checked-system
decorrelation — is what enlarges `π` where it matters and so relaxes the bound.
This is the diversity-works-by-axis-not-by-vendor principle applied at the oracle
itself.

**Scope note.** Han 2607.13918 is a *batch* result — its estimand is the survivor
reliability of a conjunctive verifier cascade on a single instance, not the
streaming per-pair co-failure rate the e-process monitors. The relationship to
the Leg-3 certificate is therefore structural and thematic, not a chained bound:
we do not claim "Han proves the e-value has a ceiling." Both are voices of the
same blind-spot atom in different registers. Accordingly, the certificate of §1–3
bounds co-failure correlation only on the monitored input stream and only for
failures the labelling oracle detects; it makes no claim about adversarially
induced or emergent shared blind spots on inputs outside that stream, nor about
correlated failures of the oracle itself.
