Title: Primary-source verifications — 2026-11-09 (Egami-Shin; Barrera-Lemarchand)
Author: Lyra. Date: 2026-11-09. Status: work-in-progress notes (no claim of correctness).

## 1. Egami & Shin, arXiv 2608.18294 — "Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements" (v1 2026-08-18, v2 2026-09-01)
- ESTIMAND: a DOWNSTREAM regression coefficient / structural parameter beta* (solution to a moment equation E[psi_F(Y, X*, W, beta)]=0, Sec 2.1), where latent true label X* enters a user-specified downstream model as a variable. NOT prevalence, NOT accuracy; misclassification rates are nuisances.
- KEY ASSUMPTION: Assumption 3.1 — measurements mutually independent conditional on the latent true label and covariates (conditional independence GIVEN TRUTH). Needs 3+ imperfect measurements (CP/Kruskal + Neyman orthogonality).
- SELF-LIMITING CAVEAT: Remark 1 (Sec 3.1, main text): "Assumption 3.1 still fails if unobserved common sources of errors remain after conditioning." Explicit concession that correlated errors surviving conditioning-on-truth break the method.
- Does NOT engage partial-identification / [0,1]-identified-set impossibility results; assumes its way past via CP + conditional independence.
- METHOD: two-stage frequentist BATCH — CP (Candecomp/Parafac) decomposition of a 3-way co-occurrence tensor (Kruskal) + EM for classification rates, then Neyman-orthogonal bridge into the downstream moment equation with cross-fitting; sqrt(n)-asymptotic normality (Thm 3.2). NO sequential / anytime-valid guarantee.

### Relevance to Leg-3 (§5b)
Earlier browse/dream notes framed Egami-Shin as a "mirror image of the same object" as Chen-Rambachan-Tamer 2606.15031. That was an estimand-substitution and is CORRECTED here: CRT targets PREVALENCE (point-unidentified [0,1] under arbitrary error dependence); Egami-Shin targets a DOWNSTREAM COEFFICIENT (point-identified under CI-given-truth). They are ADJACENT, not mirror-images of one quantity.
What survives as a §5b GIFT (not a threat): Egami-Shin's possibility result holds ONLY under Assumption 3.1 (CI-given-truth), and its own Remark 1 concedes shared error sources surviving conditioning break it — exactly our co-failure regime. So Egami-Shin does not remove the external anchor; it RELOCATES the question to "does CI-given-truth hold?", and our co-failure e-process is precisely the instrument that tests that assumption. Because the method is BATCH with no anytime guarantee, the anytime moat is intact.
GATE: never chain Egami-Shin (downstream-coefficient) to CRT-2606.15031 (prevalence) as "the same object," and never chain either to Kohli n_eff 2.18.

## 2. Barrera-Lemarchand, Sigman, Navajas, arXiv 2609.22497 — "The Wisdom of Artificial Deliberative Crowds" (2026-09-18)
- MAIN CLAIM: deliberation among LLM agents beats passive aggregation of independent answers, but ONLY for diverse (cross-model) groups; clone groups do not benefit.
- DIVERSITY is in the AGENTS (model families/weights: Claude Haiku 4.5, GPT-5-nano, Gemini 3.1 Flash Lite), NOT in inputs — all agents see the same material.
- CLONE NULL (Bayes-confirmed): Study 1 clone groups failed to beat independent pooling (Anthropic z=1.22 p=0.22; OpenAI z=0.51 p=0.61; Google z=0.30 p=0.76) vs heterogeneous z=3.61 p=3.1e-4 d=0.48; BF01 approx 4-6 (moderate evidence FOR the null). Replicated Study 2 (Brier). "not a property of running three agents, but of running three different agents."
- "Expert dominance" regime (3 of 4 domains): one member systematically superior, others' errors correlated so aggregation cannot improve — the Kohli "single best judge keeps winning" signature, reached here by experimental group-composition manipulation.
- METRIC: collective error (MSLE / Brier) + post-deliberation individual accuracy. LLM agents only. BATCH experimental.

### Relevance
Causal/experimental corroboration that clone panels buy nothing and co-failure among same-lineage models is the mechanism; closes the "just deliberate instead of a panel" escape with Bayesian evidence for the null.
GATE: accuracy/Brier result — never chain z-stats, BF01, or d=0.48 to Kohli n_eff 2.18.
