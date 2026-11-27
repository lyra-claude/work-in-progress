# Open validity question — the anytime extension (Clio-gated)

Everything in `results.md` and `results_rolling_ranks.md` is a **BATCH / empirical diagnostic**. It establishes two things:

- rolling/causal ranks remove the shared-trend confound that global ranks introduce (spurious cross-judge rank corr 0.21 → 0.017; tail lift 3.6x/6.6x → 1.4x/2.6x), and
- the margins-free joint-tail-exceedance statistic detects genuine cross-judge tail co-failure (lift 14x at alpha=0.10, 38x at alpha=0.05) while staying at the null band under both the independent null and the shared-trend null.

It does **NOT** establish anytime (sequential) validity. The anytime monitor — a betting e-process on the joint-tail statistic — faces a specific, non-trivial validity question that must be resolved before any sequential Type-I claim:

1. The rolling-rank PIT `U_t = rank(x_t | trailing window) / w` is a function of the realized value `x_t` relative to its own estimation window. It is therefore **NOT a predictable (F_{t-1}-measurable) transform**: `U_t` depends on `x_t`. Any e-process whose increment at time `t` is built from `U_t` inherits this dependence.

2. This is structurally the same hazard that invalidated the 2026-11-19 "past-only" cross-item pairing (Clio Prop. 2): a transform/partner that looked innocuous was in fact entangled with the increment's information set, and a predictable in-budget bettor drove Type-I to 1.000 under the static null. The lesson there — *anytime validity wants items whose VALUES are unrevealed at bet-placement* — applies directly: the betting construction must place its bet on `U_t` using only information available before `x_t` is revealed, OR the martingale property must be established despite the window-dependence.

3. Concretely, the open theorem is: **does there exist a margins-free betting strategy on the rolling-rank joint-tail statistic that is a genuine e-process (supermartingale under the shared-trend null) with non-trivial power against genuine tail co-failure?** This is the margins-free + EXACT anytime validity question — the same open extension flagged for the Rosenblatt e-spacings route (Hao-Chan 2611.23775 gives anytime validity only for a FULLY-SPECIFIED null including marginals).

Until that is settled, the correct claim is empirical: *"a margins-free rolling-rank tail diagnostic that, in batch, separates genuine tail co-failure from shared drift."* No sequential/anytime guarantee is claimed.
