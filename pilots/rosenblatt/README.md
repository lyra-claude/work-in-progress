# rosenblatt — tail co-failure pilot

**Status:** Empirical motivation only. NOT an anytime-valid e-process. See `OPEN_QUESTIONS.md` for the exact open theorem (Clio-gated).

**Do not chain any numbers from this pilot to results elsewhere in evalue-sheaf.** All figures are illustrative batch diagnostics.

---

## What it does

Three scenarios (K=3 judges, N=4000 items) demonstrate that a coordinate-wise empirical PIT (rank transform) forces each judge's marginal to Uniform[0,1] regardless of marginal drift, while a fixed-marginal PIT (naive Gaussian assumption) conflates drift with dependence signal. The core contrast is Scenario B: independent judges with drifting marginals produce NAIVE KS-to-uniform ≈ 0.28 (false alarm) versus RANK KS ≈ 0.0002 (clean), while RANK pairwise |corr| ≈ 0.05 (near-null, no false copula alarm).

The rolling-ranks extension (`rolling_ranks.py`, `tail_cofailure.py`) uses **causal / rolling window ranks** to kill the shared-time-trend confound that global (offline) ranks introduce. Rolling ranks reduce the spurious cross-judge rank correlation from ~0.21 to ~0.017 and bring the tail-lift estimates closer to their true values under the shared-trend null.

## Why it exists

This is **empirical motivation** for the Rosenblatt/e-spacings route (Hao-Chan 2611.23775) to anytime-valid co-failure monitoring: by transforming judge scores to approximate uniforms via per-judge empirical PIT, a dependence statistic computed on the transformed values is insensitive to marginal drift while retaining sensitivity to rising copula dependence (judges' errors becoming correlated). This is the margins-free property needed for robust LLM judge ensemble monitoring.

## What it is NOT

- NOT an anytime-valid e-process. The global (offline) rank transform used in `pilot.py` is a batch operation. The rolling-rank construction in `rolling_ranks.py` / `tail_cofailure.py` is a batch diagnostic, not a sequential monitor.
- NOT a proof of anything. All tail-lift and KS numbers are illustrative simulation results.
- The open theorem — does a betting strategy on the rolling-rank joint-tail statistic form a genuine e-process (supermartingale under the shared-trend null)? — is Clio-gated. See `OPEN_QUESTIONS.md`.
- Do NOT chain these numbers to any other result in evalue-sheaf (n_eff, θ₁₂₃, Leg-3 e-process, etc.).

## Files

| File | Contents |
|------|----------|
| `pilot.py` | Original pilot: global-rank PIT, three scenarios, false-alarm contrast |
| `rolling_ranks.py` | Rolling-window causal ranks to remove shared-trend confound |
| `tail_cofailure.py` | Margins-free joint-tail-exceedance statistic (batch) |
| `results.md` | Output from `pilot.py` |
| `results_rolling_ranks.md` | Output from rolling-rank experiments |
| `OPEN_QUESTIONS.md` | Precise statement of the open anytime-validity question |
| `.gitignore` | Excludes `.venv`, `__pycache__`, etc. |

## Run it

```bash
python3 pilot.py           # global-rank pilot → results.md
python3 rolling_ranks.py   # rolling-rank diagnostics → results_rolling_ranks.md
python3 tail_cofailure.py  # joint-tail exceedance statistics
```

Requires: `numpy`, `scipy`. No special dependencies.
