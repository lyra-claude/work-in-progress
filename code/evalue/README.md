# Co-failure E-process — Monte-Carlo Simulation

## What this is

A synthetic 2-judge co-failure e-process experiment. Two LLM judges score N items
sequentially. Each item has a latent difficulty θ ~ Uniform(0,1). The null hypothesis
(H0) is conditional independence of judge errors given θ. The alternative (H1) adds
genuine excess co-failure via a Marshall-Olkin common shock: with probability γ both
judges fail together regardless of difficulty.

## The three methods

- **Method A (stratified cross-item):** Baseline term V_s = W_i^s × W_j^{t(s)} where
  t(s) pairs each item with a *different* item in the *same difficulty stratum*. Stratum
  slack δ_k = 2ε_k guarantees E[increment] ≤ 1 under H0. This is the proposed method.
- **Method B (unstratified cross-item):** Same cross-item pairing but t(s) is a global
  derangement over the whole stream; δ = 0. Ablation — inflates Type-I under H0.
- **Method C (naive plug-in):** V_s = â × b̂ (global sample means). Ablation — badly
  inflates Type-I under drifting marginals.

## Headline results (N=2000, R=500, α=0.05, seed=42)

- Method A Type-I: **0.000** (valid under H0, even with drifting marginals)
- Method A power (γ=0.10, Q=5 strata): **0.000** — conservative at coarse stratification
- Method A power vs Q sweep (run_qsweep.py, γ=0.10, mixture λ): rises from 0 to **0.92** as Q goes 5→80
- Method C Type-I under drift: 0.316 (fixed λ) / 0.746 (mixture) — badly inflated

## How to reproduce

```bash
cd /home/lyra/projects/evalue-sheaf/code/evalue
python3 run_experiment.py    # main validity/power table → results.csv, trajectory.png
python3 run_qsweep.py        # power-vs-Q sweep → results_qsweep.csv, power_vs_q.png
```

Seeds are fixed in each script (SEED=42). No external dependencies beyond numpy and matplotlib.

**Note on ε:** In this simulation ε is computed from the *true* p(θ). In deployment ε
would be estimated from data, introducing additional conservatism.

## Files

| File | Purpose |
|------|---------|
| `dgp.py` | DGPs: null (H0), Marshall-Olkin alt (H1), drifting-marginals variant |
| `eprocess.py` | Three e-process methods, wealth accumulation, Ville threshold |
| `run_experiment.py` | Monte-Carlo driver: Type-I / power table, CSV, trajectory PNG |
| `run_qsweep.py` | Power-vs-Q sweep across stratification granularity |
