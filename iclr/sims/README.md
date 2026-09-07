# Section 7 simulation

`section7_sim.py` — de Finetti two-atom sweep validating the cross-item
co-failure e-process. Three panels:

- **(a)** Power curve: P(e-process crosses 1/alpha) vs induced co-failure rho.
- **(b)** Naive plug-in false-fires under drifting margins; margins-free
  cross-item e-process holds its size.
- **(c)** FailureScope real-data anchor: n_eff = 1.635, phi-bar = 0.534.

## Run

```
python3 section7_sim.py
```

Writes `panel_a_power.png`, `panel_b_falsefire.png`, `panel_c_failurescope.png`
and prints the headline numbers. Fixed RNG seed `20260907`.

## Requirements

- numpy
- matplotlib

(Panel c additionally reads the FailureScope corpus at
`/home/lyra/data-scratch/failurescope/failurescope-single-turn-v1.0.0/failures.json`
and reproduces the Kish n_eff computed by
`/home/lyra/projects/evalue-sheaf/code/evalue/failurescope_probe.py`.)

## e-process definition (source)

From `memos/2026-09-03-sec5b-stratification-draft.tex`:

    e_t = prod_i ( 1 + lambda ( U_i - V_i - delta_k ) )

  U_i = 1{both judges fail on item i}
  V_i = W^s_a * W^t_b, distinct items a != b  (cross-item pairing, margins-free)
  delta_k = 2*eps,  lambda in [0, 1/(1+2 eps)]

Under item independence E[V] = a*b with no fitted nuisance (margins-free null).
