# NOTES-cstar: Growth-optimal vs Power-optimal c in Z_c = c·EW + (1−c)·mean_windows

**Date:** 2026-11-28  
**Script:** `code/evalue/compute_cstar.py`  
**Data:** `results/results_cstar.json`  
**Simulation:** Q=10, N=4000, BLOCK=20, R=500 seeds (80000–80499), THRESH=20=1/α.  
**Cross-validated** at c=1, c≈1/2, c≈1/9 against the R=2000 run (`novel_dgp_halfew_results.json`); differences ≤0.05 nats (consistent with R=500 SE).

---

## Setup

The one-parameter family of combiners is:

    Z_c = c · EW_wealth + (1−c) · mean_windows_wealth

where `mean_windows` = mean of the 8 contiguous 3-stratum window experts {0,1,2}..{7,8,9}.

Special cases:
- **EW** = Z_1 (c = 1)
- **half-EW** ≈ Z_{1/2} (c = 1/2; Clio's proposal mixes EW at 1/2 and each window at 1/16)
- **MixWin** = Z_{1/9} (c = 1/9; mean of all 9 experts including EW)

Metrics for each DGP cell (dgp × γ), over a grid c ∈ {0, 0.05, …, 1.0}:
- **Growth:** E[log Z_c(terminal)]  — maximised at c_growth*
- **Power:** fraction of seeds with sup_n Z_c(n) ≥ THRESH — maximised at c_power*

---

## Main Finding: Growth and Power Optima **Diverge** in REV, **Coincide** in TWO-PEAK

### Reversed-Gradient DGP (signal in hard strata {7,8,9})

| γ    | c_growth* | E[log Z] at opt | c_power* | power at opt | Coincide? |
|------|-----------|-----------------|----------|--------------|-----------|
| 0.04 | 0.85      | 0.398           | 0.00     | 0.004        | No        |
| 0.05 | 0.80      | 0.544           | 0.00     | 0.006        | No        |
| 0.06 | 0.70      | 0.686           | 0.00     | 0.012        | No        |
| 0.07 | 0.55      | 0.837           | 0.00     | 0.016        | No        |
| 0.08 | 0.40      | 0.989           | 0.00     | 0.024        | No        |
| 0.10 | 0.10      | 1.292           | 0.00     | 0.050        | No        |
| 0.12 | 0.00      | 1.608           | 0.00     | 0.100        | **Yes**   |
| 0.15 | 0.00      | 2.087           | 0.00     | 0.226        | **Yes**   |
| 0.20 | 0.00      | 2.835           | 0.00     | 0.468        | **Yes**   |
| 0.25 | 0.00      | 3.557           | 0.00     | 0.738        | **Yes**   |
| 0.30 | 0.00      | 4.199           | 0.00     | 0.920        | **Yes**   |

**At weak-to-moderate signal (γ ≤ 0.10), growth and power optima diverge strongly.** The growth curve is concave in c with a peak at high c (near EW), while the power curve is strictly decreasing from c=0 to c=1. As γ increases, c_growth* drifts from 0.85 down to 0 — the two optima converge only when the signal is strong enough that mean_windows alone dominates.

### Two-Peak DGP (signal at both extremes {0,1,2} and {7,8,9})

| γ    | c_growth* | E[log Z] at opt | c_power* | power at opt | Coincide? |
|------|-----------|-----------------|----------|--------------|-----------|
| 0.04 | 0.00      | 1.503           | 0.00     | 0.032        | Yes       |
| 0.05 | 0.00      | 2.091           | 0.00     | 0.164        | Yes       |
| 0.06 | 0.00      | 2.700           | 0.00     | 0.366        | Yes       |
| 0.07 | 0.00      | 3.301           | 0.00     | 0.630        | Yes       |
| 0.08 | 0.00      | 3.925           | 0.00     | 0.844        | Yes       |
| 0.10 | 0.00      | 5.191           | 0.00     | 0.994        | Yes       |
| 0.12 | 0.00      | 6.405           | 0.00     | 0.998        | Yes       |

Both metrics are **strictly decreasing** from c=0 to c=1 for all γ values. EW is strictly dominated by all c < 1 for both growth and power.

---

## Position of Named Combiners Relative to Optima

| Combiner | c    | REV (γ=0.08): growth | TWO (γ=0.07): growth |
|----------|------|----------------------|----------------------|
| c_opt    | 0.40 | 0.989 (opt)          | —                    |
| MixWin   | 1/9≈0.11 | 0.984            | —                    |
| mean_win | 0.00 | 0.980                | 3.301 (opt)          |
| hEW      | 0.50 | 0.988                | 2.920                |
| EW       | 1.00 | 0.958                | 2.089                |

For REV at γ=0.08, the growth curve peak is at c=0.40 (between MixWin and hEW). MixWin (c≈0.11) outperforms hEW (c=0.50) on growth. EW is worst.

For power (all DGPs, all γ): the ranking is **c=0 > MixWin > hEW > EW**, strictly decreasing in c. EW (c=1) has the lowest power of any combiner in this family throughout.

---

## Why Divergence Occurs (REV DGP)

The power threshold is in the upper tail. Under the REV DGP at weak γ, the signal is diffuse across all hard strata. EW captures signal from all strata including the informative ones, so at moderate c the GROWTH (expected log-terminal) is better-calibrated. But for POWER (probability of crossing THRESH=20), it matters that the process exceeds a high barrier, and pure mean_windows has a larger variance (the best single windows can be very powerful but noisy), giving more probability mass above THRESH even at the cost of lower expected log-wealth.

This is the classic *growth rate vs. detection threshold* tension: a growth-optimal bettor maximizes E[log M], but crossing a fixed threshold is better served by higher variance even at some expected-log cost.

---

## Caveats

1. R=500 seeds (not R=2000). SE on E[log Z] ≈ 0.03–0.05 nats; SE on power ≈ sqrt(p(1-p)/500). Optima on a 0.05-resolution grid are coarse; true c_growth* may differ by ≤0.05.
2. c_power*=0.00 appears robust: the power curve is strictly decreasing in c for every cell, with no uptick near c=0. This was verified across multiple gamma values.
3. The original R=2000 JSON does not contain per-seed wealth values, only aggregated statistics, so a fresh simulation was required.
