# Rosenblatt Pilot — Results

**Setup:** N=4000 items, K=3 judges, seed=20261123.

## Results Table

| Scenario | True rho | Raw |Spearman| on raw scores | NAIVE KS-to-Unif | RANK KS-to-Unif | NAIVE pairwise |corr| | RANK pairwise |corr| |
|----------|----------|-------------------------------|------------------|-----------------|----------------------|---------------------|
| **A** — Dependence rising, marginals fixed | 0.4000 | 0.3762 | 0.0092 | 0.0002 | 0.3766 | 0.3762 |
| **B** — Marginals drift, dependence zero (rho=0 by construction) | 0.0000 | 0.0512 | 0.2824 | 0.0002 | 0.0505 | 0.0512 |
| **C** — Null control (iid N(0,1)) | 0.0000 | 0.0201 | 0.0126 | 0.0002 | 0.0199 | 0.0201 |


**Columns:**
- **True rho** — generative copula parameter (known from construction; zero for B and C by design).
- **Raw |Spearman| on raw scores** — empirical average absolute pairwise Spearman on the raw X-values. NOTE: in scenario B this is confounded by shared marginal drift trends and does NOT measure copula dependence — it is reported to show that naive statistics on raw scores are also unreliable.
- **NAIVE KS-to-Unif** — KS distance of NAIVE-transformed U-values to Uniform[0,1]; measures marginal non-uniformity leaking through the fixed-marginal transform.
- **RANK KS-to-Unif** — same for the empirical-PIT (rank) transform; near-zero by construction.
- **NAIVE pairwise |corr|** — average absolute pairwise Pearson correlation of NAIVE U-values.
- **RANK pairwise |corr|** — same for RANK U-values.

## Interpretation

The primary contrast is scenario B (marginals drift, rho=0 by construction). NAIVE KS-to-uniform is **large** (0.2824), confirming that the fixed-marginal transform cannot absorb the drift and produces a large marginal non-uniformity signal — a NAIVE detector would fire even though the judges are genuinely independent. RANK KS-to-uniform is **~0** (0.0002), confirming that the empirical PIT forces marginals to U[0,1] regardless of drift. RANK pairwise |corr| is **0.0512** — not fully zero (check independence of drift construction). In scenario A (true rho ramping 0→0.8, mean 0.4) both transforms correctly detect rising copula dependence (RANK |corr| = 0.3762), and both keep marginals near-uniform (no drift to absorb: RANK KS = 0.0002). Scenario C (null, iid N(0,1)) is clean for both transforms. **Caveat**: this uses offline global ranks (batch empirical PIT) and is empirical motivation only. It does NOT establish anytime-valid e-process properties for sequential monitoring, which remains the open theory question.

## Core Contrast (Scenario B)

| Check | Expected | Actual | Verdict |
|-------|----------|--------|---------|
| B: NAIVE KS > 0.1 (drift leaks into fixed-marginal PIT) | >0.10 | 0.2824 | PASS |
| B: RANK KS ~0 (empirical PIT absorbs drift in marginals) | <0.03 | 0.0002 | PASS |
| B: RANK pairwise \|corr\| near-zero (no substantial false alarm) | <0.07 | 0.0512 | PASS |

Main marginal-uniformity contrast held: **YES**

---
*Empirical motivation only. NOT an anytime-valid e-process or proof of sequential validity.*
*A causal/rolling rank transform is needed for full non-stationarity robustness in a sequential setting.*
