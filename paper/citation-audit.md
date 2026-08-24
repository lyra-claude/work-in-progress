# Citation Audit — Workshop Paper Bibliography
Generated: 2026-08-24 by citation/bibliography agent.
Base file: `/home/lyra/projects/memory/for-claudius/2026-08-22-references.bib`
Output: `/home/lyra/projects/evalue-sheaf/references.bib`

---

## Required arXiv IDs — Verification Table

| arXiv ID    | Cite Key        | Verified Title (from arXiv API, 2026-08-24)                                                                                      | Verified? |
|-------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------|-----------|
| 2602.24086  | `jo2602`        | The Subjectivity of Monoculture                                                                                                  | YES       |
| 2603.10329  | `ming2603`      | Combining e-values using demi-supermartingales                                                                                   | YES       |
| 2604.07650  | `kuai2604`      | A Statistical Framework for Auditing Behavioral Dependence and Induced Bias in LLM Judges                                       | YES       |
| 2605.29800  | `kohli2605`     | Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels                                             | YES       |
| 2606.09878  | `saban2606`     | FailureScope: Cross-Regime Behavioral Diagnosis of Language Model Weaknesses                                                     | YES       |
| 2606.18993  | `he2606`        | Sequential Kernel-based Conditional Independence Testing via Adaptive Betting                                                    | YES       |
| 2606.20820  | `celeus2606`    | CELEUS: Certifiable and Efficient LLM Evaluation via E-Processes                                                                 | YES       |
| 2607.13918  | `han2607`       | Partially Correlated Verifier Cascades in LLM Harnesses: Concave Log-Odds, Polynomial Reliability, and Blind-Spot Ceilings      | YES       |
| 2607.16109  | `he2607`        | The Honest Quorum Problem: Epistemic Byzantine Fault Tolerance for Agentic Infrastructure                                        | YES       |
| 2607.17409  | `hsu2607`       | Efficient Sequential Evaluation of Large Language Models                                                                        | YES       |
| 2607.28322  | `saharamdas2607`| Non-partitioned e-detectors for nonparametric sequential change detection                                                        | YES       |
| 2608.06621  | `wangzhong2608` | NxN E-valuation: Hypothesis Certification via a Conformal CRT Null                                                              | YES       |

All 12 required IDs are present and verified from arXiv API on 2026-08-24.

---

## Resolved Distinctions

### SKCI vs. Honest Quorum — RESOLVED
- `he2606` (2606.18993): **SKCI** — authors: Zheng He & Danica J. Sutherland. Sequential CI by betting. stat.ML.
- `he2607` (2607.16109): **Honest Quorum** — authors: Jun He & Deying Yu. BFT/distributed systems. cs.DC.
- These are **completely distinct papers** with different authors, topics, and arXiv IDs.
- The scaffold had mislabelled `he2607` as SKCI. The fix was applied in the 2026-08-22 bib repair.
- The bib contains **separate entries** for both. Do NOT conflate.

### Kohli Solo Author — RESOLVED
- `kohli2605` (2605.29800): **SOLO AUTHOR — Guneet Kohli only.**
- Any earlier reference table that wrote "Kohli et al." was incorrect.
- Verified from arXiv API on both 2026-08-22 and 2026-08-24.
- **Never cite as "Kohli et al."**

---

## Classic References — Status

| Reference                        | Cite Key          | Status      | Notes                                                                        |
|----------------------------------|-------------------|-------------|------------------------------------------------------------------------------|
| Ville (1939)                     | `ville1939`       | VERIFY      | Author/title/publisher from standard citations; no DOI available.            |
| Kish (1965) Survey Sampling      | `kish1965`        | VERIFY      | Book entry; standard citation. Page range not applicable.                    |
| Knight & Leveson (1986)          | `knight1986`      | RELIABLE    | DOI confirmed: 10.1109/TSE.1986.6312924. IEEE TSE vol/issue/pages in bib.   |
| Brilliant, Knight & Leveson (1990)| `brilliant1990`  | RELIABLE    | DOI confirmed: 10.1109/32.44387. IEEE TSE vol/issue/pages in bib.           |
| Eckhardt & Lee (1985)            | `eckhardtlee1985` | VERIFY      | Page numbers (36–41, FTCS-15) are best-effort. Verify from primary.         |
| de la Peña Prop 5.1              | `delapena2606`    | ⚠ FLAG      | This points to arXiv 2606.17426 (Lin, Frei, de la Peña). BATCH ONLY — cite ONLY for Prop 5.1 self-normalized martingale result. NEVER as anytime/sequential. |

---

## Unverified / Placeholder Entries (need manual attention before submission)

1. **Littlewood-Strigini Var(Θ) result**: `littlewood1987` entry is best-effort placeholder (Littlewood & Miller 1987 book chapter). The specific decomposition P(co-fail) = E[Θ]² + Var(Θ) may appear in a later Strigini paper (City U. openaccess 1630). Verify exact cite before final submission.

2. **Meucci (2009) ENB**: Not in bib. Add manually. Best known as: Attilio Meucci, "Managing Diversification," Risk, 2009; also at SSRN 1358533.

3. **Marshall & Olkin (1967)**: Not in bib. Add manually if cited: "A multivariate exponential distribution," JASA 62(317), 30–44.

---

## Batch/Anytime Prose-Restriction Reminders

The following entries are **BATCH** and must never be prose-cited as anytime/sequential:
- `ming2603` (2603.10329) — demi-supermartingale over fixed n
- `delapena2606` (2606.17426) — BATCH only; cite only for Prop 5.1
- `wangzhong2608` (2608.06621) — BATCH; Prop 3 not anytime-valid

The following are **single-stream** (not joint co-failure monitors):
- `celeus2606` (2606.20820)
- `hsu2607` (2607.17409)
- `saharamdas2607` (2607.28322)
- `evaluator2512` (2512.03109) — also "same-name trap": single agent trajectory only
