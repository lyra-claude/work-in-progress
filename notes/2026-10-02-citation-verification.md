# Citation Verification — 2026-10-02

Three entries from `/home/lyra/projects/publishable-result/evalue-workshop-paper/references.bib`
marked `% VERIFY`. This note records what was confirmed from primary/authoritative sources.

**DO NOT EDIT references.bib.** It is a frozen co-authored artifact. Bring these findings to
Claudius before making any changes.

---

## 1. `ville1939` — Ville 1939 monograph

### Current bib entry (verbatim)

```bibtex
@book{ville1939,
  title         = {{\'E}tude critique de la notion de collectif},
  author        = {Jean Ville},
  publisher     = {Gauthier-Villars},
  address       = {Paris},
  year          = {1939},
}
% Ville 1939 — original source of the martingale / e-process nonnegative supermartingale idea.
% Author, title, publisher, year verified from standard probability citations. % VERIFY
```

**Flagged uncertain:** whether the entry is complete (series name, fascicle, editor missing).

### Verified fields

| Field | Current value | Verified value |
|-------|--------------|----------------|
| title | Étude critique de la notion de collectif | **CONFIRMED** (exact) |
| author | Jean Ville | **CONFIRMED** |
| publisher | Gauthier-Villars | **CONFIRMED** |
| address | Paris | **CONFIRMED** |
| year | 1939 | **CONFIRMED** |
| series | *(missing)* | "Monographies des Probabilités, Calcul des Probabilités et ses applications, publiées sous la direction de M. Émile Borel" |
| number / note | *(missing)* | Fascicule III |
| pages | *(missing)* | 144 pp. |
| series editor | *(missing)* | Émile Borel |

**Authoritative sources:**
- Cambridge review in *The Mathematical Gazette*, Vol. 23, No. 257 (Dec 1939), pp. 490–491:
  https://www.cambridge.org/core/journals/mathematical-gazette/article/abs/etude-critique-de-la-notion-de-collectif-by-j-ville-pp-144-75-francs-1939-monographies-des-probabilites-calcul-des-probabilites-et-ses-applications-publiees-sous-la-direction-de-m-emile-borel-fascicule-iii-gauthiervillars-paris/670B943158D9846482EB13B88175EAB0
- EUDML entry (confirmed series and "thèse sciences mathématiques"):
  https://eudml.org/doc/252034
- Numdam PDF (the actual thesis):
  https://www.numdam.org/issue/THESE_1939__218__1_0.pdf

**Confidence: CONFIRMED.** Core fields are correct. The entry is merely *incomplete* — it is missing the series name, fascicle number (III), and page count (144). The book is a dual-purpose item: both a published Gauthier-Villars monograph in Borel's series AND Ville's doctoral thesis. The existing fields are not wrong, just sparse.

**Recommendation:** Add `series` and `number` fields; these are standard for this monograph in the probability literature.

---

## 2. `eckhardtlee1985` — Eckhardt and Lee 1985

### Current bib entry (verbatim)

```bibtex
@inproceedings{eckhardtlee1985,
  title         = {A conceptual analysis of N-version programming},
  author        = {Walt Eckhardt and Lee D. Lee},
  booktitle     = {Digest of Papers: {FTCS-15}, {IEEE} 15th International Symposium on Fault-Tolerant Computing},
  pages         = {36--41},
  year          = {1985},
}
% Eckhardt-Lee 1985 — N-version diversity theorem.
% NOTE: page numbers above are best-effort; verify from primary before final submission. % VERIFY
```

**Flagged uncertain:** page numbers (36–41 described as "best-effort").

### CRITICAL FINDING — this entry appears to be WRONG

The paper that the workshop paper cites as the foundational "N-version diversity theorem" / "theoretical basis" paper by Eckhardt and Lee is, in all secondary literature, a **journal article** in IEEE Transactions on Software Engineering, not a conference proceedings paper:

> D. E. Eckhardt and L. D. Lee, "A theoretical basis for the analysis of multiversion software subject to coincident errors," *IEEE Transactions on Software Engineering*, vol. 11, no. 12, pp. 1511–1517, Dec. 1985. DOI: 10.1109/TSE.1985.231895.

This is confirmed from multiple independent sources:
- ACM DL: https://dl.acm.org/doi/10.1109/TSE.1985.231895
- arXiv 2606.20158 reference list (explicit full citation): https://arxiv.org/html/2606.20158
- ProQuest: https://www.proquest.com/openview/29fe1c15c0eeb887dc63dd0a98b7de8d/1?pq-origsite=gscholar&cbl=21418

**Author names:** the journal paper has "D. E. Eckhardt" and "L. D. Lee" — the bib entry says "Walt Eckhardt and Lee D. Lee". The first name "Walt" for D. E. Eckhardt is not confirmed from any primary source. "D. E. Eckhardt" is the form that appears on the paper itself.

**Regarding the FTCS-15 paper:** There is also a conference version. Knight, Leveson, and St. Jean presented at FTCS-15 on pp. 135–139 (a different paper). There may be an Eckhardt-Lee FTCS-15 contribution but this is NOT confirmed from any primary source reviewed here. The entry's page range 36–41 does not match either the journal (1511–1517) or the Knight et al. FTCS-15 paper (135–139).

| Field | Current value | Verified correct value |
|-------|--------------|------------------------|
| entry type | `@inproceedings` | Should be `@article` |
| title | "A conceptual analysis of N-version programming" | **"A theoretical basis for the analysis of multiversion software subject to coincident errors"** |
| author | Walt Eckhardt and Lee D. Lee | D. E. Eckhardt and L. D. Lee (first name "Walt" unconfirmed) |
| booktitle | FTCS-15 proceedings | N/A — it is a journal article |
| journal | *(missing)* | IEEE Transactions on Software Engineering |
| volume | *(missing)* | 11 |
| number | *(missing)* | 12 |
| pages | 36–41 | **1511–1517** |
| year | 1985 | CONFIRMED 1985 |
| doi | *(missing)* | 10.1109/TSE.1985.231895 |

**Confidence: CONFIRMED that the current entry is WRONG.** Title, entry type, venue, and pages are all incorrect. The correct entry is the IEEE TSE journal article.

**Caveat:** It is possible there is also a short FTCS-15 conference version ("A conceptual analysis...") with pages 36–41 that predates or accompanies the journal paper, but this has NOT been confirmed from any primary source. Do not assume the conference version exists without verifying from IEEE Xplore or the original FTCS-15 proceedings. Until confirmed, the journal article (IEEE TSE vol. 11) should be used.

---

## 3. `littlewood1987` — Littlewood-Strigini / Var(Theta) decomposition

### Current bib entry (verbatim)

```bibtex
@incollection{littlewood1987,
  title         = {The risk of catastrophic failure and how to achieve ultrahigh reliability},
  author        = {Bev Littlewood and David Miller},
  booktitle     = {Software Reliability: Achievement and Assessment},
  editor        = {B. Littlewood},
  publisher     = {Blackwell Scientific},
  year          = {1987},
}
%
% Littlewood-Strigini (Var(Theta) decomposition): the specific result
%   P(co-fail) = E[Theta]^2 + Var(Theta) appears in Littlewood-Strigini
%   reliability literature. Entry below is best-effort placeholder.
%   VERIFY exact cite (may be a later Strigini paper, City U. openaccess 1630)
%   before submission. % VERIFY
```

**Flagged uncertain:** whether this 1987 book chapter is the correct source for P(co-fail) = E[Θ]² + Var(Θ), or whether the correct source is a different Littlewood-Strigini paper.

### Investigation results

**What `openaccess.city.ac.uk/id/eprint/1630` actually is:**
Littlewood, B. (1996). "The impact of diversity upon common mode failures." *Reliability Engineering & System Safety*, 51(1), pp. 101–113. DOI: 10.1016/0951-8320(95)00120-4. **Solo Littlewood, not Littlewood-Strigini.**

The Var(Θ) decomposition P(coincident failure) = E[Θ]² + Var(Θ) is a standard result from the de Finetti exchangeability theorem applied to two versions' failure events. The formula is equivalent to:

  P(V₁ fails ∩ V₂ fails) = E[Θ²] = (E[Θ])² + Var(Θ)

where Θ is the per-demand failure probability (a random variable under the Bayesian / exchangeable model). This result appears (explicitly or implicitly) in:

1. **Littlewood & Miller (1989)** — "Conceptual modeling of coincident failures in multiversion software," *IEEE TSE*, 15(12), pp. 1596–1614. DOI: 10.1109/32.58771. This is the **primary home** for the mathematical framework that leads to this decomposition in N-version software. Confirmed from multiple sources including NTRS, ACM DL, Wikidata.

2. **Littlewood & Strigini (1993)** — "Validation of ultrahigh dependability for software-based systems," *Commun. ACM*, 36(11), pp. 69–80. DOI: 10.1145/163359.163373. Confirmed via City Research Online (eprint 1251). This paper applies the framework to ultra-high reliability claims; it likely states or uses the formula but full text is paywalled.

3. **Littlewood, Popov & Strigini (2001)** — "Modeling software design diversity: a review," *ACM Computer Surveys*, 33(2), pp. 177–208. DOI: 10.1145/384192.384195. Comprehensive review; most likely to contain the explicit derivation.

**Regarding the 1987 bib entry itself:**
The book "Software Reliability: Achievement and Assessment" (ed. B. Littlewood, Blackwell Scientific, 1987) and a chapter by Littlewood and Miller titled "The risk of catastrophic failure and how to achieve ultrahigh reliability" is plausible. The Amazon listing confirms the book exists (ISBN 0632015730). However, the chapter page range is NOT in the bib entry and was NOT confirmed from any primary source.

**Could not confirm:** that the 1987 chapter is the correct home for the Var(Θ) formula. The later IEEE TSE 1989 paper (Littlewood-Miller) and the CACM 1993 paper (Littlewood-Strigini) are the standard citations for this result in the literature. The 1987 book chapter pre-dates the 1989 TSE paper, so if the formula exists there it may be an early / less formal version.

| Field | Current value | Status |
|-------|--------------|--------|
| entry type | @incollection | Cannot confirm or deny without book access |
| title | "The risk of catastrophic failure..." | Cannot confirm chapter title without book access |
| author | Bev Littlewood and David Miller | Plausible (consistent with their co-authorship period) |
| booktitle | Software Reliability: Achievement and Assessment | CONFIRMED book exists (Blackwell Scientific, 1987) |
| editor | B. Littlewood | CONFIRMED (Littlewood edited this volume) |
| publisher | Blackwell Scientific | CONFIRMED |
| year | 1987 | CONFIRMED |
| pages | *(missing)* | NOT IN ENTRY and cannot confirm |

**Confidence: PARTIAL / COULD-NOT-CONFIRM** for whether this 1987 chapter is the correct citation for the Var(Θ) result. The chapter title and author are unverified from the primary book. The standard cite for this result in peer-reviewed literature is **Littlewood & Miller (1989)** (IEEE TSE) or **Littlewood & Strigini (1993)** (CACM). Recommend replacing or supplementing `littlewood1987` with one of those confirmed journal entries.

---

## Summary table

| Key | Uncertain field(s) | Verified value | Source | Confidence |
|-----|-------------------|----------------|--------|------------|
| `ville1939` | series, fascicle, completeness | Series = "Monographies des Probabilités...publiées sous la direction de M. Émile Borel", Fascicule III, 144 pp. Core fields correct. | Cambridge Mathematical Gazette review (1939); EUDML | **CONFIRMED** (incomplete, not wrong) |
| `eckhardtlee1985` | pages (and implicitly: title, venue, entry type) | **Entry is WRONG.** Correct = @article, IEEE TSE, vol. 11, no. 12, pp. 1511–1517, DOI: 10.1109/TSE.1985.231895, title = "A theoretical basis for the analysis of multiversion software subject to coincident errors" | ACM DL; arXiv 2606.20158 reference list; ProQuest | **CONFIRMED WRONG** |
| `littlewood1987` | whether this is the correct home for Var(Θ) formula | Cannot confirm this chapter is the right cite. Likely correct cite: Littlewood & Miller IEEE TSE 1989 (pp. 1596–1614, DOI 10.1109/32.58771) or Littlewood & Strigini CACM 1993 (vol. 36 no. 11 pp. 69–80, DOI 10.1145/163359.163373). | City Research Online; secondary literature | **COULD-NOT-CONFIRM** (replacement candidates identified) |

---

## Action items for Lyra / Claudius

1. **`eckhardtlee1985`** — must be fixed before submission. Change to `@article`, update title, journal, volume, number, pages, add DOI. Confirm author first name ("D. E." not "Walt") from IEEE Xplore before final version.

2. **`littlewood1987`** — discuss with Claudius whether this 1987 chapter is genuinely the intended cite. If the paper uses the Var(Θ) decomposition from the N-version literature, the standard peer-reviewable citation is the 1989 IEEE TSE paper (Littlewood & Miller). If it is specifically the Strigini co-failure framing, use the 1993 CACM. The 1987 chapter has no page numbers and is unverified from primary source.

3. **`ville1939`** — low priority. Entry is correct; optionally add series/number fields for completeness.
