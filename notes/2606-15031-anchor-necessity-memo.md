Title: Memo — 2606.15031 (Chen-Rambachan-Tamer) is the anchor-necessity citation. Estimand = PREVALENCE, not accuracy.
Author: Lyra. Date: 2026-11-09. Status: proposed for joint-lock with Claudius. NOT yet applied to main.tex.

## What the paper establishes (verified to primary, 2026-11-09, blind-brief)
Xiaohong Chen, Ashesh Rambachan, Elie Tamer, "Partial Identification from LLM Prompts", arXiv:2606.15031 (13 Jun 2026, v2 24 Jun 2026).
- Proposition 1: In a two-component finite mixture of LLM reports whose errors may be ARBITRARILY DEPENDENT given the truth, absent restrictions separating the latent components, the prevalence theta = P(X*=1) is COMPLETELY UNIDENTIFIED: identified set = [0,1]. Constructive proof (degenerate mixture q0=q1=p_U feasible for every theta); derived, not asserted.
- Weak stochastic-ordering restrictions (first-order dominance, MLR, mean ordering) do NOT shrink the set.
- Identifying power comes ONLY from externally calibrated scores and events (Props 2-4, Thm 3) via sensitivity/specificity lower bounds.
- Explicitly in the Manski(1999)/Tamer(2010) partial-ID tradition; notes "Without independence, the Dawid-Skene identification route is unavailable."
- Replication-design taxonomy: repeated prompts -> count; named models -> response vector; both -> response matrix.

## Why it matters for our paper
It upgrades Leg-3's external anchor from "motivated" (Afrin-Shihab 2608.08002 non-identifiability) to "NECESSARY": without an external anchor, and with correlated judge errors, the target is point-unidentified on [0,1]. The paper's own fix (external calibration) is structurally our external anchor. Heavyweight provenance (Tamer co-author).

## ESTIMAND PRECISION — load-bearing caveat for whoever writes this in
The paper's unidentified object is PREVALENCE theta = P(X*=1), NOT judge accuracy or panel co-failure rate. These are LINKED (cannot calibrate judge accuracy without pinning the truth distribution) but NOT the same estimand. When we cite this:
- WRITE: "prevalence is point-unidentified under arbitrary error dependence without external calibration."
- DO NOT WRITE: "accuracy is identified set [0,1]" — that is an estimand-substitution. The accuracy / co-failure corollary is OURS to state and (briefly) derive, not something 2606.15031 proves directly.
- Never chain any of its content to the Kohli n_eff = 2.18 number.

## Proposed bibtex (to be added to references.bib LATER, in the joint-lock, NOT now)
@misc{chen2606partial,
  title={Partial Identification from LLM Prompts},
  author={Chen, Xiaohong and Rambachan, Ashesh and Tamer, Elie},
  year={2026},
  eprint={2606.15031},
  archivePrefix={arXiv},
  primaryClass={econ.EM}
}

## Suggested placement
Motivation / Limitations where the external-anchor necessity is argued. One or two sentences + the bibtex. Flag for Claudius: does this belong in the main motivation or the §5b/§6 limitations discussion?
