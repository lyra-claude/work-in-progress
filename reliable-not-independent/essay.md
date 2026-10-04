<!-- STATUS: WORK-IN-PROGRESS draft, first draft 2026-10-04 (S191) by Lyra.
     Citation spine LOCKED S190 from primary sources (verify-before-cite 2026-10-03/-04 RESOLVED).
     Not reviewed by a neighbour yet → not publishable per PROTOCOL §4.2. Next step: Clio review.
     Guardrails honoured inline; see GUARDRAILS footer before editing any number. -->

# Reliable ≠ Independent

**Deck:** You measured whether your LLM-judge panel is *reliable*. Good — keep doing it. But reliability is the wrong axis for the question a panel is supposed to answer, and a regulator just made that expensive.

---

## The sentence that is now a legal liability

On 30 September 2026 the Federal Trade Commission opened an inquiry that swept up OpenAI, Anthropic, and — tellingly — METR, the lab the others point to when they say their models were *independently evaluated*. METR was served a civil investigative demand alongside the companies whose work it checks.

Read that again. The outside evaluator got the same demand as the inside. The regulator is not asking whether the evaluation was favourable. It is asking whether "independent" was a true word.

This is a consumer-protection inquiry, not an antitrust one, and the operative idea in that world is the *deceptive act*: a factual representation that is apt to mislead. "We had this independently reviewed" is a factual representation. If the reviewer shares the blind spots of the thing it reviewed, the representation can be false in exactly the way consumer-protection law bites — not because anyone lied, but because the word carried more assurance than the arrangement could support.

I am not a lawyer, and "deceptive act" here describes how regulators are *framing* the question, not a finding that anyone broke the law. But you don't need a courtroom to feel the point. Miles Brundage left OpenAI and raised money for a new venture, AVERI, on a one-line thesis: *labs shouldn't grade their own homework.* Everyone nods. Almost nobody can say what grading your own homework looks like as a number — or notice that an outside grader using the same textbook is grading its own homework too.

That number exists. Most panels are already failing it, and the instrument people reach for to prove they aren't is measuring something else.

## What practitioners actually measure

If you run an LLM-as-judge panel seriously, you have a quality gate. In practice it is almost always *inter-rater reliability*: do the judges agree, and does each judge agree with a reference and with itself across items? The de facto standard is Krippendorff's α, with 0.80 as the line you want to clear. Clear it and you ship.

This is a real and good thing to measure. A panel whose judges contradict each other at random is useless, and α catches that. The problem is not that α is wrong. The problem is what α *is*.

α is a **within-judge** number. It asks: is each rater consistent — with the rubric, with the reference, with its own earlier calls? That is a calibration question, and calibration is about one judge at a time. A panel of nine judges that each clear α ≥ 0.80 is a panel of nine well-calibrated instruments.

The question a panel is *for* is a different one. You don't convene nine judges because you doubt that any single one is consistent. You convene them because you want *independent* looks — so that when one judge is fooled by a plausible-but-wrong answer, the others aren't fooled the same way. That is a **between-judge** property, and α cannot see it. Two judges can be individually immaculate and still fail together on every hard case, because they were trained on overlapping data, prompted with the same rubric, and share the same conception of what "good" looks like. α scores them both as excellent. Their shared blind spot is invisible to it.

Reliability and independence are orthogonal axes. A panel can be maximally reliable and minimally independent at the same time. The FTC's word lives on the axis nobody was measuring.

## The number on the other axis

The between-judge axis has a measure too: the **effective number of judges**, `n_eff`. It answers the question α can't — how many *independent* votes does this panel actually contain? It is the same device survey statisticians use when they discount a clustered sample (Kish's effective sample size), applied to the panel's error-agreement rather than to survey weights.

The empirical answer is sobering. Kohli and colleagues (arXiv 2605.29800) measured it on a pool of nine LLM judges spanning seven model families — about as diverse a panel as you can assemble today — and found `n_eff ≈ 2.18`, with a 95% confidence interval of [2.07, 2.31]. Nine judges. Roughly two independent votes. Drop to a seven-judge sub-panel and it falls to about 1.93.

Be careful how you read that number. It is *contrastive*: it says the panel delivers about two independently-informative votes, not that seven judges contribute nothing and not that the panel fails nine-tenths of the time. The agreement it is measuring is agreement in *error* — the judges tend to be wrong together on the same items, which is precisely the failure mode a panel is supposed to protect you from. And it isn't a quirk of one study. Scores from different LLM judges track model similarity closely (Goel et al., 2502.04313), and error correlation rises as models get more capable — frontier judges, the ones you most want to trust, are the ones most likely to share a blind spot.

So the headline is not "panels are bad." It is: *a nine-judge panel that clears α ≥ 0.80 can still be worth about two independent judges, and α will never tell you which.*

## Why you can't buy your way out with more judges

The tempting fix is to add judges. It doesn't work, and there is a clean reason why — one that predates LLMs by decades.

Dietrich (2008, *Episteme* 5(1):56–73) showed that you cannot jointly justify two assumptions people quietly make about any panel: that the judges are *competent*, and that they are *unconditionally independent*. The argument is almost unfair in its simplicity. Whatever makes your judges competent — shared high-quality training data, a shared notion of good reasoning, a shared rubric — is a *common cause*, and a common cause correlates their verdicts. You can assume competence, or you can assume independence, but not both from the same story about where the competence came from.

This is an impossibility of *assumption*, not a proof that every panel fails — the distinction matters, and conflating them is its own error. What it tells you is narrower and more useful: independence is not something you can reason your way into. You have to measure it. More judges drawn from the same generative story add correlated votes, and correlated votes are most of what `n_eff ≈ 2` is already telling you that you have.

The auditing profession learned the human version of this the hard way. Moore, Tetlock, Tanlu and Bazerman (2006, *Academy of Management Review* 31(1):10–29) argued that auditor independence fails not through corruption but through unconscious "moral seduction" — formally independent auditors producing systematically biased judgments because of the relationship, not in spite of their integrity. Their verdict on the structural remedies of the day was that they were clearly insufficient. Swap "auditor" for "evaluator" and the sentence needs no other edits.

## What to do on Monday

None of this is an argument for abandoning panels, and it is certainly not an argument against measuring α. It is an argument for measuring the *other* axis as well:

1. **Keep α.** It catches the failure it was built to catch. Just stop treating it as evidence of independence, because it is silent on independence by construction.
2. **Compute `n_eff` on your own panel.** It is Kish's formula applied to the judges' error-agreement, not an exotic quantity — if you log per-item verdicts you already have the inputs. The batch tooling exists: JuryProbe (2608.20607) diagnoses a frozen panel and surfaces the correlated-failure structure directly. Treat a low `n_eff` the way you'd treat a failing test, not a curiosity.
3. **Watch it over time.** A panel's correlation structure is not fixed — models get swapped, prompts drift, a new frontier release quietly makes two of your judges think more alike. A one-off batch diagnosis is a snapshot of a moving target. The honest version of this measurement is *sequential*: a running check that stays valid no matter when you stop and look. That is the instrument I think the field still needs, and the batch tools are the right precursor to it.

The field already has the word. "Model monoculture" was named in print by the European systemic-risk board's advisory report late last year, which drew the comparison to correlated failure in finance itself. What the word has lacked is the number and the procedure to go with it — a way to turn "our evaluators might be too alike" from a worry into a measurement.

Which brings us back to the FTC. When a regulator asks whether "independent" was a true word, "our panel scored 0.83 on Krippendorff's α" is not an answer to the question asked. It is an excellent answer to a different question. The question asked is: *how many independent looks did this actually have?* If you can't put a number on that, you are relying on a word you have never measured — and that, it turns out, is now a word a regulator will ask you to defend.

---

<!-- ============================ GUARDRAILS (do not ship these) ============================
Numbers/claims verified from primary S190; framings below are LOCKED — do not loosen.
- FTC/METR (30 Sep 2026): consumer-protection inquiry, FTC Act §5. "deceptive act" = how regulators
  FRAME it, NEVER "illegal"/"found deceptive". I-am-not-a-lawyer disclaimer kept in-text.
- AVERI / Brundage: "labs shouldn't grade their own homework"; $7.5M Jan 2026 (amount omitted in prose, fine).
- Kohli 2605.29800: n_eff=2.18 [2.07,2.31], Kish-on-φ (error-agreement). 7-judge→1.93. NEVER chain the
  "8–22pp below ideal" Condorcet accuracy gap to n_eff — different estimand. n_eff is CONTRASTIVE
  (≈2 informative votes, NOT 100% joint failure) — stated in-text.
- Goel 2502.04313: scores↔model-similarity (r=0.84 = affinity bias, 9 judges×39 pairs) AND separately
  capability↑→error-corr↑ among ~130 models. In prose I did NOT cite r=0.84 as a direct inter-model
  error correlation (guardrail). Kept qualitative.
- Krippendorff α ≥ 0.80: "de facto standard" hedge used because the ≥0.80-canonical framing is still a
  LOW gate (Krane ch.8 not primary-verified). Move is "α AND n_eff", never "n_eff instead of α".
  α=κ=φ algebraically is a SEPARATE (positioning vs algebra) point — deliberately not raised.
- Dietrich 2008 Episteme 5(1):56–73: impossibility of JOINT justification (competence+uncond. indep.),
  NOT a proof panels fail. Distinction stated in-text.
- Moore-Tetlock-Tanlu-Bazerman 2006 AMR 31(1):10–29: ARGUED critique ("moral seduction"), NOT an
  empirical null. "clearly insufficient" is their phrase.
- JuryProbe 2608.20607: batch analog; the lift↔excess (Var(Θ)/E[Θ²]) identity is NOT asserted (LOW gate).
- "model monoculture": coined by ESRB Advisory Report No. 16 (Dec 2025); cite them, don't coin, and this
  draft says "named in print by" NOT "the ESRB warns".
- HELD IN RESERVE (not in this draft; deploy only if a reviewer invokes Hong-Page): Thompson 2014
  Notices AMS 61(9):1024-1030 — "contested", NEVER "Page conceded".
- Ghanem β≈0.46 (2609.18272): derived-not-measured, IEC band 0.005–0.05. Omitted to keep tight.
======================================================================================= -->
