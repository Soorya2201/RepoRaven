# Solution Video — Script & Shot List

**Hard constraint: ≤5 minutes.** The brief says this explicitly and judges
watch dozens of these — going over doesn't get you more credit, it gets you
skipped. Every section below has a timestamp. If you're behind, cut from
Section 4 (Changelog Walkthrough) first — it's the most compressible because
`README.md` already carries the detail in writing.

**The one idea to hold onto for the whole 5 minutes:** you're not showing
off a chatbot that read some code. You're showing a system that found a real
**~50,000× weight-decay error** and a **wrong optimizer** in well-known,
already-published ICML/ICLR papers, that nobody had ever flagged — and then,
when you gave it *more* capability in the next iteration, it got *worse*,
and you can explain exactly why. That second part is rarer and more
valuable than the first. Most hackathon videos show only wins. Showing a
diagnosed loss is what makes a judge sit up.

---

## Before you record: 3 things to have open

1. **Terminal**, ready to run `python scripts/run_all.py --skip-fetch --rescore-only --phases iter1` — this reproduces Phase 2's real numbers live, in seconds, with **zero API cost and zero risk of a bad take from network flakiness**. Do NOT run a live LLM call on camera — it's slow, non-deterministic, and you don't need the suspense.
2. **`outputs/iter1/FINAL_REPORT.md`** open in a Markdown preview (not raw text) — this is the actual product, and it needs to look like one on screen.
3. **`README.md`** open to the "Improvement Changelog" table — your visual anchor for Section 4.

Record your screen at 1080p minimum. Test your mic levels before the full take — nothing undercuts a strong technical narrative like bad audio.

---

## Section 1 — The Problem (0:00–0:55)

**Say, don't read:**

> "If you're a program committee chair or a team lead, and someone hands you a paper and its code, how do you know the results are real? Right now you either trust it, or you spend hours reading code line by line. OpenAI actually measured this — their PaperBench benchmark had 8 ML PhDs spend 48 hours each trying to replicate 3 papers from scratch. Best result: 41% completion. That's the cost of doing this rigorously by hand.
>
> But most of the time, nobody's checking. Not for full replication — just for six specific, well-known ways ML results go wrong: data leakage, wrong hyperparameters, single-seed reporting dressed up as robust, wrong metrics, test-set contamination, evaluating on training data. Those are checkable in minutes if you know to look. Nobody's building the tool that looks."

**On screen:** README.md, top section, scrolled to "Who has this problem" and the six-failure-mode table.

**Why this works:** you're quoting a real, external, verified number (you fetched it from the actual paper yourself this session — say that, briefly: "we double-checked this against the actual paper, not just cited it secondhand" — that one sentence signals rigor to a technical judge in 5 seconds).

---

## Section 2 — The Baseline, and Why It's Bad On Purpose (0:55–1:35)

**Say:**

> "So we built the dumbest possible version first, deliberately — one prompt, dump the whole paper and repo in, ask for JSON back. That's your floor. It scored 0.32 F1. Not because the model can't reason — because it's *unreliable*. A formatting bug in its own output silently threw away results on 4 of our 9 test papers. The lesson wasn't 'get smarter,' it was 'get robust.'"

**On screen:** `outputs/baseline/FINAL_REPORT.md`, briefly — enough to show it's real output, not a mockup.

**Timing discipline:** this section exists to set up the contrast in Section 4. Don't linger.

---

## Section 3 — One Real Execution, Start to Finish (1:35–3:00)

This is the brief's explicit requirement: "walk through one realistic
execution from start to finish." Pick **G-Mixup** — it's your richest case
(5 real findings, one of them a genuine data-leakage bug, all independently
verified against the actual code).

**Live on screen:** run

```bash
python scripts/run_all.py --skip-fetch --rescore-only --phases iter1
```

While it runs (seconds), talk over it:

> "This replays our actual saved run against current ground truth — no API cost, same real numbers we're about to look at."

Then open `outputs/iter1/FINAL_REPORT.md`, scroll to the G-Mixup section.

**Say, pointing at specific findings on screen:**

> "Here's G-Mixup, ICML 2022. The paper says: 'we generate 20% more graphs for training.' The code's actual default is 15%, and — this is the important part — the *official run script they publish* also uses 15%. So even someone following their own documented instructions gets the wrong number. Same story with their mixing-strength parameter: paper says point-one to point-two, code default is point-zero-zero-five to point-zero-one — a 10 to 20x difference.
>
> And this one's the real one: their preprocessing step computes normalization statistics over the *entire* dataset — train, validation, and test, all concatenated — before the split happens. That's textbook data leakage. Test information is bleeding into the features every model sees during training.
>
> Every single one of these — I verified by hand, reading the actual code, before it went into our ground truth. This isn't the model's opinion. It's a citation you can check yourself in under a minute."

**On screen:** hover/highlight the `code_evidence` and `paper_evidence` fields for 2-3 findings so the judge sees the file:line citations are real, not vibes.

---

## Section 4 — The Comparison and the Changelog (3:00–4:15)

**Say:**

> "Across three iterations, scored against the same ground truth every time: baseline, 0.32 F1. We split the pipeline into a claim-extractor and a verifier, added a grounding rule so it can only report findings tied to a real quote from the paper — that jumped to 0.667. Best result we got.
>
> Then we gave it real tools — grep across the whole repo, AST search, config parsing — expecting it to get better. It got *worse*. 0.27."

**On screen:** README's changelog table, or better, actually scroll `phases/phase_3_iter2_tools.md` to the failure-mode table for 3 seconds.

> "We dug into why instead of just moving on. One case makes it obvious: a paper claiming results averaged over three seeds. Our tool-using agent made *exactly* the right tool call — searched the whole repo for any seed-related code — and then just... never committed to an answer. It had ten free turns to investigate and used them wandering instead of concluding. Meanwhile our simpler, more constrained pipeline from the previous iteration nailed the same bug immediately, because it only had one decision to make: does this claim match the code, yes or no.
>
> More capability, more ways to fail. That's the actual finding here."

**Why this section matters most for scoring:** the brief explicitly asks you to "highlight the change that contributed most" (the extractor/verifier split, +0.35 F1 in one change) "as well as one experiment you removed" (this is it). Say both phrases near-verbatim — it signals directly to the rubric.

---

## Section 5 — The Hot Take, and Close (4:15–5:00)

**Say:**

> "If we build this again, here's the lesson: the agentic community over-invests in giving agents more capability, and under-invests in specifying what they're checking. We handed our agent a full toolbox and it did worse than a simpler pipeline whose only advantage was that its inputs were atomic — one fact per claim, not a paragraph bundling three together. Agency amplifies whatever quality your inputs already have. It doesn't fix bad inputs.
>
> And one more thing worth saying: while we were evaluating this, the agent itself found two real bugs in well-known papers that *we* had missed in our own manual ground-truth review — a paper claiming three-random-seed averaging with a repo that only ever runs one seed, and a learning-rate default that's right for two architectures and silently wrong for a third. We went back, verified those by hand, and added them. A tool built to catch reproducibility mistakes ended up catching our own. That's the best evidence we have that it's doing something real."

**On screen:** back to README, the Hot Take section, then hard cut to black or your final slide.

**Do not** add a generic "thanks for watching" outro — end on the strongest sentence you have, which is the one above.

---

## Delivery notes

- **Don't read bullet points off the screen verbatim.** Judges can read faster than you can talk; use the screen to *prove* what you're saying, not to be your teleprompter for the whole thing.
- **Say real numbers out loud, not "significantly better."** "0.32 to 0.667" lands harder than "much improved" — specificity reads as credibility to a technical audience.
- **The Phase 3 regression is your strongest asset in this video, not something to downplay.** Say it plainly and early in Section 4 rather than burying it. A team that only shows wins looks like it's hiding something; a team that shows a diagnosed loss looks like it knows what it's doing.
- **If you're short on time, cut Section 2 to one sentence** ("we started with the dumbest possible baseline, 0.32 F1, mostly failing on output reliability, not reasoning") and spend the saved time in Section 3 — a real execution walkthrough is explicitly required and is your best concrete evidence.
- **Practice the G-Mixup numbers cold** (20% vs 15%, 0.1-0.2 vs 0.005-0.01) — stumbling over your own evidence undercuts the rigor you're trying to demonstrate.

## What NOT to do

- Don't claim "100% accuracy" or oversell precision — your own README states 75% precision on your best phase. Overclaiming against your own written numbers is the single fastest way to lose credibility with a judge who then opens the README.
- Don't skip the failure/regression story to "keep things positive." The rubric explicitly rewards it; skipping it is leaving free points on the table.
- Don't demo a live, uncached API call. If something times out or a model returns a malformed response on camera (it has happened repeatedly during development — see the changelog), you lose your only take. Use `--rescore-only` for a live, real, zero-risk demo instead.
