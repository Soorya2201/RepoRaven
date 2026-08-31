# RepoAudit — Problem Statement & Project Status

This follows the micro1 brief's four-question framework, applied to RepoAudit
(not the brief's own hiring-evaluation example — that was a template for
structuring this doc, not a task we're doing).

---

## 01. Who has this problem?

Journal editors, PC (program committee) chairs, and ML team leads who have to
decide whether a paper+repo pair's reported results can be trusted, before
publishing it, citing it, building on it, or basing a product decision on it.

Concretely: a reviewer who has 15 minutes to sanity-check a submission's code
before recommending accept/reject. An ML team lead deciding whether to build
a new project on top of a paper's claimed method. A journal editor triaging
which of 200 submitted repos need a deeper look before publication.

## 02. What bottleneck makes it worth solving?

Right now this is done by hand, if it's done at all. OpenAI's PaperBench
measured a 48-hour human baseline for full replication, hitting only 41.4%
completion — and that's for *replicating a paper from scratch*, a much harder
task than what we're doing. Auditing an *existing* repo against its paper for
a handful of *known* failure patterns (data leakage, hyperparameter drift,
single-seed reporting, metric misimplementation, test-set contamination,
eval-on-train) is a narrower, cheaper, more tractable question — but nobody
has built a tool that answers just that question. Existing benchmarks
(PaperBench, CORE-Bench, REPRO-Bench, MLE-bench, NatureBench) all ask "can
the agent do the research," not "can the agent tell whether the research was
done right." That's the gap.

The bottleneck isn't that these bugs are subtle to a domain expert reading
the code carefully — it's that nobody has the 4-8 hours per paper to do that
reading, so it mostly doesn't happen. (Concrete evidence for this from our
own eval-set construction: two dedicated research passes searching ReScience
C reports and GitHub issues found real, citable, pre-documented instances of
only 2 of our 6 target failure modes. The rest — including a genuinely
serious bug we found by hand in an official ICML 2024 repo, `lbcs`: a paper
claiming 20-run mean±std reporting whose shipped code hardcodes one seed with
no way to reproduce the aggregation — were sitting undiscussed in public,
well-known repos. Nobody had filed an issue about them.)

## 03. Does the agent solve it well?

That's what Phases 1-5 measure, not yet fully answered. The design: start
with a deliberately bad single-prompt baseline (Phase 1, in progress), then
iterate — split extraction/verification (Phase 2), give the verifier real
tools like grep/AST search instead of just reading a dump (Phase 3), replace
the single verifier with 6 specialized checker subagents, one per failure
mode (Phase 4), add a consensus/human-approval gate for high-severity
findings before they're surfaced (Phase 5, satisfying the brief's ground
rule #4/#5 on human review for consequential findings). Every phase is
scored against the same 9-paper eval set with hand-verified ground truth, so
"does it solve it well" gets a real, comparable number at every step, not
just at the end.

## 04. Can another person reproduce the result?

That's the target for Phase 6 (`REPRODUCTION.md` + `scripts/run_all.py`),
not built yet. What exists now: every finding in `data/ground_truth.json`
cites an exact file:line plus the paper text it contradicts; every "clean"
label is backed by an actual code read, not just trusting a third-party
report's summary tag; every eval-set paper's repo is pinned to a specific
commit SHA; the mechanical selection filters (public, <500MB, Python-majority,
has README, arXiv HTML available) are re-checked programmatically by
`scripts/curate_eval_set.py`, not asserted by hand.

---

## Where I stand right now

**Phase 0 (approved, v2):** 9-paper eval set (8 real + 1 synthetic fixture
covering the 3 failure modes we couldn't find real public examples of),
hand-verified ground truth, environment set up with live-verified NIM models
(the spec's original model names were deprecated 4 days before we started).

**Phase 1 (in progress):** baseline agent built (`src/baseline.py`) — one
LangGraph node, one prompt, paper + README + head of every `.py` file dumped
in, one LLM call, asks for JSON findings. `src/eval.py` (the scoring harness
used by every future phase) is built and working.

First real run: **6 of 9 papers technically failed** — not because the
agent's reasoning was bad, but because of two infrastructure bugs: (1)
ChatNVIDIA's default 60s HTTP timeout was too short for large-context calls
to a reasoning model (fixed: raised to 240s), and (2) the initial 6000-token
response budget wasn't enough room for the model's `reasoning_content` plus
the actual JSON answer on non-trivial repos — it would hit `finish_reason:
"length"` mid-thought and never emit valid JSON (fixed: raised to 16000,
verified empirically sufficient). A second full run is in progress now with
both fixes applied; results pending.

One useful thing that came out of debugging this: while manually re-running
`lottery_ticket_hypothesis` at the higher token budget to diagnose the
timeout, the baseline agent independently re-derived the real hyperparameter
bug we already had in ground truth, *and* surfaced a second real issue we'd
missed in our own hand-annotation (the code never constructs a validation
split at all, contradicting the paper's validation-based early-stopping
claim) — verified by reading the code, now added to `ground_truth.json`. The
eval process is already finding real gaps in our own ground truth, which is
a good sign for the underlying idea even before the "real" iterations start.

## Pain points, stated plainly

- **Eval set is 9 papers, not the 12 targeted.** Category C (your own
  TRACE/PMNet/planted-defect repos) was skipped by explicit direction and is
  still not filled in — this is the single biggest gap against the brief's
  "one challenging case" ask.
- **3 of 6 failure modes (metric_misimplementation, test_set_contamination,
  eval_on_train) have only synthetic, not real, ground-truth positives** —
  real-world research genuinely couldn't source citable examples for these
  within two dedicated search passes. Honestly logged, not hidden.
- **NIM's actual $/credit rate is unverified** — the pricing page wouldn't
  render for an automated fetch. Token usage is tracked precisely
  (`src/llm.py`); dollar-cost estimates carry a caveat until this is resolved.
- **The baseline needed hand-tuning to even produce valid output** (timeout,
  token budget) before it could be scored meaningfully at all — this is
  itself informative: a "deliberately simple" one-shot approach against a
  large reasoning model on a nontrivial repo isn't just inaccurate, it's
  operationally fragile. That's a real argument for Phase 2+'s more
  structured approach, not just a nice-to-have.
- **Timeline is tight.** Event ends 2026-08-31. Phases 2-5 and the Phase 6
  deliverables (README, `REPRODUCTION.md`, solution video, agent
  trajectories) are all still ahead.
