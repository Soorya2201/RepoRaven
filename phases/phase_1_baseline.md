# Phase 1 — Baseline Agent

## 1. What we built

The deliberately simple baseline per instructions.md §5: a single LangGraph
node, one prompt, one LLM call. For each paper we dump the paper's full text,
the repo's README, and the first 200 lines of every `.py` file into one
prompt, ask the model (temperature 0) to return a JSON list of findings, and
parse the result. No tools, no verification, no follow-up — this is the floor
every later phase has to beat.

Getting this to run cleanly took two infrastructure fixes, not agent-design
changes (see §5 below): ChatNVIDIA's default 60s HTTP timeout was too short
for large-context calls to a reasoning model (raised to 240s), and the
initial 6000-token response budget wasn't enough room for the model's hidden
`reasoning_content` plus the actual JSON answer on non-trivial repos (raised
to 16000, empirically verified sufficient for every paper in this set).

## 2. Prompt used (verbatim)

> You are auditing an ML research paper and its code repository for reproducibility issues. Read the paper and the code below. Return a JSON list of findings. Each finding must have: `failure_mode` (one of: data_leakage_global_preprocessing, hyperparam_drift, single_seed_reporting, metric_misimplementation, test_set_contamination, eval_on_train, other), `severity` (high/medium/low), `paper_evidence` (short quote or section reference), `code_evidence` (file:line or file), `explanation` (1-2 sentences). Return `[]` if you find nothing. Return only the JSON, no prose.

(Sent as the system message; the paper+README+code dump is the user message.)

**Note added after Phase 2:** the numbers below were rescored (not re-run —
same saved predictions, `src.eval.rescore_phase`) against a ground truth that
changed after this phase closed. Phase 2's agent found real bugs in
`g_mixup` (2 more hyperparameter mismatches) and `birt_continual_learning`
(previously annotated clean — it wasn't) that our Phase 0 hand-annotation
missed. Rescoring the baseline fairly against the same corrected ground
truth Phase 2 is measured against actually makes the baseline look *worse*
(more real bugs it never had a chance to catch), which is the honest
result — see phases/phase_2_iter1_extractor_verifier.md for the full story.

## 3. Results table

| paper_id | n_expected | n_predicted | TP | FP | FN | F1 |
|---|---|---|---|---|---|---|
| proto2proto | 0 | 3 | 0 | 3 | 0 | 0.0 |
| g_mixup | 4 | 0 (parse error) | 0 | 0 | 4 | 0.0 |
| lottery_ticket_hypothesis | 2 | 1 | 1 | 0 | 1 | 0.667 |
| network_deconvolution | 0 | 0 (parse error) | 0 | 0 | 0 | 1.0* |
| birt_continual_learning | 2 | 0 (parse error) | 0 | 0 | 2 | 0.0 |
| cifar_10_100n | 0 | 1 | 0 | 1 | 0 | 0.0 |
| synthetic_multi_defect | 3 | 3 | 3 | 0 | 0 | 1.0 |
| lbcs | 1 | 2 | 0 | 2 | 1 | 0.0 |
| sample_specific_masks | 1 | 0 (parse error) | 0 | 0 | 1 | 0.0 |
| **MICRO-AVG** | — | — | **4** | **6** | **9** | **0.348** |

`*` — network_deconvolution scores a "perfect" 1.0 here, but **this is a
masked failure, not a correct assessment**: the call hit the JSON parse bug
described below (§5), returned zero findings, and on a paper with zero
expected findings that coincidentally scores as if the agent correctly said
"this is clean." It didn't — it never produced a readable answer at all.
(birt_continual_learning hit the same parse bug, but since it's no longer a
zero-expected-findings paper after the ground-truth correction above, its
row correctly shows FN=2 instead of a masked "1.0".)

## 4. Aggregate metrics

**Micro-avg: precision 0.400, recall 0.308, F1 0.348.** False positive rate
on clean papers (now just proto2proto + network_deconvolution + cifar_10_100n,
since birt_continual_learning is no longer counted as clean): **1.333**
predicted findings per clean paper on average — driven almost entirely by
proto2proto's 3 hallucinated findings.

By failure mode:

| failure_mode | TP | FP | FN | F1 |
|---|---|---|---|---|
| data_leakage_global_preprocessing | 0 | 0 | 1 | 0.0 |
| hyperparam_drift | 0 | 2 | 5 | 0.0 |
| single_seed_reporting | 0 | 1 | 3 | 0.0 |
| metric_misimplementation | 1 | 1 | 0 | 0.667 |
| test_set_contamination | 2 | 0 | 0 | 1.0 |
| eval_on_train | 1 | 0 | 0 | 1.0 |

`hyperparam_drift` and `single_seed_reporting` — the two modes with the most
real (non-synthetic) ground-truth instances — score **0.0 recall**: the
baseline missed every single real instance of these two modes across the
whole eval set. It only caught findings on the fully-synthetic paper
(`synthetic_multi_defect`, all 3 caught) and on the paper it happened to
fully parse.

## 5. Failure analysis

**Case 1 — a systematic JSON-formatting bug hid 4 of 9 papers' worth of results.**
`g_mixup`, `network_deconvolution`, `birt_continual_learning`, and
`sample_specific_masks` all failed to parse for the *exact same reason*: the
model consistently drops the opening quote on the `"explanation"` field's
value (e.g. `"explanation": The paper describes...` instead of
`"explanation": "The paper describes...`), which is invalid JSON no amount of
lenient parsing can recover. This is not random — it happened identically
across 4 independent calls to different papers. We deliberately did **not**
patch around this with a regex repair: the baseline is supposed to be naive,
and a real, reproducible formatting failure mode is exactly the kind of thing
worth measuring and reporting honestly rather than engineering away before
we've even established the floor. Inspecting the truncated snippets, several
of these hidden responses looked substantively reasonable before cutting off
(e.g. network_deconvolution's hidden finding correctly cited `arg_parser.py`
and `main.py`'s seed handling) — meaning some real reasoning was likely
thrown away purely by a formatting slip, not a reasoning failure.

**Case 2 — proto2proto: 3 hallucinated findings, illustrating why we kept
this "documented as broken, but not for our reasons" paper as a negative
control.** The baseline claimed `single_seed_reporting` ("the paper reports
results from a single random seed") — but we verified in Phase 0 that the
paper's text contains zero mentions of "seed" anywhere; there's no claim to
contradict. The model appears to be pattern-matching "this paper had
reproducibility problems" (visible in surrounding text/README signals) into
a plausible-sounding but factually ungrounded specific claim, rather than
actually checking the paper for a multi-seed statement.

**Case 3 — lbcs: missed the real bug, hallucinated two different ones.** The
real, verified bug (Phase 0) is `single_seed_reporting`: three run scripts
each hardcode a single seed with zero mechanism anywhere in the repo to run
or aggregate the paper's claimed 20 repeats. The baseline never flagged this
at all. Instead it predicted two specific `hyperparam_drift` claims (e.g.
"paper reports Adam lr=0.001... code uses lr=0.005") that don't match
anything in our ground truth. Our hypothesis: the real bug here is an
*absence* — no `--seed` flag, no wrapper script, no aggregation logic
*anywhere in the 30+-file repo* — and the baseline only sees the first 200
lines of each file with no way to reason about repo-wide absence of a
mechanism. Spotting "this thing is missing across the whole codebase" is a
fundamentally different (harder) task than spotting "this specific line
looks wrong," and a flat 200-line-per-file dump doesn't support it.

## 6. What this tells us

- **The baseline's biggest problem isn't reasoning quality, it's reliability.** Nearly half the eval set (4/9) produced zero usable signal purely from a JSON-formatting slip, and 2 of those coincidentally scored as "correct" on clean papers despite never actually assessing them. A next iteration needs either structured/tool-enforced output or a repair-and-retry step, not just a better prompt.
- **Absence-based bugs (missing seed loop, missing validation split) are effectively invisible to a flat context dump.** Every real finding the baseline *did* catch was a presence-based pattern (a specific wrong line); every real absence-based bug it missed. This is a strong argument for Phase 3's tool-based approach (a checker that can grep the *whole* repo for "is there a --seed flag anywhere," not just read the first 200 lines of each file).
- **The baseline hallucinates plausible-sounding claims not grounded in the actual paper text** (proto2proto's phantom seed claim). A verifier step that has to point to an actual quote and gets checked against it (Phase 2's extractor/verifier split) should catch this directly.

## 7. How to reproduce Phase 1

```bash
.venv/bin/python -m src.eval baseline
```

Writes `outputs/baseline/<paper_id>.{json,txt}` and
`outputs/baseline/_summary.{json,txt}`. Took 1069.9s wall clock, 484,577
total tokens (417,482 prompt / 67,095 completion) across 9 calls in this run.

## 8. Next phase gate

Phase 2 splits the single node into an extractor (paper claims → structured
list, fast model) + verifier (claims + repo → findings, primary model). Given
that the single biggest loss here was a formatting/reliability failure, not
a reasoning failure, **the F1 threshold to clear is a low bar (>0.421) but
the real target is eliminating the parse-failure category entirely** — if
Phase 2 gets every paper to produce *some* readable answer, that alone should
meaningfully move recall even before the extractor/verifier split improves
actual detection quality.
