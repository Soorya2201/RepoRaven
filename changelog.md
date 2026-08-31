# RepoAudit Improvement Changelog

## Phase 0 — Setup, eval set curation, ground truth

**What we tried and why:** Set up the project skeleton and NIM wrapper, then curated an eval set of real papers with documented and independently-verified reproducibility issues (data leakage, hyperparam drift) plus known-clean papers, to have honest ground truth to measure every later phase against. Two research passes searching ReScience C, GitHub issues, and the Kapoor & Narayanan leakage taxonomy found real, citable, code-verified instances of only 2 of our 6 target failure modes (data leakage, hyperparam drift); the other 4 (single-seed reporting, metric misimplementation, test-set contamination, eval-on-train) were not found documented anywhere in the wild within the time budget.

**Evidence:** No prior phase to compare against — this phase establishes the eval set itself. 7 papers total (3 real category-A with 3 real findings, 3 category-B clean, 1 synthetic fixture with 4 planted findings covering the 4 missing modes) vs. the original target of 12. Every finding is cited to an exact file:line plus the paper text it contradicts; every "clean" label is backed by an actual code read, not just a ReScience "reproduced" tag (which turned out, on inspection, to sometimes track something other than our 6-mode taxonomy — see `phase_0_setup.md` for the Proto2Proto case).

**Decision:** kept, with one adjustment: built one clearly-labeled synthetic fixture (`synthetic_multi_defect`) rather than leaving 4 of 6 failure modes with zero ground-truth positives. This was a genuine build-time decision, not a fallback of convenience — see next line.

**What this taught us:** the failure modes that are *easiest to find pre-documented* (via ReScience reports, GitHub issues) are not necessarily the ones that matter most for reproducibility risk — several genuinely serious bugs (G-Mixup's train/test leakage, Lottery Ticket Hypothesis's wrong optimizer *and* wrong learning rate) were sitting undiscussed in public repos with real published papers, found only by directly reading the code against the paper's claims. That's the strongest early argument for why a tool like RepoAudit is worth building: the bugs it's meant to catch are exactly the ones nobody files a GitHub issue about.

## Phase 0 (continued) — Sourcing eval papers from PaperBench instead of ad-hoc search

**What we tried and why:** switched from ReScience-driven candidate search to sourcing papers directly from PaperBench's (arXiv 2504.01848, OpenAI) official 20-paper ICML-2024 list — a more credible, citable source pool than manual search, per explicit direction. Also investigated REPRO-Bench and NatureBench as possible sources.

**Evidence:** REPRO-Bench rejected outright (social science / Stata-R, not Python, no hyperparameter concepts our taxonomy needs; its own repo doesn't even expose paper metadata). NatureBench doesn't apply (measures beating SOTA, not reproducibility auditing). 11 of PaperBench's 20 papers passed our filters; hand-audited 2 (`lbcs`, `sample_specific_masks`) and found both genuinely exhibit `single_seed_reporting` — paper claims N-seed mean±std, code hardcodes one seed with no aggregation script. Eval set grew from 7 to 9 papers; `single_seed_reporting` ground truth moved from 100% synthetic to 100% real (2 instances). Retired the synthetic fixture's planted version of this bug and fixed its code to a genuine 5-seed loop, so a correct checker isn't falsely penalized against stale ground truth.

**Decision:** kept — real, benchmark-sourced findings are strictly better evidence than our own synthetic fixture wherever we can get them, and this closes one more of the 3 remaining synthetic-only failure modes.

**What this taught us:** the same pattern held again — `single_seed_reporting` bugs (hardcoded seed despite a paper's mean±std claim) were sitting in official ICML 2024 competition-grade repos, unremarked, in 2 of the first 2 papers we checked from a random-ish subset. This is starting to look less like a fluke and more like the norm for this specific failure mode.

## Phase 1 — Baseline agent

**What we tried and why:** single LangGraph node, one prompt, paper + README + first 200 lines of every `.py` file dumped in, one LLM call, temperature 0, asked for JSON findings. Deliberately unsophisticated — this is the floor every later phase has to beat.

**Evidence:**
| Metric | Prev phase | This phase | Delta |
|---|---|---|---|
| Overall F1 | n/a (no prior agent) | 0.320 | — |
| Precision / Recall | n/a | 0.400 / 0.267 | — |
| FPR on clean papers | n/a | 2.0 findings/paper | — |
| Papers with usable output | n/a | 5/9 (4 hit a JSON parse bug) | — |
| Wall clock (9 papers) | n/a | 1069.9s | — |
| Total tokens (9 papers) | n/a | 484,577 | — |

(Numbers above were rescored after Phase 2 against a corrected ground truth
— Phase 2 and Phase 3 agents found real bugs in `birt_continual_learning` and
`g_mixup` that our original Phase 0 annotation missed. Original in-phase
numbers were F1 0.421 / precision 0.400 / recall 0.444 / FPR 1.0; rescoring
against the more complete ground truth makes the baseline look honestly
worse, not better, since it never had a chance to catch bugs nobody had
found yet either. See Phase 2's entry below and
phases/phase_2_iter1_extractor_verifier.md §2.)

Two infrastructure bugs had to be fixed before any real numbers were possible: ChatNVIDIA's 60s default timeout was too short for large-context reasoning-model calls (raised to 240s), and a 6000-token response budget wasn't enough for `reasoning_content` overhead on non-trivial repos, causing `finish_reason: "length"` cutoffs before any JSON was ever emitted (raised to 16000).

**Decision:** kept as the baseline (that's the point), moving to Phase 2. Notably: `hyperparam_drift` and `single_seed_reporting` — the two modes with the most *real* (non-synthetic) ground truth — scored 0.0 recall. The baseline caught zero real instances of either.

**What this taught us:** the baseline's dominant failure mode wasn't bad reasoning, it was unreliable output — a systematic JSON-formatting bug (the model consistently drops the opening quote on the `"explanation"` field) silently discarded results on 4 of 9 papers, including plausible-looking findings that got thrown away purely on a syntax slip. Separately, every real bug the baseline *did* catch was a "presence" pattern (one wrong line); every bug it *missed* was an "absence" pattern (no seed loop anywhere in a 30-file repo, no validation split anywhere in the codebase) — invisible to a flat 200-line-per-file dump with no way to reason across the whole repo. Both are direct arguments for Phase 2 (structured extractor/verifier, catches ungrounded hallucination) and Phase 3 (tools that can search a whole repo for absence, not just read heads of files).

## Phase 2 — Iteration 1: Extractor + verifier split

**What we tried and why:** split the single-shot baseline into a claim extractor (fast model, paper text only, extracts specific checkable claims each with an exact verbatim quote) and a verifier (primary model, checks each claim against the repo). Two targeted robustness mechanisms: a post-hoc check that discards any finding not grounded in a real extracted quote (directly targets Phase 1's hallucination), and an LLM-based JSON repair pass on any parse failure (directly targets Phase 1's 44% parse-failure rate). Also confirmed NIM's endpoint doesn't support native structured output / tool-calling (both return HTTP 400/500) — the repair-pass approach is the portable alternative.

**Evidence:**
| Metric | Prev phase (rescored) | This phase | Delta |
|---|---|---|---|
| Overall F1 | 0.320 | 0.667 | +0.347 |
| Precision / Recall | 0.400 / 0.267 | 0.750 / 0.600 | +0.350 / +0.333 |
| FPR on clean papers | 2.0 | 0.5 | −1.5 (4x fewer false alarms) |
| hyperparam_drift F1 (by-mode) | 0.0 | 1.0 | +1.0 |
| Wall clock (9 papers) | 1069.9s | 1910.8s | +840.9s (2 LLM calls/paper now) |
| Total tokens (9 papers) | 484,577 | 403,417 | −81,160 (fewer per-call, more calls) |

**Decision:** kept. This is the single biggest jump in the whole changelog so far, across every metric simultaneously, not a tradeoff.

**What this taught us:** two things, one about the design and one about our own process. On design: claim-by-claim systematic checking essentially solved `hyperparam_drift` detection (0.0 → 1.0 F1) — forcing the model to extract every specific numeric claim first, then check each one, beats holistic "does anything look wrong" scanning. On process: while evaluating this phase's results, we found the agent surfacing real bugs in `birt_continual_learning` and `g_mixup` that weren't in our own Phase 0 ground truth — verified both by hand before accepting them, then rescored Phase 1 fairly against the corrected data (making Phase 1 look worse, honestly, not better). A reproducibility-auditing tool ended up auditing its own evaluation harness. We also found a real, still-open limitation: the context builder only reads `.py` files, so a verifier call incorrectly flagged `g_mixup` for single-seed reporting when the real seed loop lives in a `.sh` script it never saw — a concrete, evidence-backed argument for Phase 3's tool-based approach rather than a prompt fix.

## Phase 3 — Iteration 2: Real tools (grep / AST / config) — REGRESSED, kept as a negative result

**What we tried and why:** Phase 2's failure analysis pointed at the context layer, not the reasoning layer: the verifier could only see `.py` file heads, so a real seed loop living in `run_gmixup.sh` was invisible and absence-checks ("is there a `--seed` flag anywhere in this 30-file repo?") were impossible. So we gave the verifier real tools — whole-repo grep across every file type, tree-sitter AST search for exact function calls/assignments, structured argparse/YAML parsing, plus a `ReadFile` tool — and turned it into a multi-turn ReAct agent with native tool-calling. This also required a model change: `nemotron-3-super` returns HTTP 400/500 on tool-calling for this endpoint, so the verifier moved to `openai/gpt-oss-120b` (verified with a real multi-turn tool-call loop).

**Evidence:** (all phases rescored against identical current ground truth)
| Metric | Phase 2 | Phase 3 | Delta |
|---|---|---|---|
| Overall F1 | 0.667 | 0.273 | **−0.394** |
| Precision / Recall | 0.750 / 0.600 | 0.429 / 0.200 | −0.321 / −0.400 |
| Wall clock (9 papers) | 1910.8s | 1142.5s | −768s |
| Total tokens | 403,417 | 983,688 | **+580,271 (2.4x)** |

Worse on every quality metric, at 2.4x the token cost.

**Decision:** revised, not kept. The tools ship forward into Phase 4 — they demonstrably work (during development the tool-using verifier read `run_gmixup.sh` and caught that it loops over **8 seeds while the paper claims ten runs**, a subtler and more precise bug than Phase 2's crude and *wrong* "no seed loop exists"; that's now verified finding #5 in g_mixup's ground truth, and Phase 2 was structurally incapable of finding it). What does not ship forward is the single-shared-extractor + one-general-verifier architecture.

**What this taught us:** the regression's cause is fully isolated by `birt_continual_learning` (Phase 2: 2/2 findings; Phase 3: 0/2 despite 9 successful tool calls). Phase 2's extractor produced *atomic* claims — `"weight decay of 1e-6"` and `"averaged across three random seeds"` as separate items — and caught both bugs. Phase 3's extractor, with a byte-identical prompt, happened to bundle three hyperparameters into one compound quote and dropped the seed claim entirely. Since the grounding rule requires every finding to map to an extracted claim, the seed bug became *structurally unfindable* no matter how good the tools were, and the compound claim diluted the weight-decay check (2 of its 3 values match the code, so the sentence reads as broadly correct). **Better tools cannot compensate for worse inputs — the bottleneck was never evidence gathering, it was evidence specification.** Separately, 10 free investigative turns produced analysis paralysis on 2 of 9 papers (0 findings after exhausting the budget). Both findings point the same direction: Phase 4's specialized per-failure-mode checkers, each owning one narrow atomic question, structurally enforce the atomicity this phase lost by accident.
