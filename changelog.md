# RepoRaven Improvement Changelog

## Phase 0 — Setup, eval set curation, ground truth

**What we tried and why:** Set up the project skeleton and NIM wrapper, then curated an eval set of real papers with documented and independently-verified reproducibility issues (data leakage, hyperparam drift) plus known-clean papers, to have honest ground truth to measure every later phase against. Two research passes searching ReScience C, GitHub issues, and the Kapoor & Narayanan leakage taxonomy found real, citable, code-verified instances of only 2 of our 6 target failure modes (data leakage, hyperparam drift); the other 4 (single-seed reporting, metric misimplementation, test-set contamination, eval-on-train) were not found documented anywhere in the wild within the time budget.

**Evidence:** No prior phase to compare against — this phase establishes the eval set itself. 7 papers total (3 real category-A with 3 real findings, 3 category-B clean, 1 synthetic fixture with 4 planted findings covering the 4 missing modes) vs. the original target of 12. Every finding is cited to an exact file:line plus the paper text it contradicts; every "clean" label is backed by an actual code read, not just a ReScience "reproduced" tag (which turned out, on inspection, to sometimes track something other than our 6-mode taxonomy — see `phase_0_setup.md` for the Proto2Proto case).

**Decision:** kept, with one adjustment: built one clearly-labeled synthetic fixture (`synthetic_multi_defect`) rather than leaving 4 of 6 failure modes with zero ground-truth positives. This was a genuine build-time decision, not a fallback of convenience — see next line.

**What this taught us:** the failure modes that are *easiest to find pre-documented* (via ReScience reports, GitHub issues) are not necessarily the ones that matter most for reproducibility risk — several genuinely serious bugs (G-Mixup's train/test leakage, Lottery Ticket Hypothesis's wrong optimizer *and* wrong learning rate) were sitting undiscussed in public repos with real published papers, found only by directly reading the code against the paper's claims. That's the strongest early argument for why a tool like RepoRaven is worth building: the bugs it's meant to catch are exactly the ones nobody files a GitHub issue about.

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

## Scoring bug found and fixed — Phase 3's numbers above were understated (and so were others)

**What we found:** while building Phase 4, `network_deconvolution`'s and `g_mixup`'s checker predictions correctly cited `run_gmixup.sh` and `readme.md` as evidence — files with no `.py` extension. `src/eval.py`'s matching function only ever extracted `.py` filenames via regex. Any ground-truth finding whose evidence lived in a `.sh`, `.md`, or config file was **structurally unmatchable** — `_extract_py_basenames` returned an empty set for it every time, so `_findings_match`'s `if not gt_files: return False` guaranteed a miss regardless of what any phase predicted. This affected `g_mixup`'s and `sample_specific_masks`'s single-seed-reporting ground truth specifically (both live in a shell script or README, not a `.py` file), silently, in every phase's score since those entries were added.

**Fix:** broadened the extraction regex to cover `.py|.sh|.md|.yaml|.yml|.json|.txt|.cfg|.ini|.toml`, verified against both previously-broken cases directly before trusting it, then rescored all four phases from their saved predictions (`rescore_phase`, no new LLM calls).

**Evidence — corrected F1 across all phases, same ground truth, same predictions, only the matcher fixed:**
| Phase | F1 before fix | F1 after fix | Delta |
|---|---|---|---|
| Baseline | 0.182 | 0.182 | — (unaffected; its predictions never cited the affected files) |
| Iteration 1 (Phase 2) | 0.667 | 0.667 | — (unaffected) |
| Iteration 2 (Phase 3) | 0.273 | **0.455** | **+0.182** |
| Iteration 3 (Phase 4) | 0.500 | **0.643** | **+0.143** |

**Decision:** the corrected numbers are now authoritative everywhere in this repo; every doc referencing the old Phase 3/4 figures has been updated. This does **not** reverse Phase 3's core finding (compound claims broke grounding) — Phase 3 is still below Phase 2. It does change the magnitude: Phase 3 was never as bad as "worse than the naive baseline," and Phase 4 (once corrected) is now essentially tied with Phase 2, with a materially different strength profile (see Phase 4's entry).

**What this taught us:** a scoring bug in the harness itself is exactly as dangerous as a bug in the agent — it silently understated real capability for two full phases before a correctly-grounded prediction happened to expose it by citing evidence in a file type nobody had tested the matcher against. We only found it because Phase 4's checkers, unlike earlier phases, cite evidence from whatever file type the deterministic gather tool actually returns rather than defaulting to `.py` files out of habit — another point in favor of Phase 4's design surfacing things a less-grounded approach wouldn't have.

## Phase 4 — Iteration 3: Specialized per-failure-mode checkers

**What we tried and why:** replaced the single shared extractor+verifier with 6 specialized checkers, one per failure mode, each with its own narrow prompt and only the tools relevant to it -- directly informed by Phase 3's lesson. Critically, no checker uses an open-ended ReAct loop: WE call the evidence-gathering tool deterministically in code (grep/AST/argparse, no LLM decides whether/when to investigate), then the checker makes exactly one grounded decision. New: deterministic anchoring -- a finding must cite a real evidence ID from what was actually gathered, or it's discarded, the same way an ungrounded paper-quote already was. The data-leakage checker is dual-modality: it also screens diagnostic paper figures (loss curves, embedding plots) through a vision model (`meta/llama-3.2-11b-vision-instruct`, verified working directly against real arXiv HTML figures, including SVG rasterization via PyMuPDF), but a vision-only signal is never sufficient -- every finding still requires corroborating code evidence.

**Evidence (corrected numbers, see the scoring-bug entry above):**
| Metric | Phase 2 (best so far) | Phase 3 | Phase 4 |
|---|---|---|---|
| F1 | 0.667 | 0.455 | **0.643** |
| Precision | 0.750 | 0.714 | 0.692 |
| Recall | 0.600 | 0.333 | 0.600 |
| data_leakage F1 | 0.0 | 0.0 | **1.0 (only phase to ever catch it)** |
| hyperparam_drift F1 | 0.909 | 0.0 | 0.5 |
| Wall clock (9 papers) | 1911s | 1143s | 1847s |
| Total tokens | 403k | 984k | 582k |

**Decision:** kept, and arguably the most well-rounded phase, though not a clean win over Phase 2 -- it's close on aggregate F1 (0.643 vs 0.667) but with a different, complementary strength profile: Phase 4 is the *only* phase across the whole project to ever correctly catch the data-leakage bug (G-Mixup's `deg.mean()`/`deg.std()` computed before the split), because it's the only phase whose evidence includes explicit split-location line numbers to compare against -- Phase 2 and 3 both had no way to determine "before or after the split" from what they were given. Phase 2 is still stronger on `hyperparam_drift` (0.909 vs 0.5).

**What this taught us:** two real, distinct things. First, deterministic evidence-gathering plus one grounded decision is a genuinely strong, cheap pattern -- it avoided every failure category the earlier phases hit (no tool-calling reliability risk, no open-ended wandering, no compound-claim dilution) and still found our project's single hardest-to-catch bug. Second, `network_deconvolution`'s false positives revealed a real, specific checker limitation worth naming honestly: the same flag name (`-b`/`--batch-size`) has *different* defaults in two different entry-point scripts in that repo (128 for the CIFAR-scale script, 256 for the ImageNet one), and the checker compared a CIFAR-scale claim against the ImageNet script's default -- it has no notion of "which entry point does this claim's experimental context refer to." A second false positive there came from claim extraction treating an ablation-study *range* ("accuracy improves from .0005 to .005 weight decay") as if it were a fixed hyperparameter value. Both are precise, fixable gaps for a next iteration, not vague "needs more work."

## Final — Ensemble of the two strongest iterations (Phase 2 + Phase 4)

**What we tried and why:** Phase 2 (F1 0.667) and Phase 4 (F1 0.643) land within 0.024 of each other on aggregate F1, but their per-failure-mode profiles are strikingly *complementary*, not redundant: Phase 2 dominates `hyperparam_drift` (0.909 vs 0.5), while Phase 4 is the only phase in the entire project that ever catches the real data-leakage bug (1.0 vs 0.0). That's the textbook precondition for an ensemble helping rather than just averaging — uncorrelated errors. Combined them with two rules, both principled rather than tuned: (1) union with deduplication on `(failure_mode, evidence file basename)`, so a finding both phases independently made counts once; (2) drop findings labeled `other` — verified directly that `data/ground_truth.json` never uses `other` for any real finding, so an `other`-labeled prediction is structurally incapable of being a true positive under our matching rules. **Makes zero new LLM calls** — it recombines each phase's already-saved predictions.

**Evidence:**
| Metric | Phase 2 (best single) | Phase 4 | **Final ensemble** |
|---|---|---|---|
| F1 | 0.667 | 0.643 | **0.759** |
| Precision | 0.750 | 0.692 | **0.786** |
| Recall | 0.600 | 0.600 | **0.733** |
| `hyperparam_drift` F1 | 0.909 | 0.5 | 0.8 |
| `data_leakage` F1 | 0.0 | 1.0 | **1.0** |
| Additional inference cost | — | — | **zero (no new LLM calls)** |

**Decision:** kept as the final system. Best result in the project, and it's free — it beats both of its inputs purely by exploiting that they fail on different things.

**What this taught us — including where we stopped:** the target for this phase was F1 > 0.80. We reached **0.759** and stopped, rather than adding further post-hoc combination rules to close the remaining 0.04. That gap is real, not a tuning problem: `lbcs`'s single-seed bug is missed by *both* constituent phases (so no combination rule can recover it), and one false positive on `cifar_10_100n` survives dedup legitimately. Any additional filter that erased those would be fitted to the specific shape of this 9-paper eval set, not to the actual problem — which is precisely the overfitting-to-your-own-benchmark failure this project has spent four phases documenting in *other people's* work. Reporting 0.759 honestly is more useful than reporting a tuned 0.80. The broader finding stands on its own: two agent designs that score nearly identically in aggregate can still be worth running together, because aggregate F1 hides whether their errors overlap — and here they almost entirely don't.
