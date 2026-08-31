# Phase 4 — Iteration 3: Specialized Per-Failure-Mode Checkers

## 1. What we built

Six specialized checkers replace the single shared verifier from Phases 2-3
— one per failure mode, each with its own narrow prompt and only the
evidence-gathering relevant to it (`src/agents/checkers/`).

**The core design change from Phase 3, directly informed by its regression:**
no checker runs an open-ended ReAct loop. Instead, for each relevant claim:

1. **We call the evidence-gathering tool deterministically, in code** — the
   LLM never decides whether/when/how to investigate. `hyperparam_drift`
   always runs `GetArgparseDefaults` once; `single_seed_reporting` always
   greps for "seed" across every file type once; etc.
2. **The checker makes exactly one grounded decision** given the claim + the
   evidence gathered, and stops.

This eliminates every failure category Phase 3 hit by construction: no
tool-calling reliability risk (nothing decides to call a tool), no
open-ended wandering (there's no loop to get lost in), and — new — **no
hallucinated file:line citations**: a finding's `code_evidence` must be a
real evidence ID from what was actually gathered (`[E3]` etc.), or it's
discarded before scoring. This is the same enforcement principle as Phase
2's paper-quote grounding, now applied to code evidence too.

**The data-leakage checker is dual-modality**, per the original design
brief for this phase: it screens diagnostic paper figures (loss curves,
embedding plots — filtered from decorative ones via a cheap caption check,
no LLM call needed for that step) through a vision model
(`meta/llama-3.2-11b-vision-instruct`), verified working end-to-end against
a real figure from a real paper before being built into the pipeline,
including SVG rasterization via PyMuPDF (arXiv HTML embeds many figures as
SVG, not raster images — confirmed by testing, not assumed). **A vision-only
signal is never sufficient**: every leakage finding still requires
corroborating code evidence citing a real preprocessing call. This mirrors
exactly how G-Mixup's actual leak was originally found in this project — by
reading code, not by eyeballing a chart.

Two things were deliberately **not** built into this phase, after review: a
dynamic-execution sandbox (running each repo's training loop to trace
tensors at runtime) was rejected on safety grounds — executing untrusted
code from 8 external repos with no sandbox infrastructure — and on a
practical one found immediately after: this project's environment has none
of the ML libraries (PyTorch, PyTorch Geometric, TensorFlow) these repos
actually need, so even importing one would mean installing a different,
often conflicting dependency stack per paper.

## 2. A scoring bug was found and fixed during this phase

While debugging Phase 4's results, two "false positives" turned out to be
**correctly-grounded, real findings that could never have scored as a match
against any phase** — a bug in `src/eval.py`, not the agent. Full detail in
`changelog.md`'s dedicated entry; summary: the matcher only ever extracted
`.py` filenames from `code_evidence` strings via regex, so ground-truth
findings whose evidence lives in a `.sh` file (`g_mixup`'s real seed loop,
in `run_gmixup.sh`) or a `.md` file (`sample_specific_masks`'s, in
`readme.md`) were **structurally unmatchable in every phase**, silently,
since those entries were added. Fixed and rescored all four phases from
saved predictions (no new LLM calls). **All numbers in this document and
throughout the repo are the corrected ones.**

## 3. Results

| paper_id | n_expected | n_predicted | TP | FP | FN | F1 |
|---|---|---|---|---|---|---|
| proto2proto | 0 | 0 | 0 | 0 | 0 | 1.0 |
| g_mixup | 5 | 4 | 3 | 1 | 2 | 0.667 |
| lottery_ticket_hypothesis | 2 | 0 | 0 | 0 | 2 | 0.0 |
| network_deconvolution | 1 | 4 | 1 | 3 | 0 | 0.4 |
| birt_continual_learning | 2 | 0 | 0 | 0 | 2 | 0.0 |
| cifar_10_100n | 0 | 1 | 0 | 1 | 0 | 0.0 |
| synthetic_multi_defect | 3 | 3 | 3 | 0 | 0 | 1.0 |
| lbcs | 1 | 0 | 0 | 0 | 1 | 0.0 |
| sample_specific_masks | 1 | 1 | 1 | 0 | 0 | 1.0 |
| **MICRO-AVG** | — | — | **9** | **5** | **7** | **0.643** |

## 4. Phase 2 vs Phase 3 vs Phase 4 (identical, corrected ground truth)

| Metric | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|
| F1 | **0.667** | 0.455 | 0.643 |
| Precision | **0.750** | 0.714 | 0.692 |
| Recall | 0.600 | 0.333 | **0.600** |
| Wall clock (9 papers) | 1911s | 1143s | 1847s |
| Total tokens | 403k | 984k | 582k |

By failure mode:

| failure_mode | Phase 2 F1 | Phase 3 F1 | Phase 4 F1 |
|---|---|---|---|
| data_leakage_global_preprocessing | 0.0 | 0.0 | **1.0** |
| hyperparam_drift | **0.909** | 0.0 | 0.5 |
| single_seed_reporting | 0.333 | 0.667 | 0.571 |
| metric_misimplementation | 0.667 | **1.0** | **1.0** |
| test_set_contamination | 0.667 | 0.667 | 0.667 |
| eval_on_train | **1.0** | **1.0** | **1.0** |

Phase 4 is not a clean win over Phase 2 — it's close (0.643 vs 0.667) with a
genuinely different strength profile, not a strictly better or worse one.

## 5. Failure analysis

**Case 1 — the win: data leakage, uniquely.** Phase 4 is the only phase
across this entire project to correctly catch G-Mixup's real leak
(`deg.mean()`/`deg.std()` computed over the full dataset at
`src/gmixup.py:49`, before the split at line 250). The reason is specific
and fixable-in-hindsight: this checker's evidence explicitly includes
labeled SPLIT entries with their own line numbers alongside PREPROCESSING
entries, so the model can directly compare "does E1's line number come
before E10's" — Phase 2 and Phase 3 never gave their verifier any
evidence about *where the split happens*, only about preprocessing calls in
isolation, so neither had the information needed to determine leakage even
when they saw the right preprocessing call. This was found and fixed
mid-build: an earlier version of this checker gathered the correct
preprocessing evidence and still produced zero findings, for exactly this
reason, before split-location evidence was added.

**Case 2 — `network_deconvolution`'s false positives, precisely diagnosed.**
Two distinct, nameable causes, not vague noise:
- **Notation collision.** The paper says "we set B = 64" referring to the
  deconvolution *block size* (a real hyperparameter, `--block`/
  `--num-groups`). The checker matched this against the `--batch-size`
  flag instead — a plausible confusion since both use "B" and "batch" as
  shorthand, but a real, specific gap: the checker has no way to
  distinguish which hyperparameter a claim's "B" refers to.
- **Wrong entry-point script.** The repo has two scripts with a
  `--batch-size`/`-b` flag with *different* defaults (128 in the
  CIFAR-scale `arg_parser.py`, 256 in `main_imagenet.py`). A claim about
  the CIFAR-scale experiment (batch=128, which the code correctly matches)
  was compared against the ImageNet script's default instead, producing a
  false mismatch. The checker's evidence gathering has no concept of "which
  entry point does this claim's experimental context belong to" — it pools
  every argparse default across the whole repo into one flat list.
- A third false positive came from claim extraction treating an
  ablation-study range ("accuracy improves with weight decay from .0005 to
  .005") as if it were a single fixed hyperparameter claim to check against
  a default value, rather than a sweep result.

All three are precise, actionable gaps for a next iteration — not "the
model made mistakes."

## 6. How to reproduce Phase 4

```bash
.venv/bin/python -m src.eval iter3
```

## 7. What this tells us, project-wide

Four phases in, the pattern is now clear enough to state as a finding, not
a hunch: **agent quality here is bottlenecked by what evidence a checker is
given and how atomically claims are specified, far more than by how much
autonomy or tooling it has.** Phase 2 (constrained, one decision) and Phase
4 (constrained, six decisions, richer per-mode evidence) are the two best
results in the project. Phase 3 (unconstrained, one agent, six tools, an
open budget of decisions) is the worst of the four real iterations. Phase
4's unique success on data leakage came specifically from giving it *more
of the right evidence* (split-location line numbers), not more autonomy to
go find it itself.
