# Phase 2 — Iteration 1: Extractor + Verifier Split

## 1. What we built

Per instructions.md §6 Phase 2, but built as strong as we reasonably could
within this one, isolable design change (explicit decision, see
conversation record: build one strong Phase 2, not a merged mega-iteration —
keeping the changelog attributable was judged more valuable than maximizing
single-shot strength).

Two-node LangGraph pipeline (`src/graph.py`):

1. **Extractor** (`src/agents/extractor.py`, `NIM_MODEL_FAST`) — reads only
   the paper text, extracts a list of specific, checkable claims (5 types:
   `hyperparameter`, `seed_protocol`, `metric`, `data_split_protocol`,
   `eval_protocol`), each carrying an **exact verbatim quote** from the
   paper as evidence the claim was really made. Explicitly instructed not to
   infer or assume claims not actually stated.
2. **Verifier** (`src/agents/verifier.py`, `NIM_MODEL_PRIMARY`) — given the
   extracted claims + repo (README + code heads, same context as Phase 1),
   checks each claim against the code. Every finding must reuse one of the
   claims' exact quotes as `paper_evidence`.

**Two targeted robustness mechanisms, both aimed directly at Phase 1's two
documented failure categories:**

- **Anti-hallucination (targets Phase 1's proto2proto false positives):**
  a finding can only be grounded in a claim that was actually extracted from
  the paper. This isn't just a prompt instruction — `run_verifier` does a
  **post-hoc check**, discarding any finding whose `paper_evidence` doesn't
  match a real extracted quote (normalized for whitespace/case) before it's
  ever counted. We verified this fires: on proto2proto, 1 finding was
  produced and dropped by this check, and the extractor correctly found no
  `seed_protocol` claim at all (the paper never mentions seeds) — so there
  was structurally nothing for a phantom seed-related finding to hang on.
- **JSON repair pass (targets Phase 1's 44% parse-failure rate):**
  `src/parsing.py`'s `parse_json_list_with_repair` tries a normal parse
  first; on failure, sends the broken text to the fast model with "fix this
  JSON's syntax, change nothing else" and reparses. We confirmed this
  recovers the exact bug pattern from Phase 1 (missing opening quote) in a
  synthetic test before running the real eval.

We also checked whether NIM's hosted endpoint supports native structured
output or tool-calling, which would have been an even stronger fix. It
doesn't: both `guided_json` mode and `bind_tools`/forced tool-choice
returned HTTP 400/500 errors on this backend for our primary model, tested
directly. The repair-pass approach above is the portable alternative.

## 2. An important side effect: this phase found real bugs in our own Phase 0 ground truth

While diagnosing errors during this phase's eval run, the agent produced
findings on `birt_continual_learning` and `g_mixup` that weren't in our
existing ground truth. We didn't just count these as false positives — we
went back and verified each one against the actual paper text and code
before deciding:

- **`birt_continual_learning`** (originally annotated *clean*, category B):
  the paper explicitly states weight decay of `1e-6` and results "averaged
  across three random seeds." The code's actual defaults are weight decay
  `0.05` (a ~50,000x difference) and a single hardcoded `--seed 42` example
  with no multi-seed mechanism anywhere in the repo. Our original Phase 0
  annotation checked the code's *internal* self-consistency (README example
  vs. code default) but never cross-referenced those defaults against the
  paper's actual claims — a real gap. **Reclassified from clean (B) to
  category A, 2 new real findings added.**
- **`g_mixup`**: 2 more real hyperparameter mismatches beyond the one we'd
  already found (augmentation ratio: paper claims 20%, code and the
  *official run script* both use 15%; mixing-strength range λ: paper claims
  `[0.1, 0.2]`, code default is `[0.005, 0.01]` and the official script
  doesn't override it). Both verified directly against the code and
  `run_gmixup.sh` before being accepted.

Because Phase 1's saved predictions were scored against stale ground truth,
we rescored (not re-ran — `src.eval.rescore_phase`, same saved LLM outputs,
new ground truth) both Phase 1 and Phase 2 against the corrected data so the
comparison below is fair. **Phase 1's numbers in `phase_1_baseline.md` were
updated accordingly** (its own score got worse, since it never had a chance
to catch bugs we hadn't found yet either).

## 3. Results table (rescored against corrected ground truth)

| paper_id | n_expected | n_predicted | TP | FP | FN | F1 |
|---|---|---|---|---|---|---|
| proto2proto | 0 | 0 | 0 | 0 | 0 | 1.0 |
| g_mixup | 4 | 5 | 3 | 2 | 1 | 0.667 |
| lottery_ticket_hypothesis | 2 | 1 | 1 | 0 | 1 | 0.667 |
| network_deconvolution | 0 | 0 | 0 | 0 | 0 | 1.0 |
| birt_continual_learning | 2 | 2 | 2 | 0 | 0 | 1.0 |
| cifar_10_100n | 0 | 1 | 0 | 1 | 0 | 0.0 |
| synthetic_multi_defect | 3 | 3 | 3 | 0 | 0 | 1.0 |
| lbcs | 1 | 0 (timeout) | 0 | 0 | 1 | 0.0 |
| sample_specific_masks | 1 | 0 | 0 | 0 | 1 | 0.0 |
| **MICRO-AVG** | — | — | **9** | **3** | **4** | **0.720** |

## 4. Aggregate metrics: Phase 1 vs Phase 2, same corrected ground truth

| Metric | Phase 1 (baseline) | Phase 2 (iter1) | Delta |
|---|---|---|---|
| Precision | 0.400 | 0.750 | **+0.350** |
| Recall | 0.308 | 0.692 | **+0.385** |
| F1 | 0.348 | 0.720 | **+0.372** |
| FPR on clean papers | 1.333 | 0.333 | **−1.0 (4x fewer)** |
| Papers with usable output | 5/9 | 6/9 (3 hit infra errors, see §5) | +1 |

By failure mode:

| failure_mode | Phase 1 F1 | Phase 2 F1 |
|---|---|---|
| data_leakage_global_preprocessing | 0.0 | 0.0 |
| hyperparam_drift | 0.0 | **1.0** |
| single_seed_reporting | 0.0 | 0.4 |
| metric_misimplementation | 0.667 | 0.667 |
| test_set_contamination | 1.0 | 0.667 |
| eval_on_train | 1.0 | 1.0 |

`hyperparam_drift` — the mode with the most real ground truth (5 instances
across 3 papers) — went from **0.0 to a clean 1.0**: the claim-extraction
step forces the verifier to check every specific numeric claim
systematically instead of eyeballing the whole repo at once, and it worked.
`data_leakage_global_preprocessing` (g_mixup's leakage bug) is still missed
— see failure analysis.

## 5. Failure analysis

**Case 1 — the `.sh`-file blind spot (a real, honest new limitation, not a
reasoning failure).** On the g_mixup retry, the verifier confidently claimed
`single_seed_reporting` ("the code sets a single seed and runs training
once; there is no loop"). This is **factually wrong** — `run_gmixup.sh`
genuinely loops over 8 seeds (verified in Phase 0). The reason: our context
builder (`src/context.py`) only includes `.py` file contents, never `.sh`
files. The README mentions `run_gmixup.sh` by name, but its actual loop
logic is invisible to the model. This is a clean, direct argument for
Phase 3's tool-based approach — a tool that can read/grep the *whole* repo,
not a fixed set of file-type heads, would have caught this.

**Case 2 — "unimplemented" vs. "misimplemented" (a taxonomy boundary, not a
bug).** The same g_mixup run flagged `metric_misimplementation`: the paper
mentions AUROC for a molecular-property-prediction experiment, but the repo
never computes AUROC anywhere. We verified this is real (confirmed the
paper's claim, confirmed zero AUROC-related code exists) but rejected it
from ground truth: the repo simply never implements that experiment at all
(no script, no README mention) — that's "missing," not "wrong," and doesn't
match any of our 6 modes as defined. Identical pattern to Phase 1's
`cifar_10_100n` false positive (paper compares to methods the repo doesn't
locally reimplement). Worth naming as a recurring, systematic thing this
class of agent needs to learn to distinguish, not a one-off.

**Case 3 — infrastructure still causes real losses, now for different
reasons than Phase 1.** 3 of 9 papers still didn't produce a usable result:
`g_mixup`'s first attempt hit a transient NVIDIA `503 Service temporarily
overloaded` (not our bug — retried manually, succeeded); `lbcs` hit a
`ReadTimeout` even at our already-raised 240s timeout (a large, ~30-file
repo — the verifier call itself needs either a higher timeout or a trimmed
context for repos this size); `lottery_ticket_hypothesis`'s extractor
response needed the repair pass (which worked) but the verifier still only
surfaced 1 of 2 known real findings in this run, plausibly at-random
variance in a MoE reasoning model rather than a design flaw — a repeated
verifier call on this exact paper in Phase 1 debugging did catch both.

## 6. What this tells us

- **Claim-grounded verification is a real, working anti-hallucination
  mechanism**, not just a nicer prompt — proto2proto now scores 0 findings
  for the *correct* reason (no groundable claims exist), not by accident.
  The post-hoc quote-matching check is doing real work (it fired and
  dropped a finding on the exact test case built to probe it).
- **Systematic claim-by-claim checking beats holistic "does anything look
  wrong" scanning** for hyperparameter bugs specifically (F1 0.0 → 1.0) —
  the single clearest, cleanest win of this iteration.
- **The context-building layer, not just the agent design, is now the
  binding constraint.** Two of this phase's remaining gaps (the `.sh`-blind
  spot, and — unfixed since Phase 1 — the "absence across a 30-file repo"
  problem for `lbcs`/`sample_specific_masks`'s single-seed bugs) aren't
  things a better prompt fixes. This is the strongest, most concrete
  argument yet for Phase 3's actual tools (grep across the whole repo,
  not just `.py` heads; AST search; config parsing).
- **This phase's own evaluation process improved our ground truth twice**
  (birt, g_mixup) — worth stating plainly in the eventual Hot Take: a tool
  built to audit reproducibility ended up auditing its own evaluator.

## 7. How to reproduce Phase 2

```bash
.venv/bin/python -m src.eval iter1
```

Writes `outputs/iter1/<paper_id>.{json,txt}` and
`outputs/iter1/_summary.{json,txt}`. Original run: 1910.8s wall clock,
403,417 tokens across 17 calls (2 per paper, minus errors, plus repair
passes) for 9 papers; g_mixup's retry after the transient 503 is not
included in that wall-clock figure. To rescore either phase against updated
ground truth without re-calling the LLM: `src.eval.rescore_phase(phase_name)`.

## 8. Next phase gate

Phase 3 gives the verifier real tools: `src/tools/grep_tool.py` (search the
*entire* repo, not just `.py` heads — directly fixes Case 1 above),
`src/tools/ast_tool.py` (Python AST search — find function calls/imports/
assignments precisely rather than reading raw text), `src/tools/config_tool.py`
(parse YAML/JSON/argparse defaults directly rather than eyeballing them from
a text dump). Expected gain per instructions.md §6: "big jump on
hyperparam_drift and eval_on_train specifically" — hyperparam_drift is
already at 1.0 here, so the more meaningful target is closing the
`data_leakage_global_preprocessing` and `single_seed_reporting` gaps, both
of which trace directly to context/tooling limits identified in this
phase's failure analysis, not reasoning quality.
