# Phase 3 — Iteration 2: Real Tools (grep / AST / config)

**Correction, found during Phase 4:** a scoring bug in `src/eval.py` (only
`.py` files were recognized in evidence citations) made some of this
phase's *correct* predictions unmatchable against ground truth. Fixed and
rescored — see `changelog.md`'s dedicated entry. **Corrected F1: 0.455**,
not the 0.273 originally reported below. The regression finding and its
root cause (§3) are unaffected and still hold; only the magnitude changes.
All numbers below this note are the ORIGINAL, now-superseded figures,
preserved for the record of what was found at the time — see
`phases/phase_4_iter3_specialized_checkers.md` for the corrected comparison.

## Verdict up front (at the time): this iteration REGRESSED. F1 0.667 → 0.273 (now 0.455, see correction above).

It is kept in the changelog as a documented negative result, not quietly
dropped. The cause is identified, evidence-backed, and directly informs
Phase 4. Per the micro1 brief: "Include experiments you later removed and
explain what they taught you about the problem."

## 1. What we built

Replaced Phase 2's fixed context dump with a tool-using ReAct verifier that
can actively investigate the repo:

- **`src/tools/grep_tool.py`** — regex search across the ENTIRE repo, every
  file type. Built specifically to fix Phase 2's documented `.sh` blind spot.
  Also added `ReadFile` (read a whole file / line range) after observing the
  model hallucinate a nonexistent `OpenFile` tool when it needed one.
- **`src/tools/ast_tool.py`** — tree-sitter Python AST search: find exact
  function calls with their arguments, and assignments to a given variable.
- **`src/tools/config_tool.py`** — structured extraction of every
  `argparse.add_argument` default, plus YAML/JSON config parsing.
- **`src/agents/verifier_tools.py`** — multi-turn ReAct loop (max 10
  iterations), native tool-calling via `bind_tools`.

**Model change:** `NIM_MODEL_PRIMARY` (nemotron-3-super-120b-a12b) returns
HTTP 400/500 on both guided-JSON and tool-calling on this endpoint (tested
directly). We tested every accessible NIM model and found
`openai/gpt-oss-120b` supports real native tool-calling, verified with a full
multi-turn tool-call → result → answer loop. Phases 1–2 keep the original
model so their published numbers stay reproducible.

## 2. Results (all phases scored against identical current ground truth)

| paper_id | n_expected | Phase 2 F1 | Phase 3 F1 |
|---|---|---|---|
| proto2proto | 0 | 1.0 | 1.0 |
| g_mixup | 5 | 0.600 | 0.0 |
| lottery_ticket_hypothesis | 2 | 0.667 | 0.0 |
| network_deconvolution | 1 | 0.0 | 0.0 |
| birt_continual_learning | 2 | **1.0** | **0.0** |
| cifar_10_100n | 0 | 0.0 | 0.0 |
| synthetic_multi_defect | 3 | 1.0 | 1.0 |
| lbcs | 1 | 0.0 | 0.0 |
| sample_specific_masks | 1 | 0.0 | 0.0 |
| **MICRO-AVG** | — | **0.667** | **0.273** |

| Metric | Phase 2 | Phase 3 | Delta |
|---|---|---|---|
| Precision | 0.750 | 0.429 | −0.321 |
| Recall | 0.600 | 0.200 | −0.400 |
| F1 | 0.667 | 0.273 | **−0.394** |
| Wall clock (9 papers) | 1910.8s | 1142.5s | −768s (faster) |
| Total tokens | 403,417 | 983,688 | +580,271 (2.4x more) |

Phase 3 is worse on every quality metric while costing 2.4x the tokens.

## 3. Root cause: three distinct failure modes, not one

**Correction to an earlier draft of this document.** The first version of
this analysis attributed the regression primarily to compound claim
extraction. A per-paper audit of the saved runs shows that was
*incomplete* — Phase 3 failed in three mechanically different ways, plus a
fourth "completed cleanly but still wrong" bucket:

| Failure mode | Papers | What happened |
|---|---|---|
| Extractor total failure | `network_deconvolution` | 0 claims extracted → nothing for the verifier to check |
| Extractor schema violation | `lottery_ticket_hypothesis` | 15 claims returned, **every one missing `claim_type`** |
| Verifier never committed | `lbcs` | Correct claim, correct tool call, exhausted 10 turns without answering |
| Completed but still wrong | 4 of the remaining 6 | Full clean run, wrong output |

Only 2 of 9 papers both completed cleanly *and* scored well
(`synthetic_multi_defect` 3/3; `proto2proto` correctly clean).

**`lbcs` is the most damning case, because nothing was missing.** The
extractor supplied the right claim (`"All experiments are repeated 20 times
on NVIDIA GTX3090 GPUs with PyTorch."`). The verifier then made *exactly the
correct* investigative move — `GrepRepo(pattern="seed", file_glob="*")`,
the whole-repo search this phase was built to enable. And then it exhausted
its iteration budget without ever producing a conclusion. Right claim, right
tool, right query, no answer.

**The structural reason Phase 2 wins:** Phase 2's verifier had exactly one
decision to make — *given this claim and this code, is there a
contradiction?* Phase 3's verifier had to additionally decide what to
investigate, which of six tools to use, whether the evidence so far was
sufficient, and when to stop. Four extra decisions per claim, each an
opportunity to wander. On `lottery_ticket_hypothesis` (15 claims) that
compounds into paralysis. **More capability created more ways to fail; the
constraint was the feature.**

### The extractor degradation (still real, still worth reading)

`birt_continual_learning` remains the cleanest illustration of the *input
quality* half of the problem (Phase 2: 2/2 findings; Phase 3: 0/2 despite 9
successful tool calls):

**Phase 2 extracted these as separate, atomic claims:**
- `"weight decay of 1e-6"` → verifier checked it → found code default `0.05` → **caught**
- `"averaged across three random seeds"` → verifier checked it → **caught**

**Phase 3 extracted this compound claim instead:**
> `"We train models with a learning rate of 5e-4, a batch size of 128, and a weight decay of 1e-6."`

...and extracted **no `seed_protocol` claim at all** (verified:
`claim_types = [hyperparameter, metric, data_split_protocol, eval_protocol]`).

Two consequences, both fatal:
1. **The seed bug became unfindable.** The grounding rule (Phase 2's
   anti-hallucination mechanism, deliberately kept) means a finding must map
   to an extracted claim. No seed claim → no possible seed finding, no matter
   how good the tools are. The verifier never had a chance.
2. **The compound claim diluted the check.** Bundling 3 hyperparameters into
   one quote means the verifier evaluates "is this whole sentence consistent?"
   rather than "is weight_decay exactly 1e-6?". Two of three values (lr,
   batch_size) DO match the code, so the sentence reads as broadly correct and
   the one wrong value (1e-6 vs 0.05) gets lost.

The extractor prompt is byte-identical between Phase 2 and Phase 3. The only
difference is run-to-run nondeterminism in the fast model — which is itself
the finding: **the pipeline's weakest link is a cheap model doing unglamorous
extraction work, and no amount of downstream tooling can recover information
that was never extracted.**

## 4. Secondary failures (real, but not the main story)

- **`lottery_ticket_hypothesis` / `lbcs`: "Tool loop ended with no final text
  response"** — the model exhausted 10 tool iterations investigating without
  ever committing to an answer. `lottery_ticket_hypothesis` extracted 15
  claims (vs. Phase 2's handful) and got lost trying to check them all. More
  claims + a fixed iteration budget = analysis paralysis.
- **`network_deconvolution`: extraction returned prose, not JSON**, twice
  (retry included), so 0 claims → 0 findings. The repair pass couldn't fix
  output that was never JSON-shaped.
- **Four bugs found and fixed during this phase** (all real, all in the
  shipped code now): a stale-process run testing pre-fix code; dict-wrapped
  JSON (`{"findings": [...]}`) being rejected; the model emitting fake
  tool-call text (`"Tool call: OpenFile(...)"`) instead of using the real
  mechanism, which the loop then accepted as a final answer; and the missing
  `ReadFile` tool that caused the hallucinated-tool behavior in the first
  place.

## 5. What genuinely improved (worth keeping from this phase)

Not everything regressed — two real wins, both preserved for Phase 4:

- **The `.sh` blind spot is fixed and demonstrably so.** During development,
  the tool-using verifier read `run_gmixup.sh` via whole-repo grep and caught
  that the shipped script loops over **8 seeds while the paper claims ten
  runs** — a subtler, more precise bug than Phase 2's crude (and wrong)
  "no seed loop exists at all." This is now finding #5 in g_mixup's verified
  ground truth. Phase 2 was structurally incapable of finding it.
- **The tools themselves are correct and reusable.** `ReadFile` was observed
  landing directly on `mnist_fc/constants.py` (the exact file with the Lottery
  Ticket bug); `config_tool` extracts exact argparse defaults;
  `ast_tool` pinpoints `f1_score(y, preds)` with its real arguments. The
  evidence-gathering layer is sound — it's the claim layer feeding it that
  failed.

## 6. What this teaches us (feeds Phase 4 and the Hot Take)

- **Better tools cannot compensate for worse inputs.** We upgraded the
  verifier's ability to investigate and got a 39-point F1 drop, because the
  upstream extractor silently degraded. The bottleneck was never evidence
  *gathering* — it was evidence *specification*.
- **Atomic claims beat compound claims.** One claim per checkable fact is
  worth more than a richer, more "complete" quote. This is directly
  actionable: Phase 4's specialized per-failure-mode checkers each own one
  narrow question, which structurally forces atomicity and removes the
  single-shared-extractor bottleneck this phase exposed.
- **Open-ended agency has a cost.** Giving the model 10 free investigative
  turns produced analysis paralysis on 2 of 9 papers (0 findings after
  exhausting the budget). Constrained, purpose-built checkers should be more
  reliable than a general agent with a big toolbox and a vague mandate.

## 7. How to reproduce Phase 3

```bash
.venv/bin/python -m src.eval iter2          # ~1140s, ~984k tokens
.venv/bin/python -c "from src.eval import rescore_phase; rescore_phase('iter1')"   # fair comparison, no LLM calls
```

## 8. Decision

**Revised, not kept as-is.** The tools ship forward into Phase 4 (they work).
The single shared extractor + one general tool-using verifier does not —
Phase 4 replaces it with specialized per-failure-mode checkers, each owning
one narrow, atomic question. Phase 3's regression is the direct evidence
motivating that change.
