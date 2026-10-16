# RepoRaven

**Published ML papers regularly claim results their own code can't produce — wrong hyperparameters, single-run numbers reported as multi-seed averages, test data leaking into training. RepoRaven finds these automatically, and cites the exact line of code and the exact sentence it contradicts.**

Built for the micro1 Agentic Workflows Hackathon (2026).

---

### What you actually get — one real finding, unedited

```
### 3. [HIGH] data_leakage_global_preprocessing (confidence: 90%)

> Paper says: We split the dataset into train/val/test data by 7 : 1 : 2.

Code: src/gmixup.py:49

The preprocessing call deg.mean() at line 49 computes statistics on deg
(derived from the full dataset) before the train/val/test split at line
250 (train_dataset = dataset[:train_nums]), causing global preprocessing
leakage.
```

That's an unedited excerpt from `outputs/final/FINAL_REPORT.md` — a real bug in G-Mixup (ICML 2022)'s official code, found automatically, cited to an exact line, checkable in under a minute. The full report covers all 9 eval-set papers; run `python scripts/generate_report.py final` to regenerate it yourself.

---

## Results at a glance

| | Simple baseline | Final system |
|---|---:|---:|
| F1 vs. hand-verified ground truth | 0.320 | **0.759** |
| Precision / recall | 0.400 / 0.267 | 0.786 / 0.733 |
| Data-leakage detection (F1) | 0.0 | **1.0** |
| Cost per paper | — | ~109k tokens, ~7 min, on the NVIDIA NIM free tier |

- **Evaluation set:** 9 papers (ICML/ICLR repos, including 2 from PaperBench's official ICML 2024 list), 2 negative controls, and a synthetic fixture with planted bugs. There are 15 hand-verified findings in total.
- **Architecture:** atomic claim extraction → six specialized, deterministic evidence-gathering checkers → grounded LLM decisions, ensembled with an extractor-verifier pipeline. Built with LangGraph, NVIDIA NIM (Nemotron, gpt-oss-120b) and tree-sitter.
- **The agent found bugs our own annotation missed**, three times. Each was verified by hand, then every earlier phase was rescored, which made earlier phases look *worse*.

---

## The user and their bottleneck

**Who has this problem:** journal editors, program-committee chairs, and ML team leads who have to decide whether a paper's reported results can be trusted — before publishing it, citing it, or building a product on top of it.

**The bottleneck:** today this is done by hand, if at all. OpenAI's PaperBench (Starace et al., 2025) measured this directly: on a 3-paper subset, 8 ML PhD participants (best of 3 attempts) hit only **41.4% replication after 48 hours of effort each** — a number we checked directly against the paper's own text rather than trusting a secondhand citation — and that's the *harder* task, full from-scratch replication, not the narrower question we're asking. The practical question — *"does this repo exhibit any of the specific bugs that make ML results unreliable?"* — has no tooling at all. A reviewer with 15 minutes has no way to check it, so it mostly doesn't get checked.

**Why solving it is valuable:** the bugs are real and they're sitting in plain sight. While building this project's eval set we hand-verified **12 findings across 6 well-known public papers**, including official ICML 2024 repos:

- **G-Mixup** (ICML 2022): node-degree normalization statistics computed over train+val+test *concatenated*, before the split — textbook data leakage. Plus 4 hyperparameter mismatches, including the shipped run script looping **8 seeds while the paper claims ten runs**.
- **The Lottery Ticket Hypothesis** (ICLR 2019): the paper says Adam @ lr 0.0012 for Lenet/MNIST; the code uses plain SGD @ lr 0.1 — wrong optimizer *and* wrong learning rate.
- **BiRT** (ICML 2023): paper states weight decay 1e-6; code default is 0.05 — a ~50,000× difference. Also claims 3-seed averaging with no mechanism to produce it.
- **Network Deconvolution** (ICLR 2020): one universal learning-rate default that's correct for 2 of 3 ImageNet architectures and silently wrong for the third.

**Two dedicated research passes searching ReScience C reports, GitHub issues, and the Kapoor & Narayanan leakage taxonomy found pre-documented public examples of only 2 of our 6 target failure modes.** The rest we found ourselves, by reading code against paper claims. Nobody had filed an issue. That gap *is* the case for this tool.

---

## What it checks

Six failure modes, from the Kapoor & Narayanan (2023) leakage taxonomy plus community-known ML paper pathologies:

| Failure mode | What it means |
|---|---|
| `data_leakage_global_preprocessing` | Feature statistics fit on train+test together, before the split |
| `hyperparam_drift` | Paper claims one value; code default is another |
| `single_seed_reporting` | Paper reports mean±std over N runs; code has no way to produce N runs |
| `metric_misimplementation` | Paper says F1-macro; code computes something else |
| `test_set_contamination` | Test data influences training or model selection |
| `eval_on_train` | The "test" evaluation is actually run on training data |

**Output:** a structured report where every finding cites a `file:line` in the repo *and* the verbatim paper sentence it contradicts, sorted by how much it's worth your time. See `outputs/iter1/FINAL_REPORT.md` for the actual output on all 9 eval-set papers, generated by `scripts/generate_report.py`.

### Findings are triage, not adjudication

RepoRaven runs at **79% precision** — about 1 in 5 findings is a false positive — so every finding cites an exact `file:line` and the verbatim paper sentence specifically so it's checkable in under a minute. It points a reviewer at the 3 things worth their time; it reads public repositories read-only and never modifies, contacts, or publishes anything on its own.

---

## Improvement Changelog

**Primary metric: F1** against `data/ground_truth.json` — it's the one number the user actually needs, since a journal editor cares equally about missed real bugs (recall) and wasted-time false alarms (precision), and F1 is the standard way to force both to matter at once. Precision, recall, FPR-on-clean, and the per-failure-mode breakdown are reported alongside it as supporting diagnostics — they explain *why* F1 moved, not what "winning" means. Full detail with per-paper tables in [`changelog.md`](changelog.md) and `phases/phase_*.md`. All figures below are scored against **identical current ground truth** (`src.eval.rescore_phase` replays saved predictions when ground truth changes, so comparisons stay fair).

**Why we stopped after Phase 4 (iteration 3) rather than building a Phase 5:** the original plan included one more iteration — consensus checking across checkers, plus a human-approval gate. After weighing it directly against finishing well, we chose to close out the submission instead: record the video, verify Phase 4's results, and polish what's built, rather than add a fifth iteration on the last day. That human-approval gate is the one piece of unbuilt work we consider a real, still-open gap — see [Findings are triage, not adjudication](#findings-are-triage-not-adjudication).

| Stage | What we tried and why | Evidence | Decision / learning |
|---|---|---|---|
| **Baseline** (Phase 1) | One prompt, one call: paper + README + first 200 lines of every `.py` file dumped in, ask for JSON findings. | **F1 0.320** · P 0.400 / R 0.267 · FPR-on-clean 2.0 · 4/9 papers produced no usable output | Established the floor. Failure wasn't reasoning, it was *reliability* — a systematic JSON bug silently discarded 4 papers' results. |
| **Iteration 1** (Phase 2) | Split into a claim **extractor** (atomic claims + verbatim quotes) and a **verifier**. Added a post-hoc grounding check that discards findings not tied to a real extracted quote, plus an LLM JSON-repair pass. | **F1 0.667** (+0.347) · P 0.750 / R 0.600 · FPR-on-clean 0.5 (4× fewer) · `hyperparam_drift` F1 0.0 → **0.909** | **Best configuration on raw F1.** Claim-grounding structurally prevents hallucination; systematic claim-by-claim checking is by far the strongest approach to hyperparameter detection specifically. |
| **Iteration 2** (Phase 3) | Gave the verifier real tools — whole-repo grep (all file types), tree-sitter AST search, structured argparse/config parsing — as a multi-turn ReAct agent with native tool-calling. | **F1 0.455** (−0.212) · P 0.714 / R 0.333 · **2.4× the tokens** | **Revised, not kept as-is.** A documented negative result — see below. The *tools* work and ship forward into Phase 4; the single-general-agent architecture around them doesn't. |
| **Iteration 3** (Phase 4) | Replaced the single verifier with **6 specialized checkers**, one per failure mode. Each deterministically gathers its own evidence in code (no LLM-driven tool loop) and makes exactly one grounded decision. New: deterministic anchoring (findings must cite a real evidence ID, not a generated file:line). The leakage checker is dual-modality — screens paper figures via a vision model, but vision alone is never sufficient without corroborating code. | **F1 0.643** (+0.188 vs. Phase 3) · P 0.692 / R 0.600 · **`data_leakage` F1 0.0 → 1.0 (the only phase to ever catch it)** | **Kept.** Near-tied with Phase 2 on aggregate F1, but with a genuinely complementary strength — see below. |
| **Final** (ensemble) | Combined the two strongest iterations (Phase 2 + Phase 4), which have **complementary** strengths. Two principled rules: union with dedup on (failure_mode, evidence file), and drop `other`-labeled findings — verified that ground truth never uses `other`, so those can never be a true positive by construction. **Makes zero new LLM calls** — it combines already-saved predictions. | **F1 0.759** — best result in the project · P 0.786 / R 0.733 · `hyperparam_drift` 0.8 · `data_leakage` 1.0 | **Kept as the final system.** Beats both inputs (0.667, 0.643) because their errors are genuinely uncorrelated, at zero additional inference cost. |

*Phase 3 and 4's numbers above were corrected after a scoring bug was found and fixed mid-project: `src/eval.py`'s matcher only recognized `.py` file citations, so ground-truth findings living in a `.sh` or `.md` file were structurally unmatchable in every phase, silently, until a correctly-grounded Phase 4 prediction exposed it. Fixed and all four phases rescored from saved predictions — full detail in `changelog.md`. Baseline's own F1 is also not stable run-to-run — a second run scored 0.182 on identical papers/code/prompt (see `outputs/trajectories/README.md`), because Phase 1's single-shot design has no repair mechanism for the transient API errors and formatting bugs documented as its central finding. **What is stable across every run observed: Iterations 1 and 3 are clearly the two strongest configurations, each with different strengths.***

### Simple baseline vs. agent solution, at a glance

| Metric | Simple baseline (Phase 1) | Agent solution (Final ensemble) | Change |
|---|---|---|---|
| Primary outcome (F1) | 0.320 | 0.759 | **+0.439** |
| Human time per task | Reference point, not our own measurement: OpenAI's PaperBench had 8 ML PhDs spend **~48 hours each** attempting full manual replication of one paper (best result 41% completion). We don't check full replication — only 6 specific failure modes — so this isn't apples-to-apples, but it's the honest scale of the manual alternative. | ~7 min of agent compute per paper (iter1 + iter3 combined, the two phases the final ensemble depends on: (1,911 s + 1,847 s) ÷ 9 papers), plus a few minutes of human review of the flagged findings — every finding ships with a paper quote and a code citation specifically so that review is fast. | Manual full replication is hours; targeted review of this tool's output is minutes. Not a fair head-to-head (different scope), stated as an order-of-magnitude reference, not a controlled comparison. |
| Cost per task | Not applicable — fully manual, no API cost. | ~109,000 tokens/paper on average ((403,417 + 581,545 tokens) ÷ 9 papers, iter1+iter3 combined). Fits NVIDIA NIM's free tier for all 9 papers across all phases. We deliberately do **not** publish a dollar figure — NIM's pricing page did not render for automated fetching during this project, so we could not verify a per-token rate and won't state a number we haven't confirmed (see `REPRODUCTION.md` §5). | Token cost is real and reproducible; dollar cost is knowingly left unstated rather than estimated. |

### The most instructive result: Phase 3 regressed, Phase 4 recovered differently than expected

Phase 3 gave the agent strictly *more* capability and it got substantially worse. `birt_continual_learning` isolates the cause exactly — Phase 2 caught 2/2 findings; Phase 3 caught 0/2 **despite making 9 successful tool calls**:

- Phase 2's extractor produced **atomic** claims: `"weight decay of 1e-6"` and `"averaged across three random seeds"` as separate items. Both bugs caught.
- Phase 3's extractor (byte-identical prompt) bundled three hyperparameters into **one compound quote** and extracted **no seed claim at all**.

Because the grounding rule requires every finding to map to an extracted claim, the seed bug became *structurally unfindable* — no amount of tooling could recover it. **Better tools cannot compensate for worse inputs.**

Phase 4 didn't fix this by giving the agent more autonomy back — it fixed it by giving *narrower* agents *more of the right evidence*. The clearest result in the whole project: Phase 4 is the **only** phase across all four iterations to correctly catch G-Mixup's real data-leakage bug (`deg.mean()`/`deg.std()` computed before the train/test split). Not because it reasoned better — because it's the only checker whose evidence explicitly includes the split's own line number alongside the preprocessing call's, so the model could directly compare "does E1 come before E10." Phase 2 and Phase 3 both had no way to answer that question from what they were given, no matter how they were built. Phase 4 does lose ground to Phase 2 on `hyperparam_drift` (0.5 vs. 0.909) — `network_deconvolution`'s repo defines the same `--batch-size` flag with *different* defaults in two different entry-point scripts, and the checker had no way to know which one a given claim's experimental context referred to.

### The final system: an ensemble, because the two best phases fail differently

Phase 2 and Phase 4 land within 0.024 F1 of each other but are **not** redundant — Phase 2 owns hyperparameter detection (0.909 vs 0.5), Phase 4 uniquely owns data leakage (1.0 vs 0.0). Combining them (union + dedup, dropping the structurally-unmatchable `other` category) gives **F1 0.759**, better than either alone, **at zero extra inference cost** since it reuses saved predictions.

**Honest note on the target:** we set out to reach 0.80 with this ensemble and landed at 0.759. We stopped there rather than adding further post-hoc combination rules to close the last 0.04 — the two remaining errors are genuine (`lbcs`'s single-seed bug is missed by *both* phases, and one false positive on `cifar_10_100n` survives dedup), and inventing filters to erase them would be tuning combination rules to this specific 9-paper set rather than building something that generalizes. The number reported is the one the method actually produces.

---

## An honest note on the evaluation

Ground truth was corrected **three times** during development, because the agent surfaced real bugs our own hand-annotation had missed (`birt_continual_learning`, `g_mixup`, `network_deconvolution`). Each was verified by hand against the actual code before being accepted, and **all earlier phases were rescored against the corrected data** — which made earlier phases look *worse*, not better.

A tool built to audit reproducibility ended up auditing its own evaluator. That's the strongest evidence we have that it does something real.

---

## Main failure mode and Hot Take

**Main failure mode:** the pipeline's quality is capped by its weakest upstream component — a cheap model doing unglamorous claim extraction. Every downstream improvement (better verifier, real tools, more agency) is worthless if the claim it needed to check was never extracted, or was extracted as a mushy compound sentence instead of an atomic fact.

**Hot take:** *the agentic community over-invests in giving agents more capability and under-invests in specifying what they're checking.* Phase 3 handed one agent whole-repo grep, AST search, config parsing, and 10 free investigative turns — and it did worse than a simpler pipeline whose only advantage was that its inputs happened to be atomic. Phase 4 confirmed the fix isn't "more autonomy," it's "more of the right evidence, given deterministically": six narrow checkers, each making exactly one decision with no freedom to wander, and one of them — the only one across four iterations — caught our project's hardest bug (a real data-leakage bug) *specifically because* it was handed the split's line number alongside the preprocessing call's, not because it was smarter or more agentic. Agency amplifies input quality; it does not substitute for it, and it isn't even the thing that improves outcomes on its own — evidence design is.

The corollary for evaluation: run-to-run nondeterminism in a "minor" component produced a 39-point F1 swing with an unchanged prompt. If your agent's score moves and you only instrumented the parts you think are interesting, you cannot tell improvement from noise.

---

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # add your NVIDIA_API_KEY
.venv/bin/python scripts/run_all.py
```

Full setup, expected runtime, and cost: **[`REPRODUCTION.md`](REPRODUCTION.md)**.

## What's here

```
src/agents/       extractor, verifier (Phase 2), tool-using verifier (Phase 3), 6 specialized checkers (Phase 4)
src/tools/        grep (whole-repo), tree-sitter AST search, argparse/YAML parsing
src/eval.py       scoring harness + rescore_phase for fair cross-phase comparison
data/ground_truth.json    15 hand-verified findings across 9 papers
phases/           per-phase writeups with failure analysis
outputs/trajectories/     agent trajectories (instructions → tool calls → result)
```

## What existed before vs. what we built

**Used as-is:** LangGraph, LangChain, NVIDIA NIM models (`nemotron-3-super-120b-a12b`, `nemotron-3-nano-30b-a3b`, `gpt-oss-120b`), tree-sitter, and the 9 external paper repos as unmodified audit subjects.

**Built for this hackathon:** the entire audit pipeline, all four agent architectures, the tool layer (grep/AST/config/vision), the eval harness and scoring logic, the hand-verified ground truth, and the synthetic fixture.

## Related work

PaperBench (Starace et al., 2025), CORE-Bench (Siegel et al., 2024), MLE-bench (Chan et al., 2024), SciReplicate-Bench (Xiang et al., 2025), REPRO-Bench (Hu et al., 2025), ML Reproducibility Challenge / ReScience C, and Kapoor & Narayanan (2023) for the leakage taxonomy.

Two of our eval papers (`lbcs`, `sample_specific_masks`) are sourced from PaperBench's official ICML-2024 list. But note the positioning:

> **These benchmarks ask *can the agent do the research?* RepoRaven asks *can the agent tell whether the research was done right?*** The second question is cheaper, faster, and closer to what a reviewer or team lead actually pays for.
