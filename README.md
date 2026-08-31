# RepoAudit

**An agentic system that audits ML paper + code-repo pairs for known reproducibility failure modes — and traces every finding to an exact file:line and the paper sentence it contradicts.**

Built for the micro1 Agentic Workflows Hackathon (2026).

---

## The user and their bottleneck

**Who has this problem:** journal editors, program-committee chairs, and ML team leads who have to decide whether a paper's reported results can be trusted — before publishing it, citing it, or building a product on top of it.

**The bottleneck:** today this is done by hand, if at all. OpenAI's PaperBench measured a 48-hour human baseline for full paper replication at 41.4% completion — and that's the *harder* task. The narrower, more practical question — *"does this repo exhibit any of the specific bugs that make ML results unreliable?"* — has no tooling at all. A reviewer with 15 minutes has no way to check it, so it mostly doesn't get checked.

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

**Output:** a structured report where every finding cites a `file:line` in the repo *and* the verbatim paper sentence it contradicts. See `outputs/<phase>/<paper_id>.txt`.

### ⚠️ Findings are advisory. A qualified human must review before any action.

RepoAudit produces **claims that named researchers made specific errors in published work.** Those claims are reputationally consequential and this system is measurably imperfect — our best configuration runs at 0.750 precision, meaning roughly **1 in 4 findings is wrong.** We have observed it hallucinate confident, specific, entirely false claims (see `phases/phase_1_baseline.md`, the proto2proto case: it asserted a paper reported single-seed results when the paper never mentions seeds at all).

Accordingly:

- **Every finding is designed to be checkable in ~30 seconds** — it cites an exact `file:line` and the verbatim paper sentence. Check it before you believe it.
- **No output of this system should be sent to an author, editor, or committee without a qualified human verifying the cited code and quote first.**
- The intended use is *triage* — pointing a reviewer at the 3 things worth their 15 minutes — **not** adjudication.
- This project reads public repositories read-only. It never modifies, contacts, or publishes anything on its own.

A consensus check plus an explicit human-approval gate for high-severity findings was the planned Phase 5 (`instructions.md` §6) and **is not built** — the deadline arrived first. Until it exists, the human review step is procedural, described here, rather than enforced in code. We consider this the most significant open gap in the system; see `MICRO1_JUDGE_AUDIT.txt`.

---

## Improvement Changelog

Full detail with per-paper tables in [`changelog.md`](changelog.md) and `phases/phase_*.md`. All figures below are scored against **identical current ground truth** (`src.eval.rescore_phase` replays saved predictions when ground truth changes, so comparisons stay fair).

| Stage | What we tried and why | Evidence | Decision / learning |
|---|---|---|---|
| **Baseline** (Phase 1) | One prompt, one call: paper + README + first 200 lines of every `.py` file dumped in, ask for JSON findings. | **F1 0.320** · P 0.400 / R 0.267 · FPR-on-clean 2.0 · 4/9 papers produced no usable output | Established the floor. Failure wasn't reasoning, it was *reliability* — a systematic JSON bug silently discarded 4 papers' results. |
| **Iteration 1** (Phase 2) | Split into a claim **extractor** (atomic claims + verbatim quotes) and a **verifier**. Added a post-hoc grounding check that discards findings not tied to a real extracted quote, plus an LLM JSON-repair pass. | **F1 0.667** (+0.347) · P 0.750 / R 0.600 · FPR-on-clean 0.5 (4× fewer) · `hyperparam_drift` F1 0.0 → **1.0** | **Kept — best configuration.** Claim-grounding structurally prevents hallucination; systematic claim-by-claim checking solved hyperparameter detection outright. |
| **Iteration 2** (Phase 3) | Gave the verifier real tools — whole-repo grep (all file types), tree-sitter AST search, structured argparse/config parsing — as a multi-turn ReAct agent with native tool-calling. | **F1 0.273** (−0.394) · P 0.429 / R 0.200 · **2.4× the tokens** | **Revised, not kept.** A documented negative result — see below. The *tools* work and ship forward; the architecture around them doesn't. |

### The most instructive result: Phase 3 regressed, and why

Phase 3 gave the agent strictly *more* capability and it got substantially worse. `birt_continual_learning` isolates the cause exactly — Phase 2 caught 2/2 findings; Phase 3 caught 0/2 **despite making 9 successful tool calls**:

- Phase 2's extractor produced **atomic** claims: `"weight decay of 1e-6"` and `"averaged across three random seeds"` as separate items. Both bugs caught.
- Phase 3's extractor (byte-identical prompt) bundled three hyperparameters into **one compound quote** and extracted **no seed claim at all**.

Because the grounding rule requires every finding to map to an extracted claim, the seed bug became *structurally unfindable* — no amount of tooling could recover it. And the compound claim diluted the weight-decay check: two of its three values match the code, so the sentence reads as broadly correct.

**Better tools cannot compensate for worse inputs.** The bottleneck was never evidence *gathering* — it was evidence *specification*, in the cheapest, least glamorous component of the pipeline.

### What Phase 3 did prove

The tool layer itself is sound and ships forward. The tool-using verifier read `run_gmixup.sh` via whole-repo grep and caught that it loops **8 seeds against the paper's claimed ten runs** — a subtler, more precise bug than Phase 2's crude (and factually *wrong*) "no seed loop exists." Phase 2 was structurally incapable of finding it, because it only ever read `.py` files.

---

## An honest note on the evaluation

Ground truth was corrected **three times** during development, because the agent surfaced real bugs our own hand-annotation had missed (`birt_continual_learning`, `g_mixup`, `network_deconvolution`). Each was verified by hand against the actual code before being accepted, and **all earlier phases were rescored against the corrected data** — which made earlier phases look *worse*, not better.

A tool built to audit reproducibility ended up auditing its own evaluator. That's the strongest evidence we have that it does something real.

---

## Main failure mode and Hot Take

**Main failure mode:** the pipeline's quality is capped by its weakest upstream component — a cheap model doing unglamorous claim extraction. Every downstream improvement (better verifier, real tools, more agency) is worthless if the claim it needed to check was never extracted, or was extracted as a mushy compound sentence instead of an atomic fact.

**Hot take:** *the agentic community over-invests in giving agents more capability and under-invests in specifying what they're checking.* We handed our agent whole-repo grep, AST search, config parsing, and 10 free investigative turns — and it did worse than a simpler pipeline whose only advantage was that its inputs happened to be atomic. Agency amplifies input quality; it does not substitute for it. If we built this again, we'd spend the effort on making claim extraction deterministic and atomic (one claim per checkable fact, schema-enforced) *before* adding a single tool.

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
src/agents/       extractor, verifier (Phase 2), tool-using verifier (Phase 3)
src/tools/        grep (whole-repo), tree-sitter AST search, argparse/YAML parsing
src/eval.py       scoring harness + rescore_phase for fair cross-phase comparison
data/ground_truth.json    15 hand-verified findings across 9 papers
phases/           per-phase writeups with failure analysis
outputs/trajectories/     agent trajectories (instructions → tool calls → result)
MICRO1_JUDGE_AUDIT.txt    our own adversarial self-audit against the brief
```

## What existed before vs. what we built

**Used as-is:** LangGraph, LangChain, NVIDIA NIM models (`nemotron-3-super-120b-a12b`, `nemotron-3-nano-30b-a3b`, `gpt-oss-120b`), tree-sitter, and the 9 external paper repos as unmodified audit subjects.

**Built for this hackathon:** the entire audit pipeline, all three agent architectures, the tool layer, the eval harness and scoring logic, the hand-verified ground truth, and the synthetic fixture.

## Related work

PaperBench (Starace et al., 2025), CORE-Bench (Siegel et al., 2024), MLE-bench (Chan et al., 2024), SciReplicate-Bench (Xiang et al., 2025), REPRO-Bench (Hu et al., 2025), ML Reproducibility Challenge / ReScience C, and Kapoor & Narayanan (2023) for the leakage taxonomy.

Two of our eval papers (`lbcs`, `sample_specific_masks`) are sourced from PaperBench's official ICML-2024 list. But note the positioning:

> **These benchmarks ask *can the agent do the research?* RepoAudit asks *can the agent tell whether the research was done right?*** The second question is cheaper, faster, and closer to what a reviewer or team lead actually pays for.
