# Phase 0 — Setup, Eval Set Curation, Ground Truth

## 1. Environment

- **Python:** 3.11 (installed via Homebrew specifically for this project; system default was 3.9.6, which instructions.md flagged as risky for LangGraph). Virtualenv at `.venv/`.
- **Dependencies:** pinned per `requirements.txt` (langgraph, langchain-nvidia-ai-endpoints, pymupdf, tree-sitter, gitpython, etc.), installed clean, all imports verified.
- **NIM models — deviated from the spec's defaults, verified live:**
  - `instructions.md` specified `meta/llama-3.3-70b-instruct` (primary) and `meta/llama-3.1-8b-instruct` (fast). Both returned **HTTP 410 Gone** — deprecated as of 2026-08-26 — when tested against the live API.
  - I pulled the full model listing (`GET /v1/models`, 83 models) and live-tested candidates against the actual account (many listed models 404 as "not entitled" for this free-tier key, so listing ≠ access).
  - **Primary:** `nvidia/nemotron-3-super-120b-a12b` (120B total params, ~12B active, MoE, reasoning model). Confirmed working, 70B+-class reasoning capacity as instructed.
  - **Fast:** `nvidia/nemotron-3-nano-30b-a3b` (30B total, ~3B active, MoE). Confirmed working, smallest/fastest entitled model in the same family for consistency.
  - **Important wiring detail:** both are *reasoning* models — they emit a `reasoning_content` field (hidden chain-of-thought) separately from `content` (the final answer), and spend part of `max_tokens` on it. `src/llm.py` handles this (`extract_text()` strips reasoning; `DEFAULT_MAX_TOKENS` padded above the instructions.md baseline of 4096 to leave room for it) and tracks running token usage (`get_usage_totals()`) for the credit estimates §8 requires.
- **NIM pricing / free-tier credits:** could not be confirmed from NVIDIA's own pricing page — `build.nvidia.com/pricing` and a specific model page both timed out via automated fetch (appears to be a JS-heavy SPA that doesn't serve static content to a fetcher). Third-party aggregators report "1,000 credits on signup, forever-free tier, 40 RPM limit, $0.04–$1.20/M tokens depending on model" but I'm flagging this as **unverified** rather than asserting it — Soorya, please check the exact balance/rate at your account dashboard directly if credit budget matters for later phases. `src/llm.py` tracks real token counts regardless, so `eval.py` can report actual usage even without a confirmed $/credit conversion.

## 2. Eval set

**Deviation from target:** instructions.md §4.1 targeted 12 papers (8 with issues + 4 clean), split across category A (documented issues, 4-5), B (known clean, 3-4), and C (your own repos, 2-3). Per your direction, **category C was skipped entirely** (no TRACE/PMNet/planted-defect repos supplied) and I proceeded with A+B only, adjusting targets upward on A to compensate.

**Sourcing evolved in two stages.** First pass: two dedicated research passes searched ReScience C reports, Kapoor & Narayanan's leakage taxonomy, GitHub issue trackers, and Papers With Code — found 6 real papers + built 1 synthetic fixture (4 planted bugs) to cover failure modes nothing real could be found for. Second pass, per your explicit request to use established benchmarks instead of ad-hoc search: verified PaperBench's (arXiv 2504.01848) official 20-paper ICML-2024 list against our filters (11 of 20 pass), then hand-audited 2 of those 11 (`lbcs`, `sample_specific_masks`) and found both genuinely exhibit `single_seed_reporting` — real enough that the synthetic fixture's planted version of that same bug was retired (code fixed to a real 5-seed loop) to avoid a stale ground-truth entry that would falsely penalize a correct checker. (REPRO-Bench was investigated and rejected outright — it's social science, Stata/R-based, no Python-majority repos, no hyperparameter/train-test-split concepts our taxonomy needs. NatureBench doesn't apply either — it measures beating published SOTA, not reproducibility auditing.)

Net result: **8 real papers + 1 synthetic fixture = 9 total**, closer to but still short of 12.

| paper_id | arXiv ID | repo | commit | category | expected findings | source |
|---|---|---|---|---|---|---|
| proto2proto | 2204.11830 | archmaester/proto2proto | 2fdc4fb2 | A | 0 (taxonomy-clean; see note) | ReScience search |
| g_mixup | 2202.07179 | ahxt/g-mixup | 0813bb32 | A | 2 | ReScience search |
| lottery_ticket_hypothesis | 1803.03635 | google-research/lottery-ticket-hypothesis | 1f172798 | A | 1 | GitHub issue search |
| lbcs | 2311.08675 | xiaoboxia/LBCS | 338a31dc | A | 1 | PaperBench ICML-2024 list |
| sample_specific_masks | 2406.03150 | tmlr-group/SMM | eeac33af | A | 1 | PaperBench ICML-2024 list |
| synthetic_multi_defect | n/a (synthetic) | local, constructed by us | n/a | A | 3 | constructed |
| network_deconvolution | 1905.11926 | yechengxi/deconvolution | d221c221 | B | 0 | ReScience search |
| birt_continual_learning | 2305.04769 | NeurAI-Lab/BiRT | 1340a97b | B | 0 | ReScience search |
| cifar_10_100n | 2110.12088 | UCSC-REAL/cifar-10-100n | 49df7d8a | B | 0 | ReScience search |

All 8 real candidates passed the mechanical filters (public, <500MB, arXiv HTML or usable source, majority Python, has README) via `scripts/curate_eval_set.py`.

## 3. Ground truth summary

**Total findings: 8**, across all 6 failure modes. **3 of 6 modes now have only real, code-verified positive instances** (no synthetic fallback needed):

| failure_mode | count | source |
|---|---|---|
| data_leakage_global_preprocessing | 1 | real (g_mixup) |
| hyperparam_drift | 2 | real (g_mixup, lottery_ticket_hypothesis) |
| single_seed_reporting | 2 | real (lbcs, sample_specific_masks) |
| metric_misimplementation | 1 | synthetic |
| test_set_contamination | 1 | synthetic |
| eval_on_train | 1 | synthetic |

**Every finding was verified by me reading the actual code and cross-referencing the actual paper text** — none were accepted just because a ReScience report said "not reproduced." That distinction mattered: Proto2Proto is ReScience-flagged "not reproduced" but its code shows no instance of our 6 modes (the real issue was missing weights/visualization code); I kept it in the set anyway as a deliberate negative control, so we can see whether the agent hallucinates findings on a paper that IS documented as broken, just not in a way we check for.

## 4. Known issues

- **Category C skipped entirely** (per your direction) — no "one challenging case" from your own repos yet. This is the biggest single gap relative to the original spec; worth revisiting if time allows before submission, since the micro1 brief explicitly asks for "one challenging case and explain what it revealed."
- **`fairness_demographic_shift` (Fairness Guarantees under Demographic Shift, OpenReview wbPObLm6ueA) — rejected.** No arXiv listing exists (OpenReview-only), and OpenReview's PDF endpoint returned HTTP 403 to automated `curl` (with and without a browser User-Agent) — bot-blocked, not something I forced through. Repo (`sgiguere/Fairness-Guarantees-under-Demographic-Shift`) is otherwise fine (Python, 42MB, README present).
- **Two more category-A leads rejected for failing the Python-source filter:** `imandrealombardo/FACT-AI` (Fairness Attacks reproduction) and `ph-mehdi/BioAct-Het` are both majority-Jupyter-Notebook repos — Phase 4's planned AST-based checkers assume `.py` source, so these were excluded rather than special-cased.
- **Real-world research genuinely could not source citable examples of 4 of the 6 failure modes** (single_seed_reporting, metric_misimplementation, test_set_contamination, eval_on_train) within two dedicated search passes (~40 min, ~40 tool calls total). The second research fork's own conclusion, which matches my independent experience finding the G-Mixup and Lottery Ticket Hypothesis bugs by hand: *these bugs are typically only found by direct code audit, not self-reported in GitHub issues or reproducibility reports* — which is arguably itself a small piece of supporting evidence for why a tool like RepoRaven is useful. **Decision:** built one clearly-labeled synthetic fixture (`data/synthetic/synthetic_multi_defect/`, `data/papers/synthetic_multi_defect.html`) with one deliberately planted, runnable, verified instance of each of the 4 missing modes. Every file, the README, and `ground_truth.json` mark it `"synthetic": true` and state plainly it is not a real paper — this is a constructed test fixture for eval coverage (analogous to a unit test), not a fabricated result about real research.
- **`cifar_10_100n` has conflicting reproducibility verdicts** across two ReScience C reports (one says reproduced, a newer one from 2026 says only partial). I did not adjudicate which is right; I cited both and independently verified via my own code audit that it's clean with respect to our specific 6-mode taxonomy, which is a narrower and different claim than "fully reproducible."
- **NIM pricing page unreachable** via automated fetch — noted above; credit-rate estimates in later phases will carry this same caveat unless resolved.
- **`.gitignore` currently excludes `outputs/`** per instructions.md §4.1 step 1, but micro1's ground rule #10 ("give judges enough access to run the project and reproduce the main result") and the required Phase 6 deliverable #4 (agent trajectories under `outputs/trajectories/`) imply at least some `outputs/` content needs to ship with the submission. Flagging now rather than fixing unilaterally — the current plan (per instructions.md's own Phase 6 design) is that `REPRODUCTION.md` + `scripts/run_all.py` regenerate everything from a clean environment, so gitignoring `outputs/` during development should be fine as long as that reproduction path actually works end to end. Will re-verify this assumption in Phase 6.
- The nested git repo for this project was initialized fresh inside `micro1_hackathon/`, separate from the large personal monorepo at `/Volumes/Soorya_1_TB/Projects` (whose git root is one level up and tracks many unrelated projects) — needed a clean, independent repo for a hackathon submission.

## 5. How to reproduce Phase 0

```bash
cd micro1_hackathon
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in your own NVIDIA_API_KEY
.venv/bin/python scripts/curate_eval_set.py   # downloads papers, clones repos, re-validates filters
.venv/bin/python -c "from src.ingest import load_paper, load_repo; p = load_paper('g_mixup'); r = load_repo('g_mixup'); print(p['title'], len(r['python_files']), 'python files')"
```

Expected: `curate_eval_set.py` reports 6 accepted (0 rejected), 1 synthetic; the smoke test prints the G-Mixup paper title and `4 python files`.

Note: `synthetic_multi_defect` is not fetched by `curate_eval_set.py` (it has no external source) — it already exists at `data/synthetic/synthetic_multi_defect/` and `data/papers/synthetic_multi_defect.html` as committed files.

## 6. Next phase gate

Phase 1 builds the deliberately-dumb baseline agent (§5 of instructions.md): one LangGraph node, one prompt, paper + README + head of each `.py` file dumped in, ask for JSON findings. It will run against all 7 eval-set papers via `src/eval.py`.

**What "success" looks like for Phase 1:** not high scores — the baseline is supposed to be bad. Success is a clean, honest F1/precision/recall readout (probably low, possibly near-zero on some failure modes given only 1 real example each for 4 of the 6 modes) that gives Phase 2+ something concrete to beat. Given the thin per-mode sample size (n=1 for 4 of 6 modes), per-failure-mode metrics in early phases should be read as directional signal, not statistically robust — I'll say this plainly in the Phase 1 write-up rather than overstate precision on an n=1 category.
