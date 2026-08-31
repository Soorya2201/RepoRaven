# Reproduction Guide

Written for someone starting from a **clean environment** with no prior setup.

---

## 1. Prerequisites

| Requirement | Version / notes |
|---|---|
| Python | **3.11** (3.9 will not work — LangGraph requires 3.10+; 3.12 untested) |
| git | any recent version (used to clone the 8 eval-set repos) |
| Disk | ~250 MB (repos ~35 MB, papers ~5 MB, venv ~200 MB) |
| Network | required — fetches arXiv HTML, clones GitHub repos, calls the NIM API |
| NVIDIA NIM API key | free tier at https://build.nvidia.com (key starts `nvapi-`) |

On macOS, if you don't have Python 3.11: `brew install python@3.11`

---

## 2. Setup

```bash
git clone <this-repo> && cd micro1_hackathon

python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# Edit .env and set NVIDIA_API_KEY=nvapi-...  (the model IDs are pre-filled)
```

### Verify your setup before spending credits

```bash
.venv/bin/python -c "
from src.llm import get_llm, invoke_with_retry, extract_text
print(extract_text(invoke_with_retry(get_llm(max_tokens=50), [{'role':'user','content':'Reply with exactly: OK'}])))
"
```

Expected output: `OK`. If you get HTTP 404/410, your account may not be entitled to a model listed in `.env` — see §7.

---

## 3. Which data is required

Nothing to download manually. `scripts/curate_eval_set.py` (run automatically as step 1) fetches everything:

- **9 papers** — arXiv HTML into `data/papers/` (already committed, so this is idempotent)
- **8 repos** — shallow-cloned into `data/repos/` at **pinned commit SHAs** recorded in `data/ground_truth.json`
- **1 synthetic fixture** — `data/synthetic/` (committed; not fetched, has no external source)

`data/repos/` is gitignored (repos are re-fetchable). `data/ground_truth.json` — the 15 hand-verified findings everything is scored against — **is** committed.

---

## 4. Run everything

```bash
.venv/bin/python scripts/run_all.py
```

This fetches data, runs all three phases in order, and exports agent trajectories.

### Useful variants

```bash
# One phase only
.venv/bin/python scripts/run_all.py --phases iter1

# Skip re-fetching papers/repos (they're already local)
.venv/bin/python scripts/run_all.py --skip-fetch

# Re-score saved predictions against current ground truth — NO LLM calls, NO cost
.venv/bin/python scripts/run_all.py --rescore-only

# Run a single phase directly
.venv/bin/python -m src.eval baseline    # or iter1, iter2
```

`--rescore-only` is how every cross-phase comparison in this project stays fair: when ground truth changes, earlier phases are re-scored from their **saved predictions** rather than re-run, so no phase gets an unfair advantage from a different ground truth.

---

## 5. Expected runtime and cost

Measured on the actual runs reported in `changelog.md` (9 papers each):

| Phase | Wall clock | Tokens | LLM calls |
|---|---|---|---|
| `baseline` | ~1,070 s (18 min) | 484,577 | 9 |
| `iter1` | ~1,911 s (32 min) | 403,417 | 17 |
| `iter2` | ~1,143 s (19 min) | 983,688 | 91 |
| **All three** | **~70 min** | **~1.87 M** | **117** |

**Cost:** all three phases fit comfortably within NVIDIA NIM's free tier. We deliberately do **not** publish a dollar figure — NVIDIA's pricing page did not render for automated fetching, so we could not verify a per-token rate, and we won't state a number we haven't confirmed. `_summary.json` reports exact token counts (`token_usage`) with `estimated_nim_credits: null` rather than a fabricated estimate.

**Variance warning:** wall-clock varies significantly run to run (NIM latency, occasional HTTP 503s that trigger retries). Results also vary somewhat even at `temperature=0` — see §7.

---

## 6. What output to expect

```
outputs/<phase>/FINAL_REPORT.md     the actual deliverable -- polished, for the intended end user
outputs/<phase>/_summary.txt        aggregate metrics + per-paper table (printed to stdout too)
outputs/<phase>/_summary.json       same, machine-readable
outputs/<phase>/<paper_id>.txt      developer-facing audit report per paper (debug artifact)
outputs/<phase>/<paper_id>.json     findings + score + raw response + tool calls (debug artifact)
outputs/trajectories/               one representative trajectory per phase
```

`FINAL_REPORT.md` is what the intended user (a PC chair, 15 minutes) would actually read: findings sorted by severity/confidence, the paper quote and code citation for each, and the human-review disclaimer repeated per-paper. Generate it directly with `python scripts/generate_report.py <phase>`.

A `_summary.txt` looks like:

```
paper_id                   n_expected    n_predicted      TP    FP    FN      F1
-------------------------  ------------  -------------  ----  ----  ----  ------
proto2proto                0             0                 0     0     0  1
g_mixup                    5             5                 3     2     2  0.6
...
MICRO-AVG                  -             -                 9     3     6  0.6667
```

### The headline result you should be able to reproduce

| Phase | F1 | Precision | Recall | FPR on clean papers |
|---|---|---|---|---|
| `baseline` | 0.320 | 0.400 | 0.267 | 2.0 |
| **`iter1`** | **0.667** | **0.750** | **0.600** | **0.5** |
| `iter2` | 0.273 | 0.429 | 0.200 | 0.5 |

**`iter1` (Phase 2) is the best configuration.** `iter2` regressing is a real, documented result, not a mistake — see `phases/phase_3_iter2_tools.md` for the root-cause analysis.

---

## 7. Known issues and caveats

**Results are not bit-for-bit deterministic, and the swings can be large, not just "a few points."** Every call uses `temperature=0`, but the NIM-hosted MoE models still vary run to run. Observed real effects: a byte-identical extractor prompt produced atomic claims one run and compound claims another (this is the entire cause of Phase 3's regression — see `phases/phase_3_iter2_tools.md` §3). A second baseline run, made to capture a fuller trajectory (see `outputs/trajectories/README.md`), scored **F1 0.182** against the reported 0.320 — a 0.14 swing on the same 9 papers, same code, same prompt, driven by a `ReadTimeout`, a transient HTTP 503, and the same JSON-formatting bug already documented as Phase 1's central finding, just landing on different papers. **Expect F1 to move substantially between baseline runs specifically** — Phase 1's single-shot design has no repair/retry mechanism, so it's far more exposed to transient failures than Phase 2 or 3. Iteration 1 (Phase 2, F1 0.667) has clearly and consistently been the best result across every run observed; the relative ordering of baseline vs. iteration 2 is *not* guaranteed to reproduce (0.320 vs 0.273 in the reported run; 0.182 vs 0.273 in the second baseline run) — don't treat "baseline beats Phase 3" as a stable finding, only "Phase 2 beats both" is.

**Model availability changes.** The models named in `instructions.md` (`meta/llama-3.3-70b-instruct`, `meta/llama-3.1-8b-instruct`) were **deprecated mid-project** (HTTP 410 Gone, EOL 2026-08-26). `.env.example` names the replacements we verified live. Model listing ≠ entitlement: many models in `GET /v1/models` return 404 for a given free-tier account. To find what your account can actually use:

```bash
curl -s https://integrate.api.nvidia.com/v1/models -H "Authorization: Bearer $NVIDIA_API_KEY" | python3 -m json.tool
```

Note `NIM_MODEL_TOOLCALLING` is a separate variable on purpose: `nemotron-3-super-120b-a12b` returns HTTP 400/500 for tool-calling and guided-JSON on this endpoint, so Phase 3 uses `openai/gpt-oss-120b`, which we verified with a real multi-turn tool-call loop. Phases 1–2 keep the original model so their published numbers stay reproducible.

**Transient API errors are normal.** HTTP 503 ("Service temporarily overloaded") and read timeouts occur. `invoke_with_retry` retries with exponential backoff; occasional papers still fail a run and are logged in `_summary.json`'s `errors` field rather than silently dropped.

**External repos could disappear.** Reproduction shallow-clones 8 public GitHub repos. If one is deleted or made private, that paper will fail to fetch. Commit SHAs are pinned in `data/ground_truth.json` for verification.

**One eval case is synthetic.** `synthetic_multi_defect` is a small script we wrote ourselves, clearly labeled `"synthetic": true`, covering the 3 failure modes no real public example could be found for. It is not a real paper and is never presented as one.

**Licensing.** This project reads public GitHub repos and feeds excerpts to an LLM for read-only analysis; it never redistributes their code. Of the 8 real eval-set repos, 5 carry an explicit license (Apache 2.0 ×3, MIT ×1, CC BY-NC 4.0 ×1 — the last one non-commercial, consistent with this being a non-commercial hackathon evaluation); 3 have none. NVIDIA's free-tier API Catalog is explicitly positioned for research, prototyping, and hackathon use (not production), which is the use case here.

---

## 8. Verifying our claims yourself

Every finding in `data/ground_truth.json` cites an exact `file:line` plus the paper text it contradicts. To spot-check the most striking one:

```bash
# BiRT: paper says weight decay 1e-6, code default is 0.05
grep -n "weight-decay" data/repos/birt_continual_learning/main.py

# G-Mixup: paper claims "ten runs", shipped script loops 8 seeds
grep -n "for seed" data/repos/g_mixup/run_gmixup.sh

# Lottery Ticket: paper says Adam @ 0.0012, code uses SGD @ 0.1
grep -n "OPTIMIZER_FN" data/repos/lottery_ticket_hypothesis/mnist_fc/constants.py
```

Each corresponding paper quote is in `data/ground_truth.json` under that paper's `expected_findings[].paper_evidence`.
