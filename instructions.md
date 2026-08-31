# RepoAudit — Build Instructions for Claude Code

**Project:** An agentic system that audits ML research papers + their code repos for known reproducibility failure modes (data leakage, hyperparameter drift, single-seed reporting, metric misimplementation, test-set contamination, eval-on-train).

**Submission target:** micro1 Agentic Workflows Hackathon (2026). Rubric buckets we are optimizing for: Problem & User Value (15), Agent Solution & Engineering (30), End-to-End Quality (20), Measured Improvement (15), Reproducibility (15), Hot Take (5) = 100.

**You (Claude Code) are the builder.** Soorya is the reviewer. Follow the phased protocol below strictly.

---

## 0. CRITICAL PROTOCOL — READ FIRST

1. **Never skip a phase.** Phases run in order: 0 → 1 → 2 → 3 → 4 → 5 → 6.
2. **Hard stop after every phase.** When a phase ends, write its `phase_N_<name>.md` file, print a `=== PHASE N COMPLETE — AWAITING APPROVAL ===` banner, and **do not start the next phase** until Soorya says "approved, proceed to phase N+1" (or gives edits and then approves).
3. **Log every decision to `changelog.md`.** One entry per phase, using the schema in §9. This file *is* the micro1 improvement changelog and must survive to submission.
4. **Never fabricate results.** Every metric in a phase README must be produced by running `src/eval.py`, not estimated. If a run fails, write down the failure — do not guess numbers.
5. **Ask before installing anything not in `requirements.txt` §3.** If you need a new dependency, add it to `requirements.txt`, mention it in the phase README, and continue.
6. **Never commit API keys.** Read from `.env` only. `.env` is git-ignored; `.env.example` is committed.
7. **Deterministic where possible.** Set temperature=0 for all LLM calls unless a phase explicitly requires sampling. Set every seed you can.
8. **If a paper/repo is missing or broken during eval, skip it and log it** in the phase README's "known issues" section. Never invent replacement cases.

---

## 1. Problem Statement (for the README, don't paraphrase)

Journal editors, PC chairs, and ML team leads currently audit paper+repo pairs for reproducibility issues by hand. A senior reviewer spends 4–8 hours per paper; OpenAI's PaperBench human baseline used 48 hours per paper and hit only 41.4% completion. Existing agent work (PaperBench, CORE-Bench, SciReplicate-Bench, MLE-bench, REPRO-Bench) evaluates agents on *replicating* papers from scratch or given a repo. **None of them ask the practical question:** "given this paper and this repo, is there evidence of the specific known failure modes that make ML results unreliable?"

That is the gap RepoAudit fills. Input: paper HTML/PDF + repo. Output: a structured audit report where every finding traces to a file:line in the repo and a claim location in the paper.

**Failure modes checked (from Kapoor & Narayanan 2023 leakage taxonomy + community-known ML paper pathologies):**

1. **Data leakage — global preprocessing** (fit_transform on train+test concatenated)
2. **Hyperparameter drift** (paper claims LR=3e-4, code default is 1e-3)
3. **Single-seed reporting** (paper reports mean±std, code has no seed loop)
4. **Metric misimplementation** (paper says F1-macro, code uses default 'binary')
5. **Test-set contamination** (test loader used during training)
6. **Eval-on-train** (evaluation loop iterates train_loader)

---

## 2. Repository Structure (create in Phase 0)

```
repoaudit/
├── README.md                       # Master project readme (final version written in Phase 6)
├── instructions.md                 # This file (do not modify)
├── changelog.md                    # Improvement changelog (append-only across phases)
├── requirements.txt
├── .env.example
├── .gitignore
├── phases/
│   ├── phase_0_setup.md
│   ├── phase_1_baseline.md
│   ├── phase_2_iter1_extractor_verifier.md
│   ├── phase_3_iter2_tools.md
│   ├── phase_4_iter3_specialized_checkers.md
│   ├── phase_5_iter4_consensus.md
│   └── phase_6_final.md
├── src/
│   ├── __init__.py
│   ├── llm.py                      # NVIDIA NIM client wrapper
│   ├── ingest.py                   # Paper (HTML/PDF) + repo loading
│   ├── baseline.py                 # Phase 1 baseline agent
│   ├── graph.py                    # LangGraph orchestration (grows each phase)
│   ├── tools/                      # Populated in Phase 3
│   │   ├── __init__.py
│   │   ├── grep_tool.py
│   │   ├── ast_tool.py
│   │   └── config_tool.py
│   ├── agents/                     # Populated in Phase 2+
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   ├── verifier.py
│   │   └── checkers/               # Populated in Phase 4
│   │       ├── __init__.py
│   │       ├── leakage.py
│   │       ├── hyperparams.py
│   │       ├── seeds.py
│   │       ├── metric.py
│   │       ├── test_contamination.py
│   │       └── eval_on_train.py
│   ├── consensus.py                # Populated in Phase 5
│   └── eval.py                     # Scoring against ground truth
├── data/
│   ├── papers/                     # <paper_id>.html or .pdf
│   ├── repos/                      # <paper_id>/ (cloned)
│   └── ground_truth.json           # Manual annotations per paper
├── outputs/
│   ├── baseline/                   # Phase 1 outputs
│   ├── iter1/                      # Phase 2 outputs
│   ├── iter2/                      # Phase 3
│   ├── iter3/                      # Phase 4
│   ├── iter4/                      # Phase 5
│   └── final/                      # Phase 6
└── scripts/
    ├── curate_eval_set.py          # Phase 0
    └── run_all.py                  # Phase 6 one-command reproduction
```

Every `outputs/<phase>/` folder must contain, per paper:
- `<paper_id>.txt` — human-readable audit report
- `<paper_id>.json` — structured findings (schema in §7)
- `_summary.json` — aggregate metrics across all papers for that phase
- `_summary.txt` — human-readable summary

---

## 3. Environment Setup

### 3.1 Python & dependencies
Python 3.11 (LangGraph is well-supported; avoid 3.12 edge cases).

Create `requirements.txt`:
```
langgraph>=0.2.50
langchain>=0.3.0
langchain-nvidia-ai-endpoints>=0.3.0
langchain-core>=0.3.0
python-dotenv>=1.0.0
requests>=2.32.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
pymupdf>=1.24.0
pypdf>=4.0.0
tree-sitter>=0.23.0
tree-sitter-python>=0.23.0
gitpython>=3.1.40
pydantic>=2.9.0
tenacity>=9.0.0
tabulate>=0.9.0
scikit-learn>=1.5.0
rich>=13.7.0
```

Install: `pip install -r requirements.txt`

### 3.2 NVIDIA NIM API
- Sign in at https://build.nvidia.com. Free tier gives monthly credits per account (currently ~1000 credits for personal accounts as of the free tier at build.nvidia.com; verify at signup).
- Get an API key from the account dashboard (starts with `nvapi-`).
- API is OpenAI-compatible. Base URL: `https://integrate.api.nvidia.com/v1`.

Create `.env.example`:
```
NVIDIA_API_KEY=nvapi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
NIM_MODEL_PRIMARY=meta/llama-3.3-70b-instruct
NIM_MODEL_FAST=meta/llama-3.1-8b-instruct
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
```

**Model selection guidance for Claude Code:** In Phase 0, `curl` the model listing endpoint (or check https://build.nvidia.com/models) and confirm the two IDs above are currently free. If either has been deprecated, pick a replacement in the same size class:
- Primary (heavy reasoning): a 70B-class instruct model (llama-3.3, nemotron-super, or qwen2.5-72b-instruct).
- Fast (lightweight extraction): an 8–12B-class instruct model.
Log the choice and reason in `phase_0_setup.md`.

### 3.3 LangGraph + NIM wiring (`src/llm.py`)

Use `langchain-nvidia-ai-endpoints`'s `ChatNVIDIA` — this integrates cleanly with LangGraph.

Minimum contents:
```python
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

def get_llm(model: str | None = None, temperature: float = 0.0, max_tokens: int = 4096):
    """Return a ChatNVIDIA instance configured for our NIM endpoint."""
    return ChatNVIDIA(
        model=model or os.environ["NIM_MODEL_PRIMARY"],
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url=os.environ["NIM_BASE_URL"],
        temperature=temperature,
        max_tokens=max_tokens,
    )

@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=30))
def invoke_with_retry(llm, messages):
    return llm.invoke(messages)
```

All phases use this wrapper. Never call the LLM directly.

---

## 4. Phase 0 — Setup, Eval Set Curation, Ground Truth

**Goal:** produce a clean project skeleton, a curated eval set of 10–15 papers, and hand-annotated ground truth.

**Steps:**

1. Create the directory tree from §2. Add `.gitignore` (`.env`, `data/repos/`, `__pycache__/`, `*.pyc`, `outputs/`, `.venv/`).
2. Write `requirements.txt`, `.env.example`, `src/llm.py`, `src/ingest.py` skeleton (see §4.2).
3. **Curate the eval set** — see §4.1 for the exact selection protocol. Target 12 papers total: 8 with known issues + 4 known clean.
4. Download HTML (or PDF fallback) for each paper into `data/papers/<paper_id>.html`.
5. `git clone` each repo into `data/repos/<paper_id>/`. Do a shallow clone (`--depth 1`) and pin to a specific commit — record the commit SHA in ground truth.
6. Hand-annotate `data/ground_truth.json` (see schema §4.3). For each paper, list every failure-mode instance that exists with the exact file/line and a short justification.
7. Write `phase_0_setup.md` (see §4.4 for what goes in it).
8. **STOP. Await Soorya's approval.**

### 4.1 Eval set curation protocol

Do **not** ask the user for paper links unless the automated curation fails. Sources, in priority order:

**A. Papers with documented, published reproducibility issues (target 4-5)**
- The Kapoor & Narayanan (2023) "Leakage and the reproducibility crisis in machine-learning-based science" paper lists specific studies with data leakage. Pull 2-3 candidates whose code is on GitHub. Their taxonomy paper is at https://www.sciencedirect.com/science/article/pii/S2666389923001599.
- Papers in the [ML Reproducibility Challenge](https://reproml.org/) with reports at [ReScience C](https://rescience.github.io/) marked as "partially reproduced" or "did not reproduce." Cross-reference the original paper's GitHub repo.
- Papers on [Papers With Code](https://paperswithcode.com/) with community-flagged issues in the discussion.

**B. Papers with known-clean reproducibility (target 3-4)**
- Papers from the ML Reproducibility Challenge marked as "successfully reproduced."
- Papers listed on [Papers With Code](https://paperswithcode.com/) SOTA leaderboards where multiple independent implementations exist and agree.

**C. Your own repos (target 2-3, ask Soorya for these)**
- **Ask Soorya in the Phase 0 README** to supply:
  - His TRACE paper (arXiv ID + repo URL) — likely clean or with 1-2 known minor issues
  - PMNet repo (paper if available)
  - One repo where he'll inject a known bug (leakage or HP mismatch) so we have a ground-truth-with-a-planted-defect case
- Do not proceed past Phase 0 without these supplied. This is our "one challenging case" per the micro1 brief.

**Selection filters (apply to all candidates):**
- Repo must be publicly accessible (no auth)
- Repo must be < 500 MB
- Paper must have arXiv HTML available at `https://arxiv.org/html/<id>` OR usable PDF
- Code must be Python (Phase 4 checkers assume Python AST)
- Repo must have a README (any quality)

Reject any candidate that fails these. Log rejections.

### 4.2 `src/ingest.py` skeleton

Two functions to define now (implement in Phase 0):

```python
def load_paper(paper_id: str) -> dict:
    """Return {'title': str, 'sections': dict[str, str], 'raw_text': str, 'source': 'html'|'pdf'}."""

def load_repo(paper_id: str) -> dict:
    """Return {'root': Path, 'files': list[Path], 'readme': str, 'python_files': list[Path], 'config_files': list[Path]}."""
```

HTML parsing: use `beautifulsoup4` + `lxml` on `https://arxiv.org/html/<id>`. Fall back to `pymupdf` (`fitz`) for PDF text extraction. Preserve section headings.

### 4.3 `data/ground_truth.json` schema

```json
{
  "paper_id_1": {
    "arxiv_id": "2401.12345",
    "title": "...",
    "repo_url": "https://github.com/org/repo",
    "repo_commit": "abc123def",
    "paper_source": "html",
    "expected_findings": [
      {
        "failure_mode": "data_leakage_global_preprocessing",
        "severity": "high",
        "paper_evidence": "Section 3.2, para 2: 'We normalize features using StandardScaler.'",
        "code_evidence": "src/data.py:47 — scaler.fit_transform(np.vstack([X_train, X_test]))",
        "notes": "Feature normalization fit on concat of train+test before split."
      }
    ]
  }
}
```

An empty `expected_findings: []` means the paper is annotated as clean.

### 4.4 `phase_0_setup.md` contents

Sections (all required):
1. **Environment** — Python version, dependencies pinned, NIM model chosen and why.
2. **Eval set** — table of the 12 papers: paper_id, arxiv ID, repo URL, commit, category (has-issue / clean / own), count of expected findings.
3. **Ground truth summary** — total findings by failure mode across the set.
4. **Known issues** — anything rejected or that failed to download, and why.
5. **How to reproduce Phase 0** — exact commands: `pip install -r requirements.txt`, `python scripts/curate_eval_set.py`, `python -c "from src.ingest import load_paper, load_repo; ..."` smoke test.
6. **Next phase gate** — what Phase 1 will build and what "success" looks like there.

**STOP after writing this file.**

---

## 5. Phase 1 — Baseline Agent

**Goal:** a deliberately simple baseline that puts a floor under all comparisons. This is what we must beat.

**Design:** single LangGraph node. One prompt. Paper text + entire repo README + first 200 lines of every Python file dumped in as context, then ask the LLM to list reproducibility issues.

**Steps:**

1. Implement `src/baseline.py`:
   - Function `run_baseline(paper_id: str) -> BaselineResult`.
   - Loads paper via `ingest.load_paper`, repo via `ingest.load_repo`.
   - Constructs one prompt with the paper + README + head of each `.py` file (truncate any single file to 200 lines, cap total context to fit NIM_MODEL_PRIMARY's context window minus 2k tokens of response space).
   - Calls the primary model, temperature 0.
   - Parses the response into the finding schema (§7). Instruct the model to return JSON.
2. Baseline prompt (put in `src/baseline.py` as a constant):
    > You are auditing an ML research paper and its code repository for reproducibility issues. Read the paper and the code below. Return a JSON list of findings. Each finding must have: `failure_mode` (one of: data_leakage_global_preprocessing, hyperparam_drift, single_seed_reporting, metric_misimplementation, test_set_contamination, eval_on_train, other), `severity` (high/medium/low), `paper_evidence` (short quote or section reference), `code_evidence` (file:line or file), `explanation` (1-2 sentences). Return `[]` if you find nothing. Return only the JSON, no prose.
3. Implement `src/eval.py`:
   - `score_findings(predicted: list, ground_truth: list) -> dict` — see §7.2 for the matching rules and metrics (precision, recall, F1 on failure_mode + file match, plus severity-weighted precision).
   - `run_phase(phase_name: str, agent_fn) -> None` — iterates the eval set, writes `<paper_id>.txt` and `<paper_id>.json` per paper, aggregates into `_summary.json` and `_summary.txt`.
4. Run Phase 1: `python -m src.eval baseline`. Writes to `outputs/baseline/`.
5. Write `phases/phase_1_baseline.md` (see §5.1).
6. Append entry to `changelog.md` per §9.
7. **STOP. Await approval.**

### 5.1 `phase_1_baseline.md` contents

Sections:
1. **What we built** — one paragraph.
2. **Prompt used** — verbatim.
3. **Results table** — per paper: paper_id, # true findings, # predicted, TP, FP, FN, F1. Bottom row: micro-avg.
4. **Aggregate metrics** — precision, recall, F1 overall; F1 broken down by failure mode.
5. **Failure analysis** — the 3 most representative failure cases with:
   - The paper_id
   - What ground truth said
   - What the baseline said
   - Our hypothesis for why it failed (missed context? hallucinated? wrong file?)
6. **What this tells us** — 2-3 bullets on what the next iteration must fix.
7. **How to reproduce Phase 1** — exact command.
8. **Next phase gate** — what Phase 2 will change and the F1 threshold we expect to clear.

**STOP after writing this file.**

---

## 6. Phases 2–5 — Iteration Plan (expand each on approval)

**Do not build Phases 2–5 until Phase 1 is approved.** The plan below is fixed at the level of goals; the specific prompts/code will be spelled out in the corresponding phase README when you get there.

### Phase 2 — Iteration 1: Split extractor + verifier
- Add `src/agents/extractor.py`: given paper text, output a list of *claims* the paper makes about training setup (splits, hyperparams, seeds, metrics). Uses `NIM_MODEL_FAST`.
- Add `src/agents/verifier.py`: given claims + repo, verify each claim against the code. Uses `NIM_MODEL_PRIMARY`.
- Build a two-node LangGraph in `src/graph.py`.
- Run, evaluate, write `phase_2_iter1_extractor_verifier.md`.
- **Expected gain:** F1 up 10–20 points over baseline (structure alone helps).
- **STOP.**

### Phase 3 — Iteration 2: Give the verifier real tools
- Implement `src/tools/grep_tool.py` (ripgrep-style regex over repo), `src/tools/ast_tool.py` (Python AST search via tree-sitter — find function calls, imports, assignments), `src/tools/config_tool.py` (parse YAML/JSON/argparse defaults).
- Verifier becomes a ReAct-style agent that can call these tools before deciding.
- Run, evaluate, write `phase_3_iter2_tools.md`.
- **Expected gain:** big jump on hyperparam_drift and eval_on_train specifically.
- **STOP.**

### Phase 4 — Iteration 3: Specialized checker subagents (the 6)
- Replace the single verifier with 6 specialized subagents in `src/agents/checkers/`, one per failure mode. Each has a tight prompt targeting only its failure mode, and calls only the tools relevant to it.
- Orchestrator (LangGraph) fans out to all 6 in parallel, gathers findings.
- Run, evaluate, write `phase_4_iter3_specialized_checkers.md`.
- **Expected gain:** precision up (each checker is less prone to drift into other failure modes); recall stable or up.
- **STOP.**

### Phase 5 — Iteration 4: Consensus + human checkpoint (or removed experiment)
- Add `src/consensus.py`: high-severity findings require a second-model check (call the primary model a second time with a different prompt phrasing; only keep the finding if both agree). This addresses hallucinated flags on clean repos.
- Add a "human review" gate for HIGH-severity findings that flags them for approval before appearing in the final report. Complies with micro1 ground rule #4.
- Run, evaluate, write `phase_5_iter4_consensus.md`.
- **Expected outcome:** one of two things — (a) precision jumps enough to be worth keeping, or (b) it doesn't and you *keep the negative result in the changelog as a removed experiment*. Both outcomes are wins for the micro1 rubric ("include experiments you later removed and explain what they taught you").
- **STOP.**

---

## 7. Data Schemas

### 7.1 Finding schema (used by all phases)

```json
{
  "failure_mode": "data_leakage_global_preprocessing",
  "severity": "high",
  "paper_evidence": "Section 3.2: 'We standardize features before splitting.'",
  "code_evidence": "src/preprocess.py:23",
  "explanation": "StandardScaler.fit_transform is called on the full dataset before train/test split at line 23, then split at line 41.",
  "confidence": 0.85
}
```

### 7.2 Matching rules for `eval.py`

A predicted finding matches a ground-truth finding iff:
- `failure_mode` is identical, AND
- `code_evidence` file path matches (line number match not required — the LLM won't get lines exactly right and that's fine)

For each paper:
- **TP** = predicted findings matching a ground truth finding (one-to-one, each GT finding matches at most one prediction)
- **FP** = predicted findings not matching any GT finding
- **FN** = GT findings not matched by any prediction

Metrics:
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2 · P · R / (P + R)
- **Severity-weighted precision** = sum of severity weights for TP / sum of severity weights for (TP+FP), with weights high=3, medium=2, low=1
- **False positive rate on clean papers** = # predicted findings on clean papers / # clean papers (should be near 0)

Report micro-avg across the whole set + breakdown by failure mode.

### 7.3 Per-paper `.txt` report format

```
=================================================================
RepoAudit Report — <paper_id>
Paper: <title>
arXiv: <id>   Repo: <url> @ <commit>
Phase: <phase_name>
=================================================================

FINDINGS: <N>

[1] <severity> — <failure_mode>
    Paper:  <paper_evidence>
    Code:   <code_evidence>
    Why:    <explanation>
    Confidence: <0.0–1.0>

[2] ...

=================================================================
Ground truth: <N_expected> findings expected.
Match: <TP> true positive, <FP> false positive, <FN> false negative.
F1: <score>
=================================================================
```

---

## 8. Evaluation Harness

`src/eval.py` must expose a CLI:

```
python -m src.eval <phase_name>
```

Where `<phase_name>` is one of: `baseline`, `iter1`, `iter2`, `iter3`, `iter4`, `final`. It:
1. Imports the appropriate agent entry point (`baseline.run_baseline`, `graph.run_iter1`, etc.).
2. Iterates the eval set from `data/ground_truth.json`.
3. Writes per-paper `.txt` and `.json` into `outputs/<phase_name>/`.
4. Writes `_summary.json` and `_summary.txt` with aggregate metrics.
5. Prints the summary table to stdout using `tabulate`.

`_summary.json` schema:
```json
{
  "phase": "baseline",
  "model_primary": "...",
  "model_fast": "...",
  "n_papers": 12,
  "per_paper": {
    "paper_id_1": {"tp": 1, "fp": 3, "fn": 2, "precision": 0.25, "recall": 0.33, "f1": 0.29}
  },
  "micro_avg": {"precision": ..., "recall": ..., "f1": ..., "severity_weighted_precision": ..., "fpr_on_clean": ...},
  "by_failure_mode": {
    "data_leakage_global_preprocessing": {"tp": 3, "fp": 2, "fn": 1, "f1": 0.66}
  },
  "wall_clock_seconds": 234.5,
  "estimated_nim_credits": 12.3
}
```

Track wall-clock and credit usage from Phase 1 onward. Credits: log token counts from every LLM call and estimate at NIM's posted per-model rate (find this in Phase 0 and record in `phase_0_setup.md`).

---

## 9. Changelog format (`changelog.md`)

One entry per phase. Append at bottom.

```markdown
## Phase N — <short name>

**What we tried and why:** <2-3 sentences>

**Evidence:**
| Metric | Prev phase | This phase | Delta |
|---|---|---|---|
| Overall F1 | 0.30 | 0.48 | +0.18 |
| Severity-weighted precision | 0.40 | 0.55 | +0.15 |
| FPR on clean papers | 0.60 | 0.30 | −0.30 |
| Wall clock / paper (s) | 12 | 28 | +16 |
| NIM credits / paper | 1.2 | 3.4 | +2.2 |

**Decision:** kept / revised / removed. <One sentence why.>

**What this taught us:** <One sentence — this feeds the Hot Take.>
```

---

## 10. Phase 6 — Final Deliverable (do not touch until Phase 5 approved)

Assembles submission per micro1's four required deliverables:

1. **Complete solution code + improvement changelog** — the repo itself, with `README.md` (top-level) that: introduces the user (journal editor / ML team lead), states the bottleneck, walks the reader through the changelog, states the final F1, links to reproduction guide, ends with the hot take.
2. **Reproduction guide** — `REPRODUCTION.md`: clean environment → clone → `pip install -r requirements.txt` → set `.env` → `python scripts/run_all.py` → expected outputs and wall-clock time. Test this on a fresh venv before submitting.
3. **Solution video (≤5 min)** — record a scripted walkthrough: problem, baseline, one paper end-to-end, final comparison table, changelog highlights, one experiment removed, hot take. `scripts/run_all.py` should be runnable during the video for a live demo of the final phase on one paper.
4. **Agent trajectories** — for every phase, save one representative agent trace: the paper it audited, the tool calls it made with inputs/responses, the final report. Save to `outputs/trajectories/<phase>_<paper_id>.json`.

Also produce `phase_6_final.md` with the head-to-head baseline-vs-final comparison table, and the hot take (5 rubric points).

---

## 11. Research references to cite (in README and paper writeup)

Cite these in the top-level README under "Related work / positioning" — establishes we know the landscape:

- Kapoor & Narayanan (2023) — Leakage and the reproducibility crisis in ML-based science. Source of the 8-type leakage taxonomy.
- Siegel et al. (2024) — **CORE-Bench**: 270 tasks / 90 papers across CS/social sci/medicine. Focuses on *reproducing* results given the repo.
- Starace et al. (2025) — **PaperBench**: 20 ICML 2024 papers, 8,316 gradable sub-tasks. Best AI agent 21.0%, ML PhD humans 41.4% in 48 hours. Focuses on from-scratch *replication*.
- Chan et al. (2024) — **MLE-bench**: 75 Kaggle competitions.
- Xiang et al. (2025) — **SciReplicate-Bench**: algorithm reproduction from paper descriptions.
- Hu et al. (2025) — **REPRO-Bench**: end-to-end social-science reproducibility.
- ML Reproducibility Challenge / ReScience C — source of clean-vs-not annotations for the eval set.

**Positioning line to use in README and hot take:** "These benchmarks ask *can the agent do the research?* RepoAudit asks *can the agent tell whether the research was done right?* The second question is cheaper, faster, and closer to what a reviewer or team lead actually pays for."

---

## 12. Timeline & discipline

Given event ends Aug 31: Phases 0–1 should be same-day. Phases 2–5 one per session. Phase 6 last. If any phase is running long, cut scope inside that phase (e.g., 4 checkers instead of 6 in Phase 4) but never skip a phase — the changelog *is* the story.

---

## 13. Questions Claude Code should ask Soorya before starting Phase 0

1. Confirm you have an NVIDIA API key and it's live (test with a `curl` before we build).
2. Supply the 2-3 own-repo cases described in §4.1.C (TRACE arxiv+repo, PMNet arxiv+repo if any, one repo where you'll plant a defect and tell me exactly what defect at what file:line).
3. Confirm you want me to `git clone` repos to your local disk (`data/repos/`) or if you'd rather I store them in a different location.

After those three answers, start Phase 0.

---

**End of instructions. Do not proceed to any phase without explicit approval to start it.**
