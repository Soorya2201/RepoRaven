"""data_leakage_global_preprocessing checker: the dual-modality one.

Two evidence sources, both gathered deterministically:
1. Code: calls to fit_transform/StandardScaler/PCA/etc. anywhere in the repo
   (the textbook leakage pattern -- fitting a preprocessor on combined
   train+test data before the split).
2. Vision: diagnostic figures (loss/accuracy curves, embedding plots) run
   through a vision model against specific, checkable visual signatures.

Design principle (deliberately conservative): vision alone never produces a
finding. A suspicious chart is a hint, not proof -- the code is the proof.
This mirrors exactly how the leakage bug in our own eval set (G-Mixup) was
actually found in earlier phases: by reading code, not by eyeballing a plot.
Vision evidence is included in the decision prompt as supporting context
when available, but the deterministic anchoring rule still requires the
final finding to cite a real CODE evidence ID, never a figure alone.
"""

import os

from src.agents.checkers.base import DECISION_PROMPT_TEMPLATE, decide_with_retry
from src.agents.checkers.evidence import Evidence, evidence_by_id, gather_function_call_evidence, gather_grep_evidence, render_evidence_block
from src.llm import get_llm, invoke_with_retry, extract_text
from src.tools.vision_tool import fetch_image_b64_png, list_figures

# Patterns that typically mark where a train/test split actually happens --
# without this, the checker has no way to tell whether a preprocessing call's
# line number is before or after the split, which is the entire question.
# Found missing by direct case study: G-Mixup's real leak was in the
# evidence the first version of this checker gathered, but the checker still
# produced zero findings, because it had no split-location evidence to
# compare line numbers against.
SPLIT_PATTERNS = r"train_test_split|train_dataset\s*=|X_train\s*=|train_idx|train_indices"

PREPROCESSING_FUNCTIONS = [
    "fit_transform", "fit", "StandardScaler", "MinMaxScaler", "PCA",
    "Normalizer", "RobustScaler",
    # Leakage isn't always a named sklearn transformer -- hand-rolled
    # normalization (raw .mean()/.std() calls fitting statistics manually)
    # is just as real a leak. Found by direct case study: G-Mixup's actual
    # leakage bug (verified in Phase 0/2) is exactly this pattern -- a raw
    # `deg.mean()`/`deg.std()` over the full dataset, no sklearn involved at
    # all -- and was invisible to this checker until these were added.
    "mean", "std", "normalize",
]

MAX_FIGURES_TO_ANALYZE = 4  # cap vision calls per paper -- cost/time control

VISION_PROMPT = (
    "You are screening this figure from an ML paper for visual signatures of data leakage. "
    "Answer ONLY based on what's actually visible in the image. Check specifically: "
    "(1) If this is a loss or accuracy curve, does the validation/test curve start suspiciously "
    "close to its final converged value at epoch/step 1, rather than a normal learning trajectory? "
    "(2) If this is an embedding/t-SNE/cluster plot with train and test points distinguished, are "
    "test points suspiciously perfectly interleaved within training clusters rather than showing "
    "independent structure? "
    "If neither applies or nothing looks suspicious, say so plainly. Answer in 1-2 sentences, "
    "and start with either 'SUSPICIOUS:' or 'NORMAL:'."
)

INSTRUCTIONS = """The claim describes the data split protocol (e.g. "we split into train/val/test by 7:1:2"). The evidence includes THREE kinds, each labeled: PREPROCESSING entries (calls to fit_transform/StandardScaler/PCA/mean/std/normalize, which compute statistics on whatever data they're given), SPLIT entries (lines that mark where train/test separation actually happens, e.g. `train_dataset = dataset[:train_nums]` or `train_test_split(...)`), and possibly VISION entries (a vision model's read of a diagnostic figure, prefixed SUSPICIOUS or NORMAL).

Data leakage means a preprocessing entry's line number comes BEFORE a split entry's line number IN THE SAME FILE, computing statistics on the dataset variable before it's been separated into train/test -- meaning those statistics are fit on combined data. Compare the actual line numbers given in the evidence to determine order; do not assume order from anything else. If a preprocessing call's line number is clearly after every split entry in that file, or operates on a variable that's already been sliced to train-only (e.g. a `train_dataset` variable), that is NOT leakage.

IMPORTANT: a finding here MUST cite a PREPROCESSING CODE evidence ID, and your explanation MUST reference the specific SPLIT evidence ID whose line number it precedes, by number, to justify the ordering claim. A SUSPICIOUS vision result alone, with no corroborating code evidence, is NOT sufficient -- do not report a finding based on vision evidence alone. Vision evidence is supporting context only: use it to raise or lower your confidence in a code-based finding, never as the sole basis for one."""


def _gather_vision_evidence(paper_id: str, existing: list[Evidence]) -> list[Evidence]:
    figures = [f for f in list_figures(paper_id) if f.is_diagnostic_candidate][:MAX_FIGURES_TO_ANALYZE]
    if not figures:
        return []

    vision_model = os.environ.get("NIM_MODEL_VISION", "meta/llama-3.2-11b-vision-instruct")
    llm = get_llm(model=vision_model, temperature=0.0, max_tokens=200)

    out = []
    for fig in figures:
        b64 = fetch_image_b64_png(fig)
        if b64 is None:
            continue
        try:
            response = invoke_with_retry(llm, [{"role": "user", "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}])
            verdict = extract_text(response).strip()
        except Exception:  # noqa: BLE001 -- a vision call failing shouldn't crash the checker
            continue
        eid = f"E{len(existing) + len(out) + 1}"
        out.append(Evidence(id=eid, file="(paper figure)", line=fig.index, snippet=f"Fig.{fig.index} [{fig.caption[:60]}]: {verdict[:150]}", kind="figure"))
    return out


def run_leakage_checker(paper_id: str, claims: list[dict]) -> dict:
    """Returns {'findings': list, 'evidence_gathered': int, 'error': str|None}."""
    relevant_claims = [c for c in claims if isinstance(c, dict) and c.get("claim_type") == "data_split_protocol"]
    if not relevant_claims:
        return {"findings": [], "evidence_gathered": 0, "error": None}

    primary_model = os.environ["NIM_MODEL_PRIMARY"]
    llm = get_llm(model=primary_model, temperature=0.0, max_tokens=2000)

    all_findings = []
    total_evidence = 0
    errors = []

    for claim in relevant_claims:
        preprocessing_evidence = gather_function_call_evidence(paper_id, PREPROCESSING_FUNCTIONS)
        split_evidence = gather_grep_evidence(paper_id, SPLIT_PATTERNS, file_glob="*.py")
        vision_evidence = _gather_vision_evidence(paper_id, preprocessing_evidence + split_evidence)
        # Each gather_*_evidence call independently numbers its own list
        # starting at E1 -- re-key split and vision IDs to continue after
        # preprocessing's, so all three lists share one ID space with no
        # collisions (caught by inspection while wiring this up, not by a
        # failed run -- worth being paranoid about since a collision here
        # would make deterministic anchoring silently point at the wrong
        # evidence instead of failing loudly).
        offset = len(preprocessing_evidence)
        for i, e in enumerate(split_evidence):
            e.id = f"E{offset + i + 1}"
        offset += len(split_evidence)
        for i, e in enumerate(vision_evidence):
            e.id = f"E{offset + i + 1}"
        evidence = preprocessing_evidence + split_evidence + vision_evidence
        total_evidence += len(evidence)
        if not preprocessing_evidence:
            continue  # no preprocessing code evidence at all -- nothing groundable, correctly silent regardless of vision

        labeled_block = (
            "-- PREPROCESSING entries --\n" + render_evidence_block(preprocessing_evidence) +
            "\n\n-- SPLIT entries --\n" + render_evidence_block(split_evidence) +
            ("\n\n-- VISION entries --\n" + render_evidence_block(vision_evidence) if vision_evidence else "")
        )
        prompt = DECISION_PROMPT_TEMPLATE.format(
            failure_mode="data_leakage_global_preprocessing",
            mode_instructions=INSTRUCTIONS,
            claim_quote=claim.get("quote", ""),
            claim_type=claim.get("claim_type", ""),
            evidence_block=labeled_block,
        )

        findings, parse_err = decide_with_retry(llm, prompt, os.environ["NIM_MODEL_FAST"])
        if parse_err and not findings:
            errors.append(f"data_leakage_global_preprocessing: {parse_err}")
            continue

        for f in findings:
            if not isinstance(f, dict):
                continue
            eid = str(f.get("code_evidence", "")).strip()
            ev = evidence_by_id(evidence, eid)
            if ev is None:
                errors.append(f"data_leakage_global_preprocessing: dropped finding citing unknown evidence ID {eid!r}")
                continue
            if ev.kind == "figure":
                # Enforcement of the "vision alone is not enough" rule: even if
                # the model tried to cite a figure as its evidence, reject it.
                errors.append("data_leakage_global_preprocessing: dropped finding grounded only in vision evidence (code corroboration required)")
                continue
            f["code_evidence"] = f"{ev.file}:{ev.line}"
            f["failure_mode"] = "data_leakage_global_preprocessing"
            all_findings.append(f)

    return {
        "findings": all_findings,
        "evidence_gathered": total_evidence,
        "error": "; ".join(errors) if errors else None,
    }
