"""Extract real figures from a paper's arXiv HTML and screen them for
diagnostic relevance before spending a vision-model call on them.

arXiv's HTML rendering embeds figures two ways, both handled here (verified
against real papers, not assumed): a plain <img src="...png"> for raster
figures, and <object data="...svg" type="image/svg+xml"> for vector figures
(common in figure-heavy empirical papers, e.g. Lottery Ticket Hypothesis's
own accuracy-vs-pruning curves). SVGs are rasterized with PyMuPDF (already a
dependency -- it opens and renders SVG natively, no extra system library
needed) before being sent to the vision model, since vision-language models
expect raster pixels, not vector markup -- confirmed directly: sending raw
SVG bytes to the vision model failed outright; the rasterized PNG worked and
was read correctly.

Most figures in an ML paper are architecture diagrams or qualitative
examples, not diagnostic charts (loss curves, embedding/t-SNE plots) --
running vision analysis on all of them wastes calls and dilutes signal. We
pre-filter on the caption text (cheap, deterministic, no LLM call) before
fetching or analyzing an image.
"""

from dataclasses import dataclass

import pymupdf
import requests
from bs4 import BeautifulSoup

from src.ingest import PAPERS_DIR

DIAGNOSTIC_CAPTION_KEYWORDS = [
    "loss", "accuracy", "curve", "epoch", "training", "validation",
    "embedding", "t-sne", "tsne", "cluster", "distribution", "convergence",
    "learning rate", "confusion matrix", "pruning", "sparsity",
]

REQUEST_TIMEOUT = 15
MAX_IMAGE_BYTES = 8_000_000


@dataclass
class Figure:
    index: int
    caption: str
    image_url: str
    is_svg: bool
    is_diagnostic_candidate: bool


def _caption_looks_diagnostic(caption: str) -> bool:
    lower = caption.lower()
    return any(kw in lower for kw in DIAGNOSTIC_CAPTION_KEYWORDS)


def _resolve_url(src: str) -> str:
    return src if src.startswith("http") else f"https://arxiv.org/html/{src}"


def list_figures(paper_id: str) -> list[Figure]:
    """Parse the paper's saved HTML for every <figure> with a real graphic
    (<img> or <object data=...svg>), return them with a cheap caption-based
    relevance flag. Does not fetch any image bytes."""
    html_path = PAPERS_DIR / f"{paper_id}.html"
    if not html_path.exists():
        return []
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")

    figures = []
    for i, fig in enumerate(soup.find_all("figure")):
        caption_tag = fig.find("figcaption")
        caption = caption_tag.get_text(separator=" ", strip=True) if caption_tag else ""

        img = fig.find("img")
        obj = fig.find("object")
        if img is not None and img.get("src"):
            figures.append(Figure(
                index=i, caption=caption, image_url=_resolve_url(img["src"]),
                is_svg=False, is_diagnostic_candidate=_caption_looks_diagnostic(caption),
            ))
        elif obj is not None and obj.get("data"):
            figures.append(Figure(
                index=i, caption=caption, image_url=_resolve_url(obj["data"]),
                is_svg=str(obj.get("type", "")).endswith("svg+xml") or obj["data"].endswith(".svg"),
                is_diagnostic_candidate=_caption_looks_diagnostic(caption),
            ))
        # else: figure has no graphic we recognize (e.g. a table-as-figure) -- skip.
    return figures


def fetch_image_b64_png(fig: Figure) -> str | None:
    """Fetch a figure's image and return PNG bytes, base64-encoded. Rasterizes
    SVG via PyMuPDF first. Returns None on any failure -- a missing/broken
    figure should degrade to 'skip this figure', never crash the checker."""
    import base64
    try:
        resp = requests.get(fig.image_url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200 or len(resp.content) > MAX_IMAGE_BYTES:
            return None
    except requests.RequestException:
        return None

    if fig.is_svg:
        try:
            doc = pymupdf.open(stream=resp.content, filetype="svg")
            pix = doc[0].get_pixmap(dpi=150)
            png_bytes = pix.tobytes("png")
        except Exception:  # noqa: BLE001 -- malformed SVG shouldn't crash the checker
            return None
    else:
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return None
        png_bytes = resp.content  # good enough for jpg/png; the vision API accepts both

    return base64.b64encode(png_bytes).decode()
