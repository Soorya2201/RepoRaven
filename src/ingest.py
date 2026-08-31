"""Paper (HTML/PDF) + repo loading."""

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers"
REPOS_DIR = DATA_DIR / "repos"
# Synthetic fixtures (constructed by this project, not cloned from an
# external source) live outside data/repos/ because that whole directory is
# gitignored -- cloned repos are re-fetchable via curate_eval_set.py, but a
# synthetic fixture is not, so it must be committed. See data/synthetic/.
SYNTHETIC_DIR = DATA_DIR / "synthetic"

CODE_EXTENSIONS = {".py"}
CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".cfg", ".ini", ".toml"}
SKIP_DIR_NAMES = {".git", "__pycache__", "node_modules", ".venv", "venv", ".eggs", "egg-info"}


def _fetch_arxiv_html(arxiv_id: str) -> str | None:
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "RepoAudit/0.1"})
        if resp.status_code == 200 and len(resp.text) > 500:
            return resp.text
    except requests.RequestException:
        pass
    return None


def _strip_mathml_annotations(soup: BeautifulSoup) -> None:
    """arXiv HTML embeds raw LaTeX source in <annotation> tags inside MathML;
    left in place, get_text() concatenates it with the rendered unicode glyph
    (e.g. title becomes '𝒢\\mathcal{G}-Mixup...'). Drop them before extracting
    any text."""
    for tag in soup.find_all("annotation"):
        tag.decompose()


def _parse_html_sections(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    _strip_mathml_annotations(soup)
    title_tag = soup.find("h1", class_="ltx_title") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "UNKNOWN"

    sections = {}
    for sec in soup.find_all(["section", "div"], class_=re.compile(r"ltx_section")):
        heading = sec.find(["h2", "h3", "h4"])
        heading_text = heading.get_text(strip=True) if heading else f"section_{len(sections)}"
        text = sec.get_text(separator=" ", strip=True)
        if text:
            sections[heading_text] = text

    raw_text = soup.get_text(separator=" ", strip=True)
    return {"title": title, "sections": sections, "raw_text": raw_text}


def _parse_pdf(pdf_path: Path) -> dict:
    import pymupdf

    doc = pymupdf.open(pdf_path)
    raw_text = "\n".join(page.get_text() for page in doc)
    title = doc.metadata.get("title") or pdf_path.stem
    # No reliable section splitting for PDF; downstream code should treat
    # `sections` as best-effort and fall back to raw_text.
    sections = {"full_text": raw_text}
    return {"title": title, "sections": sections, "raw_text": raw_text}


def load_paper(paper_id: str) -> dict:
    """Return {'title': str, 'sections': dict[str, str], 'raw_text': str, 'source': 'html'|'pdf'}."""
    html_path = PAPERS_DIR / f"{paper_id}.html"
    pdf_path = PAPERS_DIR / f"{paper_id}.pdf"

    if html_path.exists():
        parsed = _parse_html_sections(html_path.read_text(encoding="utf-8"))
        parsed["source"] = "html"
        return parsed
    if pdf_path.exists():
        parsed = _parse_pdf(pdf_path)
        parsed["source"] = "pdf"
        return parsed
    raise FileNotFoundError(f"No paper found for {paper_id} in {PAPERS_DIR}")


def download_paper(paper_id: str, arxiv_id: str) -> str:
    """Fetch arXiv HTML (fallback: leave for manual PDF drop) and save to data/papers/. Returns source type."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    html = _fetch_arxiv_html(arxiv_id)
    if html:
        (PAPERS_DIR / f"{paper_id}.html").write_text(html, encoding="utf-8")
        return "html"
    return "missing"


def load_repo(paper_id: str) -> dict:
    """Return {'root': Path, 'files': list[Path], 'readme': str, 'python_files': list[Path], 'config_files': list[Path]}."""
    root = REPOS_DIR / paper_id
    if not root.exists():
        root = SYNTHETIC_DIR / paper_id
    if not root.exists():
        raise FileNotFoundError(
            f"No repo cloned for {paper_id} at {REPOS_DIR / paper_id} "
            f"(and no synthetic fixture at {SYNTHETIC_DIR / paper_id})"
        )

    files = []
    python_files = []
    config_files = []
    readme = ""

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        files.append(path)
        if path.suffix in CODE_EXTENSIONS:
            python_files.append(path)
        elif path.suffix in CONFIG_EXTENSIONS:
            config_files.append(path)
        elif path.stem.lower().startswith("readme") and not readme:
            try:
                readme = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass

    return {
        "root": root,
        "files": files,
        "readme": readme,
        "python_files": python_files,
        "config_files": config_files,
    }
