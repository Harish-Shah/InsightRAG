"""PDF extraction pipeline (Phase 1, Tasks 1.1-1.6).

Turns the annual-report PDFs in ``PDF_DIR`` into a single ``metadata.json``
manifest plus cropped/rendered PNG assets under ``ASSETS_DIR``. Each emitted
*unit* is a first-class retrievable item:

    {id, content_type, source_pdf, report_year, page, section, caption, text, asset_path?}

with ``content_type in {"text", "table", "image", "figure"}``.

The four extractors share one per-page loop (Task 1.1):
  * 1.2 text     - page text, cleaned, with a section heading from font sizes.
  * 1.3 table    - table finder -> Markdown + a cropped PNG of the table bbox.
  * 1.4 image    - get_images -> RGB PNG, tiny logos filtered, captioned.
  * 1.5 figure   - caption-anchored page-region render (captures vector charts
                   that get_images misses), conservative (explicit captions).
  * 1.6 manifest - deterministic ids, per-type/PDF counts, integrity asserts.

Engine note (deviation from the plan, for performance):
  The plan named pdfplumber as the primary text/table engine. On these
  graphics-heavy reports pdfplumber's per-page parse is ~1s/page and tens of
  seconds on image/vector-dense pages (a full run took ~46 min). PyMuPDF's
  native ``get_text`` (~0.02s/page) and ``find_tables`` (~0.1s/page) produce
  equivalent text and the *same* real tables (while rejecting pdfplumber's
  thin-line false positives), so we use PyMuPDF for both. Table bboxes are then
  already in PyMuPDF coordinates, so crops need no coordinate conversion.

Run the whole corpus::

    python -m ingestion.extract               # all PDFs in PDF_DIR
    python -m ingestion.extract --pdf 2022_23 # one PDF (substring match)
    python -m ingestion.extract --pdf 2022_23 --pages 40-60   # a page range (test)

No NVIDIA credentials are required - embeddings/LLM are not used here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

# Allow ``python ingestion/extract.py`` as well as ``-m ingestion.extract``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402

# --- Tunable heuristics ------------------------------------------------------
RENDER_DPI = 150                 # crop/figure render resolution
MIN_PAGE_CHARS = 40              # below this (and no visual) a page is skipped
MIN_IMG_PX = 120                 # drop raster images smaller than this (w or h)
MAX_IMAGES_PER_PAGE = 12         # keep only the largest N raster images per page
# Table acceptance filters (points; page is 612x792)
TBL_MIN_WIDTH = 60.0
TBL_MIN_HEIGHT = 22.0
TBL_MIN_COLS = 2
TBL_MIN_ROWS = 2
TBL_MIN_FILLED = 0.30            # min fraction of non-empty cells
TBL_FULLPAGE_FRAC = 0.95         # bbox covering >= this fraction of page = layout junk
# Figure region geometry (points)
FIG_ABOVE = 320.0                # how far above a caption a chart may extend
FIG_BELOW = 24.0                 # small margin below the caption
FIG_MAX_HEIGHT = 380.0
PAGE_MARGIN = 36.0
MIN_FIG_DRAWINGS = 12            # region must contain >= this many vector drawings

CAPTION_RE = re.compile(
    r"\b(?:fig(?:ure)?|chart|graph|exhibit|diagram|illustration)\s*\.?\s*\d+",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(20\d{2})[ _\-]?(\d{2})")
_WS_RE = re.compile(r"[ \t ]+")  # includes non-breaking space
_MULTINL_RE = re.compile(r"\n{3,}")
_DEHYPHEN_RE = re.compile(r"(\w)-\n(\w)")
_PAGENUM_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.IGNORECASE)


# --- Unit record -------------------------------------------------------------
@dataclass
class Unit:
    id: str
    content_type: str            # text | table | image | figure
    source_pdf: str              # original filename
    report_year: str             # friendly, e.g. "Annual Report 2022-23"
    page: int                    # 1-based
    section: Optional[str] = None
    caption: Optional[str] = None
    text: str = ""               # embeddable content
    asset_path: Optional[str] = None  # relative to ASSETS_DIR (visuals only)


# --- Provenance helpers (Task 1.1) ------------------------------------------
def year_slug_from_filename(filename: str) -> str:
    """``Annual_Report_2021_22 1.pdf`` -> ``2021-22`` (handles space/trailing 1)."""
    m = _YEAR_RE.search(filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return re.sub(r"[^A-Za-z0-9]+", "-", Path(filename).stem).strip("-")


def report_year_label(year_slug: str) -> str:
    """``2021-22`` -> ``Annual Report 2021-22``."""
    if re.fullmatch(r"20\d{2}-\d{2}", year_slug):
        return f"Annual Report {year_slug}"
    return year_slug.replace("-", " ")


# --- Text cleaning & headings (Task 1.2) ------------------------------------
def _raw_page_text(fitz_page) -> str:
    try:
        return fitz_page.get_text("text") or ""
    except Exception:
        return ""


def _normalize_line(line: str) -> str:
    """Normalize a line for header/footer frequency matching (digits -> #)."""
    s = _WS_RE.sub(" ", line.strip().lower())
    return re.sub(r"\d+", "#", s)


def build_header_footer_set(raw_texts: list[str], n_pages: int) -> set[str]:
    """Detect repeating header/footer lines (top/bottom 3 lines of each page)."""
    freq: Counter[str] = Counter()
    for txt in raw_texts:
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        edge = lines[:3] + lines[-3:]
        for ln in set(_normalize_line(ln) for ln in edge):
            if 0 < len(ln) <= 80:
                freq[ln] += 1
    threshold = max(5, int(0.25 * n_pages))
    return {ln for ln, c in freq.items() if c >= threshold}


def clean_text(raw: str, header_footer: set[str]) -> str:
    """Collapse whitespace, de-hyphenate, drop header/footer + page-number lines."""
    raw = _DEHYPHEN_RE.sub(r"\1\2", raw)
    out: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if _PAGENUM_RE.match(stripped):
            continue
        if _normalize_line(stripped) in header_footer:
            continue
        out.append(_WS_RE.sub(" ", stripped))
    text = "\n".join(out)
    return _MULTINL_RE.sub("\n\n", text).strip()


def detect_section_heading(fitz_page) -> Optional[str]:
    """Largest short line on the page, a little bigger than body text."""
    try:
        d = fitz_page.get_text("dict")
    except Exception:
        return None
    size_chars: dict[float, int] = defaultdict(int)
    spans: list[tuple[float, float, str]] = []  # (size, top, text)
    for blk in d.get("blocks", []):
        for line in blk.get("lines", []):
            if not line.get("spans"):
                continue
            line_text = "".join(s.get("text", "") for s in line["spans"])
            size = round(max(s.get("size", 0) for s in line["spans"]), 1)
            top = line["spans"][0].get("bbox", [0, 0, 0, 0])[1]
            size_chars[size] += len(line_text)
            t = line_text.strip()
            if t:
                spans.append((size, top, t))
    if not size_chars or not spans:
        return None
    body_size = max(size_chars.items(), key=lambda kv: kv[1])[0]
    candidates = [
        (sz, top, t)
        for (sz, top, t) in spans
        if sz >= body_size + 1.0 and 1 <= len(t.split()) <= 14 and len(t) <= 90
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], c[1]))  # larger font, then higher
    heading = candidates[0][2].strip(" .:-")
    return heading or None


# --- Geometry helpers --------------------------------------------------------
def _clamp_rect(x0: float, y0: float, x1: float, y1: float, page_rect) -> fitz.Rect:
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return fitz.Rect(x0, y0, x1, y1) & page_rect


def _render_clip(fitz_page, rect: fitz.Rect, out_path: Path) -> bool:
    if rect.is_empty or rect.width < 8 or rect.height < 8:
        return False
    try:
        pix = fitz_page.get_pixmap(clip=rect, dpi=RENDER_DPI)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(out_path)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    ! render failed for {out_path.name}: {exc}")
        return False


def _text_near(fitz_page, rect: fitz.Rect, pad: float = 60.0, limit: int = 400) -> str:
    """Grab text just above/below a region for visual captions/descriptions."""
    page_rect = fitz_page.rect
    band = fitz.Rect(page_rect.x0, max(page_rect.y0, rect.y0 - pad),
                     page_rect.x1, min(page_rect.y1, rect.y1 + pad))
    try:
        txt = fitz_page.get_text("text", clip=band) or ""
    except Exception:
        txt = ""
    return _WS_RE.sub(" ", txt.replace("\n", " ")).strip()[:limit]


def _text_above(fitz_page, rect: fitz.Rect, height: float = 64.0, limit: int = 140) -> str:
    """Closest non-empty text line just above a region (a table/figure title)."""
    page_rect = fitz_page.rect
    band = fitz.Rect(page_rect.x0, max(page_rect.y0, rect.y0 - height),
                     page_rect.x1, max(page_rect.y0, rect.y0 - 2))
    if band.is_empty or band.height < 4:
        return ""
    try:
        txt = fitz_page.get_text("text", clip=band) or ""
    except Exception:
        return ""
    lines = [_WS_RE.sub(" ", ln).strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return ""
    # The two closest lines (title + units note, e.g. "Schedule 16 ... (Amount in ...)").
    chosen = lines[-2:] if len(lines) >= 2 else lines[-1:]
    return " ".join(chosen)[:limit]


# --- Table extraction (Task 1.3) --------------------------------------------
def _cells_to_markdown(rows: list[list[Optional[str]]]) -> tuple[str, float]:
    """Build a Markdown table from cells; return (markdown, filled_ratio)."""
    norm = [[_WS_RE.sub(" ", (c or "").replace("\n", " ")).strip().replace("|", "\\|")
             for c in row] for row in rows]
    ncols = max((len(r) for r in norm), default=0)
    if ncols == 0:
        return "", 0.0
    norm = [r + [""] * (ncols - len(r)) for r in norm]
    total = sum(len(r) for r in norm)
    filled = sum(1 for r in norm for c in r if c)
    filled_ratio = filled / total if total else 0.0
    header, body = norm[0], norm[1:]
    md = ["| " + " | ".join(header) + " |",
          "| " + " | ".join("---" for _ in header) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md), filled_ratio


def extract_tables(fitz_page, year_slug: str, page_no: int, section: Optional[str],
                   assets_root: Path) -> tuple[list[Unit], list[fitz.Rect]]:
    """Return (table units, accepted bboxes) using PyMuPDF's native table finder."""
    units: list[Unit] = []
    accepted_rects: list[fitz.Rect] = []
    page_rect = fitz_page.rect
    page_area = page_rect.width * page_rect.height
    try:
        tables = fitz_page.find_tables().tables
    except Exception as exc:  # noqa: BLE001
        print(f"    ! find_tables error p{page_no}: {exc}")
        return units, accepted_rects

    seen_bboxes: set[tuple] = set()
    idx = 0
    for tbl in tables:
        try:
            rows = tbl.extract()
        except Exception:
            continue
        if not rows:
            continue
        x0, y0, x1, y1 = tbl.bbox
        w, h = abs(x1 - x0), abs(y1 - y0)
        ncols = max((len(r) for r in rows), default=0)
        nrows = len(rows)
        if w < TBL_MIN_WIDTH or h < TBL_MIN_HEIGHT:
            continue
        if ncols < TBL_MIN_COLS or nrows < TBL_MIN_ROWS:
            continue
        if (w * h) >= TBL_FULLPAGE_FRAC * page_area and nrows < 6:
            continue
        md, filled = _cells_to_markdown(rows)
        if filled < TBL_MIN_FILLED or not md:
            continue
        key = (round(x0), round(y0), round(x1), round(y1))
        if key in seen_bboxes:
            continue
        seen_bboxes.add(key)

        idx += 1
        rect = (_clamp_rect(x0, y0, x1, y1, page_rect) + (-4, -4, 4, 4)) & page_rect
        rel = f"{year_slug}/p{page_no:03d}_tbl{idx:02d}.png"
        asset_path = rel if _render_clip(fitz_page, rect, assets_root / rel) else None
        # Caption = the title line just above the table, else the section heading.
        caption = (_text_above(fitz_page, rect) or section or "").strip() or None
        desc_parts = [p for p in (caption, md) if p]
        units.append(Unit(
            id=f"{year_slug}_p{page_no:03d}_tbl{idx:02d}",
            content_type="table", source_pdf="", report_year="", page=page_no,
            section=section, caption=caption, text="\n\n".join(desc_parts),
            asset_path=asset_path,
        ))
        accepted_rects.append(rect)
    return units, accepted_rects


# --- Raster image extraction (Task 1.4) -------------------------------------
def extract_images(doc, fitz_page, year_slug: str, page_no: int, section: Optional[str],
                   assets_root: Path, doc_seen_xrefs: set[int]) -> list[Unit]:
    units: list[Unit] = []
    try:
        imgs = fitz_page.get_images(full=True)
    except Exception:
        return units
    candidates = []
    for item in imgs:
        xref, w, h = item[0], item[2], item[3]
        if w < MIN_IMG_PX or h < MIN_IMG_PX:
            continue
        if xref in doc_seen_xrefs:
            continue
        candidates.append((w * h, xref))
    candidates.sort(reverse=True)
    candidates = candidates[:MAX_IMAGES_PER_PAGE]

    idx = 0
    for _area, xref in candidates:
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.alpha or (pix.colorspace and pix.colorspace.n >= 4):
                pix = fitz.Pixmap(fitz.csRGB, pix)  # CMYK/alpha -> RGB
        except Exception as exc:  # noqa: BLE001
            print(f"    ! image xref {xref} p{page_no}: {exc}")
            continue
        doc_seen_xrefs.add(xref)
        idx += 1
        rel = f"{year_slug}/p{page_no:03d}_img{idx:02d}.png"
        out = assets_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            pix.save(out)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! save image p{page_no} xref {xref}: {exc}")
            idx -= 1
            continue
        caption = None
        try:
            rects = fitz_page.get_image_rects(xref)
        except Exception:
            rects = []
        if rects:
            caption = _text_near(fitz_page, rects[0], pad=50, limit=200) or None
        desc_parts = [p for p in (section, caption) if p]
        text = " | ".join(desc_parts) if desc_parts else f"Image on page {page_no}"
        units.append(Unit(
            id=f"{year_slug}_p{page_no:03d}_img{idx:02d}",
            content_type="image", source_pdf="", report_year="", page=page_no,
            section=section, caption=caption, text=text, asset_path=rel,
        ))
    return units


# --- Caption-anchored figure regions (Task 1.5) -----------------------------
def extract_figures(fitz_page, cleaned_text: str, year_slug: str, page_no: int,
                    section: Optional[str], assets_root: Path,
                    avoid_rects: list[fitz.Rect]) -> list[Unit]:
    """Render a region around each explicit visual caption (vector charts)."""
    units: list[Unit] = []
    captions = {m.group(0).strip() for m in CAPTION_RE.finditer(cleaned_text)}
    if not captions:
        return units
    page_rect = fitz_page.rect
    # Vector drawings on this page (only computed for caption pages -> cheap overall).
    try:
        draw_rects = [fitz.Rect(d["rect"]) for d in fitz_page.get_drawings()
                      if d.get("rect") is not None]
    except Exception:
        draw_rects = []
    idx = 0
    used: list[fitz.Rect] = []
    for cap in sorted(captions):
        try:
            hits = fitz_page.search_for(cap)
        except Exception:
            hits = []
        if not hits:
            continue
        cap_rect = hits[0]
        region = fitz.Rect(
            page_rect.x0 + PAGE_MARGIN,
            max(page_rect.y0, cap_rect.y0 - FIG_ABOVE),
            page_rect.x1 - PAGE_MARGIN,
            min(page_rect.y1, cap_rect.y1 + FIG_BELOW),
        )
        if region.height > FIG_MAX_HEIGHT:
            region.y0 = region.y1 - FIG_MAX_HEIGHT
        region = region & page_rect
        region_area = max(region.width * region.height, 1.0)

        def _overlap_frac(r: fitz.Rect) -> float:
            inter = region & r
            return 0.0 if inter.is_empty else (inter.width * inter.height) / region_area

        if any(_overlap_frac(r) > 0.6 for r in avoid_rects + used):
            continue
        # Conservative gate: only render if the region actually holds a graphic
        # (filters body-text "Figure N" references with no chart on the page).
        n_draw = sum(1 for r in draw_rects if not (region & r).is_empty)
        if n_draw < MIN_FIG_DRAWINGS:
            continue
        idx += 1
        rel = f"{year_slug}/p{page_no:03d}_fig{idx:02d}.png"
        if not _render_clip(fitz_page, region, assets_root / rel):
            idx -= 1
            continue
        context = _text_near(fitz_page, region, pad=10, limit=320)
        text = f"{cap}: {context}" if context else cap
        units.append(Unit(
            id=f"{year_slug}_p{page_no:03d}_fig{idx:02d}",
            content_type="figure", source_pdf="", report_year="", page=page_no,
            section=section, caption=cap, text=text, asset_path=rel,
        ))
        used.append(region)
    return units


# --- Per-PDF driver ----------------------------------------------------------
@dataclass
class PdfStats:
    pages_total: int = 0
    pages_processed: int = 0
    pages_skipped: int = 0
    counts: Counter = field(default_factory=Counter)


def process_pdf(pdf_path: Path, assets_root: Path,
                page_range: Optional[tuple[int, int]] = None,
                do_tables: bool = True, progress_every: int = 50) -> tuple[list[Unit], PdfStats]:
    filename = pdf_path.name
    year_slug = year_slug_from_filename(filename)
    report_year = report_year_label(year_slug)
    stats = PdfStats()
    units: list[Unit] = []

    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
        stats.pages_total = n
        lo, hi = (1, n) if not page_range else (max(1, page_range[0]), min(n, page_range[1]))
        full_run = page_range is None

        # Clear stale assets for a full run so re-runs are idempotent.
        sub = assets_root / year_slug
        if full_run and sub.exists():
            for f in sub.glob("*.png"):
                f.unlink()

        # Pass 1: page text (fast with PyMuPDF), reused for header/footer + body.
        raw_texts = [_raw_page_text(doc[i]) for i in range(n)]
        header_footer = build_header_footer_set(raw_texts, n)

        doc_seen_xrefs: set[int] = set()
        last_heading: Optional[str] = None
        skipped_pages: list[int] = []

        # Pass 2: per-page extraction.
        for i in range(lo - 1, hi):
            page_no = i + 1
            fitz_page = doc[i]
            cleaned = clean_text(raw_texts[i], header_footer)
            heading = detect_section_heading(fitz_page)
            if heading:
                last_heading = heading
            section = heading or last_heading

            table_units, table_rects = ([], [])
            if do_tables:
                table_units, table_rects = extract_tables(
                    fitz_page, year_slug, page_no, section, assets_root)
            image_units = extract_images(
                doc, fitz_page, year_slug, page_no, section, assets_root, doc_seen_xrefs)
            figure_units = extract_figures(
                fitz_page, cleaned, year_slug, page_no, section, assets_root,
                avoid_rects=table_rects)

            has_visual = bool(table_units or image_units or figure_units)
            if len(cleaned) < MIN_PAGE_CHARS and not has_visual:
                stats.pages_skipped += 1
                skipped_pages.append(page_no)
                continue

            page_units: list[Unit] = []
            if len(cleaned) >= MIN_PAGE_CHARS:
                page_units.append(Unit(
                    id=f"{year_slug}_p{page_no:03d}_text", content_type="text",
                    source_pdf="", report_year="", page=page_no, section=section,
                    caption=None, text=cleaned, asset_path=None,
                ))
            page_units += table_units + image_units + figure_units
            for u in page_units:
                u.source_pdf = filename
                u.report_year = report_year
                stats.counts[u.content_type] += 1
            units.extend(page_units)
            stats.pages_processed += 1

            if progress_every and page_no % progress_every == 0:
                print(f"  .. p{page_no}/{hi} units={sum(stats.counts.values())}", flush=True)

        if skipped_pages:
            print(f"  skipped {len(skipped_pages)} empty pages: {skipped_pages}")
    finally:
        doc.close()
    return units, stats


# --- Manifest assembly + integrity (Task 1.6) -------------------------------
def write_manifest(units: list[Unit], assets_root: Path, manifest_path: Path,
                   per_pdf: dict[str, PdfStats]) -> dict:
    ids = [u.id for u in units]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    assert not dupes, f"Duplicate unit ids: {dupes[:10]}"
    missing = [u.asset_path for u in units
               if u.asset_path and not (assets_root / u.asset_path).exists()]
    assert not missing, f"Missing asset files: {missing[:10]}"

    by_type = Counter(u.content_type for u in units)
    by_pdf_type: dict[str, Counter] = defaultdict(Counter)
    for u in units:
        by_pdf_type[u.report_year][u.content_type] += 1

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "assets_dir": str(assets_root),
        "counts": {
            "total_units": len(units),
            "by_content_type": dict(by_type),
            "by_report_year": {yr: dict(c) for yr, c in by_pdf_type.items()},
            "pages": {
                yr: {"total": s.pages_total, "processed": s.pages_processed,
                     "skipped": s.pages_skipped}
                for yr, s in per_pdf.items()
            },
        },
        "units": [asdict(u) for u in units],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    return manifest


# --- CLI ---------------------------------------------------------------------
def _parse_pages(s: Optional[str]) -> Optional[tuple[int, int]]:
    if not s:
        return None
    if "-" in s:
        a, b = s.split("-", 1)
        return (int(a), int(b))
    v = int(s)
    return (v, v)


def find_pdfs(pdf_dir: Path, filter_sub: Optional[str]) -> list[Path]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if filter_sub:
        pdfs = [p for p in pdfs if filter_sub.lower() in p.name.lower()]
    return pdfs


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract PDFs -> metadata.json + assets")
    ap.add_argument("--pdf", help="substring to select one PDF (e.g. 2022_23)")
    ap.add_argument("--pages", help="page range like 40-60 (1-based; testing)")
    ap.add_argument("--out", help="manifest path (default: settings.METADATA_PATH)")
    ap.add_argument("--no-tables", action="store_true", help="skip table extraction (faster)")
    args = ap.parse_args(argv)

    pdf_dir = settings.PDF_DIR
    assets_root = settings.ASSETS_DIR
    assets_root.mkdir(parents=True, exist_ok=True)
    page_range = _parse_pages(args.pages)
    manifest_path = Path(args.out).resolve() if args.out else settings.METADATA_PATH

    pdfs = find_pdfs(pdf_dir, args.pdf)
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}" + (f" matching '{args.pdf}'" if args.pdf else ""))
        return 1

    print(f"Extracting {len(pdfs)} PDF(s) from {pdf_dir}")
    print(f"Assets -> {assets_root}")
    if page_range:
        print(f"Page range: {page_range[0]}-{page_range[1]} (manifest not written for ranges)")

    all_units: list[Unit] = []
    per_pdf: dict[str, PdfStats] = {}
    t0 = time.perf_counter()
    for pdf in pdfs:
        print(f"\n== {pdf.name} ==", flush=True)
        ts = time.perf_counter()
        units, stats = process_pdf(pdf, assets_root, page_range, do_tables=not args.no_tables)
        per_pdf[report_year_label(year_slug_from_filename(pdf.name))] = stats
        all_units.extend(units)
        print(f"  pages processed={stats.pages_processed} skipped={stats.pages_skipped} "
              f"units={sum(stats.counts.values())} {dict(stats.counts)} "
              f"({time.perf_counter()-ts:.1f}s)", flush=True)

    print(f"\nTotal units: {len(all_units)}  ({time.perf_counter()-t0:.1f}s)")
    print("By type:", dict(Counter(u.content_type for u in all_units)))

    if page_range:
        print("Page-range run: skipping manifest write (use a full run for metadata.json).")
        return 0

    manifest = write_manifest(all_units, assets_root, manifest_path, per_pdf)
    print(f"\nManifest written: {manifest_path}")
    print("Counts:", json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
