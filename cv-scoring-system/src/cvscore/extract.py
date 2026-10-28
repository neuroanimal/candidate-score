"""Document ingestion: PDF (text layer or OCR), DOCX and plain text.

Designed for CVs exported from Adobe InDesign, Canva, Word and similar tools, which
cause the usual extraction problems:

* **multi-column layouts and sidebars** - text frames are emitted in creation order,
  not reading order. Blocks are re-ordered with a recursive XY-cut that prefers
  vertical cuts (columns) but refuses to split off a narrow column such as a column
  of dates next to job entries, which must stay on the same line as its entry;
* **text converted to outlines or placed as images** - there is no text layer, so
  the page is rendered and OCR-ed with Tesseract (``pol+eng``);
* **fonts without a ToUnicode map** - the text layer is garbage; detected with
  :func:`cvscore.textnorm.garbage_ratio` and OCR-ed instead;
* **hyperlinks hidden behind icons** - portfolio, Behance or Instagram links exist
  only as link annotations, so they are collected from the annotations too.

The result is a :class:`Document` holding the cleaned text, per-line style
information used by section detection, and extraction diagnostics that feed the
"ATS parseability" metrics.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from . import textnorm

try:  # PyMuPDF
    import pymupdf
except ImportError:  # pragma: no cover - dependency is mandatory, message for humans
    pymupdf = None

try:
    import pytesseract
    from PIL import Image
except ImportError:  # pragma: no cover
    pytesseract = None
    Image = None

URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>()\"']+|"
    r"\b(?:behance\.net|instagram\.com|facebook\.com|linkedin\.com|github\.com|"
    r"dribbble\.com|artstation\.com|pinterest\.[a-z]+|tiktok\.com|youtube\.com|vimeo\.com|"
    r"issuu\.com|saatchiart\.com|etsy\.com)/[^\s<>()\"']*",
    re.IGNORECASE,
)


@dataclass
class Line:
    """One logical line of text with the style information needed for heading detection."""

    text: str
    page: int
    size: float = 0.0
    bold: bool = False
    x0: float = 0.0
    y0: float = 0.0
    last_y: float | None = None  # top of the last physical line after re-joining wrapped lines


@dataclass
class PageInfo:
    """Extraction diagnostics for one page."""

    number: int
    method: str  # "text", "ocr", "empty"
    chars: int
    garbage: float
    columns: int
    image_coverage: float
    images: int
    ocr_confidence: float | None = None
    # image boxes normalised to the page: (x0, y0, x1, y1) in 0..1 - used to spot a photo
    image_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class Document:
    """Extracted document with diagnostics."""

    path: str
    kind: str  # "pdf", "docx", "txt"
    text: str
    lines: list[Line] = field(default_factory=list)
    pages: list[PageInfo] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        """Number of pages (1 for DOCX/TXT)."""
        return max(1, len(self.pages))

    @property
    def ocr_pages(self) -> int:
        """Number of pages whose text came from OCR."""
        return sum(1 for p in self.pages if p.method == "ocr")

    @property
    def image_pages(self) -> int:
        """Pages that are essentially pictures (e.g. artworks in a portfolio appendix)."""
        return sum(1 for p in self.pages if p.method == "image")

    @property
    def text_pages(self) -> int:
        """Pages carrying CV text (all pages minus picture pages, at least 1)."""
        return max(1, len(self.pages) - self.image_pages)

    @property
    def max_columns(self) -> int:
        """Largest number of text columns detected on any page."""
        return max((p.columns for p in self.pages), default=1)

    @property
    def producer(self) -> str:
        """Creator/producer application, e.g. ``Adobe InDesign 19.0``."""
        return " / ".join(v for v in (self.metadata.get("creator"), self.metadata.get("producer")) if v)

    def all_links(self) -> list[str]:
        """Link annotations plus URLs found in the text, de-duplicated, order kept."""
        seen, out = set(), []
        for url in self.links + URL_RE.findall(self.text):
            key = url.rstrip("/.,;").lower()
            key = re.sub(r"^https?://(www\.)?|^www\.", "", key)
            if key and key not in seen:
                seen.add(key)
                out.append(url.rstrip(".,;"))
        return out


# --------------------------------------------------------------------------- PDF


@dataclass
class _Block:
    x0: float
    y0: float
    x1: float
    y1: float
    lines: list[Line]

    @property
    def width(self) -> float:
        """Block width in points."""
        return self.x1 - self.x0


def _gaps(intervals: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Return uncovered gaps (start, end) between the given intervals."""
    intervals = sorted(intervals)
    gaps, reach = [], intervals[0][1]
    for start, end in intervals[1:]:
        if start - reach >= min_gap:
            gaps.append((reach, start))
        reach = max(reach, end)
    return gaps


def _xy_cut(blocks: list[_Block], page_width: float, depth: int = 0) -> tuple[list[_Block], int]:
    """Order blocks in reading order. Returns (ordered blocks, number of columns found)."""
    if len(blocks) <= 1 or depth > 30:
        return blocks, 1
    # 1) vertical cut = columns. Only accept a cut whose both sides are substantial,
    #    otherwise a narrow column of dates would be separated from its entries.
    x_gaps = _gaps([(b.x0, b.x1) for b in blocks], min_gap=9.0)
    best: tuple[float, float, list[_Block], list[_Block]] | None = None
    best_gap = -1.0
    for g0, g1 in x_gaps:
        left = [b for b in blocks if b.x1 <= g0 + 0.5]
        right = [b for b in blocks if b.x0 >= g1 - 0.5]
        if not left or not right:
            continue
        lspan = max(b.x1 for b in left) - min(b.x0 for b in left)
        rspan = max(b.x1 for b in right) - min(b.x0 for b in right)
        if min(lspan, rspan) < 0.17 * page_width:
            continue
        # both sides must contain real text, not just a few icons
        if min(sum(len(ln.text) for b in side for ln in b.lines) for side in (left, right)) < 25:
            continue
        if (g1 - g0) > best_gap:
            best, best_gap = (g0, g1, left, right), g1 - g0
    if best is not None:
        left, cols_l = _xy_cut(best[2], page_width, depth + 1)
        right, cols_r = _xy_cut(best[3], page_width, depth + 1)
        return left + right, cols_l + cols_r
    # 2) horizontal cuts = bands, read top to bottom.
    y_gaps = _gaps([(b.y0, b.y1) for b in blocks], min_gap=2.0)
    if y_gaps:
        bands, cuts = [], [g[0] for g in y_gaps] + [float("inf")]
        start = -float("inf")
        for cut in cuts:
            band = [b for b in blocks if start <= (b.y0 + b.y1) / 2 < cut]
            if band:
                bands.append(band)
            start = cut
        if len(bands) > 1:
            ordered, cols = [], 1
            for band in bands:
                part, c = _xy_cut(band, page_width, depth + 1)
                ordered.extend(part)
                cols = max(cols, c)
            return ordered, cols
    # 3) not separable: row-major order (a date next to a job title on the same line)
    return sorted(blocks, key=lambda b: (round(b.y0 / 4), b.x0)), 1


def _merge_line_fragments(lines: list[Line], tol: float = 3.0) -> list[Line]:
    """Merge lines that sit on the same baseline (date + title, bullet + text)."""
    lines = sorted(lines, key=lambda ln: (round(ln.y0 / tol), ln.x0))
    out: list[Line] = []
    for ln in lines:
        if out and abs(out[-1].y0 - ln.y0) <= tol and out[-1].page == ln.page:
            prev = out[-1]
            sep = " " if not prev.text.endswith(("•", " ")) else ""
            out[-1] = Line(prev.text + sep + ln.text, prev.page, max(prev.size, ln.size),
                           prev.bold or ln.bold, prev.x0, prev.y0)
        else:
            out.append(ln)
    return out


def _page_blocks(page) -> tuple[list[_Block], list[Line]]:
    """Blocks with per-line style from PyMuPDF's ``dict`` output.

    Rotated lines (a vertical name or label along the page edge, a design favourite) are
    returned separately: as a tall narrow block they would block every horizontal cut
    of the layout analysis. They are appended after the page's other text.
    """
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
    data = page.get_text("dict", flags=flags)
    blocks, rotated = [], []
    for blk in data.get("blocks", []):
        if blk.get("type") != 0:
            continue
        lines = []
        for ln in blk.get("lines", []):
            spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            direction = ln.get("dir", (1.0, 0.0))
            if abs(direction[1]) > 0.2:
                rotated.append(Line("".join(s["text"] for s in ln["spans"]), page.number + 1,
                                    max(s.get("size", 0.0) for s in spans), False, ln["bbox"][0], ln["bbox"][1]))
                continue
            text = "".join(s["text"] for s in ln["spans"])
            size = max(s.get("size", 0.0) for s in spans)
            bold = any((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower()
                       or "black" in s.get("font", "").lower() for s in spans)
            x0, y0 = ln["bbox"][0], ln["bbox"][1]
            lines.append(Line(text, page.number + 1, size, bold, x0, y0))
        if lines:
            x0, y0, x1, y1 = blk["bbox"]
            blocks.append(_Block(x0, y0, x1, y1, lines))
    return blocks, rotated


_DATE_START = re.compile(r"^\s*(\d{1,2}[./]\d{2,4}|(19|20)\d{2}\b)")
_JOIN_END = (",", "–", "-", "—", "/", "&", "+")
_JOIN_WORDS = {"i", "oraz", "lub", "w", "z", "na", "do", "dla", "ze", "we", "od", "o", "and", "or", "of", "the", "in",
               "for", "with", "a"}


def join_wrapped(lines: list[Line]) -> list[Line]:
    """Re-join lines that a text frame wrapped ("... w Photoshopie," + "Illustratorze i InDesignie").

    A line continues the previous one when both are on the same page, have about
    the same font size, the gap is at most ~1.7 line heights, the indentation fits a
    hanging bullet, the new line is not a bullet / date / heading, and either it
    starts with a lower-case letter or the previous line ends with a comma, dash or
    a conjunction.
    """
    out: list[Line] = []
    for ln in lines:
        prev = out[-1] if out else None
        text = ln.text.strip()
        if prev is not None and text and prev.page == ln.page and not text.startswith(("•", "-", "–")):
            same_size = abs(prev.size - ln.size) <= max(0.8, 0.25 * max(prev.size, ln.size)) or not (prev.size and ln.size)
            gap = ln.y0 - (prev.last_y if prev.last_y is not None else prev.y0)
            close = 0 < gap <= max(ln.size, 6.0) * 1.8
            indent_ok = prev.x0 - 3 <= ln.x0 <= prev.x0 + 22
            ptext = prev.text.rstrip()
            last_word = ptext.split()[-1].lower() if ptext.split() else ""
            starts_lower = text[0].islower() or text[0] in "(„\"'"
            cont = starts_lower or ptext.endswith(_JOIN_END) or last_word in _JOIN_WORDS
            not_new = (not _DATE_START.match(text) and not (ln.bold and not prev.bold)
                       and not re.match(r"(www\.|https?://|\S+@\S+\.)", text))
            if same_size and close and indent_ok and cont and not_new and not ptext.endswith(":") and len(ptext) > 12:
                joined = ptext[:-1] + text if ptext.endswith("-") and starts_lower else ptext + " " + text
                out[-1] = Line(joined, prev.page, prev.size, prev.bold, prev.x0, prev.y0, ln.y0)
                continue
        out.append(ln)
    return out


def _absorb_bullets(blocks: list[_Block]) -> list[_Block]:
    """Attach tiny bullet/icon blocks to the block on their right on the same line."""
    keep, small = [], []
    for b in blocks:
        txt = "".join(ln.text for ln in b.lines).strip()
        if b.width < 18 and len(txt) <= 2:
            small.append(b)
        else:
            keep.append(b)
    for s in small:
        target = None
        for b in keep:
            if b.x0 >= s.x1 - 1 and b.x0 - s.x1 < 25 and b.y0 - 4 <= s.y0 <= b.y1:
                if target is None or b.x0 < target.x0:
                    target = b
        if target is not None:
            target.lines.append(Line("•", s.lines[0].page, s.lines[0].size, False,
                                     target.x0 - 1, s.lines[0].y0))
            target.x0 = min(target.x0, s.x0)
        # icons with no neighbour are dropped (they carry no text)
    return keep


def _image_coverage(page) -> tuple[float, int, list]:
    """Fraction of the page area covered by images, image count and normalised boxes."""
    area = abs(page.rect.width * page.rect.height) or 1.0
    covered, count, boxes = 0.0, 0, []
    pw, ph = page.rect.width or 1.0, page.rect.height or 1.0
    try:
        infos = page.get_image_info()
    except Exception:  # pylint: disable=broad-exception-caught
        infos = []
    for info in infos:
        x0, y0, x1, y1 = info.get("bbox", (0, 0, 0, 0))
        w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
        if w * h > 0.002 * area:  # ignore tiny icons
            covered += w * h
            count += 1
            boxes.append((round(x0 / pw, 3), round(y0 / ph, 3), round(x1 / pw, 3), round(y1 / ph, 3)))
    return min(1.0, covered / area), count, boxes


def ocr_available() -> bool:
    """True when Tesseract and its Python wrapper are usable."""
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _ocr_languages() -> str:
    """``pol+eng`` when both language packs are installed, else whatever is available."""
    try:
        langs = set(pytesseract.get_languages(config=""))
    except Exception:  # pylint: disable=broad-exception-caught
        return "eng"
    wanted = [lang for lang in ("pol", "eng") if lang in langs]
    return "+".join(wanted) or "eng"


_OCR_BULLET = re.compile(r"^\s*[»«*·°eo©®¢.,»>]\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])")
_OCR_PAST = re.compile(r"\b(\w{5,}[aiy])t(am|em|aś|eś)\b")


def fix_ocr_line(text: str) -> str:
    """Correct typical Tesseract errors in Polish CVs.

    * bullets read as ``»``, ``e``, ``*`` before a capitalised word -> ``•``
    * ``ł`` read as ``t`` in past-tense verbs: "prowadzitam" -> "prowadziłam"
      (only for words of 8+ letters, so "witam" or "zapytam" are left alone)
    """
    text = _OCR_BULLET.sub("• ", text)
    return _OCR_PAST.sub(lambda m: m.group(1) + "ł" + m.group(2), text)


def _ocr_page(page, dpi: int = 300) -> tuple[list[Line], float]:
    """Render and OCR one page. Returns lines (with estimated size) and mean confidence."""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    data = pytesseract.image_to_data(img, lang=_ocr_languages(), config="--psm 3",
                                     output_type=pytesseract.Output.DICT)
    groups: dict[tuple, list[int]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append(i)
    scale = 72.0 / dpi
    lines, confs = [], []
    for key in sorted(groups):
        idx = groups[key]
        text = " ".join(data["text"][i] for i in idx)
        height = statistics.median(data["height"][i] for i in idx) * scale
        x0 = min(data["left"][i] for i in idx) * scale
        y0 = min(data["top"][i] for i in idx) * scale
        confs.extend(float(data["conf"][i]) for i in idx if float(data["conf"][i]) >= 0)
        # cap-height to font-size: font size is roughly 1.4 x the median glyph height
        lines.append(Line(fix_ocr_line(text), page.number + 1, round(height * 1.4, 1), False, x0, y0))
    return join_wrapped(lines), (sum(confs) / len(confs) if confs else 0.0)


def extract_pdf(path: Path, ocr: str = "auto") -> Document:
    """Extract a PDF. ``ocr``: ``auto`` (only where needed), ``always`` or ``never``."""
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is not installed: pip install pymupdf")
    pdf = pymupdf.open(str(path))
    meta = pdf.metadata or {}  # pylint: disable=no-member
    result = Document(str(path), "pdf", "", metadata={k: v for k, v in meta.items() if v})
    can_ocr = ocr != "never" and ocr_available()
    if ocr != "never" and not can_ocr:
        result.warnings.append("Tesseract niedostępny - strony bez warstwy tekstowej zostaną pominięte.")
    all_lines: list[Line] = []
    for page in pdf:
        raw_blocks, rotated = _page_blocks(page)
        blocks = _absorb_bullets(raw_blocks)
        ordered, columns = _xy_cut(blocks, page.rect.width) if blocks else ([], 1)
        lines = []
        for b in ordered:
            lines.extend(_merge_line_fragments(b.lines))
        lines = join_wrapped(lines) + rotated
        raw = "\n".join(ln.text for ln in lines)
        garbage = textnorm.garbage_ratio(raw) if raw.strip() else 1.0
        coverage, n_images, boxes = _image_coverage(page)
        chars = len(raw.strip())
        need_ocr = ocr == "always" or chars < 40 or garbage > 0.3 or (coverage > 0.6 and chars < 300)
        info = PageInfo(page.number + 1, "text" if chars else "empty", chars, round(garbage, 3),
                        columns, round(coverage, 3), n_images, None, boxes)
        if need_ocr and can_ocr:
            ocr_lines, conf = _ocr_page(page)
            ocr_text = "\n".join(ln.text for ln in ocr_lines)
            if coverage > 0.5 and (conf < 55 or len(ocr_text.strip()) < 150) and ocr != "always":
                # a picture page (artworks in a portfolio appendix): OCR of a painting is noise.
                # Keep the few real text captions from the text layer, if any.
                info.method = "image"
                info.ocr_confidence = round(conf, 1)
            elif len(ocr_text.strip()) > chars * 0.8 or garbage > 0.3 or ocr == "always":
                lines = ocr_lines
                info.method, info.chars = "ocr", len(ocr_text.strip())
                info.garbage = round(textnorm.garbage_ratio(ocr_text), 3)
                info.ocr_confidence = round(conf, 1)
                if conf < 70:
                    result.warnings.append(
                        f"Strona {info.number}: niska pewność OCR ({conf:.0f}%), wyniki mogą być zaniżone.")
        elif need_ocr and garbage > 0.3 and chars:
            result.warnings.append(f"Strona {info.number}: tekst wygląda na uszkodzony, a OCR jest wyłączony.")
        elif need_ocr and coverage > 0.5:
            info.method = "image"
        result.pages.append(info)
        all_lines.extend(lines)
        for link in page.get_links():
            uri = link.get("uri")
            if uri:  # mailto: links are kept too - they are the e-mail behind an icon
                result.links.append(uri)
    pdf.close()
    result.lines = [Line(textnorm.clean_text(ln.text), ln.page, ln.size, ln.bold, ln.x0, ln.y0)
                    for ln in all_lines]
    result.lines = [ln for ln in result.lines if ln.text]
    result.text = textnorm.clean_text("\n".join(ln.text for ln in result.lines))
    if "indesign" in result.producer.lower():
        result.metadata["indesign"] = True
    return result


# -------------------------------------------------------------------------- DOCX


def extract_docx(path: Path) -> Document:
    """Extract paragraphs and tables from a DOCX; heading styles become bold lines."""
    import docx  # pylint: disable=import-outside-toplevel

    d = docx.Document(str(path))
    lines: list[Line] = []
    for para in d.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        bold = "heading" in style or "nagł" in style or all(r.bold for r in para.runs if r.text.strip())
        size = 16.0 if "heading" in style or "nagł" in style else 11.0
        lines.append(Line(text, 1, size, bold))
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(Line(" | ".join(dict.fromkeys(cells)), 1, 11.0, False))
    links = []
    for rel in d.part.rels.values():
        if "hyperlink" in rel.reltype and rel.is_external:
            links.append(rel.target_ref)
    text = textnorm.clean_text("\n".join(ln.text for ln in lines))
    words = len(text.split())
    pages = [PageInfo(i + 1, "text", 0, 0.0, 1, 0.0, 0) for i in range(max(1, round(words / 450 + 0.49)))]
    core = d.core_properties
    return Document(str(path), "docx", text, lines, pages, links,
                    {"creator": core.author or "", "producer": "docx"})


# --------------------------------------------------------------------------- TXT


def extract_txt(path: Path) -> Document:
    """Read a text file (UTF-8, falling back to Windows-1250 used by Polish Windows)."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1250", "iso-8859-2"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover
        text = raw.decode("utf-8", errors="replace")
    text = textnorm.clean_text(text)
    lines = [Line(t, 1, 11.0, False) for t in text.split("\n") if t.strip()]
    words = len(text.split())
    pages = [PageInfo(i + 1, "text", 0, 0.0, 1, 0.0, 0) for i in range(max(1, round(words / 450 + 0.49)))]
    return Document(str(path), "txt", text, lines, pages)


CACHE_VERSION = "3"  # bump when extraction logic changes, so old cached results are ignored


def _cache_file(path: Path, ocr: str) -> Path | None:
    """Location of the cached extraction for this file content, or None when disabled."""
    if os.environ.get("CVSCORE_CACHE", "1").lower() in ("0", "off", "no", "false"):
        return None
    base = Path(os.environ.get("CVSCORE_CACHE_DIR", Path.home() / ".cache" / "cvscore"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return base / f"{digest[:2]}" / f"{digest}_{ocr}_v{CACHE_VERSION}.pickle"


def extract(path: str | Path, ocr: str = "auto") -> Document:
    """Extract any supported document by extension.

    PDF results (OCR is slow) are cached in ``~/.cache/cvscore`` by file content, so a
    renamed or copied file is not processed again. ``CVSCORE_CACHE=0`` disables the cache.
    The cache is local and only ever read back by this program.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf" and path.is_file():
        cache = _cache_file(path, ocr)
        if cache is not None and cache.is_file():
            try:
                with open(cache, "rb") as handle:
                    doc = pickle.load(handle)
                doc.path = str(path)
                return doc
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # unreadable cache entry: extract again
        doc = extract_pdf(path, ocr=ocr)
        if cache is not None:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                with open(cache, "wb") as handle:
                    pickle.dump(doc, handle)
            except OSError:
                pass  # read-only home etc.: caching is an optimisation only
        return doc
    if suffix == ".pdf":
        return extract_pdf(path, ocr=ocr)
    if suffix == ".docx":
        return extract_docx(path)
    if suffix in (".txt", ".md", ".text"):
        return extract_txt(path)
    raise ValueError(f"Nieobsługiwany format pliku: {path.name} (obsługiwane: .pdf, .docx, .txt)")


def from_text(text: str, name: str = "<text>") -> Document:
    """Build a Document directly from a string (tests, pasted job offers)."""
    clean = textnorm.clean_text(text)
    lines = [Line(t, 1, 11.0, False) for t in clean.split("\n") if t.strip()]
    words = len(clean.split())
    pages = [PageInfo(i + 1, "text", 0, 0.0, 1, 0.0, 0) for i in range(max(1, round(words / 450 + 0.49)))]
    return Document(name, "txt", clean, lines, pages)
