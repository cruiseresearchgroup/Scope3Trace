#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chart and image reading for emission figures (vision backbone, e.g. Gemini).

The geometry layer only rebuilds year grids from the PDF text layer. Two kinds
of disclosure fall outside it:

  * charts whose data labels are printed as loose numbers ("4,740 9,403 1,126")
    with the category given only by colour or position;
  * tables embedded as raster images (no text layer at all).

For evidence pages that look like a chart or an image, the page is rendered and
a vision model reads the printed data labels into (field, category, year, value,
unit) rows. The model only *interprets*; every number is still checked
deterministically:

  1. the complete value token is printed on the page, in the PDF text layer or,
     for raster images, in Tesseract OCR of the rendered page;
  2. the year is printed on the page (text layer or OCR);
  3. the unit is an absolute emissions unit printed on the page.

Values that pass are exported with ``binding.origin == "chart_vision"``. They
are a separate evidence type: category and year come from the vision model, so
they rank below table and text-row bindings and only fill fields the report
does not state elsewhere. Bar heights are never measured; unlabeled bars are
not read.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from scope3trace.common.rules import (SCOPE3_CATEGORIES, emission_unit_multiplier, normalize_emission_to_tco2e,
                               parse_number)
from scope3trace.common.state import EvidenceLink, ExtractedValue

CHART_SYSTEM = (
    "You read greenhouse-gas emission figures from an image of one page of a corporate report. "
    "Return only JSON. Report only numbers that are printed on the page as data labels or table cells. "
    "Never estimate a value from bar heights, pie slices or line positions, never sum or convert, "
    "and omit any value whose category or year you cannot read on the page. "
    "Use a unit printed on the image; only if the image has no unit, you may use the single "
    "absolute emissions unit explicitly printed in the supplied adjacent-page context. "
    "Never take amounts, categories or years from that context.")

CHART_USER = """Page {page} of the report. Text layer of the page (may be scrambled or empty):
<<<
{text}
>>>

Adjacent-page context, ONLY for the unit when the image has no unit (empty means unavailable):
<<<
{unit_context}
>>>

From the IMAGE, list every printed emission amount that belongs to one of these fields:
scope1, scope2_location, scope2_market, scope2 (basis not stated), scope3_total,
scope3_category (GHG Protocol Scope 3 category 1-15).

Rules:
- Financed / portfolio / PCAF emissions of a bank, insurer or investor are Scope 3 category 15
  (field scope3_category, category_no 15). Their "Scope 1+2" / "Scope 3" columns are the investees'
  scopes, never the reporter's scope1/scope2/scope3_total. Give the total financed emissions (the
  Total row, total column) as category_total and sector or asset-class rows as component.
- Skip targets, milestones, budgets, forecasts, pathways, intensities, scores and percentages.

Return {{"values": [{{"field": "...", "category_no": null or 1-15, "label": "series/row label as printed",
"column": "column or series header as printed (empty for a single series)",
"year": 2024, "year_label": "year label exactly as printed, e.g. 2024 or 2023/24", "value_raw": "number exactly as printed", "unit_raw": "unit as printed",
"value_kind": "category_total" or "component", "visual": "chart" or "image_table" or "text"}}]}}
Use value_kind "component" for sub-items (a country, an activity inside a category).
Return {{"values": []}} if the page has no such figure."""

_ALLOWED = {"scope1", "scope2_location", "scope2_market", "scope2", "scope3_total", "scope3_category"}
_KEYWORDS = re.compile(r"(?i)scope\s*[123]|ghg|co2|co₂|greenhouse|emission|tco2|carbon")
_AMOUNT = re.compile(r"(?<![\w.,])\d{1,3}(?:[,.   ]\d{3})+(?:[.,]\d+)?(?![\w])|(?<![\w.,])\d+[.,]\d+(?![\w.,])")


def _tok_re(token: str) -> str:
    return r"(?<![\d.,])" + re.escape(token.strip()) + r"(?![\d]|[.,]\d)"


def chart_pages(pdf_path: str, page_numbers: List[int], parsed_by_page: Dict[int, Any],
                max_pages: int = int(os.environ.get("SCOPE3TRACE_CHART_MAX_PAGES", "2"))) -> List[int]:
    """Evidence pages that carry a chart or a raster image with emission figures.

    A page qualifies when it is drawn with a raster image or many vector shapes,
    mentions emissions, and prints amounts that no reconstructed table row
    holds (or has almost no text layer at all).
    """
    try:
        import pdfplumber
    except ImportError:
        return []
    out = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for no in sorted(set(page_numbers)):
                if not 1 <= no <= len(pdf.pages):
                    continue
                p = pdf.pages[no - 1]
                parsed = parsed_by_page.get(no)
                text = (getattr(parsed, "raw_text", None) or p.extract_text() or "")
                n_shapes = len(p.curves) + len(p.rects) + len(p.lines)
                has_image = any((im['x1'] - im['x0']) * (im['bottom'] - im['top']) > .08 * p.width * p.height
                                for im in p.images)
                if not (has_image or n_shapes >= 25):
                    continue
                if text.strip() and not _KEYWORDS.search(text):
                    continue
                # every amount inside a reconstructed table cell is read by the
                # table pipeline (base-year, target and intensity columns too);
                # an ordinary table page is not a chart
                bound = {str(c['raw']).strip() for t in (getattr(parsed, "table_structures", None) or [])
                         for r in t['rows'] for c in r['cells']}
                # "277,251 209,664 208,952": adjacent cells joined by spaces
                loose = [m.group(0) for m in _AMOUNT.finditer(text) if m.group(0).strip() not in bound
                         and not all(x in bound for x in m.group(0).split())]
                n_amounts = len(_AMOUNT.findall(text))
                if bound and n_amounts and len(loose) < .5 * n_amounts and not has_image:
                    continue  # a table page the table pipeline reads
                if len(loose) >= 3 or (has_image and len(text.strip()) < 200):
                    out.append((len(loose) + (50 if has_image and len(text.strip()) < 200 else 0), no))
    except Exception:
        pass
    # the pages with the most unbound amounts first; a small per-document cap
    # keeps vision calls (and daily quotas) bounded
    return sorted(no for _, no in sorted(out, reverse=True)[:max_pages])


def render_page(pdf_path: str, page_no: int, dpi: int = 130) -> Optional[bytes]:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        with tempfile.TemporaryDirectory(prefix="s3t_chart_") as tmp:
            stem = os.path.join(tmp, "p")
            r = subprocess.run([pdftoppm, "-r", str(dpi), "-f", str(page_no), "-l", str(page_no), "-png",
                                "-singlefile", pdf_path, stem], capture_output=True, timeout=120)
            if r.returncode == 0 and os.path.exists(stem + ".png"):
                with open(stem + ".png", "rb") as f:
                    return f.read()
    try:
        import pdfplumber
        import io
        with pdfplumber.open(pdf_path) as pdf:
            img = pdf.pages[page_no - 1].to_image(resolution=dpi)
            buf = io.BytesIO()
            img.original.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return None


def ocr_png(png: bytes) -> str:
    tess = shutil.which("tesseract")
    if not tess or not png:
        return ""
    with tempfile.TemporaryDirectory(prefix="s3t_ocr_") as tmp:
        path = os.path.join(tmp, "p.png")
        with open(path, "wb") as f:
            f.write(png)
        r = subprocess.run([tess, path, "stdout", "--psm", "11"], capture_output=True, timeout=180)
        return r.stdout.decode("utf-8", "ignore") if r.returncode == 0 else ""


def _unit_printed(unit: str, text: str) -> bool:
    core = re.sub(r"[^a-z0-9]", "", unit.lower().replace("₂", "2"))
    norm = re.sub(r"[^a-z0-9]", "", text.lower().replace("₂", "2"))
    if core and core in norm:
        return True
    # "tCO2e" against a printed "t CO2" / "tCO2eq": the same mass unit
    base = re.sub(r"(?:eq|e)$", "", core)
    if base.endswith("co2") and re.search(re.escape(base) + r"(?:eq|e)?", norm):
        return True
    mass = re.search(r"(?i)\b(kt|mt|t|tonnes?|tons?|thousand|million)\b", unit)
    return bool(mass and re.search(r"(?i)\b" + re.escape(mass.group(1)) + r"\b", text) and re.search(r"(?i)co\s?2|co₂", text))


def check_chart_row(row: Dict[str, Any], text_layer: str, ocr_text: str,
                    locale: Optional[str] = None, context_text: str = "") -> Tuple[bool, str, Dict[str, Any]]:
    """Deterministic checks for one vision-read row. Returns (ok, note, facts).

    ``context_text``: text of the adjacent report pages; a chart titled
    without a unit takes the unit only if those pages print it."""
    """Deterministic checks for one vision-read row. Returns (ok, note, facts)."""
    field = str(row.get("field") or "").strip().lower()
    if field not in _ALLOWED:
        return False, f"unsupported field {field!r}", {}
    cat = row.get("category_no")
    if field == "scope3_category":
        try:
            cat = int(cat)
        except (TypeError, ValueError):
            return False, "Scope 3 category number missing", {}
        if cat not in SCOPE3_CATEGORIES:
            return False, "Scope 3 category number out of range", {}
        field = f"scope3_cat{cat}"
    else:
        cat = None
    names = f"{row.get('label') or ''} | {row.get('column') or ''}"
    if re.search(r"(?i)target|milestone|budget|goal\b|forecast|projected|projection|pathway|trajectory|"
                 r"intensity|/\s*m\s?€|/\s*(?:m|mn|million)\b|\bper\s|\bscore\b|%", names):
        return False, "target, intensity or ratio column, not an inventory amount", {}
    if re.search(r"(?i)financ|pcaf|portfolio|investee|loan\s*book", names) and field != "scope3_cat15":
        return False, "financed-emissions figure is Scope 3 category 15, not the reporter's own scope", {}
    token = str(row.get("value_raw") or "").strip()
    value = parse_number(token, locale)
    if value is None:
        return False, "value is not a printed number", {}
    sources = []
    if text_layer and re.search(_tok_re(token), text_layer):
        sources.append("text_layer")
    if ocr_text and re.search(_tok_re(token), ocr_text):
        sources.append("ocr")
    if not sources:
        return False, "value token is not printed on the page (text layer or OCR)", {}
    try:
        year = int(row.get("year"))
    except (TypeError, ValueError):
        return False, "year missing", {}
    page_text = (text_layer or "") + "\n" + (ocr_text or "")
    # a fiscal-year label "2023/24" (or "2023/2024", "2023-24") is the period
    # ending in 2024: the year is the end year of the printed label
    split = re.search(rf"(?<!\d){year}\s?[/\u2010-\u2014-]\s?(?:{year + 1}|{str(year + 1)[2:]})(?!\d)", page_text)
    label = str(row.get("year_label") or "")
    m_lab = re.fullmatch(r"\s*(?:FY\s?)?(\d{4})\s?[/\u2010-\u2014-]\s?(\d{2}|\d{4})\s*", label)
    if m_lab and int(m_lab.group(1)) == year and re.search(re.escape(label.strip()), page_text):
        year += 1
    elif split and not re.search(rf"(?<![\d/\u2010-\u2014-]){year}(?![\d]|\s?[/\u2010-\u2014-]\s?\d)", page_text):
        year += 1  # the year is printed only as the start of fiscal-year labels
    elif re.search(rf"(?<!\d){year - 1}\s?[/\u2010-\u2014-]\s?(?:{year}|{str(year)[2:]})(?!\d)", page_text):
        pass  # the model already gave the end year of a printed "2023/24"
    elif not re.search(rf"(?<!\d){year}(?!\d)|\bFY\s?{str(year)[2:]}\b", page_text):
        return False, f"year {year} is not printed on the page", {}
    unit = str(row.get("unit_raw") or "").strip()
    mult = emission_unit_multiplier(unit)
    if mult is None:
        return False, "no absolute emissions unit", {}
    if _unit_printed(unit, page_text):
        unit_source = "page"
    elif not re.search(r"(?i)t\s?co\s?2|tonnes|tons\b|kt\b|mt\b", page_text) and context_text and _unit_printed(unit, context_text):
        unit_source = "adjacent_page"  # the chart prints no unit of its own
    else:
        return False, f"unit {unit!r} is not printed on the page", {}
    kind = "component" if str(row.get("value_kind") or "").lower() == "component" else "category_total"
    return True, "chart value: printed data label verified on the page; category and year read by vision model", {
        "field": field, "category_no": cat, "value": value, "token": token, "year": year, "unit": unit,
        "kind": kind, "token_source": sources, "unit_source": unit_source}


class ChartReader:
    """Reads chart/image pages with a vision backbone; deterministic checks per value."""

    def __init__(self, llm, state):
        self.llm = llm
        self.state = state
        self._cache: Dict[Tuple[str, int], List[ExtractedValue]] = {}

    @classmethod
    def from_env(cls, state):
        if os.environ.get("SCOPE3TRACE_CHART_READER", "1") == "0":
            return None
        model = os.environ.get("SCOPE3TRACE_CHART_MODEL", "gemini-2.5-flash")
        from scope3trace.common.llm import BoundedLLM
        llm = BoundedLLM("chart", model, timeout=180, max_retries=4, retry_delays=(30, 60, 120))
        return cls(llm, state) if llm.available else None

    @staticmethod
    def _section_text(doc, no: int, parsed_by_page: Dict[int, Any]) -> str:
        """Text of the pages around a chart (two either side) when they state
        one absolute emissions unit; otherwise empty (no unit is borrowed)."""
        texts = []
        for n in (no - 2, no - 1, no + 1, no + 2):
            t = getattr(parsed_by_page.get(n), "raw_text", None)
            if t is None:
                try:
                    from pypdf import PdfReader
                    r = PdfReader(doc.local_path)
                    t = r.pages[n - 1].extract_text() if 1 <= n <= len(r.pages) else ""
                except Exception:
                    t = ""
            texts.append(t or "")
        text = "\n".join(texts)
        units = re.findall(r"(?i)\b(?:in\s+)?(k|thousand\s+|million\s+|m)?\s?t(?:onnes?)?\s?co\s?2\s?e?q?\b", text)
        mults = {(u or "").strip().lower()[:1] for u in units}
        return text if units and len(mults) == 1 else ""

    def read_document(self, doc, page_numbers: List[int]) -> List[ExtractedValue]:
        parsed_by_page = {p.page: p for p in self.state.parsed_pages.get(doc.doc_id, [])}
        values: List[ExtractedValue] = []
        if not getattr(doc, "local_path", None):
            return values
        for no in chart_pages(doc.local_path, page_numbers, parsed_by_page):
            if (doc.doc_id, no) in self._cache:  # repair rounds re-use one reading
                import copy
                values.extend(copy.deepcopy(self._cache[(doc.doc_id, no)]))
                continue
            page_values: List[ExtractedValue] = []
            png = render_page(doc.local_path, no)
            if not png:
                continue
            parsed = parsed_by_page.get(no)
            text_layer = getattr(parsed, "raw_text", "") or ""
            ocr_text = ocr_png(png)
            context = self._section_text(doc, no, parsed_by_page)
            # Gemini 2.5 counts hidden reasoning against max_tokens; reading
            # printed labels needs none, and a larger budget avoids truncation
            obj = self.llm.complete_json(CHART_SYSTEM, CHART_USER.format(page=no, text=text_layer[:6000],
                                                                      unit_context=context),
                                         max_tokens=int(os.environ.get("SCOPE3TRACE_CHART_MAX_TOKENS", "8192")),
                                         images=[base64.b64encode(png).decode()],
                                         extra={"reasoning_effort": os.environ.get("SCOPE3TRACE_CHART_REASONING", "none")})
            if not self.llm.available:
                break  # quota exhausted: stop calling for this run
            if obj is None:
                continue  # a failed API call is not a successfully read empty chart
            rows = (obj or {}).get("values") or []
            img_sha = hashlib.sha256(png).hexdigest()
            locale = getattr(parsed, "number_locale", None)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                ok, note, f = check_chart_row(row, text_layer, ocr_text, locale, context)
                self.state.log_phase("ChartReader", "check",
                                     f"{doc.doc_id} p{no} {row.get('field')} {row.get('value_raw')}: {note}")
                if not ok:
                    continue
                tco2e = normalize_emission_to_tco2e(f["value"], f["unit"])[0]
                binding = {"origin": "chart_vision", "status": "chart_bound", "document": doc.doc_id, "page": no,
                           "field": f["field"], "year": f["year"], "parent_category": f["category_no"],
                           "value_raw": f["token"], "unit_raw": f["unit"], "row_label": str(row.get("label") or ""), "column": str(row.get("column") or ""),
                           "visual": str(row.get("visual") or ""), "token_source": f["token_source"],
                           "image_sha256": img_sha, "unit_source": f["unit_source"],
                           "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                           "text_sha256": hashlib.sha256(text_layer.encode()).hexdigest(),
                           "ocr_sha256": hashlib.sha256(ocr_text.encode()).hexdigest(),
                           "vision_model": self.llm.model, "quantity_type": "inventory"}
                ev = ExtractedValue(field=f["field"], value_raw=f["token"], unit_raw=f["unit"], value_tco2e=tco2e,
                                    category_no=f["category_no"],
                                    category_canonical=SCOPE3_CATEGORIES[f["category_no"]]["name"] if f["category_no"] else None,
                                    category_label_raw=str(row.get("label") or "") if f["category_no"] else None,
                                    category_map_method="vision" if f["category_no"] else None,
                                    report_year=f["year"], label_raw=str(row.get("label") or ""),
                                    value_kind=f["kind"],
                                    evidence=EvidenceLink(source_document=doc.doc_id, page=no,
                                                          text_snippet=f"{row.get('label')} | {f['year']} | {f['token']} {f['unit']}",
                                                          method="chart_vision", confidence=0.6,
                                                          verified=tco2e is not None, verification_note=note,
                                                          binding=binding))
                ev._chart_ocr_text = ocr_text  # kept in memory for Verify re-checks
                ev._chart_context_text = context
                page_values.append(ev)
            _one_total_per_field(page_values)
            self._cache[(doc.doc_id, no)] = page_values
            import copy
            values.extend(copy.deepcopy(page_values))
        return values


def _one_total_per_field(values: List[ExtractedValue]) -> None:
    """Several different "totals" of one field and year on one page (sector
    rows read as totals): only a row labelled Total is the total; when that
    does not decide it, all become components (nothing is guessed)."""
    groups: Dict[Tuple[str, Any], List[ExtractedValue]] = {}
    for v in values:
        if v.value_kind == "category_total" and v.value_tco2e is not None:
            groups.setdefault((v.field, v.report_year), []).append(v)
    for vs in groups.values():
        if len({v.value_tco2e for v in vs}) <= 1:
            continue
        totals = [v for v in vs if re.match(r"(?i)\s*(?:grand\s+)?total\b", v.label_raw or "")]
        if len({v.value_tco2e for v in totals}) > 1:
            # "Total | Financed emissions Total / Scope 1 + 2 / Scope 3": the
            # column that names no scope is the total column
            totals = [v for v in totals if not re.search(r"(?i)\bscopes?\s*\d",
                                                         (v.evidence.binding or {}).get("column") or "")]
        keep = totals if len({v.value_tco2e for v in totals}) == 1 else []
        for v in vs:
            if v not in keep:
                v.value_kind = "component"
                v.hierarchy_note = "chart: one of several different totals of this field on the page"


def validate_chart_binding(value, parsed) -> Tuple[bool, str]:
    """Verify Agent re-check: the stored facts still hold on the source page."""
    b = value.evidence.binding or {}
    text_layer = getattr(parsed, "raw_text", "") or ""
    ocr_text = getattr(value, "_chart_ocr_text", "") or ""
    if b.get("text_sha256") != hashlib.sha256(text_layer.encode()).hexdigest():
        return False, "chart evidence: page text changed since reading"
    row = {"field": "scope3_category" if (b.get("field") or "").startswith("scope3_cat") else b.get("field"),
           "category_no": b.get("parent_category"), "value_raw": b.get("value_raw"), "year": b.get("year"),
           "unit_raw": b.get("unit_raw"), "value_kind": value.value_kind}
    ok, note, f = check_chart_row(row, text_layer, ocr_text, getattr(parsed, "number_locale", None),
                                  getattr(value, "_chart_context_text", "") or "")
    if not ok:
        return False, "chart evidence: " + note
    if f["field"] != value.field or f["year"] != value.report_year:
        return False, "chart evidence: field or year changed"
    if value.value_tco2e != normalize_emission_to_tco2e(f["value"], f["unit"])[0]:
        return False, "chart evidence: normalized amount does not match the printed label"
    return True, note
