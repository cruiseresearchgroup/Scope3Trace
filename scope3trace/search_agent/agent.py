#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search Agent — document acquisition + evidence retrieval (returns a LIST).

Role boundary (deliberately narrow): the Search Agent
  1. acquires public sustainability / GHG-inventory report PDFs
     (flowchart "Data Acquisition"): searches for an organization's reports
     (an LLM with OpenRouter web search by default, or the Google Custom
     Search API), downloads and identity-checks them
     (``acquire_from_search``, see report_search.py), or takes local paths /
     manifest CSVs;
  2. parses each PDF to per-page text ("Parsing & OCR Toolset"), with an
     OCR fallback for scanned pages;
  3. locates emissions-relevant content ("Page-localization" /
     "Find Scope Info") and RETURNS A LIST of verbatim evidence items:
     [{doc_id, page, content_type, text, reason}].

It preserves table headings, units, scale factors, years, row labels,
numbers, and page markers exactly — it never calculates, converts, maps, or
extracts values. Extraction and estimation are the Scope Agent's job; the
Verify Agent later checks every extracted value against this evidence list.

The LLM here is bounded to page-level classification; the full evidence text
is retained deterministically from parsed/OCR pages, so verbatim fidelity is
guaranteed by construction. With ``rule_only=True`` (or no API
key) localization degrades to the keyword prefilter (recall-oriented).
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scope3trace.common.config import (
    LOCALIZE_CHARS_PER_PAGE,
    LOCALIZE_PAGES_PER_CALL,
)
from scope3trace.common.llm import BoundedLLM
from scope3trace.common.prompts import (
    PAGE_LOCALIZATION_PROMPT,
    SENTENCE_LOCALIZATION_PROMPT,
    SYSTEM_COMMON,
)
from scope3trace.common.rules import (
    clean_page_text,
    content_tokens,
    page_keyword_score,
    split_sentences,
)
from scope3trace.common.state import DocumentRecord, EvidenceItem, ParsedPage, SharedState
from scope3trace.scope_agent.table_structure import reconstruct_page, format_structures
from scope3trace.search_agent.pdf_source import open_geometry, visible_page



def _timed_page_texts(reader, pdf_path: str):
    """pypdf text per page with a time budget (SCOPE3TRACE_PAGE_TEXT_TIMEOUT
    seconds, default 3). A few graphics-heavy pages can take a minute in pypdf;
    those pages take pdfium's text instead. Returns (texts, fallback_pages)."""
    import signal
    import threading
    budget = float(os.environ.get("SCOPE3TRACE_PAGE_TEXT_TIMEOUT", "3"))
    can_alarm = budget > 0 and hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()

    class _Slow(BaseException):  # pypdf swallows Exception subclasses
        pass

    def _alarm(signum, frame):
        raise _Slow()

    texts, slow = [], []
    old = signal.signal(signal.SIGALRM, _alarm) if can_alarm else None
    try:
        for i, page in enumerate(reader.pages, 1):
            try:
                if can_alarm:
                    signal.setitimer(signal.ITIMER_REAL, budget)
                texts.append(page.extract_text() or "")
            except _Slow:
                texts.append(None)
                slow.append(i)
            finally:
                if can_alarm:
                    signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        if can_alarm:
            signal.signal(signal.SIGALRM, old)
    if slow:
        try:
            import pypdfium2 as pdfium
            d = pdfium.PdfDocument(pdf_path)
            for i in slow:
                tp = d[i - 1].get_textpage()
                texts[i - 1] = (tp.get_text_range() or "").replace("\r\n", "\n").replace("\r", "\n")
        except Exception:
            for i in slow:
                texts[i - 1] = texts[i - 1] or ""
    return [t or "" for t in texts], slow

class SearchAgent:
    """Acquires report documents and returns verbatim evidence lists."""

    name = "search"

    def __init__(self, llm: BoundedLLM, state: SharedState, rule_only: bool = False,
                 page_localization: bool = True,
                 sentence_localization: bool = True,
                 table_reconstruction: bool = True) -> None:
        self.llm = llm
        self.state = state
        self.rule_only = rule_only or (
            not llm.available and not getattr(llm, "strict_failures", False))
        self.page_localization = page_localization
        self.sentence_localization = sentence_localization
        self.table_reconstruction = table_reconstruction
        self._page_cache: Dict[str, List[str]] = {}
        self._structures = {}
        self._reconstruction_issues = {}
        self._visible_raw = {}
        self._parse_audit = {}

    # ------------------------------------------------------------------
    # 1) Data acquisition — build the document list
    # ------------------------------------------------------------------
    def acquire_local(self, pdf_paths: List[Path],
                      org_hint: Optional[str] = None,
                      org_id: Optional[str] = None,
                      year_hint: Optional[int] = None) -> List[DocumentRecord]:
        docs = []
        for p in pdf_paths:
            p = Path(p)
            if not p.exists():
                self.state.log_phase(self.name, "acquire", f"missing file skipped: {p}")
                continue
            docs.append(DocumentRecord(doc_id=p.name, local_path=str(p),
                                       org_hint=org_hint, org_id=org_id,
                                       year_hint=year_hint, source="local"))
        self.state.documents.extend(docs)
        self.state.log_phase(self.name, "acquire", f"{len(docs)} local document(s) listed")
        return docs

    def acquire_from_manifest(self, manifest_csv: Path, root: Path,
                              limit: Optional[int] = None) -> List[DocumentRecord]:
        """Ingest a manifest CSV of report PDFs (``local_path`` column; e.g. the
        ``search_manifest.csv`` written by ``acquire_from_search``), so a run is
        reproducible from a fixed document list without searching again.
        """
        docs: List[DocumentRecord] = []
        with Path(manifest_csv).open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lp = (row.get("local_path") or "").strip()
                if not lp:
                    continue
                path = (root / lp) if not Path(lp).is_absolute() else Path(lp)
                if not path.exists():
                    continue
                docs.append(DocumentRecord(
                    doc_id=path.name, local_path=str(path),
                    url=(row.get("pdf_url") or row.get("url") or None),
                    org_id=(row.get("org_id") or None),
                    org_hint=(row.get("brand_title") or row.get("org") or None),
                    year_hint=_safe_int(row.get("report_year") or row.get("year")),
                    source="manifest"))
                if limit is not None and len(docs) >= limit:
                    break
        self.state.documents.extend(docs)
        self.state.log_phase(self.name, "acquire",
                             f"{len(docs)} document(s) from manifest {Path(manifest_csv).name}")
        return docs

    def acquire_from_search(self, org_name: str, year: Optional[int], download_dir: Path,
                            max_reports: int = 1, org_id: Optional[str] = None,
                            searcher=None, downloader=None, identity_check=None,
                            backend: Optional[str] = None,
                            manifest_path: Optional[Path] = None) -> List[DocumentRecord]:
        """Find and download an organization's report PDFs (report_search.py).

        Default backend: an LLM with OpenRouter's web-search plugin; only the
        URLs the search tool returns are used. Every download must be a PDF
        whose first pages name the organization and the year. Each query, hit
        and decision is logged, and accepted files are written to a manifest
        so the run can be repeated offline."""
        from scope3trace.search_agent.report_search import make_searcher, search_reports, write_manifest
        def log(msg):
            self.state.log_phase(self.name, "search", msg)
            print(f"[search] {msg}", flush=True)
        try:
            searcher = searcher or make_searcher(backend)
            reports = search_reports(org_name, year, Path(download_dir), max_reports=max_reports,
                                     searcher=searcher, downloader=downloader, log=log,
                                     identity_check=identity_check)
        except Exception as e:  # missing key, quota, network
            log(f"{org_name} {year or ''}: report search failed ({e})")
            return []
        log(f"search backend {type(searcher).__name__} model={getattr(searcher, 'model', '-')} "
            f"calls={getattr(searcher, 'calls', 0)} tokens_in={getattr(searcher, 'tokens_in', 0)} "
            f"tokens_out={getattr(searcher, 'tokens_out', 0)}")
        if manifest_path is not None and reports:
            write_manifest(reports, Path(manifest_path))
        source = "openrouter_web_search" if type(searcher).__name__ == "OpenRouterWebSearch" else "google_search"
        docs = [DocumentRecord(doc_id=Path(r.local_path).name, local_path=r.local_path, url=r.url,
                               org_hint=org_name, org_id=org_id, year_hint=year, source=source)
                for r in reports]
        self.state.documents.extend(docs)
        log(f"{org_name} {year or ''}: {len(docs)} report(s) downloaded")
        return docs

    # ------------------------------------------------------------------
    # 2) Parsing & OCR toolset
    # ------------------------------------------------------------------
    def parse_pdf(self, doc: DocumentRecord) -> List[str]:
        """Return cleaned per-page text (cached per doc_id)."""
        if doc.doc_id in self._page_cache:
            return self._page_cache[doc.doc_id]
        try:
            try:
                from pypdf import PdfReader
            except ImportError:
                try:
                    from PyPDF2 import PdfReader  # type: ignore
                except ImportError:
                    PdfReader = None  # type: ignore
            if PdfReader is not None:
                reader = PdfReader(str(doc.local_path))
                raw_pages, slow_pages = _timed_page_texts(reader, str(doc.local_path))
                if slow_pages:
                    self.state.log_phase(self.name, "parse",
                                         f"{doc.doc_id}: pypdf text exceeded the per-page budget on pages "
                                         f"{slow_pages[:20]}; pdfium text used there")
                from scope3trace.search_agent.font_repair import repair_pages
                fffd_pages = repair_pages(raw_pages, str(doc.local_path))
                if fffd_pages:
                    self.state.log_phase(self.name, "parse",
                                         f"{doc.doc_id}: font maps digit separators to U+FFFD on "
                                         f"{len(fffd_pages)} page(s); glyph-name text used there")
            else:
                import pdfplumber  # type: ignore
                with pdfplumber.open(str(doc.local_path)) as pdf:
                    raw_pages = [p.extract_text() or "" for p in pdf.pages]
        except Exception as e:
            doc.status = "failed"
            self.state.log_phase(self.name, "parse", f"{doc.doc_id}: parse failed ({e})")
            self._page_cache[doc.doc_id] = []
            return []
        methods = ["pdf_text"] * len(raw_pages)
        for _n in (locals().get("slow_pages") or []):
            methods[_n - 1] = "pdfium_text_fallback"
        empty = [i + 1 for i, t in enumerate(raw_pages) if len(t.strip()) < 30]
        if empty:
            for page_no, text in self._ocr_pages(Path(doc.local_path), empty).items():
                raw_pages[page_no - 1] = text
                methods[page_no - 1] = "ocr"
            still_empty = [i + 1 for i, t in enumerate(raw_pages) if len(t.strip()) < 30]
            if still_empty:
                self.state.log_phase(self.name, "parse",
                                     f"{doc.doc_id}: {len(still_empty)} page(s) without "
                                     f"extractable text (OCR unavailable/failed): {still_empty[:10]}")
        original_raw_pages = list(raw_pages)
        self._visible_raw = {}
        from scope3trace.common.rules import infer_number_locale
        # Numeric convention (1.234,5 vs 1,234.5) is a document property.
        self._number_locale = infer_number_locale(raw_pages)
        from scope3trace.common.rules import page_number_locale
        self._page_locales = {i: page_number_locale(t, self._number_locale)
                              for i, t in enumerate(raw_pages, 1)}
        from scope3trace.common.rules import infer_period_end_year
        self._period_year = infer_period_end_year(raw_pages)
        tables_by_page = (
            self._reconstruct_tables(Path(doc.local_path), len(raw_pages))
            if self.table_reconstruction else {}
        )
        pages: List[str] = []
        structured: List[ParsedPage] = []
        for i, raw in enumerate(raw_pages, 1):
            raw = self._visible_raw.get(i, raw)
            tables = tables_by_page.get(i, [])
            structures = self._structures.get(i, [])
            table_text = format_structures(structures) if structures else _format_tables(tables)
            evidence_text = raw.rstrip() + (("\n\n" + table_text) if table_text else "")
            clean = clean_page_text(evidence_text)
            method = methods[i - 1] + ("+tables" if tables or structures else "")
            pages.append(evidence_text)
            structured.append(ParsedPage(
                doc_id=doc.doc_id, page=i, raw_text=raw, clean_text=clean,
                sentences=split_sentences(clean), content_tokens=content_tokens(clean),
                reconstructed_tables=tables, table_structures=structures,
                reconstruction_issues=self._reconstruction_issues.get(i, []), parse_method=method,
                original_extracted_text=original_raw_pages[i-1] if raw != original_raw_pages[i-1] else None,
                parse_audit=self._parse_audit, number_locale=self._page_locales.get(i, self._number_locale)))
        doc.n_pages = len(pages)
        doc.status = "parsed"
        self.state.parsed_pages[doc.doc_id] = structured
        from scope3trace.scope_agent.source_context import link_document_basis
        for page_no in link_document_basis(structured):
            parsed=structured[page_no-1]
            pages[page_no-1]=parsed.raw_text.rstrip()+'\n\n'+format_structures(parsed.table_structures)
            parsed.clean_text=clean_page_text(pages[page_no-1])
            parsed.sentences=split_sentences(parsed.clean_text)
            parsed.content_tokens=content_tokens(parsed.clean_text)
        self._page_cache[doc.doc_id] = pages
        return pages

    def refresh_evidence(self, doc: DocumentRecord, page_numbers: List[int]) -> List[EvidenceItem]:
        """Rebuild requested pages from physical PDF words, then expose full text.

        Repair does not alter the original linear text or reuse an LLM table.
        The orchestrator retains each previous page representation in its audit.
        """
        selected = set(page_numbers)
        items = []
        with open_geometry(doc.local_path, doc.n_pages) as (pdf, audit):
            for parsed in self.state.parsed_pages.get(doc.doc_id, []):
                if parsed.page not in selected:
                    continue
                if self.table_reconstruction:
                    vpage = visible_page(pdf.pages[parsed.page-1])
                    vpage._s3t_period_year = getattr(self, "_period_year", None)
                    _prev = next((q for q in self.state.parsed_pages.get(doc.doc_id, []) if q.page == parsed.page - 1), None)
                    vpage._s3t_prev_tables = (_prev.table_structures or None) if _prev is not None else None
                    parsed.table_structures, parsed.reconstruction_issues = reconstruct_page(
                        vpage,doc.doc_id,parsed.page,
                        number_locale=getattr(parsed, "number_locale", None))
        from scope3trace.scope_agent.source_context import link_document_basis
        selected.update(link_document_basis(self.state.parsed_pages.get(doc.doc_id,[])))
        for parsed in self.state.parsed_pages.get(doc.doc_id,[]):
            if parsed.page in selected:
                text = parsed.raw_text.rstrip()
                if parsed.table_structures:
                    text += '\n\n'+format_structures(parsed.table_structures)
                item = EvidenceItem(doc.doc_id,parsed.page,
                                    content_type='table' if parsed.table_structures or parsed.reconstructed_tables else 'text',
                                    text=text,localized_text=text,
                                    reason='bounded repair: full source page and refreshed physical table',
                                    method='repair_full_page')
                items.append(item)
                if doc.doc_id in self._page_cache:
                    self._page_cache[doc.doc_id][parsed.page-1]=text
        old = self.state.evidence.get(doc.doc_id,[])
        self.state.evidence[doc.doc_id] = sorted([it for it in old if it.page not in selected]+items,key=lambda it:it.page)
        self.state.log_phase(self.name,'repair',f'{doc.doc_id}: refreshed pages {sorted(selected)}')
        return items

    # Table reconstruction is stage 3 of the paper's pipeline (Scope Agent,
    # scope3trace.scope_agent.table_structure). It runs here, during parsing,
    # in the same pdfplumber page-layout pass that checks the visible page
    # text; the Search Agent attaches the reconstructed tables to its verbatim
    # evidence list and performs no extraction or computation on them.
    def _reconstruct_tables(self, pdf_path: Path, n_pages: int) -> Dict[int, List[List[List[str]]]]:
        """Deterministic table-structure reconstruction using pdfplumber."""
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            return {}
        out: Dict[int, List[List[List[str]]]] = {}
        self._structures, self._reconstruction_issues = {}, {}
        try:
            with open_geometry(pdf_path,n_pages) as (pdf,audit):
                self._parse_audit = audit
                only = getattr(self, "reconstruct_only", None)
                env_pages = os.environ.get("SCOPE3TRACE_RECONSTRUCT_PAGES", "").strip()
                if only is None and env_pages:
                    only = {int(x) for x in re.findall(r"\d+", env_pages)}
                if only:
                    # a table continued from the previous page needs that page too
                    only = set(only) | {n - 1 for n in only}
                prev_source = None
                for page_no, source_page in enumerate(pdf.pages, 1):
                    # free the previous page's parsed objects (large reports
                    # otherwise hold every page's characters in memory)
                    if prev_source is not None and hasattr(prev_source, "close"):
                        try:
                            prev_source.close()
                        except Exception:
                            pass
                    prev_source = source_page
                    if only and page_no not in only:
                        continue
                    page = visible_page(source_page)
                    # Off-page spread contents and overprint glyphs cannot be
                    # used as evidence for this visible PDF page.
                    if len(page.chars) != len(source_page.chars):
                        self._visible_raw[page_no] = page.extract_text() or ""
                    try:
                        page_text = page.extract_text() or ""
                        if page_keyword_score(page_text) <= 0:
                            continue
                        page._s3t_period_year = getattr(self, "_period_year", None)
                        page._s3t_prev_tables = self._structures.get(page_no - 1) or None
                        structures, issues = reconstruct_page(page, pdf_path.name, page_no,
                                                              number_locale=getattr(self, "_page_locales", {}).get(
                                                                  page_no, getattr(self, "_number_locale", None)))
                        self._structures[page_no] = structures
                        self._reconstruction_issues[page_no] = issues
                    except Exception as exc:
                        self._reconstruction_issues[page_no] = [{"error": type(exc).__name__, "message": str(exc)}]
                        continue
                    tables = []
                    try:
                        extracted = page.extract_tables() or []
                        if not structures:
                            # Keep both fallback candidates for audit. Neither
                            # is sufficient for a verified numeric binding.
                            extracted += page.extract_tables({
                                "vertical_strategy": "text",
                                "horizontal_strategy": "text",
                                "intersection_tolerance": 5,
                                "snap_tolerance": 3,
                                "join_tolerance": 3,
                                "min_words_vertical": 2,
                                "min_words_horizontal": 1,
                            }) or []
                    except Exception as exc:
                        self._reconstruction_issues.setdefault(page_no, []).append(
                            {'error':type(exc).__name__,'message':str(exc),'stage':'legacy_table_candidates'})
                        extracted = []
                    for table in extracted:
                        rows = [[clean_page_text(str(cell or "")) for cell in row]
                                for row in table if row]
                        if rows:
                            tables.append(rows)
                    if tables:
                        out[page_no] = tables
        except Exception as exc:
            self._parse_audit["error"] = type(exc).__name__+": "+str(exc)
            # Keep successfully reconstructed pages if a later page fails.
            self.state.log_phase(self.name, "reconstruct", f"{pdf_path.name}: partial reconstruction; inspect per-page issues")
        return out

    @staticmethod
    def _ocr_pages(pdf_path: Path, page_numbers: List[int]) -> Dict[int, str]:
        """OCR selected pages without making Python OCR wrappers mandatory.

        Prefer ``pytesseract``/``pdf2image`` when installed, then fall back to
        the underlying ``pdftoppm`` and ``tesseract`` executables.  The latter
        path is important in minimal evaluation environments where Poppler and
        Tesseract are available but their Python wrappers are not.
        """
        out: Dict[int, str] = {}
        try:
            import pytesseract  # type: ignore
            from pdf2image import convert_from_path  # type: ignore
        except ImportError:
            pytesseract = None  # type: ignore
            convert_from_path = None  # type: ignore
        if pytesseract is not None and convert_from_path is not None:
            for no in page_numbers:
                try:
                    images = convert_from_path(
                        str(pdf_path), first_page=no, last_page=no, dpi=200)
                    if images:
                        text = pytesseract.image_to_string(images[0])
                        if text.strip():
                            out[no] = text
                except Exception:
                    continue

        missing = [no for no in page_numbers if no not in out]
        if not missing:
            return out
        pdftoppm = _find_executable("pdftoppm", "SCOPE3TRACE_PDFTOPPM")
        tesseract = _find_executable("tesseract", "SCOPE3TRACE_TESSERACT")
        if not pdftoppm or not tesseract:
            return out
        for no in missing:
            try:
                with tempfile.TemporaryDirectory(prefix="scope3trace_ocr_") as tmp:
                    prefix = Path(tmp) / f"page_{no}"
                    subprocess.run(
                        [pdftoppm, "-f", str(no), "-l", str(no), "-r", "200",
                         "-png", "-singlefile", str(pdf_path), str(prefix)],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=120)
                    image_path = prefix.with_suffix(".png")
                    proc = subprocess.run(
                        [tesseract, str(image_path), "stdout"],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, timeout=120)
                    if proc.stdout.strip():
                        out[no] = proc.stdout
            except (OSError, subprocess.SubprocessError):
                continue
        return out

    def page_text(self, doc_id: str, page: int) -> str:
        """Verbatim page text for downstream deterministic evidence checks."""
        pages = self._page_cache.get(doc_id) or []
        return pages[page - 1] if 0 < page <= len(pages) else ""

    # ------------------------------------------------------------------
    # 3) Localization — return the evidence list
    # ------------------------------------------------------------------
    def run(self, doc: DocumentRecord) -> List[EvidenceItem]:
        """Search one document and return its verbatim evidence list."""
        pages = self.parse_pdf(doc)
        if not pages:
            self.state.evidence[doc.doc_id] = []
            return []
        located = self._localize(pages)
        items = []
        for no, ctype, reason in located:
            if not pages[no - 1].strip():
                continue
            localized_text, sentence_ids = self._localize_sentences(
                doc.doc_id, no, pages[no - 1])
            if self.rule_only:
                method = "rule_keyword"
            elif not self.page_localization:
                method = "page_localization_ablation"
            elif self.sentence_localization:
                method = "llm_page_sentence_classification"
            else:
                method = "llm_page_classification"
            items.append(EvidenceItem(
                doc_id=doc.doc_id, page=no, content_type=ctype,
                text=pages[no - 1], localized_text=localized_text,
                sentence_ids=sentence_ids, reason=reason, method=method,
            ))
        self.state.evidence[doc.doc_id] = items
        n_sentences = sum(len(item.sentence_ids) for item in items)
        self.state.log_phase(self.name, "localize",
                             f"{doc.doc_id}: returned {len(items)} evidence item(s) "
                             f"and {n_sentences} localized sentence(s) from {len(pages)} pages "
                             f"({'rule-only' if self.rule_only else self.llm.model})")
        return items

    def _localize(self, pages: List[str]) -> List[Tuple[int, str, Optional[str]]]:
        """[(page_no, content_type, reason)] for emissions-relevant pages."""
        keyword_candidates = [(i + 1, t) for i, t in enumerate(pages)
                              if page_keyword_score(t) > 0]
        candidates = [(i + 1, t) for i, t in enumerate(pages) if t.strip()]
        if not self.page_localization:
            return [(no, "table" if "[RECONSTRUCTED_TABLE" in text else "text",
                     "page localization disabled for ablation")
                    for no, text in candidates]
        if self.rule_only:
            selected = keyword_candidates or candidates
            return [(no, "table" if "[RECONSTRUCTED_TABLE" in t else "text",
                     f"keyword score {page_keyword_score(t)}") for no, t in selected]

        # Keyword-positive pages are retained regardless of the semantic
        # classifier's response. The LLM can add pages, but cannot reduce the
        # deterministic high-recall set.
        located: List[Tuple[int, str, Optional[str]]] = [
            (no, "table" if "[RECONSTRUCTED_TABLE" in t else "text",
             f"keyword score {page_keyword_score(t)}") for no, t in keyword_candidates]
        for start in range(0, len(candidates), LOCALIZE_PAGES_PER_CALL):
            batch = candidates[start:start + LOCALIZE_PAGES_PER_CALL]
            pages_block = "\n\n".join(
                f"[page {no}]\n{_localization_excerpt(text, LOCALIZE_CHARS_PER_PAGE)}"
                for no, text in batch)
            obj = self.llm.complete_json(
                SYSTEM_COMMON, PAGE_LOCALIZATION_PROMPT.format(pages_block=pages_block))
            if not obj:  # recall-preserving fallback: keep the whole batch
                located.extend((no, "table", "llm unavailable; kept by prefilter")
                               for no, _ in batch)
                continue
            batch_nos = {no for no, _ in batch}
            for item in obj.get("relevant_pages", []) or []:
                try:
                    no = int(item.get("page"))
                except (TypeError, ValueError):
                    continue
                if no in batch_nos:
                    located.append((no, str(item.get("content_type") or "text"),
                                    str(item.get("reason") or "")[:120] or None))
        seen, out = set(), []
        for no, ctype, reason in located:
            if no not in seen:
                seen.add(no)
                out.append((no, ctype, reason))
        return sorted(out, key=lambda t: t[0])

    def _localize_sentences(self, doc_id: str, page_no: int,
                            page_text: str) -> Tuple[Optional[str], List[int]]:
        """Return verbatim sentence/table content selected inside one page.

        The complete page remains in ``EvidenceItem.text`` and is used by the
        Verify Agent.  The Scope Agent receives this narrower semantic view,
        matching the paper's page- and sentence-level localization stages.
        """
        if self.rule_only or not self.sentence_localization:
            return None, []
        parsed_pages = self.state.parsed_pages.get(doc_id, [])
        parsed = next((p for p in parsed_pages if p.page == page_no), None)
        if parsed is None or not parsed.sentences:
            return None, []

        # A table is an atomic evidence block: labels, units and year headers
        # may be far from the numeric rows (or emitted after them by the PDF
        # parser). A reconstructed table does not replace the original layout.
        if parsed.reconstructed_tables or parsed.table_structures or (
                page_keyword_score(page_text) > 0
                and re.search(r"20\d{2}\s+20\d{2}", page_text)
                and sum(bool(re.search(r"\d[.,\d]*\s+\d", line))
                        for line in page_text.splitlines()) >= 2):
            return page_text, list(range(1, len(parsed.sentences) + 1))

        base = {i for i, sentence in enumerate(parsed.sentences, 1)
                if page_keyword_score(sentence) > 0}
        # Classify all sentences in bounded batches. Keyword prefiltering used
        # to hide valid category rows from the classifier altogether.
        candidate_ids = set(range(1, len(parsed.sentences) + 1))
        batches, lines, used = [], [], 0
        for idx in sorted(candidate_ids):
            line = f"[sentence {idx}] {parsed.sentences[idx - 1]}"
            if used + len(line) > 16_000 and lines:
                batches.append(lines)
                lines, used = [], 0
            lines.append(line)
            used += len(line)
        if lines:
            batches.append(lines)
        selected = set(base)
        for lines in batches:
            batch_ids = {int(re.match(r"\[sentence (\d+)\]", line).group(1)) for line in lines}
            obj = self.llm.complete_json(
                SYSTEM_COMMON,
                SENTENCE_LOCALIZATION_PROMPT.format(
                    page_no=page_no, sentences_block="\n".join(lines)),
                max_tokens=1200)
            if obj is None:
                return None, []  # recall-preserving full-page fallback
            for raw in obj.get("relevant_sentence_ids", []) or []:
                try:
                    idx = int(raw)
                except (TypeError, ValueError):
                    continue
                if idx in batch_ids:
                    selected.add(idx)
        selected = sorted(selected)

        parts = [parsed.sentences[idx - 1] for idx in selected]
        if parsed.reconstructed_tables:
            parts.append(_format_tables(parsed.reconstructed_tables))
        localized = "\n".join(part for part in parts if part.strip()).strip()
        return (localized or None), selected


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _find_executable(name: str, env_name: str) -> Optional[str]:
    """Resolve an OCR executable from an explicit override or the PATH."""
    override = os.environ.get(env_name)
    if override and Path(override).is_file():
        return override
    return shutil.which(name)


def _format_tables(tables: List[List[List[str]]]) -> str:
    blocks = []
    for idx, table in enumerate(tables, 1):
        rows = [" | ".join(cell for cell in row) for row in table]
        blocks.append(f"[RECONSTRUCTED_TABLE {idx}]\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def _localization_excerpt(text: str, limit: int) -> str:
    """Return keyword-centred page excerpts rather than a lossy page prefix."""
    if len(text) <= limit:
        return text
    lowered = text.lower()
    needles = ("scope 1", "scope 2", "scope 3", "tco2", "co2e", "greenhouse", "emission")
    positions = sorted({lowered.find(n) for n in needles if lowered.find(n) >= 0})
    if not positions:
        return text[:limit]
    window = max(180, limit // max(1, min(len(positions), 3)))
    chunks = []
    for pos in positions[:3]:
        start = max(0, pos - window // 3)
        chunks.append(text[start:start + window])
    return "\n...\n".join(chunks)[:limit]
