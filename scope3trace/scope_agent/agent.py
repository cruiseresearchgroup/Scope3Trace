#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scope Agent — extraction & estimation.

Everything between the Search Agent's evidence list and the Verify Agent's
checks happens here:

  1. Asset Pool          : compiles the building asset inventory from public
                           registries / modeled energy datasets / PDS
                           annotations (flowchart "Asset Pool");
  2. Extraction          : interprets the Search Agent's verbatim evidence
                           items ("Table Construction", "Find Scope Info &
                           Extraction") into structured org-level disclosures.
                           The LLM is bounded to semantic table interpretation
                           (copy numbers verbatim); numeric parsing, unit
                           normalization to tCO2e, and GHG-Protocol category
                           mapping are pure rules (rules.py), and every value
                           is span-grounded to its evidence item;
  3. Entity linking      : maps asset names mentioned in reports to Asset
                           Pool entries (deterministic key match → fuzzy →
                           bounded LLM adjudication for the ambiguous band);
  4. Estimation          : deterministic building-level explanatory components
                           per Appendix C (C.1 Scope 2 = kWh × EF_grid;
                           C.2 Cat 2 = floor_area × EF_embodied / 1000;
                           C.3 Cat 13 = tenant kWh × EF_grid), never
                           overwriting reported values, always labeled with
                           provenance / method / factor source (C.4).

The Verify Agent later re-checks every extracted value against the evidence
and gates the estimates with semantic relevance validation — it can flag but
never correct (Appendix E).

The building inventory itself (geocoding, satellite tiles, floor-area merge)
is assembled upstream; this agent ingests the resulting v2.1 CSV.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scope3trace.common.config import (
    DEFAULT_GRID_FACTOR_SOURCE,
    DEFAULT_GRID_FACTOR_T_PER_KWH,
    EMBODIED_FACTORS_KG_PER_SQM,
    EXTRACTION_MAX_OUTPUT_TOKENS,
    LINK_ACCEPT_SCORE,
    LINK_CANDIDATE_SCORE,
)
from scope3trace.common.llm import BoundedLLM
from scope3trace.common.prompts import (
    BUILDING_TYPE_PROMPT,
    ENTITY_LINK_PROMPT,
    SYSTEM_COMMON,
    TABLE_EXTRACTION_PROMPT,
)
from scope3trace.common.rules import (
    build_asset_key,
    category_hierarchy,
    extract_postcode_from_address,
    ground_evidence_snippet,
    map_scope3_category,
    normalize_emission_to_tco2e,
    normalize_string,
    parse_number,
    table_year_check,
    verify_value_in_text,
    verify_field_context,
)
from scope3trace.common.state import (
    AssetRecord,
    DerivedComponent,
    DocumentRecord,
    EntityLink,
    EvidenceItem,
    EvidenceLink,
    ExtractedValue,
    OrgDisclosure,
    SharedState,
)

_CANONICAL_TYPES = [t for t in EMBODIED_FACTORS_KG_PER_SQM if t != "Default"]
_YEAR_RE = re.compile(r"\b(20[0-3]\d)\b")

_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "Office": ["office", "commercial office", "tower"],
    "Retail": ["retail", "shop", "mall", "shopping"],
    "Residential Apartment": ["apartment", "residential", "flat"],
    "House/Townhouse": ["house", "townhouse", "dwelling"],
    "Warehouse": ["warehouse", "logistics", "distribution centre", "distribution center"],
    "Storage": ["storage", "self storage"],
    "Industrial": ["industrial", "factory", "manufacturing", "plant"],
    "Hospital/Clinic": ["hospital", "clinic", "medical", "health"],
    "Education": ["school", "university", "campus", "education", "college"],
    "Public Display Area": ["museum", "gallery", "exhibition", "library"],
    "Parking": ["car park", "parking", "garage"],
}


class ScopeAgent:
    """Extraction & estimation over the shared state."""

    name = "scope"

    def __init__(self, llm: BoundedLLM, state: SharedState, rule_only: bool = False,
                 llm_classify_types: bool = False,
                 evidence_grounding: bool = True,
                 table_complement: bool = True) -> None:
        self.llm = llm
        self.state = state
        self.rule_only = rule_only or (
            not llm.available and not getattr(llm, "strict_failures", False))
        self.evidence_grounding = evidence_grounding
        # Deterministic complement: verified reconstructed-table cells the
        # model skipped in a table it already read (e.g. printed zeros).
        self.table_complement = table_complement
        try:
            from scope3trace.scope_agent.chart_reader import ChartReader
            self.chart_reader = ChartReader.from_env(state) if not getattr(self, "rule_only", False) else None
        except Exception as exc:  # the reader is optional; never block extraction
            print(f"[WARN] chart reader unavailable: {exc}")
            self.chart_reader = None
        # LLM building-type classification is one bounded call PER unlabeled
        # asset — opt-in, or the pool build stalls on large inventories.
        self.llm_classify_types = llm_classify_types and not self.rule_only

    # ==================================================================
    # 1) Asset Pool (§3.3 construction phase 1)
    # ==================================================================
    def build_asset_pool(self, inventory_csv: Path,
                         entity_csv: Optional[Path] = None,
                         limit: Optional[int] = None) -> int:
        """Ingest a v2.1 building inventory CSV (long, property_id × year).

        ``entity_csv`` optionally merges facility aliases
        (facility_name -> responsible_entity, building_owner).
        """
        import pandas as pd

        df = pd.read_csv(inventory_csv, low_memory=False)
        alias_map = self._load_entity_aliases(entity_csv) if entity_csv else {}

        assets: Dict[str, AssetRecord] = {}
        n_rows = 0
        for _, row in df.iterrows():
            pid = _s(row.get("property_id"))
            if not pid:
                continue
            rec = assets.get(pid)
            if rec is None:
                if limit is not None and len(assets) >= limit:
                    continue
                rec = self._new_asset(pid, row, alias_map)
                assets[pid] = rec
            self._add_year_activity(rec, row)
            n_rows += 1

        self.state.asset_pool = list(assets.values())
        self._classify_building_types()
        self.state.log_phase(self.name, "asset_pool",
                             f"{len(assets)} assets from {n_rows} rows ({Path(inventory_csv).name})"
                             + (f" + aliases from {Path(entity_csv).name}" if entity_csv else ""))
        return len(assets)

    def _new_asset(self, pid: str, row: Any, alias_map: Dict[str, List[str]]) -> AssetRecord:
        name = _s(row.get("building_name"))
        address_raw = _s(row.get("address_standardized")) or _s(row.get("address"))
        postcode = extract_postcode_from_address(address_raw or "")
        names = [n for n in [name] if n]
        for alias in alias_map.get(normalize_string(name or ""), []):
            if alias and alias not in names:
                names.append(alias)
        floor_area, fa_source = self._pick_floor_area(row)
        lat, lon = _pick_coordinates(row)
        rec = AssetRecord(
            property_id=pid,
            names=names,
            address_raw=address_raw,
            address_std=normalize_string(address_raw or "") or None,
            postcode=postcode,
            grid_region=_s(row.get("state")) or _s(row.get("region"))
                        or _infer_au_state(address_raw),
            lat=lat,
            lon=lon,
            building_type=_s(row.get("primary_usage")),
            building_type_method="inventory" if _s(row.get("primary_usage")) else None,
            floor_area_m2=floor_area,
            floor_area_source=fa_source,
            year_built=_i(row.get("year_built")),
            satellite_image=_s(row.get("image_path")),
            org_id=_s(row.get("org_id")),
        )
        for fld, col in (("floor_area_m2", "floor_area_source"),
                         ("scope2", "scope2_quality"), ("scope3", "scope3_quality")):
            q = _s(row.get(col))
            if q:
                rec.provenance[fld] = q
        return rec

    @staticmethod
    def _pick_floor_area(row: Any) -> Tuple[Optional[float], Optional[str]]:
        for col, src in (("floor_area_sqm", "reported"),
                         ("gba_floor_area_final_m2", "modeled:gba"),
                         ("floor_area_sqm_model", "modeled")):
            v = _f(row.get(col))
            if v:
                return v, _s(row.get("floor_area_source")) or src
        return None, None

    @staticmethod
    def _add_year_activity(rec: AssetRecord, row: Any) -> None:
        year = _i(row.get("report_year"))
        if year is None:
            return
        rec.activity[str(year)] = {
            "electricity_kwh": _f(row.get("energy_electricity_kwh"))
                               if _f(row.get("energy_electricity_kwh")) is not None
                               else _f(row.get("energy_total_kwh")),
            "tenant_electricity_kwh": _f(row.get("tenant_electricity_kwh")),
            "p_com": _f(row.get("p_com")),
            "energy_source": _s(row.get("energy_data_source")),
            "scope1_tco2e_reported": _f(row.get("scope1_tco2e_value")),
            "scope2_location_tco2e_reported": _f(row.get("scope2_location_based_tco2e_value")),
            "scope2_market_tco2e_reported": _f(row.get("scope2_market_based_tco2e_value")),
            "scope2_quality": _s(row.get("scope2_quality")),
            "scope3_cat2_tco2e": _f(row.get("scope3_cat2_embodied_tco2e")),
            "scope3_cat2_source": _s(row.get("scope3_cat2_source")),
            "scope3_cat2_method": _s(row.get("scope3_cat2_method")),
            "scope3_cat2_quality": _s(row.get("scope3_cat2_quality")),
            "scope3_cat13_tco2e": _f(row.get("scope3_cat13_leased_assets_tco2e")),
            "scope3_cat13_source": _s(row.get("scope3_cat13_source")),
            "scope3_cat13_method": _s(row.get("scope3_cat13_method")),
            "scope3_cat13_quality": _s(row.get("scope3_cat13_quality")),
        }

    @staticmethod
    def _load_entity_aliases(entity_csv: Path) -> Dict[str, List[str]]:
        """facility_name (normalized) → [responsible_entity, building_owner]."""
        import csv as _csv
        aliases: Dict[str, List[str]] = {}
        with Path(entity_csv).open("r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                key = normalize_string(row.get("facility_name") or "")
                if not key:
                    continue
                vals = aliases.setdefault(key, [])
                for col in ("responsible_entity", "building_owner"):
                    v = (row.get(col) or "").strip()
                    if v and v not in vals:
                        vals.append(v)
        return aliases

    # -- text classification module (building typology) -----------------
    def _classify_building_types(self) -> None:
        n_rule = n_llm = 0
        for rec in self.state.asset_pool:
            if rec.building_type and rec.building_type_method == "inventory":
                canon = self._rule_type(rec.building_type)
                if canon:
                    rec.building_type = canon
                continue
            desc = " ".join(filter(None, [" ".join(rec.names), rec.address_raw or ""]))
            canon = self._rule_type(desc)
            if canon:
                rec.building_type, rec.building_type_method = canon, "rule_classifier"
                n_rule += 1
            elif self.llm_classify_types and desc.strip():
                obj = self.llm.complete_json(
                    SYSTEM_COMMON, BUILDING_TYPE_PROMPT.format(description=desc[:400]))
                label = (obj or {}).get("building_type")
                if label in _CANONICAL_TYPES:
                    rec.building_type, rec.building_type_method = label, "llm_classifier"
                    n_llm += 1
        if n_rule or n_llm:
            self.state.log_phase(self.name, "classify",
                                 f"building types filled: {n_rule} by rules, {n_llm} by LLM")

    @staticmethod
    def _rule_type(text: str) -> Optional[str]:
        t = normalize_string(text or "")
        if not t:
            return None
        for label, kws in _TYPE_KEYWORDS.items():
            if any(kw in t for kw in kws):
                return label
        for label in _CANONICAL_TYPES:
            if normalize_string(label) == t:
                return label
        return None

    # ==================================================================
    # 2) Extraction from the Search Agent's evidence list
    # ==================================================================
    def extract_disclosures(self, doc: DocumentRecord,
                            evidence_items: List[EvidenceItem],
                            feedback_by_page=None, extraction_round: int = 0,
                            append: bool = True) -> List[OrgDisclosure]:
        """Extract one organization record per reporting year.

        Multi-year tables are grouped by each value's explicit year, preserving
        historical observations instead of overwriting them with the latest
        column (Appendix H.6).
        """
        rows: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {"org_name": doc.org_hint, "reporting_year": doc.year_hint,
                                "reporting_boundary": None, "certification": None,
                                "restated": False, "superseded_by": None,
                                "asset_mentions": []}
        for item in evidence_items:
            if not item.text.strip():
                continue
            if self.rule_only:
                rows.extend(self._rule_extract_item(item))
                continue
            repair_context = ''
            if feedback_by_page and item.page in feedback_by_page:
                import json
                repair_context = ('\nVALIDATION FEEDBACK (diagnostics, not reference answers):\n' +
                                  json.dumps(feedback_by_page[item.page],ensure_ascii=False) +
                                  '\nRe-read the original page and table. Return fresh source-backed candidates, '
                                  'including omitted rows and years. Never guess a number, sum a total, '
                                  'or treat a previous rejected candidate as verified. If unresolved, omit it.')
            obj = self._extract_page(item.page, (item.localized_text or item.text), repair_context)
            if not obj:
                rows.extend(self._rule_extract_item(item))
                continue
            for key in ("org_name", "reporting_year", "reporting_boundary", "certification"):
                if obj.get(key) and not meta.get(key):
                    meta[key] = obj[key]
            if obj.get("restated") is True:
                meta["restated"] = True
            if obj.get("superseded_by") and not meta.get("superseded_by"):
                meta["superseded_by"] = str(obj["superseded_by"])
            for m in obj.get("asset_mentions", []) or []:
                if isinstance(m, dict) and m.get("mention"):
                    meta["asset_mentions"].append(
                        {"mention": str(m["mention"]), "page": item.page,
                         "snippet": str(m.get("snippet") or "")[:300]})
            for v in obj.get("values", []) or []:
                if isinstance(v, dict):
                    v["page"] = item.page
                    if v.get("reporting_year") is None:
                        v["reporting_year"] = obj.get("reporting_year")
                    rows.append(v)
        if meta.get("reporting_year") is None:
            meta["reporting_year"] = self._guess_year(evidence_items)
        if self.table_complement and not self.rule_only:
            rows.extend(self._table_complement_rows(doc.doc_id, rows))
        explicit_boundary = self._boundary_from_evidence(evidence_items)
        if explicit_boundary:
            # The schema field is the GHG consolidation approach, not a generic
            # description of which subsidiaries appear in the report.
            meta["reporting_boundary"] = explicit_boundary
        evidence_by_page = {it.page: it.text for it in evidence_items}
        grouped: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for row in rows:
            year = self._row_year(row, _i(meta.get("reporting_year")))
            grouped.setdefault(year, []).append(row)
        if not grouped:
            grouped[_i(meta.get("reporting_year"))] = []

        # Normalize every row first: a value bound to a reconstructed cell
        # takes the cell's column year, so records are grouped afterwards.
        normalized: Dict[Optional[int], List[Any]] = {y: [] for y in grouped}
        for year, year_rows in grouped.items():
            for row in year_rows:
                ev = self._normalize_row(doc.doc_id, row, evidence_by_page, year)
                if ev is not None:
                    normalized.setdefault(ev.report_year if ev.report_year is not None else year, []).append(ev)

        # Charts and raster images: printed data labels read by a vision
        # backbone, each number checked on the page (chart_reader.py)
        if not self.rule_only and getattr(self, "chart_reader", None) is not None:
            for ev in self.chart_reader.read_document(doc, [it.page for it in evidence_items]):
                ev.evidence.extraction_round = extraction_round
                normalized.setdefault(ev.report_year, []).append(ev)

        disclosures: List[OrgDisclosure] = []
        for year, year_evs in sorted(normalized.items(), key=lambda kv: kv[0] or 0):
            disclosure = OrgDisclosure(
                org_id=doc.org_id,
                org_name=_s(meta.get("org_name")), reporting_year=year,
                reporting_boundary=_s(meta.get("reporting_boundary")),
                certification=_s(meta.get("certification")),
                source_document=doc.doc_id, restated=bool(meta.get("restated")),
                superseded_by=_s(meta.get("superseded_by")),
                asset_mentions=list(meta["asset_mentions"]),
            )
            for ev in year_evs:
                if ev is not None:
                    ev.evidence.extraction_round = extraction_round
                    if ev.value_kind == "component":
                        disclosure.category_components.append(ev)
                    else:
                        disclosure.values.append(ev)
            # Preserve all candidates through verification. An early page's
            # rejected amount must not erase a later supported table cell.
            n_g = sum(1 for v in disclosure.values if v.evidence.verified)
            self.state.log_phase(
                self.name, "extract",
                f"{doc.doc_id}@{year}: {len(disclosure.values)} values "
                f"({n_g} span-grounded), {len(disclosure.category_components)} category components, "
                f"{len(disclosure.asset_mentions)} asset mentions")
            if append:
                self.state.org_disclosures.append(disclosure)
            disclosures.append(disclosure)
        return disclosures

    # Fields the deterministic table complement may add (Table 1 fields).
    _COMPLEMENT_FIELDS = {"scope1", "scope2_location", "scope2_market", "scope3_total"} | {
        f"scope3_cat{i}" for i in range(1, 16)}

    def _table_complement_rows(self, doc_id: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rows of a reconstructed table the model read but did not extract.

        The model interprets a page; the geometry already holds every row of
        the tables on it. When the model extracted at least one amount of a
        reconstructed table, the table's other inventory rows (same reporting
        years, absolute glyph-verified unit, reported boundary) are proposed
        for the fields the model left empty. They pass through the same
        binding, grounding and Verify checks as model rows.
        """
        from scope3trace.common.rules import parse_number as _pn
        parsed = {p.page: p for p in self.state.parsed_pages.get(doc_id, [])}
        def _year(r):
            return _i(r.get("reporting_year"))
        have = set()
        for r in rows:
            sc = str(r.get("scope") or "").lower()
            if sc == "scope3_category":
                no = map_scope3_category(r.get("category_label"))[0]
                if no:
                    have.add((f"scope3_cat{no}", _year(r)))
            elif sc:
                have.add((sc, _year(r)))
        by_page = defaultdict(list)
        for r in rows:
            if _i(r.get("page")) is not None:
                by_page[_i(r.get("page"))].append(r)
        out, added = [], set()
        for page_no, page_rows in by_page.items():
            pp = parsed.get(page_no)
            if not pp or not pp.table_structures:
                continue
            locale = getattr(pp, "number_locale", None)
            for t in pp.table_structures:
                cols = {c["id"]: c for c in t["columns"]}
                # the model used this table: one of its amounts is a model value
                used_years = set()
                for r in page_rows:
                    v = _pn(r.get("value_raw"), locale)
                    if v is None:
                        continue
                    for tr in t["rows"]:
                        for c in tr["cells"]:
                            col = cols.get(c["column_id"])
                            if col and col.get("role") == "reported" and _pn(c["raw"], locale) == v:
                                used_years.add(col["year"])
                if not used_years:
                    continue
                for tr in t["rows"]:
                    f = tr.get("field")
                    if (f not in self._COMPLEMENT_FIELDS or tr.get("value_kind") != "category_total"
                            or tr.get("quantity_type") != "inventory"
                            or (tr.get("coverage_boundary") or "reported") != "reported"
                            or tr.get("organization_boundary") in ("individual_entity",)
                            or not tr.get("unit") or tr["unit"].get("evidence_status") != "verified"
                            or not tr["unit"].get("multiplier")):
                        continue
                    # the row's own label must name the field: no inherited
                    # section/category, no breakdown or non-CO2e quantity
                    from scope3trace.scope_agent.table_structure import _category as _tcat, _field as _tfield, _explicit_category_marker as _xcat
                    lab = tr.get("label") or ""
                    if f.startswith("scope3_cat"):
                        if _tcat(lab) != int(f[10:]):
                            continue
                    elif _tfield(lab) != f:
                        continue
                    if re.search(r"(?i)\((?:t|kg|m3|mwh|gj|%)\)|landfill|waste sent|intensity|per\b|scope\s*1\s*\+", lab):
                        continue
                    for c in tr["cells"]:
                        col = cols.get(c["column_id"])
                        if not col or col.get("role") != "reported" or col.get("year") not in used_years:
                            continue
                        raw_c = str(c["raw"]).strip()
                        if col.get("boundary") == "SBTi" or _pn(raw_c, locale) is None:
                            continue
                        # footnote references "(2)", year tokens, and zeros
                        # without an explicit category marker are not amounts
                        if (re.fullmatch(r"\(\d{1,2}\)", raw_c) or re.fullmatch(r"(?:19|20)\d{2}", raw_c)
                                or (_pn(raw_c, locale) == 0 and not _xcat(lab))):
                            continue
                        key = (f, col["year"])
                        if key in have or key in added:
                            continue
                        added.add(key)
                        row = {"scope": "scope3_category" if f.startswith("scope3_cat") else f,
                               "label_raw": tr["label"], "value_raw": c["raw"],
                               "unit_raw": tr["unit"].get("text"), "reporting_year": col["year"],
                               "value_kind": "category_total", "page": page_no,
                               "table_binding": {"table_id": t["id"], "row_id": tr["id"], "column_id": col["id"]},
                               "_table_complement": True}
                        if f.startswith("scope3_cat"):
                            row["category_label"] = f"Category {f[10:]} {tr['label']}"
                        out.append(row)
        if out:
            self.state.log_phase(self.name, "extract",
                                 f"{doc_id}: {len(out)} table-complement candidate(s) from model-read tables")
        return out

    def extract_disclosure(self, doc: DocumentRecord,
                           evidence_items: List[EvidenceItem]) -> OrgDisclosure:
        """Backward-compatible single-record wrapper."""
        disclosures = self.extract_disclosures(doc, evidence_items)
        preferred = next((d for d in disclosures if d.reporting_year == doc.year_hint), None)
        return preferred or max(disclosures, key=lambda d: d.reporting_year or 0)

    _RULE_ROW_RE = re.compile(
        r"^[ \t]*((?:(?:total|gross|net)[ \t]+)?scope[ \t]*[-–]?[ \t]*([123])"
        r"(?:[ \t]+(?:total|emissions?|ghg[ \t]+emissions?)){0,2})"
        r"[ \t]*(?:[:=]|[-–][ \t]+)?[ \t]*"
        r"([-−(]?[ \t]*\d[\d,. \t ']*\d\)?|\d)"
        r"[ \t]*(t[ \t]?co2-?e?|tco2-?e?|kt[ \t]?co2-?e?|kg[ \t]?co2-?e?|"
        r"tonnes?[ \t]+co2-?e?)?(?:[ \t]|$)",
        re.IGNORECASE | re.MULTILINE)

    _RULE_CAT_MARKER_RE = re.compile(
        r"\b(?:scope\s*3\s*)?cat(?:egory)?\.?\s*(1[0-5]|[1-9])\b",
        re.IGNORECASE)
    _RULE_NUMBER_TOKEN_RE = re.compile(
        r"(?<![\d.,A-Za-z])(?:[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
        r"[-−]?\d{1,3}(?:\.\d{3})+(?:,\d+)?|"
        r"[-−]?\d{1,3}(?:[ \u00a0']\d{3})+(?:[.,]\d+)?|"
        r"[-−]?\d+(?:[.,]\d+)?)(?![\d.,A-Za-z])")
    _RULE_UNIT_RE = re.compile(
        r"\b(?:t\s*co2-?e?|tco2-?e?|kt\s*co2-?e?|kg\s*co2-?e?|"
        r"tonnes?\s+co2-?e?)\b", re.IGNORECASE)

    def _rule_extract_item(self, item: EvidenceItem) -> List[Dict[str, Any]]:
        """Regex fallback = the "Rule-only" baseline extraction path (Table 1)."""
        rows = []
        parts = re.split(r"\[RECONSTRUCTED_TABLE\s+\d+\]\s*", item.text)
        raw_text = parts[0]
        emission_table_blocks = [part for part in parts[1:]
                                 if self._RULE_UNIT_RE.search(part)]
        # Parse line by line so a flattened multi-year row can never be folded
        # into one number.  Ambiguous rows are skipped rather than guessed.
        for line in raw_text.splitlines():
            scope_match = re.search(r"\bscope\s*[-–]?\s*([123])\b", line, re.IGNORECASE)
            if not scope_match:
                continue
            normalized = normalize_string(line)
            if any(term in normalized for term in (
                    "target", "baseline", "intensity", "per m2", "per sqm",
                    "offset", "credit", "certificate", "scope 1 and scope 2",
                    "scope 1 2")) or "%" in line:
                continue
            scope_no = scope_match.group(1)
            if scope_no == "3" and "total" not in normalized:
                cat_no, _, _, _ = map_scope3_category(line)
                if cat_no is not None:
                    continue
            tail = line[scope_match.end():]
            candidates = []
            for number_match in self._RULE_NUMBER_TOKEN_RE.finditer(tail):
                token = number_match.group(0).strip()
                parsed = parse_number(token)
                if parsed is None:
                    continue
                if 2000 <= parsed <= 2035 and float(parsed).is_integer():
                    continue
                candidates.append((token, number_match.start()))
            if len(candidates) != 1:
                continue
            value_raw, value_start = candidates[0]
            label_prefix = set(normalize_string(tail[:value_start]).split())
            allowed_prefix = {
                "ghg", "emission", "emissions", "total", "gross", "net",
                "direct", "indirect", "location", "locationbased", "market",
                "marketbased", "based", "tco2e", "tco2eq", "co2e", "tonnes",
                "tons", "metric",
            }
            if label_prefix - allowed_prefix:
                # A named subcomponent (for example "Refrigerants") is not a
                # Scope total under Appendix H.4.
                continue
            unit_match = self._RULE_UNIT_RE.search(line)
            if scope_no == "1":
                scope = "scope1"
            elif scope_no == "2":
                if "location based" in normalized or "locationbased" in normalized:
                    scope = "scope2_location"
                elif "market based" in normalized or "marketbased" in normalized:
                    scope = "scope2_market"
                else:
                    scope = "scope2"
            else:
                scope = "scope3_total"
            years = [int(y) for y in _YEAR_RE.findall(line)]
            rows.append({
                "scope": scope, "category_label": None,
                "label_raw": line.strip(), "value_raw": value_raw,
                "unit_raw": unit_match.group(0) if unit_match else None,
                "reporting_year": years[0] if len(set(years)) == 1 else None,
                "text_evidence": line.strip(), "page": item.page,
                "_rule_fallback": True,
            })

        # Scope 3 category rows need a separate rule because their labels do
        # not contain the literal text "Scope 3".  Accept only rows that map
        # confidently to one canonical GHG Protocol category and contain one
        # unambiguous non-year numeric cell.  Multi-column rows are skipped
        # instead of guessing a year/value alignment.
        category_sources = [(raw_text, False)] + [
            (block, True) for block in emission_table_blocks]
        for source_text, in_reconstructed_table in category_sources:
            for line in source_text.splitlines():
                if not line.strip() or "%" in line:
                    continue
                marker = self._RULE_CAT_MARKER_RE.search(line)
                cat_no, _, _, score = map_scope3_category(line)
                if cat_no is None or (marker is None and score < 0.9):
                    continue
                unit_match = self._RULE_UNIT_RE.search(line)
                if marker is None and unit_match is None and not in_reconstructed_table:
                    # Prevent table-of-contents section numbers such as
                    # "3.11 Use of sold products ........ 83" from becoming data.
                    continue
                value_region = line[marker.end():] if marker else line
                candidates = []
                for match in self._RULE_NUMBER_TOKEN_RE.finditer(value_region):
                    token = match.group(0).strip()
                    parsed = parse_number(token)
                    if parsed is None:
                        continue
                    if 2000 <= parsed <= 2035 and float(parsed).is_integer():
                        continue
                    candidates.append((token, match.start()))
                if len(candidates) != 1:
                    continue
                value_raw, _ = candidates[0]
                years = [int(y) for y in _YEAR_RE.findall(line)]
                snippet = line.strip()
                rows.append({
                    "scope": "scope3_category",
                    "category_label": line.strip(),
                    "label_raw": line.strip(),
                    "value_raw": value_raw,
                    "unit_raw": unit_match.group(0) if unit_match else None,
                    "reporting_year": years[0] if len(set(years)) == 1 else None,
                    "text_evidence": snippet,
                    "page": item.page,
                    "_rule_fallback": True,
                })
        return rows

    @staticmethod
    def _guess_year(evidence_items: List[EvidenceItem]) -> Optional[int]:
        counts: Counter = Counter()
        for item in evidence_items[:5]:
            counts.update(int(y) for y in _YEAR_RE.findall(item.text))
        if not counts:
            return None
        recent = {y: c for y, c in counts.items() if y >= 2010}  # standards refs etc. cite old years
        pool = recent or dict(counts)
        best = max(pool.values())
        return max(y for y, c in pool.items() if c == best)  # tie-break: latest year

    @staticmethod
    def _boundary_from_evidence(evidence_items: List[EvidenceItem]) -> Optional[str]:
        text = "\n".join(item.text for item in evidence_items).lower()
        for phrase, canonical in (
            ("operational control", "Operational Control"),
            ("financial control", "Financial Control"),
            ("equity share", "Equity Share"),
        ):
            for match in re.finditer(re.escape(phrase), text):
                window = text[max(0, match.start() - 220):match.end() + 220]
                if any(term in window for term in ("ghg", "inventory", "consolidat", "boundary")):
                    return canonical
        return None

    @staticmethod
    def _row_year(row: Dict[str, Any], fallback: Optional[int]) -> Optional[int]:
        explicit = _i(row.get("reporting_year"))
        if explicit is not None:
            return explicit
        if row.get("_rule_fallback"):
            return fallback
        text = " ".join(str(row.get(k) or "")
                        for k in ("label_raw", "text_evidence"))
        years = [int(y) for y in _YEAR_RE.findall(text)]
        return years[0] if len(set(years)) == 1 else fallback

    def _declared_scope2_basis(self, doc_id: str, page_no: Optional[int]) -> Optional[str]:
        """The report's own statement of its Scope 2 method next to the table
        ("... presented using the market-based method", "* Consolidated
        Group - location-based approach"). One stated basis, or None."""
        if page_no is None:
            return None
        found = set()
        for p in self.state.parsed_pages.get(doc_id, []):
            if p.page is None or abs(p.page - page_no) > 1 or not p.raw_text:
                continue
            t = re.sub(r"\s+", " ", p.raw_text)
            for m in re.finditer(r"(?i)scope\s*2.{0,250}?\b(?:presented|calculated|reported|measured|determined|computed|disclosed)\s+"
                                 r"(?:using|with|under|according to|based on)\s+(?:the\s+)?(location|market)[\s-]*based\s+(?:method|approach)", t):
                found.add(m.group(1).lower())
            for m in re.finditer(r"(?im)^\s*\*+\s*[^\n]{0,80}?[-\u2013\u2014]\s*(location|market)[\s-]*based\s+approach\b", p.raw_text):
                found.add(m.group(1).lower())
        return found.pop() if len(found) == 1 else None

    def _normalize_row(self, doc_id: str, row: Dict[str, Any],
                       evidence_by_page: Dict[int, str],
                       report_year: Optional[int]) -> Optional[ExtractedValue]:
        """Rule-based normalization + span grounding of one raw extraction row."""
        scope = str(row.get("scope") or "").strip().lower()
        value_raw = row.get("value_raw")
        locale = self._page_locale(doc_id, _i(row.get("page")))
        value = parse_number(value_raw, locale)
        if value is None:
            return None
        value_tco2e, unit_assumed = normalize_emission_to_tco2e(value, row.get("unit_raw"))

        confidence = 1.0
        field_name: Optional[str] = None
        cat_no = cat_name = cat_method = None
        if scope in ("scope1", "scope2", "scope2_location", "scope2_market", "scope3_total"):
            field_name = scope
        elif scope == "scope3_category":
            cat_no, cat_name, cat_method, cat_score = map_scope3_category(row.get("category_label"))
            if cat_no is None:
                self.state.log_phase(self.name, "extract",
                                     f"{doc_id}: unmapped category label "
                                     f"'{row.get('category_label')}' (score {cat_score:.2f}) — dropped")
                return None
            field_name = f"scope3_cat{cat_no}"
            if cat_method == "fuzzy":
                confidence -= 0.2
        else:
            return None
        if unit_assumed:
            confidence -= 0.1

        page_no = _i(row.get("page"))
        page_text = evidence_by_page.get(page_no or -1, "")
        source_kind, hierarchy_note, row_fragment = category_hierarchy(
            cat_no, str(value_raw), row.get("label_raw"), page_text)
        value_kind = "category_total"
        if cat_no is not None:
            value_kind = (source_kind or ("component" if row.get("value_kind") == "component"
                                          else "category_total"))
        if self.evidence_grounding:
            deterministic_snippet = ground_evidence_snippet(
                str(value_raw) if value_raw is not None else None, value,
                row_fragment if value_kind == "component" and row_fragment else page_text)
            grounded, note = verify_value_in_text(
                str(value_raw) if value_raw is not None else None, value,
                deterministic_snippet, page_text)
            if not grounded:
                confidence -= 0.3
            field_supported = verify_field_context(
                field_name, str(value_raw) if value_raw is not None else None,
                value, page_text, cat_no)
            if source_kind is not None and row_fragment:
                field_supported = True
            if not field_supported:
                grounded = False
                confidence -= 0.25
                note = f"{note}; field/category label not found near numeric span"
            year_ok, year_note = table_year_check(str(value_raw), report_year, page_text)
            if year_ok is False:
                grounded = False
                confidence -= 0.25
                note = f"{note}; {year_note}"
        else:
            supplied = str(row.get("text_evidence") or "").strip()
            deterministic_snippet = supplied[:400] or None
            grounded = None
            note = "evidence grounding disabled for ablation"

        # Unit validity is a normalization constraint, not an optional evidence
        # check. Keep invalid candidates in the audit, never in reported output.
        if value_tco2e is None:
            grounded = False
            confidence = 0.0
            note += "; missing, unknown, or non-absolute emissions unit"

        evidence = EvidenceLink(
            source_document=doc_id,
            page=page_no,
            text_snippet=deterministic_snippet,
            method="rule_parser" if (self.rule_only or row.get("_rule_fallback"))
                   else "table_complement" if row.get("_table_complement")
                   else "llm_table_extraction",
            confidence=round(max(0.0, min(1.0, confidence)), 2),
            verified=grounded,          # Verify Agent re-checks and may downgrade
            verification_note=note,
        )
        extracted = ExtractedValue(
            field=field_name,
            value_raw=str(value_raw) if value_raw is not None else None,
            unit_raw=row.get("unit_raw"),
            value_tco2e=value_tco2e,
            category_no=cat_no,
            category_canonical=cat_name,
            category_label_raw=row.get("category_label"),
            category_map_method=cat_method,
            report_year=report_year,
            asset_mention=_s(row.get("asset_mention")),
            provenance="reported",
            evidence=evidence,
            label_raw=_s(row.get("label_raw")),
            value_kind=value_kind,
            hierarchy_note=hierarchy_note if cat_no is not None else None,
        )
        if self.evidence_grounding:
            from scope3trace.scope_agent.table_structure import bind_value, _noninventory_quantity
            parsed = next((p for p in self.state.parsed_pages.get(doc_id, []) if p.page == page_no), None)
            if parsed and parsed.table_structures and value_raw is not None:
                # the printed token's own table may use another number
                # convention than the page majority ("3,695" above "3.695")
                own = {t.get("number_locale") for t in parsed.table_structures
                       if any(str(c["raw"]).strip() == str(value_raw).strip() for r in t["rows"] for c in r["cells"])}
                if not own and re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", str(value_raw).strip()):
                    # a grouped integer token whose digits a table prints in
                    # its own convention: take that cell's reading
                    digits = re.sub(r"\D", "", str(value_raw))
                    readings = {parse_number(c["raw"], t.get("number_locale")) for t in parsed.table_structures
                                for r in t["rows"] for c in r["cells"]
                                if re.sub(r"\D", "", str(c["raw"])) == digits
                                and re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", str(c["raw"]).strip())}
                    readings.discard(None)
                    if len(readings) == 1 and readings != {value}:
                        value = readings.pop()
                        extracted.value_tco2e = normalize_emission_to_tco2e(value, row.get("unit_raw"))[0]
                if len(own) == 1 and None not in own and own != {locale}:
                    v2 = parse_number(value_raw, next(iter(own)))
                    if v2 is not None:
                        value = v2
                        extracted.value_tco2e = normalize_emission_to_tco2e(v2, row.get("unit_raw"))[0]
            if parsed and parsed.table_structures:
                binding, supported = bind_value(extracted, parsed.table_structures, parsed.raw_text,
                                                row.get("table_binding"),
                                                document_pages=self.state.parsed_pages.get(doc_id,[]))
                if (binding and supported is not True and str(binding.get('reason') or '').startswith('no source cell satisfies')):
                    # The year is a property of the printed column. When the
                    # labelled row holds this exact amount under exactly one
                    # other reported year, that column year is the value's year.
                    alt_years = sorted({c['year'] for t in parsed.table_structures for c in t['columns']
                                        if c.get('role') == 'reported' and c.get('year') != extracted.report_year})
                    hits = []
                    claimed = extracted.report_year
                    for yy in alt_years:
                        extracted.report_year = yy
                        b2, s2 = bind_value(extracted, parsed.table_structures, parsed.raw_text, None,
                                            document_pages=self.state.parsed_pages.get(doc_id, []))
                        if s2 is True:
                            hits.append((yy, b2))
                    extracted.report_year = claimed
                    if len(hits) == 1:
                        yy, binding = hits[0]
                        supported = True
                        binding['extraction_year'] = claimed
                        binding['year_from_column'] = True
                        extracted.report_year = yy
                if binding:
                    evidence.binding = binding
                    evidence.verified = supported is True
                    evidence.verification_note = binding.get("reason") or binding.get("source_check")
                    if binding.get("value_kind"):
                        extracted.value_kind = binding["value_kind"]
                        extracted.hierarchy_note = (f"coordinate row {binding['row_id']} in {binding['table_id']}; "
                                                    f"parent category {binding['parent_category']}; {binding['value_kind']}")
                        parent = binding.get('parent_observation') or {}
                        if parent:
                            extracted.hierarchy_note += (f"; this parent has {parent['numeric_row_count']} disclosed numeric row(s); "
                                                         f"explicit aggregate rows={parent['explicit_aggregate_rows']}")
                            if parent['numeric_row_count'] == 1:
                                extracted.hierarchy_note += "; single disclosed amount in category block, retaining the report's stated coverage"
                    if supported and binding.get('footnote_value_corrected'):
                        # "2882" = printed "288" + superscript footnote "2": the
                        # glyph-checked cell is the amount
                        binding['extraction_value_raw'] = extracted.value_raw
                        extracted.value_raw = binding['cell_raw']
                        value = parse_number(binding['cell_raw'], locale) or value
                        extracted.value_tco2e = normalize_emission_to_tco2e(value, extracted.unit_raw)[0]
                    if supported:
                        # Preserve the model label for audit; export the exact
                        # bound source row, with merged descriptors separate.
                        binding['extraction_label_raw']=extracted.label_raw
                        extracted.label_raw=binding['row_label']
                        extracted.field=binding['resolved_field']
                        if binding.get('unit_inherited'):
                            extracted.unit_raw = binding['unit']['text']
                            extracted.value_tco2e = normalize_emission_to_tco2e(value, extracted.unit_raw)[0]
                            extracted.hierarchy_note = (extracted.hierarchy_note or '') + '; absolute unit inherited from unique physical table unit'
                        binding['cell_summary'] = (f"{binding['row_label']} | {binding['year']} | "
                                                   f"{binding['cell_raw']} | {binding['unit']['text']}")
                        evidence.text_snippet = (verbatim_snippet(parsed.raw_text, binding.get('source_span'))
                                                 or binding['cell_summary'])
                        evidence.confidence = 1.0
                    else:
                        evidence.confidence = 0.0
            # Geometry is an aid, not the only admissible evidence (§3.3): when
            # no reconstructed cell holds the value, verify the source text row.
            prior = evidence.binding or {}
            fallback_ok = parsed is not None and (
                evidence.binding is None or (evidence.verified is not True and prior.get('status') == 'rejected'
                                             and prior.get('reason') in _TEXT_FALLBACK_REASONS))
            text_reason = None
            if fallback_ok and not _table_contradiction(extracted, parsed, locale):
                from scope3trace.scope_agent.text_binding import bind_text_row
                from scope3trace.scope_agent.table_structure import footnoted_tokens
                tb, ok, text_reason = bind_text_row(extracted, parsed.raw_text, locale, doc_id, page_no,
                                                    footnoted_tokens(parsed))
                if ok:
                    if extracted.value_tco2e is None:
                        extracted.unit_raw = tb['unit_text']
                        extracted.value_tco2e = normalize_emission_to_tco2e(value, extracted.unit_raw)[0]
                    if row.get("value_kind") == "component":
                        extracted.value_kind = "component"
                    tb['prior_table_check'] = prior.get('reason')
                    evidence.binding = tb
                    evidence.verified = extracted.value_tco2e is not None
                    evidence.verification_note = text_reason
                    evidence.text_snippet = verbatim_snippet(parsed.raw_text, tb['source_span'])
                    evidence.confidence = 0.8
                elif evidence.binding is not None:
                    evidence.verification_note = f"{evidence.verification_note}; text-row check: {text_reason}"
            if (extracted.field == 'scope2' and evidence.verified is True and isinstance(evidence.binding, dict)):
                basis = self._declared_scope2_basis(doc_id, page_no)
                if basis:
                    evidence.binding['declared_scope2_basis'] = basis
            if evidence.binding is None:
                item = next((it for it in self.state.evidence.get(doc_id,[]) if it.page == page_no),None)
                is_table = (parsed and (parsed.table_structures or parsed.reconstructed_tables)) or (item and item.content_type=='table')
                if is_table or row.get('table_binding') is not None:
                    evidence.binding = {'status':'rejected','reason':'table candidate has no unique validated row-column binding',
                                        'text_row_check': text_reason}
                    evidence.verified = False
                    evidence.verification_note = evidence.binding['reason']
                    evidence.confidence = 0.0
                    return extracted
                # Narrative evidence has a separate origin type; never invent
                # a table/column id for prose or a failed reconstruction.
                import hashlib
                source = parsed.raw_text if parsed else page_text
                fragment = ground_evidence_snippet(str(value_raw),value,source) or ''
                start = source.find(fragment) if fragment else -1
                evidence.binding = {"origin": "text", "status": "span_only", "document": doc_id,
                                    "page": page_no, "field": field_name, "year": report_year,
                                    "unit_raw": row.get("unit_raw"), "parent_category": cat_no,
                                    "source_fragment": fragment, 'source_span':[start,start+len(fragment)],
                                    "source_sha256":hashlib.sha256(source.encode()).hexdigest(),
                                    "value_raw":str(value_raw),"value_tco2e":value_tco2e,
                                    "quantity_type":"inventory"}
            if _noninventory_quantity(extracted.label_raw or ''):
                evidence.verified = False
                evidence.confidence = 0.0
                evidence.verification_note = 'independent erased/avoided-emissions indicator is not an inventory emissions value'
                if evidence.binding:
                    evidence.binding['quantity_type']='excluded_noninventory'
        return extracted

    def _extract_page(self, page_no: int, text: str, suffix: str = "", depth: int = 0) -> Optional[Dict[str, Any]]:
        """One bounded extraction call; a truncated response is re-requested
        per table block (and per half of a dense block) instead of failing
        the document. Every chunk keeps its table/column header lines."""
        from scope3trace.common.llm import LLMTruncatedError
        try:
            obj = self.llm.complete_json(
                SYSTEM_COMMON,
                TABLE_EXTRACTION_PROMPT.format(page_no=page_no, page_text=text + suffix),
                max_tokens=EXTRACTION_MAX_OUTPUT_TOKENS)
            truncated = obj is None and getattr(self.llm, "last_truncated", False)
        except LLMTruncatedError:
            obj, truncated = None, True
        if not truncated or depth >= 3:
            return obj
        parts = _split_extraction_text(text)
        if len(parts) < 2:
            return obj
        self.state.log_phase(self.name, "extract",
                             f"page {page_no}: output limit reached; re-requesting {len(parts)} table blocks")
        merged: Dict[str, Any] = {"values": [], "asset_mentions": []}
        for part in parts:
            sub = self._extract_page(page_no, part, suffix, depth + 1)
            if not sub:
                continue
            for key, val in sub.items():
                if key in ("values", "asset_mentions"):
                    merged[key].extend(val or [])
                elif merged.get(key) in (None, False) and val not in (None, False):
                    merged[key] = val
        return merged

    def _page_locale(self, doc_id: str, page_no: Optional[int]) -> Optional[str]:
        parsed = next((p for p in self.state.parsed_pages.get(doc_id, []) if p.page == page_no), None)
        return getattr(parsed, "number_locale", None) if parsed else None

    @staticmethod
    def _dedup(values: List[ExtractedValue]) -> List[ExtractedValue]:
        """One value per field: prefer span-grounded, then higher confidence,
        then earlier page (deterministic resolution — no LLM arbitration)."""
        best: Dict[str, ExtractedValue] = {}
        for v in values:
            if v.value_kind == "component":
                continue
            cur = best.get(v.field)
            if cur is None:
                best[v.field] = v
                continue
            key_new = (bool(v.evidence.verified), v.evidence.confidence, -(v.evidence.page or 10**6))
            key_cur = (bool(cur.evidence.verified), cur.evidence.confidence, -(cur.evidence.page or 10**6))
            if key_new > key_cur:
                best[v.field] = v
        generic = best.get("scope2")
        if generic is not None:
            explicit = [best.get("scope2_location"), best.get("scope2_market")]
            if any(v is not None and v.value_tco2e == generic.value_tco2e for v in explicit):
                best.pop("scope2", None)
        return sorted(best.values(), key=lambda v: v.field)

    # ==================================================================
    # 3) Entity linking (report mentions ↔ Asset Pool)
    # ==================================================================
    def link_report_mentions(self, disclosure: OrgDisclosure) -> List[EntityLink]:
        """Deterministic first: exact asset-key match (name+postcode), then
        fuzzy similarity over names/addresses. Only the ambiguous band
        [LINK_CANDIDATE_SCORE, LINK_ACCEPT_SCORE) goes to a bounded LLM call;
        below the band the mention stays unlinked (no guessing)."""
        links: List[EntityLink] = []
        for m in disclosure.asset_mentions:
            mention = str(m.get("mention") or "").strip()
            if not mention:
                continue
            link = self._link_one(mention, m.get("snippet") or "", m.get("page"))
            link.source_document = disclosure.source_document
            link.reporting_year = disclosure.reporting_year
            link.org_name = disclosure.org_name
            links.append(link)
        self.state.entity_links.extend(links)
        self._carry_reported_asset_values(disclosure, links)
        n_ok = sum(1 for l in links if l.property_id)
        self.state.log_phase(self.name, "entity_link",
                             f"{disclosure.source_document}: {n_ok}/{len(links)} mentions linked")
        return links

    def _carry_reported_asset_values(self, disclosure: OrgDisclosure,
                                     links: List[EntityLink]) -> None:
        """Attach explicitly asset-attributed report values before estimation."""
        by_mention = {normalize_string(l.mention): l for l in links if l.property_id}
        field_map = {
            "scope1": "scope1_tco2e",
            "scope2": "scope2_tco2e",
            "scope2_location": "scope2_tco2e",
            "scope2_market": "scope2_market_tco2e",
            "scope3_total": "scope3_total_tco2e",
            "scope3_cat2": "scope3_cat2_tco2e",
            "scope3_cat13": "scope3_cat13_tco2e",
        }
        for value in disclosure.values:
            if not value.asset_mention or value.field not in field_map:
                continue
            link = by_mention.get(normalize_string(value.asset_mention))
            if link is None or link.property_id is None:
                continue
            self._emit(DerivedComponent(
                property_id=link.property_id,
                report_year=disclosure.reporting_year,
                field=field_map[value.field], value_tco2e=value.value_tco2e,
                provenance="reported", method="direct asset disclosure",
                factor_source=None, inputs={"value_raw": value.value_raw,
                                            "unit_raw": value.unit_raw},
                evidence=value.evidence, quality="reported",
                validation_status="reported"))
        self._carry_single_source_scope_lines(disclosure, links)

    def _carry_single_source_scope_lines(self, disclosure: OrgDisclosure,
                                         links: List[EntityLink]) -> None:
        """Building level only: a summary table listing emission sources per
        scope without subtotals ("Scope 1: Refrigerants / Natural gas",
        "Scope 2: Electricity 1,409.386", "Total Emissions") gives no Scope 2
        total, but a scope with exactly one source line has that line as its
        amount. For a report naming exactly one building, such a verified line
        is that building's reported value. Nothing is summed, and the
        organization-level record and its export are unchanged."""
        linked = {l.property_id for l in links if l.property_id}
        mentions = {normalize_string(str(m.get("mention") or "")) for m in disclosure.asset_mentions}
        mentions.discard("")
        if len(linked) != 1 or len(mentions) != 1:
            return
        pid = next(iter(linked))
        for scope_field, comp_field in (("scope1", "scope1_tco2e"), ("scope2", "scope2_tco2e")):
            totals = [v for v in disclosure.values if v.field == scope_field
                      or (scope_field == "scope2" and v.field in ("scope2_location", "scope2_market"))]
            if any(v.evidence.verified is True for v in totals):
                continue  # the report states the scope total itself
            lines = [c for c in disclosure.category_components if c.field == scope_field]
            if len(lines) != 1 or lines[0].evidence.verified is not True or lines[0].value_tco2e is None:
                continue
            line = lines[0]
            if line.asset_mention and normalize_string(line.asset_mention) not in mentions:
                continue
            self._emit(DerivedComponent(
                property_id=pid, report_year=disclosure.reporting_year,
                field=comp_field, value_tco2e=line.value_tco2e,
                provenance="reported", method="single source line of the scope (no subtotal printed)",
                factor_source=None, inputs={"value_raw": line.value_raw, "unit_raw": line.unit_raw,
                                            "label": line.label_raw},
                evidence=line.evidence, quality="reported", validation_status="reported"))

    def _link_one(self, mention: str, snippet: str, page: Optional[int]) -> EntityLink:
        mention_key = build_asset_key(mention, extract_postcode_from_address(mention))
        scored: List[Tuple[float, AssetRecord]] = []
        for rec in self.state.asset_pool:
            score = self._similarity(mention, mention_key, rec)
            if score >= LINK_CANDIDATE_SCORE:
                scored.append((score, rec))
        scored.sort(key=lambda t: -t[0])

        if not scored:
            return EntityLink(mention=mention, property_id=None, score=0.0,
                              method="unlinked", evidence_page=page,
                              evidence_snippet=snippet[:300] or None,
                              note="no candidate above threshold")
        top_score, top = scored[0]
        if top_score >= LINK_ACCEPT_SCORE:
            method = "exact_key" if top_score >= 0.999 else "fuzzy"
            return EntityLink(mention=mention, property_id=top.property_id,
                              score=round(top_score, 3), method=method,
                              evidence_page=page, evidence_snippet=snippet[:300] or None)
        if self.rule_only:
            return EntityLink(mention=mention, property_id=None, score=round(top_score, 3),
                              method="unlinked", evidence_page=page,
                              evidence_snippet=snippet[:300] or None,
                              note="ambiguous band; LLM adjudication disabled")
        return self._llm_adjudicate(mention, snippet, page, scored[:5])

    @staticmethod
    def _similarity(mention: str, mention_key: str, rec: AssetRecord) -> float:
        m_norm = normalize_string(mention)
        best = 0.0
        for cand in rec.names + [rec.address_raw or ""]:
            if not cand:
                continue
            c_norm = normalize_string(cand)
            if not c_norm:
                continue
            if build_asset_key(cand, rec.postcode) == mention_key:
                return 1.0
            if m_norm and (m_norm in c_norm or c_norm in m_norm):
                best = max(best, 0.9)
            best = max(best, difflib.SequenceMatcher(None, m_norm, c_norm).ratio())
        return best

    def _llm_adjudicate(self, mention: str, snippet: str, page: Optional[int],
                        candidates: List[Tuple[float, AssetRecord]]) -> EntityLink:
        block = "\n".join(
            f"- property_id={r.property_id} | name(s)={'; '.join(r.names) or '?'} | "
            f"address={r.address_raw or '?'} | type={r.building_type or '?'} (fuzzy score {s:.2f})"
            for s, r in candidates)
        obj = self.llm.complete_json(
            SYSTEM_COMMON,
            ENTITY_LINK_PROMPT.format(mention=mention, snippet=snippet[:300],
                                      candidates_block=block))
        valid_ids = {r.property_id for _, r in candidates}
        if obj and obj.get("decision") == "match" and obj.get("match_property_id") in valid_ids:
            score = next(s for s, r in candidates if r.property_id == obj["match_property_id"])
            return EntityLink(mention=mention, property_id=str(obj["match_property_id"]),
                              score=round(score, 3), method="llm_semantic",
                              evidence_page=page, evidence_snippet=snippet[:300] or None,
                              note=str(obj.get("reason") or "")[:200] or None)
        note = str((obj or {}).get("reason") or "LLM adjudication: no confident match")[:200]
        return EntityLink(mention=mention, property_id=None,
                          score=round(candidates[0][0], 3), method="unlinked",
                          evidence_page=page, evidence_snippet=snippet[:300] or None, note=note)

    # ==================================================================
    # 4) Deterministic estimation (Appendix C — no LLM involved)
    # ==================================================================
    def estimate(self, grid_factor_csv: Optional[Path] = None) -> int:
        """Emit building-level explanatory components for every asset-year.

        Reported values from the inventory are carried through verbatim
        (provenance "reported" / their recorded quality) and never
        overwritten; estimates fill only the gaps (C.4 distinction strategy).
        """
        factors = _load_grid_factors(grid_factor_csv) if grid_factor_csv else {}
        n = 0
        for asset in self.state.asset_pool:
            for year_s, act in sorted(asset.activity.items()):
                year = int(year_s)
                ef, ef_src = _pick_grid_factor(factors, year, asset.grid_region)
                n += self._estimate_scope2(asset, year, act, ef, ef_src)
                n += self._estimate_cat2(asset, year, act)
                n += self._estimate_cat13(asset, year, act, ef, ef_src)
        self.state.log_phase(self.name, "estimate",
                             f"{n} derived components over {len(self.state.asset_pool)} assets")
        return n

    def _emit(self, comp: DerivedComponent) -> int:
        # one component per (asset, year, field); a dict lookup instead of a
        # scan of every earlier component (linear in the inventory size)
        comps = self.state.derived_components
        index = getattr(self, "_emit_index", None)
        if index is None or getattr(self, "_emit_index_of", None) is not comps or len(index) != len(comps):
            index = {(c.property_id, c.report_year, c.field): c for c in comps}
            self._emit_index, self._emit_index_of = index, comps
        key = (comp.property_id, comp.report_year, comp.field)
        existing = index.get(key)
        if existing is not None:
            priority = {"reported": 3, "modeled": 2, "estimated": 1}
            if priority.get(comp.provenance, 0) > priority.get(existing.provenance, 0):
                comps.remove(existing)
            else:
                return 0
        comps.append(comp)
        index[key] = comp
        return 1

    def _estimate_scope2(self, asset: AssetRecord, year: int, act: Dict[str, Any],
                         ef: float, ef_src: str) -> int:
        reported = act.get("scope2_location_tco2e_reported")
        if reported is not None:
            # honour the inventory's own quality label — a value the inventory
            # marks estimated/modeled must not be relabelled "reported" (C.4)
            quality = (act.get("scope2_quality") or "").lower()
            prov = quality if quality in ("reported", "modeled", "estimated") else "reported"
            return self._emit(DerivedComponent(
                property_id=asset.property_id, report_year=year, field="scope2_tco2e",
                value_tco2e=reported, provenance=prov,
                method="carried from inventory", factor_source=None,
                inputs={}, quality=act.get("scope2_quality")))
        kwh = act.get("electricity_kwh")
        if kwh is None:
            return 0
        modeled_input = "model" in (act.get("energy_source") or "").lower()
        return self._emit(DerivedComponent(
            property_id=asset.property_id, report_year=year, field="scope2_tco2e",
            value_tco2e=round(kwh * ef, 4),
            provenance="modeled" if modeled_input else "estimated",
            method="C.1: electricity_kwh x EF_grid", factor_source=ef_src,
            inputs={"electricity_kwh": kwh, "ef_grid_t_per_kwh": ef,
                    "energy_source": act.get("energy_source")},
            quality="modeled_energy_input" if modeled_input else "estimated"))

    def _estimate_cat2(self, asset: AssetRecord, year: int, act: Dict[str, Any]) -> int:
        existing = act.get("scope3_cat2_tco2e")
        if existing is not None:
            quality = (act.get("scope3_cat2_quality") or
                       act.get("scope3_cat2_source") or "estimated").lower()
            provenance = quality if quality in ("reported", "modeled", "estimated") else "estimated"
            return self._emit(DerivedComponent(
                property_id=asset.property_id, report_year=year, field="scope3_cat2_tco2e",
                value_tco2e=existing, provenance=provenance,
                method=act.get("scope3_cat2_method") or "carried from inventory (C.2)",
                factor_source=act.get("scope3_cat2_source"), inputs={},
                quality=act.get("scope3_cat2_quality"),
                validation_status="reported" if provenance == "reported" else "pending"))
        if not asset.floor_area_m2:
            return 0
        ef_emb = EMBODIED_FACTORS_KG_PER_SQM.get(
            asset.building_type or "", EMBODIED_FACTORS_KG_PER_SQM["Default"])
        return self._emit(DerivedComponent(
            property_id=asset.property_id, report_year=year, field="scope3_cat2_tco2e",
            value_tco2e=round(asset.floor_area_m2 * ef_emb / 1000.0, 4),
            provenance="estimated",
            method="C.2: floor_area_m2 x EF_embodied / 1000",
            factor_source=f"embodied:{asset.building_type or 'Default'}:{ef_emb} kgCO2e/m2",
            inputs={"floor_area_m2": asset.floor_area_m2,
                    "floor_area_source": asset.floor_area_source,
                    "building_type": asset.building_type}))

    def _estimate_cat13(self, asset: AssetRecord, year: int, act: Dict[str, Any],
                        ef: float, ef_src: str) -> int:
        existing = act.get("scope3_cat13_tco2e")
        if existing is not None:
            quality = (act.get("scope3_cat13_quality") or
                       act.get("scope3_cat13_source") or "estimated").lower()
            provenance = quality if quality in ("reported", "modeled", "estimated") else "estimated"
            return self._emit(DerivedComponent(
                property_id=asset.property_id, report_year=year, field="scope3_cat13_tco2e",
                value_tco2e=existing, provenance=provenance,
                method=act.get("scope3_cat13_method") or "carried from inventory (C.3)",
                factor_source=act.get("scope3_cat13_source"), inputs={},
                quality=act.get("scope3_cat13_quality"),
                validation_status="reported" if provenance == "reported" else "pending"))
        tenant_kwh = act.get("tenant_electricity_kwh")
        if tenant_kwh is None:
            kwh, p_com = act.get("electricity_kwh"), act.get("p_com")
            tenant_kwh = kwh * p_com if (kwh is not None and p_com is not None) else None
        if tenant_kwh is None:
            return 0
        return self._emit(DerivedComponent(
            property_id=asset.property_id, report_year=year, field="scope3_cat13_tco2e",
            value_tco2e=round(tenant_kwh * ef, 4), provenance="estimated",
            method="C.3: tenant_electricity_kwh x EF_grid", factor_source=ef_src,
            inputs={"tenant_electricity_kwh": tenant_kwh, "p_com": act.get("p_com"),
                    "ef_grid_t_per_kwh": ef}))


# ---------------------------------------------------------------------------
# Grid factor selection (region/year specific when a CSV is supplied)
# ---------------------------------------------------------------------------

def _load_grid_factors(csv_path: Optional[Path]) -> Dict[Tuple[str, int], float]:
    """CSV columns: region/state + report_year + factor in t or kg/kWh."""
    import csv as _csv
    factors: Dict[Tuple[str, int], float] = {}
    if not csv_path:
        return factors
    with Path(csv_path).open("r", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            year = _i(row.get("report_year") or row.get("year"))
            if year is None:
                continue
            t = _f(row.get("factor_t_per_kwh"))
            if t is None:
                kg = _f(row.get("factor_kg_per_kwh"))
                t = kg / 1000.0 if kg is not None else None
            if t is not None:
                region = normalize_string(row.get("region") or row.get("state") or "")
                factors[(region, year)] = t
    return factors


def _pick_grid_factor(factors: Dict[Tuple[str, int], float], year: int,
                      region: Optional[str]) -> Tuple[float, str]:
    region_key = normalize_string(region or "")
    if (region_key, year) in factors:
        factor = factors[(region_key, year)]
        return factor, f"grid_factor_csv:{region or '?'}:{year}:{factor} tCO2e/kWh"
    if region_key:
        same_region = [((r, y), f) for (r, y), f in factors.items() if r == region_key]
        if same_region:
            (picked_region, nearest), factor = min(
                same_region, key=lambda item: abs(item[0][1] - year))
            return factor, (f"grid_factor_csv:nearest:{picked_region}:{nearest} for {year}:"
                            f"{factor} tCO2e/kWh")
    same_year = [((r, y), f) for (r, y), f in factors.items() if y == year]
    if same_year:
        # Appendix I: when the state is missing, use the highest-factor state.
        (picked_region, _), factor = max(same_year, key=lambda item: item[1])
        return factor, (f"grid_factor_csv:highest-factor fallback:{picked_region or '?'}:"
                        f"{year}:{factor} tCO2e/kWh")
    if factors:
        nearest_year = min({y for _, y in factors}, key=lambda y: abs(y - year))
        candidates = [((r, y), f) for (r, y), f in factors.items() if y == nearest_year]
        (picked_region, _), factor = max(candidates, key=lambda item: item[1])
        return factor, (f"grid_factor_csv:nearest highest-factor:{picked_region or '?'}:"
                        f"{nearest_year} for {year}:{factor} tCO2e/kWh")
    return DEFAULT_GRID_FACTOR_T_PER_KWH, DEFAULT_GRID_FACTOR_SOURCE


def _infer_au_state(address: Optional[str]) -> Optional[str]:
    if not address:
        return None
    upper = str(address).upper()
    for state in ("ACT", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"):
        if re.search(rf"\b{state}\b", upper):
            return state
    return None


def _split_extraction_text(text: str) -> List[str]:
    """Split page evidence into linear text and individual table blocks;
    a single block splits into two halves that both keep its header lines."""
    blocks = re.split(r"(?=\[STRUCTURED_TABLE )", text)
    raw, tables = blocks[0], blocks[1:]
    if tables:
        return ([raw] if raw.strip() else []) + tables
    lines = text.split("\n")
    if lines and lines[0].startswith("[STRUCTURED_TABLE") and len(lines) > 6:
        head = [l for l in lines[:3] if l.startswith(("[STRUCTURED_TABLE", "COLUMNS"))]
        body = [l for l in lines[len(head):] if not l.startswith("[/STRUCTURED_TABLE")]
        mid = len(body) // 2
        return ["\n".join(head + body[:mid] + ["[/STRUCTURED_TABLE]"]),
                "\n".join(head + body[mid:] + ["[/STRUCTURED_TABLE]"])]
    if len(lines) > 40:
        mid = len(lines) // 2
        return ["\n".join(lines[:mid]), "\n".join(lines[mid:])]
    return [text]


_TEXT_FALLBACK_REASONS = {
    "no source cell satisfies row, year, unit, parent and value together",
}


def _table_contradiction(value: ExtractedValue, parsed: Any, locale: Optional[str]) -> bool:
    """True when a reconstructed row with this label prints another amount
    for the claimed year; the text path must not override that cell."""
    from scope3trace.scope_agent.table_structure import _row_similarity
    for t in getattr(parsed, "table_structures", []) or []:
        cols = {c["id"]: c for c in t["columns"]}
        for r in t["rows"]:
            if r.get("field") != value.field or r.get("parent_category") != value.category_no:
                continue
            if _row_similarity(value.label_raw, r["label"], component=True) < .8:
                continue
            mult = (r.get("unit") or {}).get("multiplier")
            for c in r["cells"]:
                col = cols.get(c["column_id"]) or {}
                if col.get("role") != "reported" or col.get("year") != value.report_year:
                    continue
                n = parse_number(c["raw"], t.get("number_locale", locale))
                if n is not None and mult and value.value_tco2e is not None \
                        and abs(n * mult - value.value_tco2e) > 1e-9 * max(1.0, abs(value.value_tco2e)):
                    return True
    return False


def verbatim_snippet(raw: str, span: Optional[List[int]], lo: int = 100, hi: int = 400) -> Optional[str]:
    """A 100-400 character verbatim excerpt around a source span (H.5)."""
    if not raw or not span or span[0] is None or span[0] < 0:
        return None
    a, b = span
    start = raw.rfind("\n", 0, a) + 1
    end = raw.find("\n", b)
    end = len(raw) if end < 0 else end
    while end - start < lo and (start > 0 or end < len(raw)):
        if start > 0:
            start = raw.rfind("\n", 0, start - 1) + 1
        if end - start < lo and end < len(raw):
            nxt = raw.find("\n", end + 1)
            end = len(raw) if nxt < 0 else nxt
    if end - start > hi:
        c = (a + b) // 2
        start = max(start, c - hi // 2)
        end = min(end, start + hi)
    return raw[start:end].strip()


def _pick_coordinates(row: Any) -> Tuple[Optional[float], Optional[float]]:
    lat, lon = _f(row.get("lat")), _f(row.get("lon"))
    if lat is not None and lon is not None:
        return lat, lon
    raw = _s(row.get("geo_point_2d")) or _s(row.get("location_lat_lon"))
    if raw and "," in raw:
        left, right = raw.split(",", 1)
        return _f(left), _f(right)
    return None, None


def _s(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        import pandas as pd
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    return s or None


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v else None  # NaN check
    except (TypeError, ValueError):
        return None


def _i(x: Any) -> Optional[int]:
    v = _f(x)
    return int(v) if v is not None else None
