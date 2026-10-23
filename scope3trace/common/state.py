#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared intermediate state for the Scope3Trace multi-agent pipeline (§3.3).

The three agents (Search, Scope, Verify) do NOT engage in free-form
conversational orchestration — an empirically validated design choice
(§3.3, Appendix F). Instead they read from and write to this shared,
typed state object, and cross-validate one another's outputs through it:

- Search Agent → writes  ``documents``                ("Data Acquisition")
                 and     ``evidence``                 (verbatim evidence lists:
                                                       "Page-localization" /
                                                       "Find Scope Info")
- Scope Agent  → writes  ``asset_pool``               ("Asset Pool")
                 and     ``org_disclosures``          ("Reported Scope Info",
                                                       extraction from evidence)
                 and     ``entity_links``             (entity linking)
                 and     ``derived_components``       ("Estimation Process")
- Verify Agent → writes  ``diagnostics``              ("Verification Process")
                 and     ``evaluation_report``        ("Evaluation Report")

Every value carries an evidence link (page + verbatim snippet + confidence,
§3.2) and a provenance label — reported / modeled / estimated (Appendix C.4).
"""

from __future__ import annotations

import re
import os
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROVENANCE_LABELS = ("reported", "modeled", "estimated")


# ---------------------------------------------------------------------------
# Search Agent outputs — document list + verbatim evidence list
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    """One acquired report document ("Data Acquisition")."""
    doc_id: str                              # basename used as key everywhere
    local_path: Optional[str] = None
    url: Optional[str] = None
    org_id: Optional[str] = None
    org_hint: Optional[str] = None           # organization the search targeted
    year_hint: Optional[int] = None
    source: str = "local"                    # local | manifest | climate_active | google_search
    n_pages: Optional[int] = None
    status: str = "acquired"                 # acquired | parsed | failed


@dataclass
class EvidenceItem:
    """One entry of the Search Agent's returned evidence list.

    The Search Agent only LOCATES content and returns it verbatim — it never
    calculates, converts, or maps values (that is the Scope Agent's job).
    ``text`` retains the parsed/OCR page plus deterministically reconstructed
    table rows, so source-cell fidelity is preserved by construction.
    """
    doc_id: str
    page: int
    content_type: str = "text"               # table | text
    text: str = ""
    # Page/sentence localization output used by the Scope Agent. ``text``
    # always retains the complete source page for deterministic verification;
    # this field contains only the verbatim sentences/tables selected for
    # extraction.  Keeping both prevents localization from weakening the
    # evidence check.
    localized_text: Optional[str] = None
    sentence_ids: List[int] = field(default_factory=list)
    reason: Optional[str] = None             # short LLM/rule justification
    method: str = "llm_page_sentence_classification"


@dataclass
class ParsedPage:
    """Structured PDF/OCR page produced by the Search Agent."""
    doc_id: str
    page: int
    raw_text: str = ""
    clean_text: str = ""
    sentences: List[str] = field(default_factory=list)
    content_tokens: List[str] = field(default_factory=list)
    reconstructed_tables: List[List[List[str]]] = field(default_factory=list)
    table_structures: List[Dict[str, Any]] = field(default_factory=list)
    reconstruction_issues: List[Dict[str, Any]] = field(default_factory=list)
    original_extracted_text: Optional[str] = None
    parse_audit: Dict[str, Any] = field(default_factory=dict)
    parse_method: str = "pdf_text"             # pdf_text | ocr | pdf_text+tables
    number_locale: Optional[str] = None        # eu | us | None (page convention, else document)


# ---------------------------------------------------------------------------
# Evidence grounding (§3.2 "evidence link")
# ---------------------------------------------------------------------------

@dataclass
class EvidenceLink:
    """Links an extracted value to its source text (traceability requirement)."""
    source_document: Optional[str] = None
    page: Optional[int] = None
    text_snippet: Optional[str] = None
    method: str = "llm_table_extraction"   # llm_table_extraction | rule_parser | inventory_column
    confidence: float = 1.0
    # Deterministic evidence check result (Verify Agent flags, never corrects):
    verified: Optional[bool] = None
    verification_note: Optional[str] = None
    binding: Optional[Dict[str, Any]] = None
    extraction_round: int = 0


@dataclass
class ExtractedValue:
    """One numeric disclosure value with normalization + provenance."""
    field: str                              # e.g. scope1 | scope2_location | scope3_total | scope3_cat6
    value_raw: Optional[str] = None         # verbatim number string from the report
    unit_raw: Optional[str] = None          # verbatim unit string
    value_tco2e: Optional[float] = None     # rule-normalized value (Appendix A.7)
    category_no: Optional[int] = None       # 1..15 for Scope 3 categories
    category_canonical: Optional[str] = None
    category_label_raw: Optional[str] = None
    category_map_method: Optional[str] = None  # exact | synonym | regex | fuzzy
    report_year: Optional[int] = None
    asset_mention: Optional[str] = None
    provenance: str = "reported"
    evidence: EvidenceLink = field(default_factory=EvidenceLink)
    label_raw: Optional[str] = None        # original data-row label, not parent category
    value_kind: str = "category_total"     # category_total | component (for Scope 3)
    hierarchy_note: Optional[str] = None


@dataclass
class OrgDisclosure:
    """One organization × reporting-year record ("Reported Scope Info", §3.3.2)."""
    org_id: Optional[str] = None
    org_name: Optional[str] = None
    reporting_year: Optional[int] = None
    reporting_boundary: Optional[str] = None
    certification: Optional[str] = None
    source_document: Optional[str] = None
    restated: bool = False
    superseded_by: Optional[str] = None
    asset_mentions: List[Dict[str, Any]] = field(default_factory=list)  # {mention,page,snippet}
    values: List[ExtractedValue] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    category_components: List[ExtractedValue] = field(default_factory=list)

    def get(self, field_name: str, verified_only: bool = False) -> Optional[ExtractedValue]:
        candidates = [v for v in self.values if v.field == field_name and v.value_kind != "component"
                      and (not verified_only or v.evidence.verified is True)]
        from scope3trace.common.perimeter import select_perimeter,reported_quantum
        candidates,_=select_perimeter(self,field_name,candidates)
        boundary_order = {'consolidated_group': 3, 'reported': 2, 'group_and_operational_control': 1}

        def rank(v):
            b = v.evidence.binding or {}
            coverage = b.get('coverage_boundary') or 'reported'
            esrs_label = bool(re.search({'scope3_total': r"(?i)gross\s+indirect\s*\(?\s*scope\s*3",
                                         'scope1': r"(?i)\bgross\s+scope\s*1\b|\bscope\s*1\s+gross\b",
                                         'scope2_location': r"(?i)\bgross\b.*\blocation[- ]based|\blocation[- ]based\s+gross\b",
                                         'scope2_market': r"(?i)\bgross\b.*\bmarket[- ]based|\bmarket[- ]based\s+gross\b"}
                                        .get(field_name, r"(?!)"), str(b.get('row_label') or v.label_raw or '')))
            # the ESRS E1-6 datapoint table itself ("Gross Scope 1 GHG
            # emissions" under an "E1-6" heading) is the regulated disclosure
            # (or titled "GHG emissions in accordance with ESRS")
            esrs_table = bool(re.search(r"E1\s*[-\u2010-\u2014]\s*6\b|(?i:\b(?:in accordance with|according to)\s+(?:the\s+)?ESRS\b)",
                                        str(b.get('organization_heading') or '')))
            return (v.evidence.verified is True,
                    esrs_table,
                    # organization-level record: the consolidated, report-wide
                    # disclosure outranks entity/segment/site tables
                    boundary_order.get(b.get('organization_boundary') or 'reported', 0),
                    coverage == 'reported',
                    b.get('column_boundary') != 'SBTi',
                    ({'location':2,'unspecified':1,'market':0}.get(b.get('accounting_basis','unspecified'),1) if field_name=='scope3_cat3'
                     # project convention: a Scope 3 total printed per Scope 2
                     # basis is recorded with its market-based figure
                     else {'market':2,'unspecified':1,'location':0}.get(
                         (b.get('accounting_basis') if b.get('accounting_basis') in ('location', 'market') else None) or ('market' if re.search(r"(?i)market[\s\-\u2010-\u2014]*based", str(b.get('row_label') or v.label_raw or ''))
                                                       else 'location' if re.search(r"(?i)location[\s\-\u2010-\u2014]*based", str(b.get('row_label') or v.label_raw or ''))
                                                       else 'unspecified'), 1) if field_name=='scope3_total' else 1),
                    # the ESRS E1-6 datapoint row ("Total gross indirect (Scope 3)
                    # GHG emissions") over other Scope 3 summaries of the report
                    bool(re.search({'scope3_total': r"(?i)gross\s+indirect\s*\(?\s*scope\s*3",
                                    'scope1': r"(?i)\bgross\s+scope\s*1\b|\bscope\s*1\s+gross\b",
                                    'scope2_location': r"(?i)\bgross\b.*\blocation[- ]based|\blocation[- ]based\s+gross\b",
                                    'scope2_market': r"(?i)\bgross\b.*\bmarket[- ]based|\bmarket[- ]based\s+gross\b"}
                                   .get(field_name, r"(?!)"), str(b.get('row_label') or v.label_raw or ''))),
                    bool(b.get('table_id')) or b.get('origin') == 'text_row',
                    -(reported_quantum(v) or float('inf')) if b.get('reporting_perimeter') else 0,
                    v.evidence.confidence, -(v.evidence.page or 10**6))
        if not candidates:
            return None
        if os.environ.get('S3T_DEBUG_RANK') == field_name:
            for v in candidates:
                print('RANK', field_name, v.value_raw, v.evidence.page, rank(v))
        best = max(candidates, key=rank)
        # A rounded summary ("4.0 Mt") restates a finer printed figure
        # ("4,019,507 t"): among equally ranked candidates whose amounts agree
        # within the coarser rounding, the most finely printed one is the
        # source amount. Different amounts are never resolved by precision.
        q_best = reported_quantum(best)
        if q_best and best.value_tco2e is not None:
            key = rank(best)[:6] + rank(best)[7:8]
            finer = [v for v in candidates if v is not best and rank(v)[:6] + rank(v)[7:8] == key
                     and v.value_tco2e is not None and (reported_quantum(v) or float('inf')) < q_best
                     # within one unit of the coarser figure: a summary rounded
                     # from unrounded parts can differ by a unit
                     and abs(v.value_tco2e - best.value_tco2e) <= q_best + 1e-9]
            if finer:
                best = min(finer, key=lambda v: (reported_quantum(v), -rank(v)[-2], v.evidence.page or 10**6))
        if field_name.startswith('scope3_cat') and best.value_tco2e is not None:
            # several category totals of one category in one table column
            # ("Total residential mortgages", ..., "Total assets"): the
            # sub-totals are parts, the largest is the category total
            bb = best.evidence.binding or {}
            same = [v for v in candidates if v.value_tco2e is not None and (v.evidence.binding or {}).get('table_id')
                    and (v.evidence.binding or {}).get('table_id') == bb.get('table_id')
                    and (v.evidence.binding or {}).get('column_id') == bb.get('column_id')
                    and rank(v)[:6] == rank(best)[:6]]
            if len(same) > 1:
                best = max(same, key=lambda v: v.value_tco2e)
        return best

    def selection_issue(self,field_name: str) -> Optional[str]:
        from scope3trace.common.perimeter import select_perimeter
        candidates=[v for v in self.values if v.field==field_name and v.value_kind!='component'
                    and v.evidence.verified is True]
        return select_perimeter(self,field_name,candidates)[1]

    def scope3_categories(self, verified_only: bool = False) -> List[ExtractedValue]:
        fields = sorted({v.field for v in self.values if v.field.startswith("scope3_cat")})
        return [v for v in (self.get(f, verified_only) for f in fields) if v is not None]

    def to_v21(self, verified_only: bool = True) -> Dict[str, Any]:
        """Serialize to v2.1 plus additive basis-unspecified Scope2 totals.

        Values explicitly rejected by the Verify Agent are excluded from the
        benchmark prediction by default.  They remain available in the full
        evidence/audit serialization, so verification never rewrites or
        destroys the Scope Agent's original extraction.
        """
        def selected(field: str) -> Optional[ExtractedValue]:
            return self.get(field, verified_only=verified_only)

        def val(field: str) -> Optional[float]:
            v = selected(field)
            return v.value_tco2e if v else None

        def first_not_none(*values: Optional[float]) -> Optional[float]:
            return next((v for v in values if v is not None), None)

        breakdown = {key: None for key in V21_CATEGORY_KEYS.values()}
        categories = self.scope3_categories(verified_only=verified_only)
        for v in categories:
            if v.category_no in V21_CATEGORY_KEYS:
                breakdown[V21_CATEGORY_KEYS[v.category_no]] = v.value_tco2e
        from scope3trace.common.rules import normalized_emission_mass_unit
        # A basis-unspecified Scope 2 amount is exported only when the report
        # gives no location- or market-based total (v2.1 has the two bases).
        scope2_total = (None if (selected("scope2_location") or selected("scope2_market"))
                        else selected("scope2"))
        # The report states its single Scope 2 method beside the total ("...
        # presented using the market-based method"): the total is that basis.
        declared = ((scope2_total.evidence.binding or {}).get('declared_scope2_basis')
                    if scope2_total and isinstance(scope2_total.evidence.binding, dict) else None)
        basis_val = {"location": val("scope2_location"), "market": val("scope2_market")}
        if declared in basis_val:
            basis_val[declared] = scope2_total.value_tco2e
            scope2_total = None
        return {
            "org_id": self.org_id,
            "org_name_standardized": self.org_name,
            "report_year": self.reporting_year,
            "reporting_boundary": self.reporting_boundary or "Unknown",
            "assurance_level": self.certification or "Unknown",
            "scope1_tco2e_value": val("scope1"),
            "scope1_provenance": "reported" if selected("scope1") else "unknown",
            # A basis-unspecified Scope2 amount is its own reported total.
            # Never copy it to either accounting-basis field.
            "scope2_total_value": scope2_total.value_tco2e if scope2_total else None,
            "scope2_total_unit": normalized_emission_mass_unit(scope2_total.unit_raw) if scope2_total else None,
            "scope2_location_tco2e_value": basis_val["location"],
            "scope2_market_tco2e_value": basis_val["market"],
            "scope2_provenance": "reported" if (
                scope2_total or basis_val["location"] is not None or basis_val["market"] is not None
                or selected("scope2_location") or selected("scope2_market")
            ) else "unknown",
            "scope3_summary": {
                "total_tco2e_value": val("scope3_total"),
                "breakdown": breakdown,
                "provenance": "reported" if (
                    selected("scope3_total") or categories) else "unknown",
            },
            "source_document": self.source_document,
            "schema_version": "v2.1",
            "evidence_pages": sorted({v.evidence.page for v in self.values
                                      if v.evidence.page is not None and (
                                          not verified_only or v.evidence.verified is True)}),
        }

    def to_v22(self) -> Dict[str, Any]:
        """Optional internal evidence-rich serialization.

        Table 1 and the paper's example annotation use v2.1.  This helper is
        retained for callers that need nested per-field evidence, but the main
        pipeline no longer emits a separate v2.2 result file.
        """
        legacy = self.to_v21(verified_only=True)

        def ev(value: Optional[ExtractedValue]) -> Optional[Dict[str, Any]]:
            if value is None:
                return None
            return {
                "page": value.evidence.page,
                "text_evidence": value.evidence.text_snippet,
            }

        scope2_location = self.get("scope2_location", verified_only=True)
        scope2_market = self.get("scope2_market", verified_only=True)
        scope2_total = self.get("scope2", verified_only=True)
        scope3_total = self.get("scope3_total", verified_only=True)
        scope3 = legacy["scope3_summary"]
        scope3["breakdown"] = {k: v for k, v in scope3["breakdown"].items()
                               if v is not None}
        scope3["total_evidence"] = ev(scope3_total)
        scope3["breakdown_evidence"] = {
            V21_CATEGORY_KEYS[v.category_no]: ev(v)
            for v in self.scope3_categories(verified_only=True)
            if v.category_no in V21_CATEGORY_KEYS
        }
        return {
            "org_id": self.org_id,
            "org_name_standardized": self.org_name,
            "report_year": self.reporting_year,
            "reporting_boundary": self.reporting_boundary or "Unknown",
            "assurance_level": self.certification or "Unknown",
            "scope1_tco2e_value": legacy["scope1_tco2e_value"],
            "scope1_provenance": legacy["scope1_provenance"],
            "scope1_evidence": ev(self.get("scope1", verified_only=True)),
            "scope2_total_value": legacy["scope2_total_value"],
            "scope2_total_unit": legacy["scope2_total_unit"],
            "scope2_location_tco2e_value": legacy["scope2_location_tco2e_value"],
            "scope2_market_tco2e_value": legacy["scope2_market_tco2e_value"],
            "scope2_provenance": legacy["scope2_provenance"],
            "scope2_evidence": {
                "total": ev(scope2_total),
                "location_based": ev(scope2_location),
                "market_based": ev(scope2_market),
            },
            "scope3_summary": scope3,
            "restated": self.restated,
            "superseded_by": self.superseded_by,
            "source_document": self.source_document,
            "schema_version": "v2.2",
        }

# Category number → v2.1 breakdown key (matches the gold annotation schema).
V21_CATEGORY_KEYS: Dict[int, str] = {
    1: "cat1_purchased_goods_tco2e", 2: "cat2_capital_goods_tco2e",
    3: "cat3_fuel_energy_tco2e", 4: "cat4_upstream_transport_tco2e",
    5: "cat5_waste_tco2e", 6: "cat6_business_travel_tco2e",
    7: "cat7_employee_commuting_tco2e", 8: "cat8_upstream_leased_tco2e",
    9: "cat9_downstream_transport_tco2e", 10: "cat10_processing_tco2e",
    11: "cat11_use_sold_tco2e", 12: "cat12_eol_sold_tco2e",
    13: "cat13_downstream_leased_tco2e", 14: "cat14_franchises_tco2e",
    15: "cat15_investments_tco2e",
}


# ---------------------------------------------------------------------------
# Asset pool (§3.3.1)
# ---------------------------------------------------------------------------

@dataclass
class AssetRecord:
    """One physical building asset in the "Asset Pool"."""
    property_id: str
    names: List[str] = field(default_factory=list)       # building name + aliases
    address_raw: Optional[str] = None
    address_std: Optional[str] = None                    # standardized (§3.3.1)
    postcode: Optional[str] = None
    grid_region: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    building_type: Optional[str] = None                  # canonical typology label
    building_type_method: Optional[str] = None           # inventory | rule_classifier | llm_classifier
    floor_area_m2: Optional[float] = None
    floor_area_source: Optional[str] = None
    year_built: Optional[int] = None
    satellite_image: Optional[str] = None                # supplementary context only (§3.3.4)
    semantic_tags: List[str] = field(default_factory=list)
    org_id: Optional[str] = None
    # per report_year activity indicators + any already-reported emissions
    activity: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)  # field -> reported/modeled/estimated


@dataclass
class EntityLink:
    """Report mention ↔ asset pool link (entity linking, §3.3.1/§3.3.2)."""
    mention: str
    property_id: Optional[str]
    score: float
    method: str                      # exact_key | fuzzy | llm_semantic | unlinked
    source_document: Optional[str] = None
    reporting_year: Optional[int] = None
    org_name: Optional[str] = None
    evidence_page: Optional[int] = None
    evidence_snippet: Optional[str] = None
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Derived components (§3.3.3, Appendix C) and diagnostics (§3.3.4, A.4)
# ---------------------------------------------------------------------------

@dataclass
class RelevanceVerdict:
    """Semantic relevance validation gate for estimation (§3.3.3)."""
    status: str                      # applicable | inapplicable | no_evidence
    score: float = 0.0
    method: str = "rule_overlap"     # rule_overlap | llm_semantic
    note: Optional[str] = None
    property_id: Optional[str] = None
    source_document: Optional[str] = None
    reporting_year: Optional[int] = None


@dataclass
class DerivedComponent:
    """One deterministic building-level estimate, always labeled (C.4)."""
    property_id: str
    report_year: Optional[int]
    field: str                       # scope2_tco2e | scope3_cat2_tco2e | scope3_cat13_tco2e
    value_tco2e: Optional[float]
    provenance: str                  # modeled | estimated  (never overwrites reported)
    method: str                      # formula id, e.g. "C.1: electricity_kwh x EF_grid"
    factor_source: Optional[str]
    inputs: Dict[str, Any] = field(default_factory=dict)
    relevance: Optional[RelevanceVerdict] = None
    quality: Optional[str] = None
    evidence: Optional[EvidenceLink] = None
    validation_status: str = "pending"  # reported | applicable | inapplicable | no_evidence
    validation_note: Optional[str] = None


@dataclass
class DiagnosticResult:
    """One consistency / logic / evidence check outcome (§3.3.4)."""
    check: str
    status: str                      # pass | warn | fail | not_applicable
    detail: str
    scope: Optional[str] = None      # org-level | building-level | cross-level


# ---------------------------------------------------------------------------
# Per-agent metering (Appendix D computational-overhead accounting)
# ---------------------------------------------------------------------------

@dataclass
class AgentUsage:
    agent: str
    model: Optional[str] = None
    wall_time_s: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    response_metadata: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The shared state object
# ---------------------------------------------------------------------------

@dataclass
class SharedState:
    documents: List[DocumentRecord] = field(default_factory=list)
    parsed_pages: Dict[str, List[ParsedPage]] = field(default_factory=dict)
    evidence: Dict[str, List[EvidenceItem]] = field(default_factory=dict)  # doc_id -> list
    asset_pool: List[AssetRecord] = field(default_factory=list)
    org_disclosures: List[OrgDisclosure] = field(default_factory=list)
    entity_links: List[EntityLink] = field(default_factory=list)
    derived_components: List[DerivedComponent] = field(default_factory=list)
    relevance_verdicts: List[RelevanceVerdict] = field(default_factory=list)
    diagnostics: List[DiagnosticResult] = field(default_factory=list)
    usage: List[AgentUsage] = field(default_factory=list)
    evaluation_report: Dict[str, Any] = field(default_factory=dict)
    phase_log: List[Dict[str, Any]] = field(default_factory=list)
    repair_history: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_issues: List[Dict[str, Any]] = field(default_factory=list)

    # -- structured, non-conversational coordination --------------------------
    def log_phase(self, agent: str, phase: str, detail: str) -> None:
        self.phase_log.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "agent": agent,
            "phase": phase,
            "detail": detail,
        })

    def asset_by_id(self, property_id: str) -> Optional[AssetRecord]:
        for a in self.asset_pool:
            if a.property_id == property_id:
                return a
        return None

    def links_for(self, disclosure: OrgDisclosure) -> List[EntityLink]:
        mentions = {m.get("mention") for m in disclosure.asset_mentions}
        return [l for l in self.entity_links
                if l.mention in mentions
                and l.source_document == disclosure.source_document
                and l.reporting_year == disclosure.reporting_year]

    def relevance_for(self, disclosure: OrgDisclosure,
                      property_id: str) -> Optional[RelevanceVerdict]:
        return next((v for v in reversed(self.relevance_verdicts)
                     if v.property_id == property_id
                     and v.source_document == disclosure.source_document
                     and v.reporting_year == disclosure.reporting_year), None)

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SharedState":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        state = cls()
        for d in raw.get("documents", []):
            state.documents.append(DocumentRecord(**d))
        for doc_id, pages in (raw.get("parsed_pages") or {}).items():
            state.parsed_pages[doc_id] = [ParsedPage(**p) for p in pages]
        for doc_id, items in (raw.get("evidence") or {}).items():
            state.evidence[doc_id] = [EvidenceItem(**it) for it in items]
        for a in raw.get("asset_pool", []):
            state.asset_pool.append(AssetRecord(**a))
        for d in raw.get("org_disclosures", []):
            collections = {}
            for key in ("values", "category_components"):
                collections[key] = []
                for v in d.pop(key, []):
                    ev = EvidenceLink(**v.pop("evidence", {}) or {})
                    collections[key].append(ExtractedValue(evidence=ev, **v))
            state.org_disclosures.append(OrgDisclosure(**collections, **d))
        for l in raw.get("entity_links", []):
            state.entity_links.append(EntityLink(**l))
        for c in raw.get("derived_components", []):
            rel = c.pop("relevance", None)
            ev = c.pop("evidence", None)
            state.derived_components.append(
                DerivedComponent(relevance=RelevanceVerdict(**rel) if rel else None,
                                 evidence=EvidenceLink(**ev) if ev else None, **c))
        for v in raw.get("relevance_verdicts", []):
            state.relevance_verdicts.append(RelevanceVerdict(**v))
        for g in raw.get("diagnostics", []):
            state.diagnostics.append(DiagnosticResult(**g))
        for u in raw.get("usage", []):
            state.usage.append(AgentUsage(**u))
        state.evaluation_report = raw.get("evaluation_report", {})
        state.phase_log = raw.get("phase_log", [])
        state.repair_history = raw.get("repair_history", [])
        state.unresolved_issues = raw.get("unresolved_issues", [])
        return state
