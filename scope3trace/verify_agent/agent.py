#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Agent — verification process (§3.3.4, Appendix A.4/E).

Supervises everything the Scope Agent produced, without ever producing or
correcting a value itself. Its LLM modules handle semantic validation only
and "can flag and return a failing value but never correct it" (Appendix E);
final numeric verification is deterministic — each extracted value is matched
against the Search Agent's verbatim evidence text by rules.

Modules (flowchart "Verification Process"):
- Evidence check          : deterministic value-in-text matching over the
                            Search evidence list + optional bounded LLM
                            semantic flagging (target/other-year/offset
                            confusion) — flags only;
- Relevance validation    : gates the Scope Agent's estimates — an estimation
                            rule counts toward cross-level alignment only when
                            the asset's attributes have a valid substitution
                            relationship with the disclosure boundary;
- Consistency Value Check : org-level logic checks + A.4 cross-level
                            constraints (building S1/S2 sums ≤ org totals,
                            market-based ≤ location-based Scope 2);
- Logic Check + report    : alignment diagnostics (applicable / inapplicable /
                            no_evidence) and the final "Evaluation Report",
                            including Appendix D style per-agent overhead.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from scope3trace.common.config import RELEVANCE_RULE_THRESHOLD
from scope3trace.common.llm import BoundedLLM
from scope3trace.common.prompts import (
    RELEVANCE_VALIDATION_PROMPT,
    SYSTEM_COMMON,
    VALUE_FLAG_PROMPT,
)
from scope3trace.common.rules import (
    content_tokens,
    category_hierarchy,
    cross_level_checks,
    ground_evidence_snippet,
    org_logic_checks,
    table_year_check,
    verify_field_context,
    verify_value_in_text,
)
from scope3trace.common.state import (
    AssetRecord,
    DiagnosticResult,
    OrgDisclosure,
    RelevanceVerdict,
    SharedState,
)


class VerifyAgent:
    """Flags, gates, and reports — never extracts, estimates, or corrects."""

    name = "verify"

    def __init__(self, llm: BoundedLLM, state: SharedState, rule_only: bool = False,
                 llm_value_flags: bool = True,
                 evidence_grounding: bool = True) -> None:
        self.llm = llm
        self.state = state
        self.rule_only = rule_only or (
            not llm.available and not getattr(llm, "strict_failures", False))
        self.evidence_grounding = evidence_grounding
        # The paper's Verify Agent performs semantic field/year validation in
        # addition to deterministic numeric grounding, so this is on by default.
        self.llm_value_flags = llm_value_flags and not self.rule_only
        self._semantic_cache = {}

    # ------------------------------------------------------------------
    # 1) Evidence check (final numeric verification — deterministic)
    # ------------------------------------------------------------------
    def evidence_check(self, disclosure: OrgDisclosure) -> List[DiagnosticResult]:
        out: List[DiagnosticResult] = []
        if not self.evidence_grounding:
            for value in disclosure.values + disclosure.category_components:
                value.evidence.verified = None
                value.evidence.verification_note = "evidence grounding disabled for ablation"
            out.append(DiagnosticResult(
                "evidence:summary", "not_applicable",
                f"{disclosure.source_document}: evidence grounding disabled",
                "org-level"))
            self.state.diagnostics.extend(out)
            return out
        items = self.state.evidence.get(disclosure.source_document or "", [])
        text_by_page = {it.page: it.text for it in items}
        n_fail = 0
        candidates = disclosure.values + disclosure.category_components
        for v in candidates:
            from scope3trace.scope_agent.table_structure import _noninventory_quantity
            page_text = text_by_page.get(v.evidence.page or -1, "")
            parsed = next((p for p in self.state.parsed_pages.get(disclosure.source_document or "", [])
                           if p.page == v.evidence.page), None)
            # A weaker downstream span match may never resurrect a rejection
            # from normalization, field context, year binding or a prior pass.
            if _noninventory_quantity(v.label_raw or ''):
                ok, note = False, 'independent erased/avoided-emissions indicator is not an inventory emissions value'
            elif v.evidence.verified is False or v.value_tco2e is None:
                ok, note = False, v.evidence.verification_note or "prior validation rejected value"
            elif v.evidence.binding and v.evidence.binding.get("table_id"):
                from scope3trace.scope_agent.table_structure import validate_binding
                ok, note = validate_binding(v, parsed.table_structures if parsed else [],
                                            parsed.raw_text if parsed else "",
                                            document_pages=self.state.parsed_pages.get(disclosure.source_document or '',[]))
            elif v.evidence.binding and v.evidence.binding.get("origin") == "text_row":
                from scope3trace.scope_agent.text_binding import validate_text_binding
                from scope3trace.scope_agent.table_structure import footnoted_tokens
                ok, note = validate_text_binding(v, parsed.raw_text if parsed else "",
                                                 getattr(parsed, "number_locale", None) if parsed else None,
                                                 footnoted_tokens(parsed) if parsed else None)
            elif v.evidence.binding and v.evidence.binding.get("origin") == "chart_vision":
                from scope3trace.scope_agent.chart_reader import validate_chart_binding
                ok, note = validate_chart_binding(v, parsed)
            elif parsed and (parsed.table_structures or parsed.reconstructed_tables):
                ok, note = False, "table candidate missing structured binding; span-only fallback prohibited"
            elif v.evidence.binding and v.evidence.binding.get('status') != 'span_only':
                ok, note = False, "binding does not have an accepted validation status"
            else:
                ok, note = verify_value_in_text(
                    v.value_raw, None if v.value_raw else v.value_tco2e, page_text)
                kind, _, fragment = category_hierarchy(v.category_no, v.value_raw, v.label_raw, page_text)
                if ok and kind == "component" and v.value_kind != "component":
                    ok, note = False, "category component cannot be exported as category total"
                if ok and not (kind and fragment) and not verify_field_context(
                        v.field, v.value_raw, None, page_text, v.category_no):
                    ok, note = False, "field/category label not found near numeric span"
                year_ok, year_note = table_year_check(v.value_raw, v.report_year, page_text)
                if year_ok is False:
                    ok, note = False, year_note
                if ok and parsed:
                    import hashlib
                    b = v.evidence.binding or {}
                    start,end = b.get('source_span',[-1,-1])
                    expected = {'document':disclosure.source_document,'page':v.evidence.page,
                                'field':v.field,'year':v.report_year,'parent_category':v.category_no,
                                'unit_raw':v.unit_raw,'value_raw':v.value_raw,'value_tco2e':v.value_tco2e,
                                'source_sha256':hashlib.sha256(parsed.raw_text.encode()).hexdigest(),
                                'quantity_type':'inventory'}
                    if (any(b.get(k)!=val for k,val in expected.items()) or start<0 or end<=start
                            or parsed.raw_text[start:end]!=b.get('source_fragment')):
                        ok,note = False,"text-origin evidence binding missing, changed or inconsistent"
            if ok:
                from scope3trace.verify_agent.feedback import boundary_check
                ok, boundary_note = boundary_check(v,disclosure.source_document,self.state)
                if not ok:
                    note = boundary_note
            # chart values were interpreted from the page image by the vision
            # backbone; a text-only semantic judge cannot see the chart
            if ok and self.llm_value_flags and (v.evidence.binding or {}).get("origin") != "chart_vision":
                ok, note = self._llm_flag(v, page_text)
            if not ok:
                n_fail += 1
                # flag, never correct: keep the value, downgrade its evidence
                v.evidence.verified = False
                v.evidence.verification_note = note
                v.evidence.confidence = round(max(0.0, v.evidence.confidence - 0.3), 2)
                out.append(DiagnosticResult(
                    "evidence:value_grounding", "fail",
                    f"{disclosure.source_document} p{v.evidence.page} {v.field}="
                    f"{v.value_raw}: {note}", "org-level"))
            else:
                v.evidence.verified = True
                v.evidence.verification_note = note
        out.append(DiagnosticResult(
            "evidence:summary", "pass" if n_fail == 0 else "warn",
            f"{disclosure.source_document}: {len(candidates) - n_fail}/"
            f"{len(candidates)} values/components evidence-verified", "org-level"))
        self.state.diagnostics.extend(out)
        return out

    def _llm_flag(self, v, page_text: str) -> tuple:
        import json
        import hashlib
        from scope3trace.verify_agent.feedback import candidate_signature
        cache_key = hashlib.sha256(json.dumps([candidate_signature(v),page_text],
                                              sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if cache_key in self._semantic_cache:
            return self._semantic_cache[cache_key]
        context = ground_evidence_snippet(v.value_raw, v.value_tco2e, page_text,
                                          min_chars=300, max_chars=1200)
        obj = self.llm.complete_json(
            SYSTEM_COMMON,
            VALUE_FLAG_PROMPT.format(field=v.field, value_raw=v.value_raw or "",
                                     field_definition=(v.category_canonical or {
                                         "scope1": "Direct GHG emissions",
                                         "scope2": "Indirect GHG emissions from purchased energy",
                                         "scope2_location": "Location-based Scope 2 GHG emissions",
                                         "scope2_market": "Market-based Scope 2 GHG emissions",
                                         "scope3_total": "Reported aggregate of Scope 3 categories",
                                     }.get(v.field, v.field)),
                                     category_no=v.category_no or "not applicable",
                                     category_canonical=v.category_canonical or "not applicable",
                                     category_label_raw=v.category_label_raw or "not applicable",
                                     label_raw=v.label_raw or "not supplied",
                                     value_kind=v.value_kind,
                                     hierarchy_note=(v.hierarchy_note or "not determined") +
                                     "\nMachine-checked binding: " + json.dumps(v.evidence.binding, ensure_ascii=False),
                                     numeric_evidence=(((v.evidence.binding or {}).get("cell_summary") or "")
                                                       + (" || source text: " if (v.evidence.binding or {}).get("cell_summary") else "")
                                                       + (v.evidence.text_snippet or context or "")),
                                     unit_raw=v.unit_raw or "",
                                     report_year=v.report_year or "unknown",
                                     # A local numeric window can omit year and
                                     # unit headers, especially on continued tables.
                                     snippet=page_text or context or v.evidence.text_snippet or ""))
        # An unavailable semantic check is an abstention, not evidence against
        # the value: the deterministic grounding above already passed (App. E:
        # final numeric verification is rule-based; the LLM may only flag).
        result = ((True, "value grounded; semantic validation unavailable (abstained)") if obj is None else
                  (False, f"semantic flag: {str(obj.get('reason') or '')[:150]}") if obj.get('supported') is not True else
                  (True, "value grounded; semantically supported"))
        self._semantic_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # 2) Semantic relevance validation (gates estimation for alignment)
    # ------------------------------------------------------------------
    def validate_relevance(self, disclosure: OrgDisclosure) -> Dict[str, RelevanceVerdict]:
        """One verdict per linked asset; attached to its derived components."""
        verdicts: Dict[str, RelevanceVerdict] = {}
        links = self.state.links_for(disclosure)
        linked_ids = sorted({l.property_id for l in links if l.property_id})

        for pid in linked_ids:
            asset = self.state.asset_by_id(pid)
            if asset is None:
                continue
            mention_snippets = "; ".join(filter(None, (
                l.evidence_snippet for l in links if l.property_id == pid)))[:600]
            verdicts[pid] = self._relevance_one(disclosure, asset, mention_snippets)

        for pid, verdict in verdicts.items():
            verdict.property_id = pid
            verdict.source_document = disclosure.source_document
            verdict.reporting_year = disclosure.reporting_year
            self.state.relevance_verdicts.append(verdict)
            self.state.diagnostics.append(DiagnosticResult(
                "alignment:relevance", "pass" if verdict.status == "applicable" else "warn",
                f"{disclosure.source_document} ↔ asset {pid}: {verdict.status} "
                f"({verdict.method}, score {verdict.score:.2f})"
                + (f" — {verdict.note}" if verdict.note else ""), "cross-level"))
        self.state.log_phase(self.name, "relevance",
                             f"{disclosure.source_document}: "
                             f"{sum(1 for x in verdicts.values() if x.status == 'applicable')}"
                             f"/{len(verdicts)} linked assets applicable")
        return verdicts

    def _relevance_one(self, disclosure: OrgDisclosure, asset: AssetRecord,
                       mention_snippets: str) -> RelevanceVerdict:
        descriptor = ", ".join(filter(None, [
            asset.building_type, " / ".join(asset.names) or None, asset.address_raw]))
        context = " ".join(filter(None, [
            disclosure.reporting_boundary, mention_snippets]))
        a_tokens, c_tokens = set(content_tokens(descriptor)), set(content_tokens(context))
        score = len(a_tokens & c_tokens) / len(a_tokens) if a_tokens else 0.0

        if not context.strip():
            return RelevanceVerdict(status="no_evidence", score=0.0, method="rule_overlap",
                                    note="no boundary text or mention snippets")
        if self.rule_only:
            status = "applicable" if score >= RELEVANCE_RULE_THRESHOLD else "inapplicable"
            return RelevanceVerdict(status=status, score=round(score, 3), method="rule_overlap")

        obj = self.llm.complete_json(
            SYSTEM_COMMON,
            RELEVANCE_VALIDATION_PROMPT.format(
                org_name=disclosure.org_name or "?",
                boundary_text=(disclosure.reporting_boundary or "")[:400],
                mention_snippets=mention_snippets[:600],
                asset_descriptor=descriptor[:300]))
        if obj and obj.get("status") in ("applicable", "inapplicable"):
            return RelevanceVerdict(status=obj["status"], score=round(score, 3),
                                    method="llm_semantic",
                                    note=str(obj.get("reason") or "")[:200] or None)
        status = "applicable" if score >= RELEVANCE_RULE_THRESHOLD else "inapplicable"
        return RelevanceVerdict(status=status, score=round(score, 3), method="rule_overlap",
                                note="LLM unavailable; rule fallback")

    # ------------------------------------------------------------------
    # 3) Consistency Value Check + Logic Check (§3.3.4, A.4)
    # ------------------------------------------------------------------
    def consistency_checks(self, disclosure: OrgDisclosure) -> List[DiagnosticResult]:
        def val(f: str) -> Optional[float]:
            v = disclosure.get(f,verified_only=True)
            return v.value_tco2e if v else None

        cat_values = {v.category_no: v.value_tco2e for v in disclosure.scope3_categories(verified_only=True)
                      if v.category_no is not None and v.value_tco2e is not None}
        def first_not_none(*values: Optional[float]) -> Optional[float]:
            return next((v for v in values if v is not None), None)

        out = org_logic_checks(
            disclosure.org_name or disclosure.source_document or "org",
            disclosure.reporting_year,
            val("scope1"), val("scope2_location"),
            val("scope2_market"),
            val("scope3_total"), cat_values, s2_total=val("scope2"))

        # A.4 cross-level constraint over evidence-linked, relevance-applicable assets
        links = self.state.links_for(disclosure)
        linked_ids = {l.property_id for l in links if l.property_id}
        year = disclosure.reporting_year
        s1_sum = s2_sum = 0.0
        s1_seen = s2_seen = False
        for pid in linked_ids:
            asset = self.state.asset_by_id(pid)
            if asset is None or year is None:
                continue
            act = asset.activity.get(str(year), {})
            if act.get("scope1_tco2e_reported") is not None:
                s1_sum += act["scope1_tco2e_reported"]
                s1_seen = True
            comp = next((c for c in self.state.derived_components
                         if c.property_id == pid and c.report_year == year
                         and c.field == "scope2_tco2e"
                         and (c.provenance == "reported" or
                              (self.state.relevance_for(disclosure, pid) is not None and
                               self.state.relevance_for(disclosure, pid).status == "applicable"))), None)
            if comp and comp.value_tco2e is not None:
                s2_sum += comp.value_tco2e
                s2_seen = True
        out.extend(cross_level_checks(
            disclosure.org_name or disclosure.source_document or "org", year,
            val("scope1"), first_not_none(val("scope2_location"), val("scope2")),
            s1_sum if s1_seen else None, s2_sum if s2_seen else None, len(linked_ids)))
        self.state.diagnostics.extend(out)
        return out

    def apply_relevance_gate(self) -> None:
        """Gate modeled/estimated components without changing their values."""
        for comp in self.state.derived_components:
            if comp.provenance == "reported":
                if comp.evidence is not None and comp.evidence.verified is True:
                    comp.validation_status = "reported"
                    comp.validation_note = "directly reported; evidence verified"
                else:
                    comp.validation_status = "rejected_evidence"
                    comp.validation_note = "reported value lacks final verified evidence"
                continue
            verdicts = [v for v in self.state.relevance_verdicts
                        if v.property_id == comp.property_id
                        and v.reporting_year == comp.report_year]
            if any(v.status == "applicable" for v in verdicts):
                comp.validation_status = "applicable"
                comp.validation_note = "at least one evidence-scoped disclosure boundary is applicable"
            elif verdicts:
                comp.validation_status = "inapplicable"
                comp.validation_note = "linked disclosure boundary rejected the substitution"
            else:
                comp.validation_status = "no_evidence"
                comp.validation_note = "no document/year-scoped attribution evidence"
        counts = Counter(c.validation_status for c in self.state.derived_components)
        self.state.log_phase(self.name, "gate", f"derived component gate: {dict(counts)}")

    # ------------------------------------------------------------------
    # 4) Evaluation Report
    # ------------------------------------------------------------------
    def build_evaluation_report(self) -> Dict[str, Any]:
        docs = []
        for d in self.state.org_disclosures:
            links = self.state.links_for(d)
            n_ver = sum(1 for v in d.values if v.evidence.verified)
            confs = [v.evidence.confidence for v in d.values]
            docs.append({
                "source_document": d.source_document,
                "org_name": d.org_name,
                "reporting_year": d.reporting_year,
                "n_evidence_items": len(self.state.evidence.get(d.source_document or "", [])),
                "n_values": len(d.values),
                "n_category_components": len(d.category_components),
                "n_verified_category_components": sum(
                    v.evidence.verified is True for v in d.category_components),
                "n_evidence_verified": n_ver,
                "mean_confidence": round(sum(confs) / len(confs), 3) if confs else None,
                "n_asset_mentions": len(d.asset_mentions),
                "n_linked_assets": sum(1 for l in links if l.property_id),
            })

        accepted = [c for c in self.state.derived_components
                    if c.validation_status in ("reported", "applicable")]
        prov = Counter(c.provenance for c in accepted)
        align = Counter(v.status for v in self.state.relevance_verdicts)
        diag = Counter(g.status for g in self.state.diagnostics)

        report = {
            "run_summary": {
                "n_documents": len(self.state.documents),
                "n_disclosures": len(self.state.org_disclosures),
                "n_assets": len(self.state.asset_pool),
                "n_entity_links": sum(1 for l in self.state.entity_links if l.property_id),
                "n_derived_components": len(self.state.derived_components),
                "n_accepted_components": len(accepted),
            },
            "per_document": docs,
            "provenance_breakdown": dict(prov),           # Appendix C.4 labels
            "alignment_status": dict(align),              # applicable/inapplicable/no_evidence
            "diagnostics_summary": dict(diag),
            "agent_overhead": [                            # Appendix D style
                {"agent": u.agent, "model": u.model, "wall_time_s": round(u.wall_time_s, 2),
                 "calls": u.calls, "tokens_in": u.tokens_in, "tokens_out": u.tokens_out,
                 "cost_usd": u.cost_usd}
                for u in self.state.usage],
        }
        self.state.evaluation_report = report
        self.state.log_phase(self.name, "report",
                             f"evaluation report built: {report['run_summary']}")
        return report
