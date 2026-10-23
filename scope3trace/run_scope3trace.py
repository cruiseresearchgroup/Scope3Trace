#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scope3Trace orchestrator — fixed, non-conversational pipeline (§3.3).

Restricting agent autonomy is a deliberate, empirically validated design
choice: under matched conditions, free-form conversational orchestration
repeatedly overwrites originally correct values (Appendix F). The agents
therefore run in a fixed order over shared state and cross-validate each
other's outputs; there is no open-ended agent dialogue anywhere.

    Phase 1  Search Agent : acquire documents, parse (+OCR), and RETURN the
                            verbatim evidence list per document.
    Phase 2  Scope Agent  : build the Asset Pool, extract org disclosures
                            from the evidence, link asset mentions, and run
                            deterministic estimation (Appendix C).
    Phase 3  Verify Agent : deterministic evidence checks (flag, never
                            correct), semantic relevance validation,
                            consistency + logic checks (A.4), and the
                            Evaluation Report.

Usage examples
--------------
# Organization name -> report search -> extraction -> verification:
python -m scope3trace.run_scope3trace --search-org "Heineken N.V." --report-year 2025 \
    --model deepseek-chat -o out/heineken_2025

# Local PDFs, Rule-only baseline (no API key needed):
python -m scope3trace.run_scope3trace --pdf-dir reports/ --rule-only -o out/rule_only

# With a building inventory (Asset Pool) for entity linking and estimation:
python -m scope3trace.run_scope3trace --pdf report.pdf --model deepseek-chat \
    --inventory-csv buildings_v2.1.csv --entity-csv entity_aliases.csv -o out/buildings

# Heterogeneous per-agent backbones (Appendix E):
python -m scope3trace.run_scope3trace --pdf report.pdf \
    --search-model gemini-2.0-flash --scope-model deepseek-chat --verify-model gpt-4o \
    -o out/hetero_run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    def load_dotenv() -> bool:
        """Minimal .env reader used when python-dotenv is not installed.

        Reads KEY=VALUE lines (``#`` comments, optional ``export`` and quotes)
        from ./.env, else from the repository root; variables already set in
        the environment are not overridden."""
        for path in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, value = (x.strip() for x in line.split("=", 1))
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                elif " #" in value:
                    value = value.split(" #", 1)[0].rstrip()
                if key and key not in os.environ:
                    os.environ[key] = value
            return True
        return False

from scope3trace.common.config import DEFAULT_BACKBONE, AgentBackbones
from scope3trace.common.llm import BoundedLLM
from scope3trace.scope_agent.agent import ScopeAgent
from scope3trace.search_agent.agent import SearchAgent
from scope3trace.common.state import SharedState
from scope3trace.evaluation.table1_eval import (
    render_score_markdown,
    score_prediction_dir,
    write_predictions_for_gold,
)
from scope3trace.verify_agent.agent import VerifyAgent

ROOT = Path(__file__).resolve().parents[1]


def collect_pdfs(args: argparse.Namespace, search: SearchAgent) -> None:
    if args.manifest:
        search.acquire_from_manifest(Path(args.manifest), ROOT, limit=args.limit)
    pdfs: List[Path] = []
    if args.pdf:
        pdfs.extend(Path(p) for p in args.pdf)
    if args.pdf_dir:
        found = sorted(Path(args.pdf_dir).glob("*.pdf"))
        if args.limit is not None:
            found = found[: args.limit]
        pdfs.extend(found)
    if pdfs:
        search.acquire_local(pdfs, org_hint=args.org_name,
                             org_id=args.org_id, year_hint=args.report_year)
    for org in (args.search_org or []):
        out_dir = Path(args.out_dir)
        search.acquire_from_search(org, args.report_year, out_dir / "downloads",
                                   max_reports=args.search_max_reports, org_id=args.org_id,
                                   backend=args.search_backend,
                                   manifest_path=out_dir / "search_manifest.csv")


def write_outputs(state: SharedState, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state.save(out_dir / "state.json")

    # Paper/gold evaluation schema (Appendix H.10).
    with (out_dir / "organization_disclosures.jsonl").open("w", encoding="utf-8") as f:
        for d in state.org_disclosures:
            f.write(json.dumps(d.to_v21(), ensure_ascii=False) + "\n")

    # full-detail org records with per-value evidence links
    with (out_dir / "organization_disclosures_evidence.jsonl").open("w", encoding="utf-8") as f:
        for d in state.org_disclosures:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

    with (out_dir / "parsed_pages.jsonl").open("w", encoding="utf-8") as f:
        for pages in state.parsed_pages.values():
            for page in pages:
                f.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")

    # derived building-level components
    comp_path = out_dir / "derived_components.csv"
    fields = ["property_id", "report_year", "field", "value_tco2e", "provenance",
              "method", "factor_source", "quality", "validation_status",
              "validation_note", "evidence_page", "evidence_text", "inputs"]

    def write_components(path: Path, include_rejected: bool) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for c in state.derived_components:
                if not include_rejected and c.validation_status not in ("reported", "applicable"):
                    continue
                if (not include_rejected and c.provenance == "reported"
                        and (c.evidence is None or c.evidence.verified is not True)):
                    continue
                w.writerow({
                    "property_id": c.property_id, "report_year": c.report_year,
                    "field": c.field, "value_tco2e": c.value_tco2e,
                    "provenance": c.provenance, "method": c.method,
                    "factor_source": c.factor_source, "quality": c.quality,
                    "validation_status": c.validation_status,
                    "validation_note": c.validation_note,
                    "evidence_page": c.evidence.page if c.evidence else None,
                    "evidence_text": c.evidence.text_snippet if c.evidence else None,
                    "inputs": json.dumps(c.inputs, ensure_ascii=False),
                })

    write_components(comp_path, include_rejected=False)
    write_components(out_dir / "derived_components_audit.csv", include_rejected=True)

    with (out_dir / "evaluation_report.json").open("w", encoding="utf-8") as f:
        json.dump(state.evaluation_report, f, ensure_ascii=False, indent=2)
    (out_dir / "evaluation_report.md").write_text(render_report_md(state), encoding="utf-8")


def render_report_md(state: SharedState) -> str:
    r = state.evaluation_report
    lines = ["# Scope3Trace Evaluation Report", ""]
    rs = r.get("run_summary", {})
    lines += [
        f"- documents: {rs.get('n_documents')}  |  disclosures: {rs.get('n_disclosures')}",
        f"- assets in pool: {rs.get('n_assets')}  |  entity links: {rs.get('n_entity_links')}",
        f"- derived components: {rs.get('n_derived_components')}",
        "",
        "## Per document", "",
        "| document | org | year | evidence items | values | verified | mean conf | mentions | linked |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in r.get("per_document", []):
        lines.append(
            f"| {d['source_document']} | {d.get('org_name') or '—'} | "
            f"{d.get('reporting_year') or '—'} | {d['n_evidence_items']} | {d['n_values']} | "
            f"{d['n_evidence_verified']} | {d.get('mean_confidence') or '—'} | "
            f"{d['n_asset_mentions']} | {d['n_linked_assets']} |")
    lines += ["", "## Provenance breakdown (Appendix C.4)",
              f"```json\n{json.dumps(r.get('provenance_breakdown', {}), indent=2)}\n```",
              "## Alignment status (§3.3.4)",
              f"```json\n{json.dumps(r.get('alignment_status', {}), indent=2)}\n```",
              "## Diagnostics summary",
              f"```json\n{json.dumps(r.get('diagnostics_summary', {}), indent=2)}\n```",
              "", "## Per-agent overhead (Appendix D)", "",
              "| agent | model | LLM wall time (s) | calls | in tok. | out tok. | cost (USD) |",
              "|---|---|---|---|---|---|---|"]
    for u in r.get("agent_overhead", []):
        lines.append(f"| {u['agent']} | {u.get('model') or '—'} | {u['wall_time_s']} | "
                     f"{u['calls']} | {u['tokens_in']} | {u['tokens_out']} | {u['cost_usd']} |")
    pt = r.get("phase_wall_time_s")
    if pt:
        lines += ["", "## Phase wall time (s)",
                  f"```json\n{json.dumps(pt, indent=2)}\n```"]
    tools = r.get("content_extraction_tools")
    if tools:
        lines += ["", "## Content-extraction tool usage", "",
                  "| tool/stage | status | detail |",
                  "|---|---|---|"]
        for name, info in tools.items():
            lines.append(
                f"| {name} | {info.get('status', 'unknown')} | "
                f"{info.get('detail', '')} |")
    table1 = r.get("table1")
    if table1:
        lines += ["", "## Table 1 exact-value score", "",
                  "| Scope 1 F1 | Scope 2 F1 | Scope 3 F1 | Category F1 | EM |",
                  "|---:|---:|---:|---:|---:|",
                  (f"| {table1['scope1_f1']:.3f} | {table1['scope2_f1']:.3f} | "
                   f"{table1['scope3_f1']:.3f} | {table1['category_f1']:.3f} | "
                   f"{table1['em']:.3f} |")]
    if state.diagnostics:
        lines += ["", "## Failed / warned checks", ""]
        for g in state.diagnostics:
            if g.status in ("fail", "warn"):
                lines.append(f"- **{g.status}** `{g.check}` — {g.detail}")
    return "\n".join(lines) + "\n"


def content_extraction_tool_usage(state: SharedState, args: argparse.Namespace,
                                  table1_scored: bool) -> dict:
    pages = [page for doc_pages in state.parsed_pages.values() for page in doc_pages]
    values = [value for disclosure in state.org_disclosures for value in disclosure.values]
    evidence = [item for items in state.evidence.values() for item in items]
    ocr_pages = sum(page.parse_method.startswith("ocr") for page in pages)
    failed_ocr = any("OCR unavailable/failed" in entry.get("detail", "")
                     for entry in state.phase_log)
    table_pages = sum(bool(page.reconstructed_tables) for page in pages)
    sentence_count = sum(len(item.sentence_ids) for item in evidence)
    llm_values = sum(value.evidence.method == "llm_table_extraction" for value in values)
    rule_values = sum(value.evidence.method == "rule_parser" for value in values)
    verified = sum(value.evidence.verified is True for value in values)
    rejected = sum(value.evidence.verified is False for value in values)

    if args.rule_only:
        sentence_status = "not_applicable_rule_only"
    elif args.no_sentence_localization:
        sentence_status = "disabled_ablation"
    else:
        sentence_status = "used" if sentence_count else "enabled_not_triggered"
    if args.no_table_reconstruction:
        table_status = "disabled_ablation"
    else:
        table_status = "used" if table_pages else "enabled_not_triggered"
    if ocr_pages:
        ocr_status = "used"
    elif failed_ocr:
        ocr_status = "attempted_failed"
    else:
        ocr_status = "enabled_not_triggered"

    return {
        "document_acquisition": {
            "status": "used",
            "detail": "local/manifest inputs; live Google Search is external to this runner",
        },
        "pdf_text_parsing": {
            "status": "used" if pages else "not_triggered",
            "detail": f"{len(pages)} page(s)",
        },
        "ocr": {"status": ocr_status, "detail": f"{ocr_pages} OCR page(s)"},
        "page_localization": {
            "status": "disabled_ablation" if args.no_page_localization else "used",
            "detail": f"{len(evidence)} evidence page(s)",
        },
        "sentence_localization": {
            "status": sentence_status,
            "detail": f"{sentence_count} selected sentence(s)",
        },
        "table_reconstruction": {
            "status": table_status,
            "detail": f"{table_pages} page(s) with reconstructed tables",
        },
        "llm_table_interpretation": {
            "status": "used" if llm_values else (
                "not_applicable_rule_only" if args.rule_only else "not_triggered"),
            "detail": f"{llm_values} retained value(s); {rule_values} rule value(s)",
        },
        "numeric_unit_normalization": {
            "status": "used" if values else "not_triggered",
            "detail": f"{len(values)} retained normalized value(s)",
        },
        "scope3_category_mapping": {
            "status": "used" if any(v.category_no for v in values) else "not_triggered",
            "detail": f"{sum(v.category_no is not None for v in values)} category value(s)",
        },
        "evidence_grounding_and_verify": {
            "status": "disabled_ablation" if args.no_evidence_grounding else "used",
            "detail": f"{verified} verified; {rejected} rejected",
        },
        "gold_scoring": {
            "status": "used" if table1_scored else "not_requested",
            "detail": "exact-value Table 1 scorer" if table1_scored else "",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scope3Trace three-agent pipeline (Search → Scope → Verify)")
    src = ap.add_argument_group("document sources (Search Agent)")
    src.add_argument("--pdf", nargs="*", default=None, help="Report PDF path(s).")
    src.add_argument("--pdf-dir", default=None, help="Directory of report PDFs.")
    src.add_argument("--manifest", default=None,
                     help="Manifest CSV with a local_path column, e.g. the "
                          "search_manifest.csv of an earlier --search-org run")
    src.add_argument("--search-org", nargs="*", default=None,
                     help="Organization name(s) whose report PDFs are searched for on the web, "
                          "downloaded and identity-checked; use with --report-year.")
    src.add_argument("--search-backend", choices=["openrouter", "google"], default=None,
                     help="openrouter (default; LLM with OpenRouter web search, OPENROUTER_API_KEY) or "
                          "google (Custom Search JSON API, GOOGLE_API_KEY + GOOGLE_CSE_ID).")
    src.add_argument("--search-max-reports", type=int, default=1,
                     help="PDFs downloaded per searched organization.")
    src.add_argument("--limit", type=int, default=None, help="Max documents.")
    src.add_argument("--org-name", default=None, help="Organization hint for local PDFs.")
    src.add_argument("--org-id", default=None, help="Organization id for local PDFs.")
    src.add_argument("--report-year", type=int, default=None,
                     help="Reporting-year hint for local PDFs.")

    inv = ap.add_argument_group("asset inventory (Scope Agent)")
    inv.add_argument("--inventory-csv", default=None,
                     help="v2.1 building inventory CSV (property_id × report_year rows).")
    inv.add_argument("--entity-csv", default=None,
                     help="Optional facility aliases CSV (facility_name, responsible_entity, building_owner).")
    inv.add_argument("--grid-factor-csv", default=None,
                     help="Optional CSV region/state,report_year,factor_t_per_kwh for EF_grid.")
    inv.add_argument("--asset-limit", type=int, default=None,
                     help="Cap number of assets ingested (smoke runs).")

    mdl = ap.add_argument_group("backbones (Appendix E)")
    mdl.add_argument("--model", default=DEFAULT_BACKBONE,
                     help=f"Shared LLM backbone (default {DEFAULT_BACKBONE}).")
    mdl.add_argument("--search-model", default=None)
    mdl.add_argument("--scope-model", default=None)
    mdl.add_argument("--verify-model", default=None)
    mdl.add_argument("--rule-only", action="store_true",
                     help="No LLM calls anywhere (= Rule-only baseline).")
    mdl.add_argument("--no-llm-value-flags", action="store_false", dest="llm_value_flags",
                     help="Disable Verify semantic field/year checks (not paper-equivalent).")
    mdl.set_defaults(llm_value_flags=True)
    mdl.add_argument("--llm-classify-types", action="store_true",
                     help="LLM fallback for unlabeled building types "
                          "(one call per unlabeled asset — costly on large pools).")
    mdl.add_argument("--strict-llm", action="store_true",
                     help="Fail the run if any LLM call is unavailable or malformed.")
    mdl.add_argument("--allow-rule-fallback", action="store_true",
                     help="Allow silent LLM-to-rule fallback in a gold evaluation "
                          "(not suitable for a paper Table 1 run).")
    # The paper's pipeline is a single Search -> Scope -> Verify pass; the
    # repair loop is an optional extension (measured: no recovered fields).
    mdl.add_argument('--max-repair-rounds',type=int,choices=range(4),default=0,
                     help='Bounded Verify -> Search/Scope repair rounds after initial extraction (0 disables).')
    mdl.add_argument('--repair-page-budget',type=int,default=8,
                     help='Maximum source pages revisited per document per repair round.')

    ablation = ap.add_argument_group("Table 1 component ablations")
    ablation.add_argument("--no-page-localization", action="store_true",
                          help="Process every non-empty page.")
    ablation.add_argument("--no-sentence-localization", action="store_true",
                          help="Pass complete located pages to extraction.")
    ablation.add_argument("--no-table-reconstruction", action="store_true",
                          help="Disable deterministic table reconstruction.")
    ablation.add_argument("--no-evidence-grounding", action="store_true",
                          help="Disable span/context grounding and Verify evidence checks.")

    evaluation = ap.add_argument_group("Table 1 gold evaluation")
    evaluation.add_argument("--gold-dir", default=None,
                            help="Gold v2.1 JSON directory. Enables extraction-only "
                                 "Table 1 output and scoring.")

    ap.add_argument("-o", "--out-dir", default="out/scope3trace_run")
    args = ap.parse_args()
    if args.repair_page_budget < 1:
        ap.error('--repair-page-budget must be positive')

    if not (args.pdf or args.pdf_dir or args.manifest or args.search_org):
        ap.error("provide at least one of --pdf / --pdf-dir / --manifest / --search-org")

    load_dotenv()
    table1_mode = bool(args.gold_dir)
    if table1_mode and not Path(args.gold_dir).is_dir():
        ap.error(f"--gold-dir is not a directory: {args.gold_dir}")
    backbones = AgentBackbones(
        search=args.search_model or args.model,
        scope=args.scope_model or args.model,
        verify=args.verify_model or args.model,
    )
    state = SharedState()
    strict_llm = bool(args.strict_llm or (
        table1_mode and not args.rule_only and not args.allow_rule_fallback))
    search_llm = BoundedLLM("search", backbones.search, strict_failures=strict_llm)
    scope_llm = BoundedLLM("scope", backbones.scope, strict_failures=strict_llm)
    verify_llm = BoundedLLM("verify", backbones.verify, strict_failures=strict_llm)
    if strict_llm:
        unavailable = [f"{llm.agent_name}: {llm.disabled_reason}"
                       for llm in (search_llm, scope_llm, verify_llm)
                       if not llm.available]
        if unavailable:
            raise SystemExit("[ERROR] strict LLM preflight failed: " + "; ".join(unavailable))
    elif not args.rule_only:
        for llm in (search_llm, scope_llm, verify_llm):
            if not llm.available:
                print(f"[WARN] {llm.agent_name} agent: backbone {llm.model} unavailable "
                      f"({llm.disabled_reason}); this agent runs with rule-based fallbacks. "
                      "Use --strict-llm to stop instead.", flush=True)
    if args.rule_only and args.search_org and \
            (args.search_backend or os.environ.get("SCOPE3TRACE_SEARCH_BACKEND") or "openrouter") == "openrouter":
        print("[NOTE] --rule-only turns off the agents' LLM calls; report search (--search-org) still "
              "uses an LLM with OpenRouter web search (use --search-backend google for a non-LLM search).",
              flush=True)
    search = SearchAgent(
        search_llm, state, rule_only=args.rule_only,
        page_localization=not args.no_page_localization,
        sentence_localization=not args.no_sentence_localization,
        table_reconstruction=not args.no_table_reconstruction)
    scope = ScopeAgent(
        scope_llm, state, rule_only=args.rule_only,
        llm_classify_types=args.llm_classify_types,
        evidence_grounding=not args.no_evidence_grounding)
    if (not args.rule_only and getattr(scope, "chart_reader", None) is None
            and os.environ.get("SCOPE3TRACE_CHART_READER", "1") != "0"):
        print(f"[NOTE] chart reader off: no API key for SCOPE3TRACE_CHART_MODEL="
              f"{os.environ.get('SCOPE3TRACE_CHART_MODEL', 'gemini-2.5-flash')}; values printed only in "
              "charts or images are not read.", flush=True)
    verify = VerifyAgent(
        verify_llm, state, rule_only=args.rule_only,
        llm_value_flags=args.llm_value_flags,
        evidence_grounding=not args.no_evidence_grounding)

    phase_t = {}
    table1_input_audit = None

    # Phase 1 — Search Agent: acquire + parse + return evidence lists
    t0 = time.time()
    collect_pdfs(args, search)
    if table1_mode:
        gold_ids = {path.stem for path in Path(args.gold_dir).glob("*.json")}
        input_ids = {Path(doc.doc_id).stem for doc in state.documents}
        skipped = sorted(input_ids - gold_ids)
        not_selected = sorted(gold_ids - input_ids)
        state.documents = [doc for doc in state.documents
                           if Path(doc.doc_id).stem in gold_ids]
        table1_input_audit = {
            "gold_records": len(gold_ids),
            "input_documents": len(input_ids),
            "matched_documents": len(state.documents),
            "skipped_input_without_gold": skipped,
            "gold_not_selected_count": len(not_selected),
            "gold_not_selected_sample": not_selected[:20],
        }
        if skipped:
            print(f"[table 1] skipped {len(skipped)} input PDF(s) without matching gold: "
                  f"{skipped[:10]}")
    if not state.documents:
        raise SystemExit("[ERROR] no documents acquired — check --pdf/--pdf-dir/--manifest")
    print(f"[phase 1 | search] {len(state.documents)} document(s)")
    for doc in state.documents:
        print(f"  - {doc.doc_id}: parsing and localizing ...", flush=True)
        items = search.run(doc)
        print(f"  - {doc.doc_id}: {len(items)} evidence item(s)", flush=True)
    phase_t["search"] = round(time.time() - t0, 2)

    # Phase 2 — Scope Agent: asset pool + extraction + linking + estimation
    t0 = time.time()
    if table1_mode and args.inventory_csv:
        print("[phase 2 | scope] Table 1 mode: --inventory-csv ignored "
              "(asset linking and estimation are outside content extraction)")
    elif args.inventory_csv:
        n_assets = scope.build_asset_pool(
            Path(args.inventory_csv),
            entity_csv=Path(args.entity_csv) if args.entity_csv else None,
            limit=args.asset_limit)
        print(f"[phase 2 | scope] asset pool: {n_assets} assets", flush=True)
    else:
        print("[phase 2 | scope] no --inventory-csv: asset pool empty "
              "(entity linking and estimation will be skipped)")
    for doc in state.documents:
        items = state.evidence.get(doc.doc_id, [])
        if not items:
            continue
        print(f"  - {doc.doc_id}: extracting from {len(items)} evidence item(s) ...", flush=True)
        disclosures = scope.extract_disclosures(doc, items)
        for disclosure in disclosures:
            print(f"  - {doc.doc_id}: {len(disclosure.values)} value(s), "
                  f"org={disclosure.org_name!r}, year={disclosure.reporting_year}", flush=True)
            if state.asset_pool:
                scope.link_report_mentions(disclosure)
    if state.asset_pool:
        n_comp = scope.estimate(Path(args.grid_factor_csv) if args.grid_factor_csv else None)
        print(f"[phase 2 | scope] estimation: {n_comp} derived component(s)")
    phase_t["scope"] = round(time.time() - t0, 2)

    # Phase 3 — Verify Agent: checks + relevance + diagnostics + report
    t0 = time.time()
    from scope3trace.verify_agent.feedback import run_verification_with_repairs
    try:
        run_verification_with_repairs(state,search,scope,verify,
                                      max_rounds=args.max_repair_rounds,max_pages=args.repair_page_budget)
    except Exception:
        # A failed service request during repair must not erase the initial
        # candidates, negative verdicts or the partially executed repair trace.
        state.save(Path(args.out_dir)/'state_failed.json')
        raise
    for disclosure in state.org_disclosures:
        if state.asset_pool:
            verify.validate_relevance(disclosure)
    verify.apply_relevance_gate()
    report = verify.build_evaluation_report()
    report['repair_summary']={'max_rounds':args.max_repair_rounds,'page_budget':args.repair_page_budget,
                              'rounds_executed':len(state.repair_history),
                              'unresolved_issues':len(state.unresolved_issues)}
    phase_t["verify"] = round(time.time() - t0, 2)

    # persist per-agent usage + phase timing
    state.usage = [search.llm.usage, scope.llm.usage, verify.llm.usage]
    # the chart reader's vision model (agent "chart"), when it was on
    chart = getattr(scope, "chart_reader", None)
    if chart is not None and getattr(chart, "llm", None) is not None:
        state.usage.append(chart.llm.usage)
    report["agent_overhead"] = [
        {"agent": u.agent, "model": u.model, "wall_time_s": round(u.wall_time_s, 2),
         "calls": u.calls, "tokens_in": u.tokens_in, "tokens_out": u.tokens_out,
         "cost_usd": u.cost_usd} for u in state.usage]
    report["phase_wall_time_s"] = phase_t

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/'repair_history.json').write_text(json.dumps(state.repair_history,ensure_ascii=False,indent=2),encoding='utf-8')
    (out_dir/'unresolved_issues.json').write_text(json.dumps(state.unresolved_issues,ensure_ascii=False,indent=2),encoding='utf-8')
    table1_scored = False
    if table1_mode:
        pred_dir = out_dir / "table1_predictions"
        record_ids = write_predictions_for_gold(
            state, Path(args.gold_dir), pred_dir)
        if not record_ids:
            raise SystemExit(
                "[ERROR] none of the input PDF stems matched a gold JSON filename")
        scores = score_prediction_dir(
            Path(args.gold_dir), pred_dir, record_ids=record_ids)
        (out_dir / "table1_scores.json").write_text(
            json.dumps(scores, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        (out_dir / "table1_scores.md").write_text(
            render_score_markdown(scores), encoding="utf-8")
        report["table1"] = scores["table1"]
        report["table1_metric_definition"] = scores["metric_definition"]
        table1_scored = True
        print("[table 1] " + json.dumps(scores["table1"], sort_keys=True))
    report["content_extraction_tools"] = content_extraction_tool_usage(
        state, args, table1_scored=table1_scored)
    if table1_input_audit is not None:
        report["table1_input_audit"] = table1_input_audit
        (out_dir / "table1_input_audit.json").write_text(
            json.dumps(table1_input_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    write_outputs(state, out_dir)
    print(f"[phase 3 | verify] diagnostics: "
          f"{json.dumps(report.get('diagnostics_summary', {}), sort_keys=True)}")
    print(f"[done] outputs in {out_dir}/ "
          f"(state.json, organization_disclosures.jsonl, derived_components.csv, "
          f"evaluation_report.md)")


if __name__ == "__main__":
    main()
