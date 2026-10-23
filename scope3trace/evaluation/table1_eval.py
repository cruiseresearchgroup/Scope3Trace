#!/usr/bin/env python3
"""Paper-aligned content-extraction evaluation for Scope3Trace Table 1.

This module deliberately excludes asset linking and building estimation.  It
adapts the shared-state disclosures to the frozen v2.1 gold-record layout,
scores Scope 1/2/3 and Scope 3 categories, and aggregates repeated runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from scope3trace.common.state import OrgDisclosure, SharedState, V21_CATEGORY_KEYS


CATEGORY_KEYS = list(V21_CATEGORY_KEYS.values())
SCOPE_FIELDS = {
    "scope1": ("scope1_tco2e_value",),
    # Method-specific records retain both bases. A report with only a generic
    # total is scored on that total, without requiring an invented basis.
    "scope2": ("scope2_location_tco2e_value", "scope2_market_tco2e_value"),
    "scope3": ("scope3_summary.total_tco2e_value",),
}


def _nested(record: Dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _category(record: Dict[str, Any], key: str) -> Any:
    return ((record.get("scope3_summary") or {}).get("breakdown") or {}).get(key)


def exact_value_match(gold: Any, pred: Any) -> bool:
    """Exact equality after JSON numeric canonicalization; no hidden tolerance."""
    if gold is None or pred is None:
        return gold is None and pred is None
    try:
        return Decimal(str(gold)).normalize() == Decimal(str(pred)).normalize()
    except (InvalidOperation, ValueError):
        return str(gold) == str(pred)


def _add(stats: Dict[str, int], gold: Any, pred: Any, units_match: bool = True) -> bool:
    """Update extraction TP/FP/FN; a wrong value is both an FP and an FN."""
    if gold is None and pred is None:
        return True
    stats["support"] += int(gold is not None)
    stats["predicted"] += int(pred is not None)
    if gold is not None and pred is not None and units_match and exact_value_match(gold, pred):
        stats["tp"] += 1
        return True
    if pred is not None:
        stats["fp"] += 1
    if gold is not None:
        stats["fn"] += 1
    return False


def _metrics(stats: Dict[str, int]) -> Dict[str, Any]:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**stats, "precision": precision, "recall": recall, "f1": f1}


def _blank_stats() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "support": 0, "predicted": 0}


def scope2_scoring_paths(gold: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[str, ...]:
    if gold.get("scope2_total_value") is not None:
        return ("scope2_total_value",)
    paths = SCOPE_FIELDS["scope2"]
    # A spurious generic prediction remains a false positive.
    return paths + (("scope2_total_value",) if pred.get("scope2_total_value") is not None else ())


def scope2_total_units_match(gold: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    # Values are already normalized to tonnes. Unknown units or CO2 versus
    # CO2e must not silently match.
    return (gold.get("scope2_total_unit") in ("tCO2e", "tCO2", "t")
            and gold.get("scope2_total_unit") == pred.get("scope2_total_unit"))


def empty_prediction(gold: Dict[str, Any], source_document: str,
                     error: str) -> Dict[str, Any]:
    return {
        "org_id": gold.get("org_id"),
        "org_name_standardized": gold.get("org_name_standardized"),
        "report_year": gold.get("report_year"),
        "reporting_boundary": "Unknown",
        "assurance_level": "Unknown",
        "scope1_tco2e_value": None,
        "scope1_provenance": "unknown",
        "scope2_total_value": None,
        "scope2_total_unit": None,
        "scope2_location_tco2e_value": None,
        "scope2_market_tco2e_value": None,
        "scope2_provenance": "unknown",
        "scope3_summary": {
            "total_tco2e_value": None,
            "breakdown": {key: None for key in CATEGORY_KEYS},
            "provenance": "unknown",
        },
        "source_document": source_document,
        "schema_version": "v2.1",
        "evidence_pages": [],
        "_table1_error": error,
    }


def write_predictions_for_gold(state: SharedState, gold_dir: Path,
                               pred_dir: Path) -> List[str]:
    """Write one prediction per input document whose stem exists in gold_dir.

    Multi-year extraction records are matched to the gold record's report year.
    Missing or ambiguous records are written as null predictions rather than
    silently disappearing from the denominator.
    """
    gold_dir, pred_dir = Path(gold_dir), Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    by_document: Dict[str, List[OrgDisclosure]] = {}
    for disclosure in state.org_disclosures:
        by_document.setdefault(disclosure.source_document or "", []).append(disclosure)

    for document in state.documents:
        record_id = Path(document.doc_id).stem
        gold_path = gold_dir / f"{record_id}.json"
        if not gold_path.exists():
            continue
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        candidates = by_document.get(document.doc_id, [])
        gold_year = gold.get("report_year")
        exact_year = [d for d in candidates if d.reporting_year == gold_year]
        if len(exact_year) == 1:
            prediction = exact_year[0].to_v21(verified_only=True)
        elif len(candidates) == 1 and candidates[0].reporting_year is None:
            prediction = candidates[0].to_v21(verified_only=True)
        else:
            detail = (f"no extraction for gold year {gold_year}; "
                      f"available years={[d.reporting_year for d in candidates]}")
            prediction = empty_prediction(gold, document.doc_id, detail)
        prediction["_table1_record_id"] = record_id
        (pred_dir / f"{record_id}.json").write_text(
            json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        written.append(record_id)
    return sorted(set(written))


def score_prediction_dir(gold_dir: Path, pred_dir: Path,
                         record_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Score predictions with exact normalized numeric matching.

    If ``record_ids`` is supplied, every id remains in the denominator even if
    its prediction file is missing.  This prevents failed documents from being
    dropped and inflating Table 1.
    """
    gold_dir, pred_dir = Path(gold_dir), Path(pred_dir)
    ids = list(record_ids) if record_ids is not None else sorted(
        p.stem for p in pred_dir.glob("*.json") if (gold_dir / p.name).exists())
    scope_stats = {name: _blank_stats() for name in SCOPE_FIELDS}
    category_stats = {key: _blank_stats() for key in CATEGORY_KEYS}
    record_rows: List[Dict[str, Any]] = []
    em_count = year_match_count = 0
    evidence_gold_pages = evidence_matched_pages = 0

    for record_id in ids:
        gold_path = gold_dir / f"{record_id}.json"
        if not gold_path.exists():
            raise FileNotFoundError(f"gold record not found: {gold_path}")
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        pred_path = pred_dir / f"{record_id}.json"
        pred = (json.loads(pred_path.read_text(encoding="utf-8"))
                if pred_path.exists()
                else empty_prediction(gold, f"{record_id}.pdf", "prediction file missing"))

        all_values_match = True
        field_matches: Dict[str, bool] = {}
        for scope_name, paths in SCOPE_FIELDS.items():
            if scope_name == "scope2":
                paths = scope2_scoring_paths(gold, pred)
            matches = []
            for path in paths:
                units_match = path != "scope2_total_value" or scope2_total_units_match(gold, pred)
                match = _add(scope_stats[scope_name], _nested(gold, path), _nested(pred, path), units_match)
                matches.append(match)
                all_values_match = all_values_match and match
            field_matches[scope_name] = all(matches)
        for key in CATEGORY_KEYS:
            match = _add(category_stats[key], _category(gold, key), _category(pred, key))
            all_values_match = all_values_match and match

        if all_values_match:
            em_count += 1
        year_match = gold.get("report_year") == pred.get("report_year")
        year_match_count += int(year_match)

        gold_pages = {int(p) for p in gold.get("evidence_pages", []) or []
                      if str(p).isdigit()}
        pred_pages = {int(p) for p in pred.get("evidence_pages", []) or []
                      if str(p).isdigit()}
        evidence_gold_pages += len(gold_pages)
        evidence_matched_pages += len(gold_pages & pred_pages)
        record_rows.append({
            "record_id": record_id,
            "value_exact_match": all_values_match,
            "report_year_match": year_match,
            "scope_match": field_matches,
            "error": pred.get("_table1_error"),
        })

    per_scope = {name: _metrics(stats) for name, stats in scope_stats.items()}
    per_category = {key: _metrics(stats) for key, stats in category_stats.items()}
    evaluable_categories = [m["f1"] for m in per_category.values()
                            if m["support"] or m["predicted"]]
    category_macro = (sum(evaluable_categories) / len(evaluable_categories)
                      if evaluable_categories else 0.0)
    category_micro = _metrics({
        key: sum(stats[key] for stats in category_stats.values())
        for key in _blank_stats()
    })
    table1 = {
        "scope1_f1": per_scope["scope1"]["f1"],
        "scope2_f1": per_scope["scope2"]["f1"],
        "scope3_f1": per_scope["scope3"]["f1"],
        # The repository's historical scorer reports macro category F1. Keep
        # that as the Table 1 value and expose micro F1 alongside it.
        "category_f1": category_macro,
        "em": em_count / len(ids) if ids else 0.0,
    }
    table1["macro_scope_f1"] = sum(
        table1[key] for key in ("scope1_f1", "scope2_f1", "scope3_f1")) / 3
    return {
        "metric_definition": {
            "value_match": "exact equality after JSON numeric canonicalization",
            "wrong_numeric_value": "one false positive plus one false negative",
            "scope2": "basis-unspecified gold uses its reported total and exact normalized unit; otherwise location/market instances pooled; no basis inference or sum",
            "category_f1": "macro over category types with gold or predicted support",
            "em": "all Scope 1/2/3 and category values exactly match; metadata excluded",
        },
        "n_records": len(ids),
        "table1": table1,
        "per_scope": per_scope,
        "per_category": per_category,
        "category_micro": category_micro,
        "report_year_accuracy": year_match_count / len(ids) if ids else 0.0,
        "evidence_page_recall": (evidence_matched_pages / evidence_gold_pages
                                 if evidence_gold_pages else None),
        "records": record_rows,
    }


def render_score_markdown(result: Dict[str, Any]) -> str:
    row = result["table1"]
    return "\n".join([
        "# Scope3Trace Table 1 extraction evaluation", "",
        f"Records evaluated: {result['n_records']}", "",
        "| Scope 1 F1 | Scope 2 F1 | Scope 3 F1 | Category F1 | EM |",
        "|---:|---:|---:|---:|---:|",
        (f"| {row['scope1_f1']:.3f} | {row['scope2_f1']:.3f} | "
         f"{row['scope3_f1']:.3f} | {row['category_f1']:.3f} | {row['em']:.3f} |"),
        "", "Metric definition:", "",
        *[f"- {key}: {value}" for key, value in result["metric_definition"].items()],
        "",
        f"Report-year accuracy: {result['report_year_accuracy']:.3f}",
        ("Evidence-page recall: not available in gold" if result["evidence_page_recall"] is None
         else f"Evidence-page recall: {result['evidence_page_recall']:.3f}"),
        "",
    ])


def aggregate_runs(score_files: Iterable[Path]) -> Dict[str, Any]:
    results = [json.loads(Path(path).read_text(encoding="utf-8"))
               for path in score_files]
    if not results:
        raise ValueError("no score files supplied")
    keys = ("scope1_f1", "scope2_f1", "scope3_f1", "category_f1", "em")
    aggregate = {}
    for key in keys:
        values = [float(result["table1"][key]) for result in results]
        aggregate[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
    return {"n_runs": len(results), "table1_mean_std": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score or aggregate Scope3Trace Table 1 runs")
    sub = parser.add_subparsers(dest="command", required=True)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--gold-dir", required=True)
    score_parser.add_argument("--pred-dir", required=True)
    score_parser.add_argument("--out", required=True)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("scores", nargs="+")
    aggregate_parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.command == "score":
        result = score_prediction_dir(Path(args.gold_dir), Path(args.pred_dir))
    else:
        result = aggregate_runs(Path(path) for path in args.scores)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
