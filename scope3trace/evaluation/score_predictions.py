#!/usr/bin/env python3
"""
Score predictions (v2.1 schema) against gold (v2.1 schema).

Produces:
  - per-value match (Scope 1, Scope 2 total or reported bases, Scope 3 total)
  - per-category match (cat1..cat15)
  - aggregate Scope F1 / Cat F1 / EM (full-record)
  - wall time / doc (if predictions carry _elapsed_s)

Usage:
  python -m scope3trace.evaluation.score_predictions \\
      --pred-dir out/table1/table1_predictions \\
      --gold-dir data/gold_annotations \\
      --out      out/table1/tolerance_scores.json
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def value_match(gold, pred, tau_abs=10.0, tau_rel=0.05):
    """Both None → match. One None → no match. Else within tolerance."""
    if gold is None and pred is None:
        return None  # un-evaluable, ignore (no signal)
    if gold is None or pred is None:
        return False
    try:
        g = float(gold); p = float(pred)
    except Exception:
        return False
    if abs(g - p) <= tau_abs:
        return True
    return abs(g - p) <= tau_rel * max(abs(g), 1)


CATS = [
    "cat1_purchased_goods_tco2e", "cat2_capital_goods_tco2e",
    "cat3_fuel_energy_tco2e",     "cat4_upstream_transport_tco2e",
    "cat5_waste_tco2e",           "cat6_business_travel_tco2e",
    "cat7_employee_commuting_tco2e", "cat8_upstream_leased_tco2e",
    "cat9_downstream_transport_tco2e", "cat10_processing_tco2e",
    "cat11_use_sold_tco2e",       "cat12_eol_sold_tco2e",
    "cat13_downstream_leased_tco2e", "cat14_franchises_tco2e",
    "cat15_investments_tco2e",
]

SCOPE_KEYS = [
    ("scope1",          "scope1_tco2e_value"),
    ("scope2_total",    "scope2_total_value"),
    ("scope2_location", "scope2_location_tco2e_value"),
    ("scope2_market",   "scope2_market_tco2e_value"),
    ("scope3_total",    None),  # nested
]


def get_scope3_total(d):
    s3 = d.get("scope3_summary") or {}
    return s3.get("total_tco2e_value")


def get_breakdown(d):
    s3 = d.get("scope3_summary") or {}
    return s3.get("breakdown") or {}


def compute_f1_set(matches, n_predicted, n_gold):
    """matches: number of correct (predicted AND gold). returns precision/recall/F1."""
    p = matches / n_predicted if n_predicted else 0.0
    r = matches / n_gold      if n_gold      else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score(pred_dir: Path, gold_dir: Path):
    gold_files = {p.stem: p for p in gold_dir.glob("*.json")}
    pred_files = {p.stem: p for p in pred_dir.glob("*.json")}
    common = sorted(set(gold_files) & set(pred_files))
    print(f"Gold files: {len(gold_files)}, Pred files: {len(pred_files)}, Common: {len(common)}")

    per_scope_match  = defaultdict(lambda: {"match": 0, "evaluable": 0,
                                            "g_present": 0, "p_present": 0, "tp": 0})
    per_cat_match    = defaultdict(lambda: {"match": 0, "evaluable": 0,
                                            "g_present": 0, "p_present": 0, "tp": 0})
    em_full = 0
    n = 0
    elapsed = []

    for k in common:
        g = json.loads(gold_files[k].read_text(encoding="utf-8"))
        p = json.loads(pred_files[k].read_text(encoding="utf-8"))
        # Skip if pred has 'error' field
        if "error" in p: continue
        n += 1
        if "_elapsed_s" in p: elapsed.append(p["_elapsed_s"])

        record_all_match = True

        # Scope-level values
        for sname, key in SCOPE_KEYS:
            if g.get("scope2_total_value") is not None and sname in ("scope2_location", "scope2_market"):
                continue
            if key:
                gv = g.get(key); pv = p.get(key)
            else:  # scope3_total
                gv = get_scope3_total(g); pv = get_scope3_total(p)
            m = value_match(gv, pv)
            if sname == "scope2_total" and gv is not None and pv is not None:
                if (g.get("scope2_total_unit") not in ("tCO2e", "tCO2", "t")
                        or g.get("scope2_total_unit") != p.get("scope2_total_unit")):
                    m = False
            if m is None:    # both null
                continue
            per_scope_match[sname]["evaluable"] += 1
            if gv is not None: per_scope_match[sname]["g_present"] += 1
            if pv is not None: per_scope_match[sname]["p_present"] += 1
            if m:
                per_scope_match[sname]["match"] += 1
                if gv is not None and pv is not None:
                    per_scope_match[sname]["tp"] += 1
            else:
                record_all_match = False

        # Category-level values
        gb = get_breakdown(g); pb = get_breakdown(p)
        for c in CATS:
            gv = gb.get(c); pv = pb.get(c)
            m = value_match(gv, pv)
            if m is None: continue
            per_cat_match[c]["evaluable"] += 1
            if gv is not None: per_cat_match[c]["g_present"] += 1
            if pv is not None: per_cat_match[c]["p_present"] += 1
            if m:
                per_cat_match[c]["match"] += 1
                if gv is not None and pv is not None:
                    per_cat_match[c]["tp"] += 1
            else:
                record_all_match = False

        if record_all_match: em_full += 1

    # Aggregate
    def f1(stats):
        return compute_f1_set(stats["tp"], stats["p_present"], stats["g_present"])

    print(f"\n=== Per-Scope value match ({n} records) ===")
    print(f"{'Scope':<22} {'Acc':>8} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 50)
    for sname, _ in SCOPE_KEYS:
        s = per_scope_match[sname]
        if s["evaluable"] == 0: continue
        acc = s["match"] / s["evaluable"]
        pp, rr, ff = f1(s)
        print(f"{sname:<22} {acc:>8.3f} {pp:>6.3f} {rr:>6.3f} {ff:>6.3f}  (eval={s['evaluable']})")

    print(f"\n=== Per-Category value match ({n} records) ===")
    print(f"{'Cat':<35} {'Acc':>8} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 65)
    for c in CATS:
        s = per_cat_match[c]
        if s["evaluable"] == 0: continue
        acc = s["match"] / s["evaluable"]
        pp, rr, ff = f1(s)
        print(f"{c:<35} {acc:>8.3f} {pp:>6.3f} {rr:>6.3f} {ff:>6.3f}  (eval={s['evaluable']})")

    # Macro F1
    scope_f1s = [f1(per_scope_match[s])[2] for s, _ in SCOPE_KEYS if per_scope_match[s]["evaluable"]]
    cat_f1s   = [f1(per_cat_match[c])[2]   for c in CATS         if per_cat_match[c]["evaluable"]]
    macro_scope = sum(scope_f1s) / len(scope_f1s) if scope_f1s else 0
    macro_cat   = sum(cat_f1s) / len(cat_f1s) if cat_f1s else 0
    em_rate     = em_full / n if n else 0

    print(f"\n=== Aggregate ===")
    print(f"  Macro Scope F1:  {macro_scope:.3f}")
    print(f"  Macro Cat F1:    {macro_cat:.3f}")
    print(f"  Full-record EM:  {em_rate:.3f}  ({em_full}/{n})")
    if elapsed:
        print(f"  Wall time/doc:   {sum(elapsed)/len(elapsed):.1f}s  (n={len(elapsed)})")

    return {
        "n_records":  n,
        "scope":      {s: dict(per_scope_match[s]) for s, _ in SCOPE_KEYS},
        "categories": {c: dict(per_cat_match[c])   for c in CATS},
        "macro_scope_f1": macro_scope,
        "macro_cat_f1":   macro_cat,
        "em_full":        em_rate,
        "wall_s_mean":    (sum(elapsed)/len(elapsed)) if elapsed else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gold-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = score(Path(args.pred_dir), Path(args.gold_dir))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2, default=str)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
