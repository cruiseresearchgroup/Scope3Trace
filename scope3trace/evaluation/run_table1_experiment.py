#!/usr/bin/env python3
"""Run and aggregate the extraction-only Scope3Trace Table 1 experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from scope3trace.evaluation.table1_eval import aggregate_runs


ABLATION_FLAGS = {
    "full": None,
    "no-page-localization": "--no-page-localization",
    "no-sentence-localization": "--no-sentence-localization",
    "no-table-reconstruction": "--no-table-reconstruction",
    "no-evidence-grounding": "--no-evidence-grounding",
}


def _render_aggregate(result: dict) -> str:
    stats = result["table1_mean_std"]

    def cell(key: str) -> str:
        item = stats[key]
        return f"{item['mean']:.3f} +/- {item['std']:.3f}"

    return "\n".join([
        "# Scope3Trace Table 1 aggregate", "",
        f"Runs: {result['n_runs']}", "",
        "| Scope 1 F1 | Scope 2 F1 | Scope 3 F1 | Category F1 | EM |",
        "|---:|---:|---:|---:|---:|",
        (f"| {cell('scope1_f1')} | {cell('scope2_f1')} | {cell('scope3_f1')} | "
         f"{cell('category_f1')} | {cell('em')} |"),
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-run, extraction-only Table 1 evaluation")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf-dir")
    source.add_argument("--manifest")
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ablation", choices=sorted(ABLATION_FLAGS), default="full")
    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument("-o", "--out-dir", required=True)
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_files: List[Path] = []

    for index in range(1, args.runs + 1):
        run_dir = out_dir / f"run_{index}"
        if run_dir.exists():
            raise SystemExit(f"refusing to overwrite existing run directory: {run_dir}")
        cmd = [
            sys.executable, "-u", "-m", "scope3trace.run_scope3trace",
            "--gold-dir", args.gold_dir,
            "--model", args.model,
            "-o", str(run_dir),
        ]
        cmd += ["--pdf-dir", args.pdf_dir] if args.pdf_dir else ["--manifest", args.manifest]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.rule_only:
            cmd.append("--rule-only")
        ablation_flag = ABLATION_FLAGS[args.ablation]
        if ablation_flag:
            cmd.append(ablation_flag)
        print(f"[table 1] run {index}/{args.runs}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        score_files.append(run_dir / "table1_scores.json")

    aggregate = aggregate_runs(score_files)
    aggregate["configuration"] = {
        "model": args.model,
        "ablation": args.ablation,
        "rule_only": args.rule_only,
        "gold_dir": str(Path(args.gold_dir)),
        "source": str(Path(args.pdf_dir or args.manifest)),
    }
    (out_dir / "table1_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "table1_aggregate.md").write_text(
        _render_aggregate(aggregate), encoding="utf-8")
    print(f"[done] aggregate written to {out_dir / 'table1_aggregate.json'}")


if __name__ == "__main__":
    main()
