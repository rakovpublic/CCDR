#!/usr/bin/env python3
"""Run only the v55 public-source fusion parsers for T26-T30.

This bypasses older broad live-discovery layers and is useful when you want to test the
new paper/source parsers directly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from tierb.fusion_public_parsers_v55 import parse_fusion_public_source
from tierb.tierb_common import ensure_dir, to_jsonable


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def main() -> None:
    p = argparse.ArgumentParser(description="Run v55 fusion public-source parsers only")
    p.add_argument("--cache", type=Path, default=Path("tierb_cache_v55"))
    p.add_argument("--outdir", type=Path, default=Path("tierb_out_v55_parsers"))
    p.add_argument("--only", nargs="*", default=["T26", "T27", "T28", "T29", "T30"])
    p.add_argument("--timeout", type=int, default=45)
    p.add_argument("--max-tables", type=int, default=80)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    ensure_dir(args.outdir)
    all_summaries: List[Dict[str, Any]] = []
    for tid in [x.upper() for x in args.only]:
        result = parse_fusion_public_source(tid, args.cache, timeout=args.timeout, force=args.force, max_tables=args.max_tables)
        rows = list(result.pop("rows", []) or [])
        normalized = list(result.pop("normalized_rows", []) or [])
        attachments = list(result.pop("attachment_candidates", []) or [])
        downloads = list(result.get("downloads", []) or [])
        summary = dict(result)
        summary["rows_artifact"] = str(args.outdir / f"{tid.lower()}_fusion_public_source_rows_v55.csv") if rows else None
        summary["normalized_rows_artifact"] = str(args.outdir / f"{tid.lower()}_fusion_public_source_normalized_rows_v55.csv") if normalized else None
        summary["attachment_candidates_artifact"] = str(args.outdir / f"{tid.lower()}_fusion_structured_attachment_candidates_v55.csv") if attachments else None
        all_summaries.append(summary)
        if rows:
            _write_csv(rows, args.outdir / f"{tid.lower()}_fusion_public_source_rows_v55.csv")
        if normalized:
            _write_csv(normalized, args.outdir / f"{tid.lower()}_fusion_public_source_normalized_rows_v55.csv")
        if attachments:
            _write_csv(attachments, args.outdir / f"{tid.lower()}_fusion_structured_attachment_candidates_v55.csv")
        if downloads:
            _write_csv(downloads, args.outdir / f"{tid.lower()}_fusion_public_parser_downloads_v55.csv")
        (args.outdir / f"{tid.lower()}_fusion_public_parser_summary_v55.json").write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    dashboard = {
        "schema": "ccdr-v55-fusion-public-parser-only-dashboard",
        "summaries": all_summaries,
        "strict_policy": "Parsed paper/PDF rows are non-confirm evidence unless a true machine-readable raw/per-shot/per-timeslice table with required columns is downloaded.",
        "recommended_priority": ["T29", "T28", "T27", "T26", "T30"],
    }
    (args.outdir / "fusion_public_parser_dashboard_v55.json").write_text(json.dumps(to_jsonable(dashboard), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(to_jsonable(dashboard), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
