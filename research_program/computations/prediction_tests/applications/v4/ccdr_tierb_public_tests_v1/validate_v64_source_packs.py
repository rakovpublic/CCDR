#!/usr/bin/env python3
"""Validate v64 exact source packs without counting templates as evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tierb.v64_exact_data_packs import (  # noqa: E402
    DEFAULT_TESTS,
    EXACT_PACKS,
    init_v64_source_packs,
    validate_v64_source_packs,
    write_next_rows_needed_v64,
)
from tierb.v67_public_source_harvesters import harvest_public_sources_v67, _quarantine_stale_generated_rows_v72  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CCDR v64 exact source packs")
    parser.add_argument("--outdir", type=Path, default=Path("tierb_out_v64_source_pack_validation"))
    parser.add_argument("--cache", type=Path, default=Path("tierb_cache"))
    parser.add_argument("--only", nargs="*", default=DEFAULT_TESTS)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any existing source row is invalid")
    parser.add_argument("--require-nonempty", action="store_true", help="Exit non-zero when any pack needed by --only has zero usable rows")
    parser.add_argument("--harvest-public-sources", action="store_true", help="Run the automated v67 public-source harvester before validation")
    parser.add_argument("--allow-network", action="store_true", help="Allow the v67 harvester to download public URLs")
    parser.add_argument("--dry-run-harvest", action="store_true", help="Plan harvests only; do not parse/write countable rows")
    parser.add_argument("--force", action="store_true", help="Re-download public URLs when --allow-network is used")
    parser.add_argument("--max-bytes", type=int, default=50_000_000)
    parser.add_argument("--max-sources-per-pack", type=int, default=12)
    parser.add_argument("--max-rows-per-source", type=int, default=5000)
    parser.add_argument("--no-quarantine-stale-auto-rows", action="store_true", help="Do not move stale invalid AUTO_PUBLIC_ROWS_V67.csv files before validation")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    harvest_summary = None
    if args.harvest_public_sources:
        harvest_summary = harvest_public_sources_v67(
            args.outdir,
            args.cache,
            only_tests=[t.upper() for t in args.only],
            allow_network=args.allow_network,
            dry_run=args.dry_run_harvest,
            force=args.force,
            max_bytes=args.max_bytes,
            max_sources_per_pack=args.max_sources_per_pack,
            max_rows_per_source=args.max_rows_per_source,
            run_validation=False,
        )
    init_v64_source_packs(args.outdir, args.cache)
    quarantine_reports = []
    if not args.no_quarantine_stale_auto_rows:
        for pack in EXACT_PACKS:
            quarantine_reports.append(_quarantine_stale_generated_rows_v72(pack, args.outdir, args.cache))
    validation = validate_v64_source_packs(args.outdir, args.cache)
    write_next_rows_needed_v64([t.upper() for t in args.only], args.outdir)

    problem_packs = [
        p for p in validation.get("pack_results", [])
        if p.get("problems_v64")
    ]
    empty_required = [
        p for p in validation.get("pack_results", [])
        if p.get("affected_tests_v64") and int(p.get("validator_usable_rows_v64") or 0) == 0
    ]
    summary = {
        "outdir": str(args.outdir),
        "validation_file": str(args.outdir / "v64_source_pack_validation.json"),
        "next_rows_needed_file": str(args.outdir / "next_rows_needed_v64.json"),
        "n_problem_packs": len(problem_packs),
        "n_empty_required_packs": len(empty_required),
        "all_existing_rows_valid_v64": validation.get("all_existing_rows_valid_v64"),
        "harvest_public_sources_v67": {
            "enabled": bool(args.harvest_public_sources),
            "summary_file": str(args.outdir / "public_source_harvest_v67.json") if harvest_summary else None,
            "allow_network_v67": bool(args.allow_network),
            "dry_run_v67": bool(args.dry_run_harvest),
            "n_rows_written_v67": harvest_summary.get("n_rows_written_v67") if isinstance(harvest_summary, dict) else None,
            "candidate_quality_v71": harvest_summary.get("candidate_quality_v71") if isinstance(harvest_summary, dict) else None,
            "adapter_quality_warnings_v71": harvest_summary.get("adapter_quality_warnings_v71") if isinstance(harvest_summary, dict) else None,
        },
        "stale_auto_row_quarantine_v72": quarantine_reports,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.strict and problem_packs:
        return 2
    if args.require_nonempty and empty_required:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
