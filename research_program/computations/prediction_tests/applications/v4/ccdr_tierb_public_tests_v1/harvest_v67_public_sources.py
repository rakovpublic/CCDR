#!/usr/bin/env python3
"""Run the v67 automated public-source harvest pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tierb.v64_exact_data_packs import DEFAULT_TESTS, EXACT_PACKS  # noqa: E402
from tierb.v67_public_source_harvesters import harvest_public_sources_v67  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest public structured sources into v64 exact source packs")
    parser.add_argument("--outdir", type=Path, default=Path("tierb_out_v67_public_source_harvest"))
    parser.add_argument("--cache", type=Path, default=Path("tierb_cache"))
    parser.add_argument("--only", nargs="*", default=DEFAULT_TESTS, help="Tests whose required packs should be harvested")
    parser.add_argument("--only-pack", nargs="*", choices=sorted(EXACT_PACKS.keys()), default=None, help="Harvest specific packs instead of deriving packs from --only")
    parser.add_argument("--allow-network", action="store_true", help="Actually download public URLs. Without this, cached files and manifests only.")
    parser.add_argument("--dry-run", action="store_true", help="Write plans/manifests only; do not parse or write countable rows.")
    parser.add_argument("--force", action="store_true", help="Re-download public URLs when --allow-network is used")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--max-bytes", type=int, default=50_000_000)
    parser.add_argument("--max-sources-per-pack", type=int, default=12)
    parser.add_argument("--max-rows-per-source", type=int, default=5000)
    parser.add_argument("--no-write-rows", action="store_true", help="Parse and validate candidates but do not update AUTO_PUBLIC_ROWS_V67.csv")
    parser.add_argument("--no-validation", action="store_true", help="Skip v64 validation after harvest")
    args = parser.parse_args()

    summary = harvest_public_sources_v67(
        args.outdir,
        args.cache,
        only_tests=[t.upper() for t in args.only],
        only_packs=args.only_pack,
        allow_network=args.allow_network,
        dry_run=args.dry_run,
        force=args.force,
        timeout=args.timeout,
        max_bytes=args.max_bytes,
        max_sources_per_pack=args.max_sources_per_pack,
        max_rows_per_source=args.max_rows_per_source,
        write_rows=not args.no_write_rows,
        run_validation=not args.no_validation,
    )
    print(json.dumps({
        "outdir": summary.get("outdir"),
        "allow_network_v67": summary.get("allow_network_v67"),
        "dry_run_v67": summary.get("dry_run_v67"),
        "n_packs_attempted_v67": summary.get("n_packs_attempted_v67"),
        "n_sources_downloaded_or_cached_v67": summary.get("n_sources_downloaded_or_cached_v67"),
        "n_structured_sources_parsed_v67": summary.get("n_structured_sources_parsed_v67"),
        "n_rows_written_v67": summary.get("n_rows_written_v67"),
        "summary_file": str(args.outdir / "public_source_harvest_v67.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

