#!/usr/bin/env python3
"""Run v64 exact-data-pack confirm extractors without broad discovery."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tierb.tierb_runner import common_header
from tierb.v64_exact_data_packs import apply_v64_result_overlay, apply_dashboard_v64, DEFAULT_TESTS, EXACT_PACKS, PACK_TESTS_V64, validate_v64_source_packs
from tierb.v67_public_source_harvesters import harvest_public_sources_v67, _quarantine_stale_generated_rows_v72

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _write_checkpoint(outdir: Path, stage: str, **payload) -> None:
    path = outdir / "confirm_only_run_summary_v72.json"
    try:
        obj = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        obj = {}
    stages = obj.get("stages_v72")
    if not isinstance(stages, list):
        stages = []
    stages.append({"stage_v72": stage, "utc_v72": _utc_now(), **payload})
    obj.update({"schema": "ccdr-v72-confirm-only-checkpoint", "last_stage_v72": stage, "last_utc_v72": _utc_now(), "stages_v72": stages[-50:]})
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("tierb_out_v64_confirm_only"))
    p.add_argument("--cache", type=Path, default=Path("tierb_cache"))
    p.add_argument("--only", nargs="*", default=DEFAULT_TESTS)
    p.add_argument("--harvest-public-sources", action="store_true", help="Run automated v67 public-source harvest before confirm overlay")
    p.add_argument("--allow-network", action="store_true", help="Allow the v67 harvester to download public URLs")
    p.add_argument("--dry-run-harvest", action="store_true", help="Plan harvests only; do not write countable rows")
    p.add_argument("--max-sources-per-pack", type=int, default=12)
    p.add_argument("--max-rows-per-source", type=int, default=5000)
    p.add_argument("--max-bytes", type=int, default=50_000_000)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tests_requested = [x.upper() for x in args.only]
    _write_checkpoint(args.outdir, "start", outdir_v72=str(args.outdir), cache=str(args.cache), tests_requested_v72=tests_requested, harvest_public_sources_v67=bool(args.harvest_public_sources))
    if args.harvest_public_sources:
        _write_checkpoint(args.outdir, "before_public_source_harvest", allow_network_v67=args.allow_network, max_sources_per_pack=args.max_sources_per_pack, max_rows_per_source=args.max_rows_per_source)
        harvest_public_sources_v67(
            args.outdir,
            args.cache,
            only_tests=tests_requested,
            allow_network=args.allow_network,
            dry_run=args.dry_run_harvest,
            max_sources_per_pack=args.max_sources_per_pack,
            max_rows_per_source=args.max_rows_per_source,
            max_bytes=args.max_bytes,
            run_validation=True,
        )
        _write_checkpoint(args.outdir, "after_public_source_harvest")
    quarantine_reports = []
    requested = set(tests_requested)
    for pack in EXACT_PACKS:
        if any(t in requested for t in PACK_TESTS_V64.get(pack, [])):
            quarantine_reports.append(_quarantine_stale_generated_rows_v72(pack, args.outdir, args.cache))
    _write_checkpoint(args.outdir, "after_stale_auto_row_quarantine", quarantine_reports_v72=quarantine_reports)
    validation_report = validate_v64_source_packs(args.outdir, args.cache)
    _write_checkpoint(
        args.outdir,
        "after_preconfirm_validation",
        all_existing_rows_valid_v64=validation_report.get("all_existing_rows_valid_v64"),
        n_invalid_rows_v64=validation_report.get("n_invalid_rows_v64"),
    )
    tests = []
    for tid in tests_requested:
        obj = common_header(tid)
        obj.update({"status": "v64_exact_data_pack_confirm_overlay", "programmatic_verdict": "v64_exact_data_pack_confirm_overlay"})
        obj = apply_v64_result_overlay(obj, args, tid)
        (args.outdir / f"{tid.lower()}_result.json").write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tests.append(obj.get("positive_dashboard_fragment_v64"))
        _write_checkpoint(args.outdir, "test_overlay_written", test_id=tid, confirm_allowed_now_v64=obj.get("confirm_allowed_now_v64"), confirmation_status_v64=obj.get("confirmation_status_v64"))
    dash = {"schema": "ccdr-tierb-positive-dashboard-v64-confirm-only-seed", "tests": tests}
    dash = apply_dashboard_v64(dash, args.outdir, args.cache, args.only)
    (args.outdir / "positive_dashboard.json").write_text(json.dumps(dash, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_checkpoint(args.outdir, "dashboard_written", confirmed_public_now=dash.get("v64_confirm_only_dashboard", {}).get("confirmed_public_now"))
    print(json.dumps({"outdir": str(args.outdir), "confirmed_public_now": dash.get("v64_confirm_only_dashboard", {}).get("confirmed_public_now")}, indent=2))

if __name__ == "__main__":
    main()
