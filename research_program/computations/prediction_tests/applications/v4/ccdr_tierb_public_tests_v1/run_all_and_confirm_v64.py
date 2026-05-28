#!/usr/bin/env python3
"""One-command v64 runner: full Tier-B run + exact-data-pack confirm check."""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from tierb.v64_exact_data_packs import PACK_TESTS_V64
from tierb.v67_public_source_harvesters import harvest_public_sources_v67

DEFAULT_CONFIRM_TESTS = [f"T{i}" for i in range(26, 61)]

def run(cmd: list[str]) -> int:
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call(cmd)

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value).upper()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out

def _checkpoint(outdir: Path, stage: str, **payload) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "v72_run_checkpoint.json"
    try:
        obj = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        obj = {}
    stages = obj.get("stages_v72")
    if not isinstance(stages, list):
        stages = []
    stages.append({"stage_v72": stage, "utc_v72": _utc_now(), **payload})
    obj.update({
        "schema": "ccdr-v72-run-checkpoint",
        "last_stage_v72": stage,
        "last_utc_v72": _utc_now(),
        "stages_v72": stages[-50:],
    })
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")

def _affected_tests_from_harvest(summary: dict | None) -> list[str]:
    if not isinstance(summary, dict):
        return []
    out: list[str] = []
    pack_quality = summary.get("pack_quality_v71")
    if isinstance(pack_quality, dict):
        for pack, quality in pack_quality.items():
            if not isinstance(quality, dict):
                continue
            usable = int(quality.get("validator_usable_rows_v64") or 0)
            written = int(quality.get("rows_written_v67") or 0)
            if usable > 0 or written > 0:
                out.extend(PACK_TESTS_V64.get(str(pack), []))
    return _unique(out)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("tierb_cache_v64_all"))
    p.add_argument("--outdir", type=Path, default=Path("tierb_out_v64_all"))
    p.add_argument("--timeout", type=str, default="240")
    p.add_argument("--max-tables", type=str, default="300")
    p.add_argument("--script-timeout", type=str, default="1800", help="Per-test subprocess timeout for the full run; use 0/none to defer to run_all_tier_b.py")
    p.add_argument("--only", nargs="*", default=None, help="Optional test subset for run_all_tier_b.py")
    p.add_argument("--confirm-only", nargs="*", default=DEFAULT_CONFIRM_TESTS)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-full-run", action="store_true")
    p.add_argument("--harvest-public-sources", action="store_true", help="Run automated v67 public-source harvest before confirm-only")
    p.add_argument("--allow-network", action="store_true", help="Allow the v67 harvester to download public URLs")
    p.add_argument("--dry-run-harvest", action="store_true", help="Plan v67 harvests only; do not write countable rows")
    p.add_argument("--max-sources-per-pack", type=int, default=12)
    p.add_argument("--max-rows-per-source", type=int, default=5000)
    p.add_argument("--max-bytes", type=int, default=50_000_000)
    args = p.parse_args(); py = sys.executable; args.outdir.mkdir(parents=True, exist_ok=True)
    requested_confirm_tests = _unique([t.upper() for t in args.confirm_only])
    _checkpoint(args.outdir, "start", outdir_v72=str(args.outdir), cache=str(args.cache), requested_confirm_tests_v72=requested_confirm_tests, harvest_public_sources_v67=bool(args.harvest_public_sources))
    full_rc = 0
    if not args.skip_full_run:
        cmd = [py, "run_all_tier_b.py", "--cache", str(args.cache), "--outdir", str(args.outdir), "--timeout", args.timeout, "--max-tables", args.max_tables]
        if args.script_timeout and args.script_timeout.lower() not in {"0", "none", "default"}:
            cmd += ["--script-timeout", args.script_timeout]
        if args.only: cmd += ["--only"] + args.only
        if args.force: cmd.append("--force")
        _checkpoint(args.outdir, "before_full_run", command_v72=cmd)
        full_rc = run(cmd)
        _checkpoint(args.outdir, "after_full_run", full_run_returncode=full_rc)
    harvest_summary = None
    if args.harvest_public_sources:
        _checkpoint(args.outdir, "before_public_source_harvest", allow_network_v67=args.allow_network, max_sources_per_pack=args.max_sources_per_pack, max_rows_per_source=args.max_rows_per_source)
        harvest_summary = harvest_public_sources_v67(
            args.outdir,
            args.cache,
            only_tests=requested_confirm_tests,
            allow_network=args.allow_network,
            dry_run=args.dry_run_harvest,
            force=args.force,
            max_sources_per_pack=args.max_sources_per_pack,
            max_rows_per_source=args.max_rows_per_source,
            max_bytes=args.max_bytes,
            run_validation=True,
        )
        _checkpoint(args.outdir, "after_public_source_harvest", public_source_harvest_v67=harvest_summary)
    auto_confirm_tests = _affected_tests_from_harvest(harvest_summary)
    confirm_tests = _unique(requested_confirm_tests + auto_confirm_tests)
    confirm_dir = args.outdir / "confirm_only_v64"
    confirm_dir.mkdir(parents=True, exist_ok=True)
    if harvest_summary is not None:
        harvest_file = args.outdir / "public_source_harvest_v67.json"
        if harvest_file.exists():
            shutil.copy2(harvest_file, confirm_dir / "public_source_harvest_v67.json")
    cmd2 = [py, "run_confirm_only_v64.py", "--outdir", str(confirm_dir), "--cache", str(args.cache), "--only"] + confirm_tests
    _checkpoint(args.outdir, "before_confirm_only", command_v72=cmd2, auto_added_confirm_tests_v72=[t for t in confirm_tests if t not in requested_confirm_tests])
    confirm_rc = run(cmd2)
    _checkpoint(args.outdir, "after_confirm_only", confirm_only_returncode=confirm_rc)
    final = {
        "schema": "ccdr-v64-one-command-result",
        "v72_wrapper_improvements": {
            "partial_checkpoint_file": str(args.outdir / "v72_run_checkpoint.json"),
            "auto_added_confirm_tests_v72": [t for t in confirm_tests if t not in requested_confirm_tests],
            "confirm_tests_executed_v72": confirm_tests,
            "rule_v72": "Any pack with rows written or validator-usable rows is added to confirm-only so post-harvest confirms are not missed.",
        },
        "full_run_returncode": full_rc,
        "confirm_only_returncode": confirm_rc,
        "outdir": str(args.outdir),
        "confirm_only_outdir": str(confirm_dir),
    }
    if harvest_summary is not None:
        final["public_source_harvest_v67"] = {
            "summary_file": str(args.outdir / "public_source_harvest_v67.json"),
            "allow_network_v67": args.allow_network,
            "dry_run_v67": args.dry_run_harvest,
            "n_rows_written_v67": harvest_summary.get("n_rows_written_v67"),
            "n_structured_sources_parsed_v67": harvest_summary.get("n_structured_sources_parsed_v67"),
            "candidate_quality_v71": harvest_summary.get("candidate_quality_v71"),
            "adapter_quality_warnings_v71": harvest_summary.get("adapter_quality_warnings_v71"),
        }
    for name in ["confirm_only_dashboard_v64.json", "claim_summary_v64.json", "public_claim_check_v64.json", "confirm_targets_v64.json", "source_pack_status_v64.json", "v64_source_pack_validation.json", "next_rows_needed_v64.json", "public_source_harvest_v67.json", "t48_provenance_appendix_v64.json", "positive_dashboard.json"]:
        src = confirm_dir / name
        if src.exists():
            dst = args.outdir / name; shutil.copy2(src, dst)
            try: final[name.replace(".json", "")] = json.loads(src.read_text(encoding="utf-8"))
            except Exception: pass
    batch_summary = args.outdir / "tier_b_batch_summary.json"
    if batch_summary.exists():
        try:
            summary_obj = json.loads(batch_summary.read_text(encoding="utf-8"))
            rows = summary_obj.get("summary", [])
            final["process_summary_v64"] = {
                "n_tests": len(rows),
                "process_timeouts": [r.get("test_id") for r in rows if isinstance(r, dict) and r.get("process_status") == "process_timeout"],
                "process_errors": [r.get("test_id") for r in rows if isinstance(r, dict) and r.get("process_status") not in {None, "ok", "process_timeout"}],
                "script_timeout_seconds": int(args.script_timeout) if args.script_timeout and args.script_timeout.isdigit() else args.script_timeout,
                "note": "Subprocess status is reported separately from public-confirm eligibility. Re-run with --skip-full-run to refresh confirm artifacts without repeating long subprocesses.",
            }
        except Exception as exc:
            final["process_summary_v64_error"] = f"{type(exc).__name__}: {exc}"
    confirmed = final.get("confirm_only_dashboard_v64", {}).get("confirmed_public_now")
    final["confirmed_public_now"] = confirmed
    claim_summary = final.get("claim_summary_v64")
    if isinstance(claim_summary, dict):
        final["claim_counts_v64"] = claim_summary.get("claim_counts_v64")
        final["timeout_attention_v64"] = claim_summary.get("timeout_attention_v64")
        final["public_claim_rule_v64"] = claim_summary.get("public_claim_rule_v64")
    (args.outdir / "v64_one_command_summary.json").write_text(json.dumps(final, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("\n=== v64 confirmed_public_now ===")
    print(json.dumps(confirmed, indent=2))
    print(f"Summary: {args.outdir / 'v64_one_command_summary.json'}")
    return 0 if confirm_rc == 0 else confirm_rc

if __name__ == "__main__":
    raise SystemExit(main())
