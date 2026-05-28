#!/usr/bin/env python3
"""One-command v60 runner: full Tier-B run + confirm-only/public-claim dashboard.

Example:
  python run_all_and_confirm_v60.py --cache tierb_cache_v60_all --outdir tierb_out_v60_all --timeout 240 --max-tables 300 --force
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path

DEFAULT_CONFIRM_TESTS = ["T31","T32","T44","T48","T53","T34","T57","T59","T45","T47","T26","T27","T28","T29","T30","T50","T51","T52","T60"]


def run(cmd: list[str]) -> int:
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("tierb_cache_v60_all"))
    p.add_argument("--outdir", type=Path, default=Path("tierb_out_v60_all"))
    p.add_argument("--timeout", type=str, default="240")
    p.add_argument("--max-tables", type=str, default="300")
    p.add_argument("--script-timeout", type=str, default=None)
    p.add_argument("--only", nargs="*", default=None, help="Optional test subset for run_all_tier_b.py")
    p.add_argument("--confirm-only", nargs="*", default=DEFAULT_CONFIRM_TESTS, help="Tests for confirm-only dashboard")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-full-run", action="store_true", help="Only regenerate confirm dashboard")
    args = p.parse_args()
    py = sys.executable
    args.outdir.mkdir(parents=True, exist_ok=True)
    full_rc = 0
    if not args.skip_full_run:
        cmd = [py, "run_all_tier_b.py", "--cache", str(args.cache), "--outdir", str(args.outdir), "--timeout", args.timeout, "--max-tables", args.max_tables]
        if args.script_timeout:
            cmd += ["--script-timeout", args.script_timeout]
        if args.only:
            cmd += ["--only"] + args.only
        if args.force:
            cmd.append("--force")
        full_rc = run(cmd)
    confirm_dir = args.outdir / "confirm_only_v60"
    cmd2 = [py, "run_confirm_only_v60.py", "--outdir", str(confirm_dir), "--cache", str(args.cache), "--only"] + args.confirm_only
    confirm_rc = run(cmd2)
    final = {"schema": "ccdr-v60-one-command-result", "full_run_returncode": full_rc, "confirm_only_returncode": confirm_rc, "outdir": str(args.outdir), "confirm_only_outdir": str(confirm_dir)}
    for name in ["confirm_only_dashboard_v60.json", "public_claim_check_v60.json", "confirm_targets_v60.json", "final_dashboard_v60.json"]:
        src = confirm_dir / name
        if src.exists():
            dst = args.outdir / name
            shutil.copy2(src, dst)
            try:
                obj = json.loads(src.read_text(encoding="utf-8"))
                key = name.replace(".json", "")
                final[key] = obj
            except Exception:
                pass
    confirmed = final.get("confirm_only_dashboard_v60", {}).get("confirmed_public_now") or final.get("final_dashboard_v60", {}).get("confirmed_public_now")
    final["confirmed_public_now"] = confirmed
    (args.outdir / "v60_one_command_summary.json").write_text(json.dumps(final, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("\n=== v60 confirmed_public_now ===")
    print(json.dumps(confirmed, indent=2))
    print(f"Summary: {args.outdir / 'v60_one_command_summary.json'}")
    return 0 if confirm_rc == 0 else confirm_rc

if __name__ == "__main__":
    raise SystemExit(main())
