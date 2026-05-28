#!/usr/bin/env python3
"""Run only the v58 confirm-target overlays without broad discovery.

This is useful after any normal run_all_tier_b.py run: it writes the v58 strict
source contracts and confirm-only dashboard artifacts from whatever cached/generated
rows currently exist.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tierb.tierb_catalog import TESTS
from tierb.tierb_runner import common_header
from tierb.v58_confirm_focus import apply_v58_result_overlay, apply_dashboard_v58


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("tierb_out_v58_confirm_only"))
    p.add_argument("--cache", type=Path, default=Path("tierb_cache"))
    p.add_argument("--only", nargs="*", default=["T31","T32","T44","T48","T53","T34","T57","T59","T45","T47","T26","T27","T28","T29","T30","T50","T51","T52","T60"])
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tests = []
    for tid in [x.upper() for x in args.only]:
        obj = common_header(tid)
        obj.update({"status": "v58_confirm_only_overlay", "programmatic_verdict": "v58_confirm_only_overlay"})
        obj = apply_v58_result_overlay(obj, args, tid)
        (args.outdir / f"{tid.lower()}_result.json").write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        tests.append(obj.get("positive_dashboard_fragment_v58"))
    dash = {"schema": "ccdr-tierb-positive-dashboard-v58-confirm-only-seed", "tests": tests}
    dash = apply_dashboard_v58(dash, args.outdir)
    (args.outdir / "positive_dashboard.json").write_text(json.dumps(dash, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"outdir": str(args.outdir), "confirmed_public_now": dash.get("v58_confirm_only_dashboard",{}).get("confirmed_public_now")}, indent=2))

if __name__ == "__main__":
    main()
