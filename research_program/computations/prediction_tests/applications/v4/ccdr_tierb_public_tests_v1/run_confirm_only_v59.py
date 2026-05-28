#!/usr/bin/env python3
"""Run v59 confirm-extractor overlays without broad discovery."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tierb.tierb_runner import common_header
from tierb.v59_confirm_extractors import apply_v59_result_overlay, apply_dashboard_v59


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("tierb_out_v59_confirm_only"))
    p.add_argument("--cache", type=Path, default=Path("tierb_cache"))
    p.add_argument("--only", nargs="*", default=["T31","T32","T44","T48","T53","T34","T57","T59","T45","T47","T26","T27","T28","T29","T30","T50","T51","T52","T60"])
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tests = []
    for tid in [x.upper() for x in args.only]:
        obj = common_header(tid)
        obj.update({"status": "v59_confirm_only_overlay", "programmatic_verdict": "v59_confirm_only_overlay"})
        obj = apply_v59_result_overlay(obj, args, tid)
        (args.outdir / f"{tid.lower()}_result.json").write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tests.append(obj.get("positive_dashboard_fragment_v59"))
    dash = {"schema": "ccdr-tierb-positive-dashboard-v59-confirm-only-seed", "tests": tests}
    dash = apply_dashboard_v59(dash, args.outdir)
    (args.outdir / "positive_dashboard.json").write_text(json.dumps(dash, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"outdir": str(args.outdir), "confirmed_public_now": dash.get("v59_confirm_only_dashboard",{}).get("confirmed_public_now")}, indent=2))

if __name__ == "__main__":
    main()
