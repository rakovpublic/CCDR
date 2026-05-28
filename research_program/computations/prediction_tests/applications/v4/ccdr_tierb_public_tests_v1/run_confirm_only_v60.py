#!/usr/bin/env python3
"""Run v60 confirm-only overlays without broad discovery."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tierb.tierb_runner import common_header
from tierb.v60_confirm_pipeline import apply_v60_result_overlay, apply_dashboard_v60, CONFIRM_ONLY_DEFAULT_TESTS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("tierb_out_v60_confirm_only"))
    p.add_argument("--cache", type=Path, default=Path("tierb_cache"))
    p.add_argument("--only", nargs="*", default=CONFIRM_ONLY_DEFAULT_TESTS)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tests = []
    for tid in [x.upper() for x in args.only]:
        obj = common_header(tid)
        obj.update({"status": "v60_confirm_only_overlay", "programmatic_verdict": "v60_confirm_only_overlay"})
        obj = apply_v60_result_overlay(obj, args, tid)
        (args.outdir / f"{tid.lower()}_result.json").write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tests.append(obj.get("positive_dashboard_fragment_v60"))
    dash = {"schema": "ccdr-tierb-positive-dashboard-v60-confirm-only-seed", "tests": tests}
    dash = apply_dashboard_v60(dash, args.outdir)
    (args.outdir / "positive_dashboard.json").write_text(json.dumps(dash, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"outdir": str(args.outdir), "confirmed_public_now": dash.get("v60_confirm_only_dashboard",{}).get("confirmed_public_now")}, indent=2))

if __name__ == "__main__":
    main()
