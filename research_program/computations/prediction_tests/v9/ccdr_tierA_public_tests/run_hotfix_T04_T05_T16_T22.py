#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

TESTS = [
    "tests/test04_p30_euclid_q1_act_dr6_density_kappa.py",
    "tests/test05_p30_planck_crosscheck_density_kappa.py",
    "tests/test16_pta_kappa_crosslink.py",
    "tests/test22_cmb_large_angle_no_map_proxy.py",
]

def main() -> int:
    ap = argparse.ArgumentParser(description="Run CCDR Tier-A v9.4 hotfixed tests T04/T05/T16/T22.")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--outdir", default="out_tierA_hotfix")
    ap.add_argument("--alm-lmax", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--nulls", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-large", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    selected = []
    only = {x.upper().replace("TEST", "T") for x in args.only}
    for t in TESTS:
        tid = "T" + Path(t).name[4:6]
        if not only or tid in only or Path(t).name in args.only:
            selected.append(t)
    rc_all = 0
    for rel in selected:
        tid = "T" + Path(rel).name[4:6]
        print(f"\n=== RUN {rel} ===", flush=True)
        cmd = [sys.executable, str(root / rel), "--cache", args.cache, "--outdir", args.outdir]
        if tid in {"T04", "T05"}:
            cmd += ["--max-rows", str(args.max_rows), "--alm-lmax", str(args.alm_lmax), "--nulls", str(args.nulls)]
        elif tid == "T16":
            cmd += ["--alm-lmax", str(args.alm_lmax), "--nulls", str(max(args.nulls, 200))]
        if args.force:
            cmd.append("--force")
        if args.allow_large:
            cmd.append("--allow-large")
        rc = subprocess.call(cmd)
        rc_all = rc_all or rc
    return rc_all

if __name__ == "__main__":
    raise SystemExit(main())
