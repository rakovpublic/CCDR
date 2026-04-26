#!/usr/bin/env python3
"""Run the lean CCDR public-literature proxy batch: MAT5, MAT4, MAT9 only."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

TESTS = [
    "test02_mat5_kibble_zurek_gb_density.py",
    "test03_mat4_bi2te3_gb_zt_gain.py",
    "test05_mat9_moire_twist_phonon_bandgap.py",
]


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="public_cache")
    p.add_argument("--outdir", default="out_public_tests")
    p.add_argument("--force", action="store_true")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for script in TESTS:
        cmd = [sys.executable, str(root / script), "--cache", str(root / args.cache), "--outdir", str(outdir)]
        if args.force:
            cmd.append("--force")
        if args.strict:
            cmd.append("--strict")
        print(f"\n=== Running {script} ===", flush=True)
        proc = subprocess.run(cmd, cwd=str(root), text=True, capture_output=True)
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        summary.append({"script": script, "returncode": proc.returncode})
        if args.strict and proc.returncode != 0:
            break
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {outdir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
