
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import os
import time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=".ccdr_round10_cache")
    ap.add_argument("--allow-large", action="store_true")
    ap.add_argument("--max-mb", type=float, default=250.0)
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="substring filter for script filename or prediction id/name")
    ap.add_argument("--stop-on-fail", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    tests = sorted((root / "tests").glob("test*.py"))
    manifest = json.loads((root / "round10_manifest.json").read_text(encoding="utf-8"))

    if args.only:
        keep = []
        for t in tests:
            m = next((x for x in manifest if x.get("file") == t.name), {})
            blob = " ".join([t.name, str(m.get("prediction_id","")), str(m.get("prediction_name",""))]).lower()
            if args.only.lower() in blob:
                keep.append(t)
        tests = keep

    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)
    results = []
    for script in tests:
        cmd = [sys.executable, str(script), "--cache-dir", args.cache_dir, "--max-mb", str(args.max_mb), "--timeout", str(args.timeout)]
        if args.allow_large:
            cmd.append("--allow-large")
        if args.quick:
            cmd.append("--quick")
        t0 = time.time()
        env = os.environ.copy()
        env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, env=env)
        elapsed = time.time() - t0
        stdout = proc.stdout.strip()
        try:
            obj = json.loads(stdout)
        except Exception:
            obj = {
                "test_id": script.stem,
                "status": "runner_parse_error",
                "returncode": proc.returncode,
                "stdout": stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        obj["runner_elapsed_s"] = round(elapsed, 3)
        obj["runner_returncode"] = proc.returncode
        results.append(obj)
        (out_dir / (script.stem + ".json")).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        print(f"{script.name}: {obj.get('status')} ({elapsed:.1f}s)")
        if args.stop_on_fail and obj.get("status") in {"broken", "runner_parse_error"}:
            break

    summary = {}
    for r in results:
        summary[r.get("status","unknown")] = summary.get(r.get("status","unknown"), 0) + 1
    bundle = {
        "round": 10,
        "n_tests_run": len(results),
        "summary": summary,
        "results": results,
    }
    (out_dir / "round10_summary.json").write_text(json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n_tests_run": len(results), "summary": summary, "summary_path": str(out_dir / "round10_summary.json")}, indent=2))

if __name__ == "__main__":
    main()
