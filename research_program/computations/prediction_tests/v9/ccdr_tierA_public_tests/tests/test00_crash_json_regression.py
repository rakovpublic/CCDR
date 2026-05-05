#!/usr/bin/env python3
"""Regression check: run_all writes either Txx.json or Txx.crash.json for each selected script.
Usage: python tests/test00_crash_json_regression.py --outdir out_regression --cache .cache
"""
from pathlib import Path
import argparse, json, subprocess, sys, tempfile

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir', default='out_crash_regression')
    ap.add_argument('--cache', default='.cache')
    ap.add_argument('--timeout', type=int, default=30)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    cp=subprocess.run([sys.executable, str(root/'run_all_tierA.py'), '--outdir', str(out), '--cache', args.cache, '--timeout', str(args.timeout), '--max-rows', '200', '--only', '03'], cwd=str(root), text=True, capture_output=True)
    ok=(out/'T03.json').exists() or (out/'T03.crash.json').exists()
    payload={'test_id':'T00','status':'passed' if ok else 'failed','checked_script':'T03','run_all_returncode':cp.returncode,'has_result_json':(out/'T03.json').exists(),'has_crash_json':(out/'T03.crash.json').exists(),'stdout_tail':cp.stdout[-1000:],'stderr_tail':cp.stderr[-1000:]}
    (out/'T00.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))
    sys.exit(0 if ok else 1)
if __name__=='__main__': main()
