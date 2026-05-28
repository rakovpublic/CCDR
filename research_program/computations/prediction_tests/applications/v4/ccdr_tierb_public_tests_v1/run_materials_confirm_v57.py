#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from tierb.v57_confirm_repairs import materials_confirm_v57


def main() -> None:
    ap = argparse.ArgumentParser(description='Standalone v57 materials confirm runner for T31/T32.')
    ap.add_argument('--outdir', type=Path, default=Path('tierb_out_materials_v57'))
    ap.add_argument('--only', nargs='*', default=['T31', 'T32'])
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for tid in args.only:
        tid = tid.upper()
        if tid not in {'T31', 'T32'}:
            continue
        res = materials_confirm_v57(tid)
        results.append({'test_id': tid, **res})
        (args.outdir / f'{tid.lower()}_materials_confirm_v57.json').write_text(json.dumps(res, indent=2, sort_keys=True, default=str), encoding='utf-8')
    dashboard = {
        'schema': 'ccdr-materials-confirm-dashboard-v57',
        'results': results,
        'confirmed_now': [r['test_id'] for r in results if r.get('strict_confirm_ready_v57')],
        'claim_policy': 'Do not use this standalone dashboard for public claims until copied into positive_dashboard.json:v57_confirm_status.confirmed_now by run_all_tier_b.py.',
    }
    (args.outdir / 'materials_confirm_dashboard_v57.json').write_text(json.dumps(dashboard, indent=2, sort_keys=True, default=str), encoding='utf-8')
    print(json.dumps(dashboard, indent=2, sort_keys=True, default=str))

if __name__ == '__main__':
    main()
