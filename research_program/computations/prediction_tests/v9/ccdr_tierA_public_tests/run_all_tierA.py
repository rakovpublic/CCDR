#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, subprocess, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone

TESTS = [
 'tests/test01_rvm_nu_audit_planck_pantheon_desi.py',
 'tests/test02_p39_wz_drift_desi_dr2_proxy.py',
 'tests/test03_pantheon_lowz_systematic_isolation.py',
 'tests/test04_p30_euclid_q1_act_dr6_density_kappa.py',
 'tests/test05_p30_planck_crosscheck_density_kappa.py',
 'tests/test06_euclid_q1_patch_spread_audit.py',
 'tests/test07_p3_euclid_filament_orientation_density_split.py',
 'tests/test08_public_cosmic_web_filament_replication.py',
 'tests/test09_p38_vast_void_wall_kurtosis.py',
 'tests/test10_erosita_desi_cluster_sigma_mixture.py',
 'tests/test11_sparc_local_a0_anchor.py',
 'tests/test12_kmos3d_highz_a0_trend.py',
 'tests/test13_p36_three_point_nu_extractor.py',
 'tests/test14_standalone_vs_joint_nu_diagnostic.py',
 'tests/test15_nanograv_beat_recheck.py',
 'tests/test16_pta_kappa_crosslink.py',
 'tests/test17_gw_spectral_index_proxy.py',
 'tests/test18_direct_detection_window_and_peak_audit.py',
 'tests/test19_geometric_peak_ratio_audit.py',
 'tests/test20_phase_space_drift_proxy.py',
 'tests/test21_cmb_mu_y_staged_distortion_bound.py',
 'tests/test22_cmb_large_angle_no_map_proxy.py',
 'tests/test23_bulk_weyl_bmode_bandpower.py',
 'tests/test24_gwosc_ringdown_overtone_residual.py',
 'tests/test25_qgp_kss_meta_analysis.py',
]

def _test_id(rel: str) -> str:
    m=re.search(r'test(\d+)', rel, re.I)
    return f"T{int(m.group(1)):02d}" if m else Path(rel).stem

def _write_crash_json(outdir: Path, rel: str, returncode: int, stdout: str, stderr: str, seconds: float):
    tid=_test_id(rel)
    payload={
        'test_id': tid,
        'script': rel,
        'status': 'crashed',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'returncode': int(returncode),
        'seconds': float(seconds),
        'error_type': 'subprocess_nonzero_return',
        'stdout_tail': stdout[-4000:],
        'stderr_tail': stderr[-8000:],
        'warnings': ['run_all_tierA.py continued after this test crashed; inspect stderr_tail.'],
    }
    (outdir/f'{tid}.crash.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir',default='out_tierA')
    ap.add_argument('--cache',default='.cache')
    ap.add_argument('--allow-large',action='store_true')
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--max-rows',type=int,default=20000)
    ap.add_argument('--timeout',type=int,default=90)
    ap.add_argument('--only',nargs='*',help='Run test numbers/names containing these tokens, e.g. 11 sparc')
    ap.add_argument('--strict-25', action='store_true', help='After the run, require one result/crash JSON for every T01..T25 script in this bundle')
    ap.add_argument('--prefer-healpy', action='store_true', help='Pass --prefer-healpy to map tests')
    ap.add_argument('--no-harmonic', action='store_true', help='Pass --no-harmonic to map tests')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    summary=[]
    for rel in TESTS:
        if args.only and not any(tok.lower() in rel.lower() for tok in args.only):
            continue
        cmd=[sys.executable,str(root/rel),'--outdir',str(outdir),'--cache',str(args.cache),'--max-rows',str(args.max_rows),'--timeout',str(args.timeout)]
        if args.allow_large: cmd.append('--allow-large')
        if args.force: cmd.append('--force')
        if args.prefer_healpy: cmd.append('--prefer-healpy')
        if args.no_harmonic: cmd.append('--no-harmonic')
        t0=time.time(); print('\n=== RUN',rel,'===')
        try:
            cp=subprocess.run(cmd,cwd=str(root),text=True,capture_output=True)
            seconds=time.time()-t0
            print(cp.stdout)
            if cp.stderr: print(cp.stderr,file=sys.stderr)
            tid=_test_id(rel)
            expected=outdir/f'{tid}.json'
            crash_path=outdir/f'{tid}.crash.json'
            if cp.returncode != 0:
                _write_crash_json(outdir, rel, cp.returncode, cp.stdout, cp.stderr, seconds)
            elif not expected.exists():
                _write_crash_json(outdir, rel, 998, cp.stdout, cp.stderr + '\nNo result JSON was written by a zero-exit script.', seconds)
            status=None; claim_strength=None
            if expected.exists():
                try:
                    _payload=json.loads(expected.read_text(encoding='utf-8'))
                    status=_payload.get('status'); claim_strength=_payload.get('claim_strength')
                except Exception:
                    status='json_read_failed'
            summary.append({'script':rel,'test_id':tid,'returncode':cp.returncode,'seconds':seconds,'status':status,'claim_strength':claim_strength,'result_json':str(expected) if expected.exists() else None,'crash_json': str(crash_path) if crash_path.exists() else None})
        except Exception as e:
            seconds=time.time()-t0
            err=traceback.format_exc()
            print(err,file=sys.stderr)
            _write_crash_json(outdir, rel, 999, '', err, seconds)
            summary.append({'script':rel,'test_id':_test_id(rel),'returncode':999,'seconds':seconds,'crash_json': str(outdir/f'{_test_id(rel)}.crash.json')})
    # Verify all 25 test ids are represented when running full suite.
    expected_ids={f'T{i:02d}' for i in range(1,26)} if not args.only else {x.get('test_id') for x in summary}
    present={x.get('test_id') for x in summary if x.get('result_json') or x.get('crash_json')}
    missing=sorted(expected_ids-present)
    status_counts={}
    for x in summary:
        key=x.get('status') or ('crashed' if x.get('crash_json') else 'missing')
        status_counts[key]=status_counts.get(key,0)+1
    meta={'generated_utc': datetime.now(timezone.utc).isoformat(), 'n_scripts': len(summary), 'expected_test_ids': sorted(expected_ids), 'missing_test_ids': missing, 'status_counts': status_counts, 'all_25_represented': (not missing and (args.only is None))}
    (outdir/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (outdir/'run_meta_summary.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    # v10: also write and print a compact meta report so pasted/merged logs include the all-25 overview.
    lines=['# CCDR Tier-A run meta summary','',f"Generated UTC: {meta['generated_utc']}",f"Scripts represented: {len(present)}/{len(expected_ids)}",f"All 25 represented: {meta['all_25_represented']}",'', '## Status counts']
    for k,v in sorted(status_counts.items()):
        lines.append(f"- {k}: {v}")
    if missing:
        lines += ['', '## Missing test ids', ', '.join(missing)]
    lines += ['', '## Per-test table', '| Test | Status | Claim strength | Result | Crash |', '|---|---:|---:|---|---|']
    for x in summary:
        lines.append(f"| {x.get('test_id')} | {x.get('status') or ('crashed' if x.get('crash_json') else 'missing')} | {x.get('claim_strength') or ''} | {Path(x.get('result_json') or '').name if x.get('result_json') else ''} | {Path(x.get('crash_json') or '').name if x.get('crash_json') else ''} |")
    report='\n'.join(lines)+'\n'
    (outdir/'run_meta_summary.md').write_text(report,encoding='utf-8')
    # Merge JSON payloads into one file for copy/paste analysis without losing run_meta_summary.
    merged=[json.dumps({'run_meta_summary':meta},indent=2)]
    for tid in sorted(expected_ids):
        jf=outdir/f'{tid}.json'; cf=outdir/f'{tid}.crash.json'
        if jf.exists(): merged.append(f'<{tid}.json>\n'+jf.read_text(encoding='utf-8')+f'\n</{tid}.json>')
        elif cf.exists(): merged.append(f'<{tid}.crash.json>\n'+cf.read_text(encoding='utf-8')+f'\n</{tid}.crash.json>')
    (outdir/'merged_results_with_meta.txt').write_text('\n\n'.join(merged),encoding='utf-8')
    print('\n=== RUN META SUMMARY ===')
    print(json.dumps(meta,indent=2))
    print(report)
    if args.strict_25 and missing:
        print('Missing expected test outputs:', missing, file=sys.stderr)
        sys.exit(2)
    print('\nWrote',outdir/'run_summary.json',outdir/'run_meta_summary.json',outdir/'run_meta_summary.md',outdir/'merged_results_with_meta.txt')
if __name__=='__main__': main()
