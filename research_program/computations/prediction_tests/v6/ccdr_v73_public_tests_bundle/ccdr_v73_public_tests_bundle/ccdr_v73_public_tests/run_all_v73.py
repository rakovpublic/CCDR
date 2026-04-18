from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    'test01_p30_non_euclid_lensing_replication_v73.py',
    'test02_p30_euclid_systematics_audit_v73.py',
    'test03_highz_a0_sparc_kmos3d_v73.py',
    'test04_nu_public_redrive_v73.py',
    'test05_p3_euclid_filament_ordering_v73.py',
    'test06_p29_multi_redshift_growth_v73.py',
    'test07_p36_mond_sequence_nu_extractor.py',
    'test08_p33_density_correlated_bao.py',
    'test09_cl2_p8c_reducing_volume_cross.py',
    'test10_p38_void_wall_cauchy_tail.py',
    'test11_hardcore_window_readiness_v73.py',
    'test12_p37_phase_space_drift_proxy.py',
]


def main() -> None:
    outdir = ROOT / 'out_v73'
    outdir.mkdir(exist_ok=True)
    failures = []
    for script in SCRIPTS:
        print(f'=== Running {script} ===')
        proc = subprocess.run([sys.executable, str(ROOT / script)], text=True, capture_output=True)
        stem = Path(script).stem
        if proc.stdout:
            (outdir / f'{stem}.json').write_text(proc.stdout, encoding='utf-8')
            print(proc.stdout)
        if proc.returncode != 0:
            err = proc.stderr or f'Process exited with status {proc.returncode}.'
            (outdir / f'{stem}.stderr.txt').write_text(err, encoding='utf-8')
            failures.append((script, proc.returncode))
            print(err, file=sys.stderr)
    if failures:
        lines = [f'{name} (exit {code})' for name, code in failures]
        raise SystemExit('One or more tests failed:\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
