#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = [
    "test01_p30_non_euclid_lensing_replication.py",
    "test02_p30_euclid_q1_systematics_audit.py",
    "test03_highz_a0_sparc_kmos3d_proxy.py",
    "test04_public_nu_redrive.py",
    "test05_euclid_q1_filament_ordering.py",
    "test06_p29_multi_redshift_growth.py",
    "test07_p36_mond_sequence_nu_extractor.py",
    "test08_p33_density_correlated_bao.py",
    "test09_cl2_p8c_reducing_volume_cross.py",
    "test10_p38_void_wall_cauchy_tail.py",
    "test11_hardcore_window_readiness.py",
    "test12_p37_phase_space_drift_proxy.py",
]

for test in TESTS:
    print(f"\n=== Running {test} ===")
    subprocess.run([sys.executable, str(ROOT / test)], check=False)
