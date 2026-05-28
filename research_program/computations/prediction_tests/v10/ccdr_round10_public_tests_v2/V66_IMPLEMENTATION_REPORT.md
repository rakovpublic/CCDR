# V66 Implementation Report

## Scope

This pass implements the improvement plan from `outputs/round10_all_tests_report.md` without loosening dashboard semantics.

## Implemented

- Added v66 runner metadata in `run_all.py`, including v66 run IDs, progress files, partial summaries, and notes.
- Added v66 overrides in `ccdr_r10_common.py`.
- P36 high-z now builds a clean physical-radius claim subset:
  - input: v65 high-z rows;
  - output: `measurements/p36_highz_clean_claim_rows_v66_AUTO_PUBLIC.json`;
  - promotion rule: use only rows with explicit physical radius >= 0.5 kpc plus velocity/source provenance;
  - tiny/proxy/invalid rows are quarantined, not counted as claim rows.
- P30 now persists a first-class recovery artifact:
  - output: `measurements/p30_confirm_recovery_v66_AUTO_PUBLIC.json`;
  - status remains gated if curl, route-sign, or residualization controls fail.
- P33 now persists exact DESI LSS access/auth failures as a durable recovery artifact:
  - output: `measurements/p33_alpha_recovery_v66_AUTO_PUBLIC.json`;
  - status remains gated until alpha/null fields are numeric.
- PTA/CL2 now emits a residual-kappa join schema artifact:
  - output: `measurements/pta_weighted_kappa_residual_recovery_v66_AUTO_PUBLIC.json`.
- P32 now emits a detector-coverage recovery artifact:
  - output: `measurements/p32_strain_detector_recovery_v66_AUTO_PUBLIC.json`.
- P40 now scans broader local/cache BB bandpower candidates:
  - output: `measurements/p40_bb_likelihood_v66_AUTO_PUBLIC.json`.
- P41 now persists structured q2-like candidates when present:
  - output: `measurements/p41_q2_wilson_likelihood_v66_AUTO_PUBLIC.json`.
- P3 now scans local/cache filament endpoint-like numeric candidates:
  - output: `outputs/p3_endpoint_recovery_v66.json`.
- Dashboard v66 reports v66 artifacts and keeps ready/coverage/constant checks out of non-SM confirm counts.

## Expected Confirm Impact

- P36 high-z (`R10-T13`/`R10-T14`) is the main new confirm candidate. With the existing v65 row artifacts, the v66 clean physical-radius subset should pass if the source bootstrap remains above local a0.
- P30, P33, PTA/CL2, P32, P40, P41, and P3 now have implemented recovery artifacts but remain honestly gated when required external data, covariance, nulls, detector splits, or likelihood rows are absent.

## Verification

Run focused tests before a full suite:

```powershell
python tests\test13_p36_kmos3d_inventory.py --cache-dir .ccdr_round10_cache --timeout 45 --quick
python tests\test14_p36_highz_a0_cross_catalogue_inventory.py --cache-dir .ccdr_round10_cache --timeout 45 --quick
python tests\test51_round10_joint_dashboard.py --cache-dir .ccdr_round10_cache --timeout 45 --quick
```

Use `python run_all.py --resume --script-timeout 180` to rebuild the dashboard from existing outputs after focused reruns.
