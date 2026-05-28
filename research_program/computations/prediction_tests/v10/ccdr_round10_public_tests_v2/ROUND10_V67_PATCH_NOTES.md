# Round 10 v67 Patch Notes

- Added v67 runner mappings for P30, P33, PTA/CL2, P32, P40, P41, and dashboard.
- Added `p40_bb_likelihood_v67_AUTO_PUBLIC.json` with real BK18 bandpower/covariance member parsing.
- Added `p41_q2_wilson_likelihood_v67_AUTO_PUBLIC.json` with numeric fit rows, CP/sign checks, delta chi2, and jackknife.
- Added `p33_alpha_measurement_v67_AUTO_PUBLIC.json` with exact LSS path search and compressed BAO mean/cov inventory.
- Added `pta_weighted_kappa_residual_v67_AUTO_PUBLIC.json` for residual-kappa join recovery.
- Added `p32_strain_detector_recovery_v67_AUTO_PUBLIC.json` for H1/L1 detector cache discovery.
- Added `p30_confirm_recovery_v67_AUTO_PUBLIC.json` for stricter mask/residualization recovery.
- Updated `run_all.py` to write v67 progress, partial, current-run ID, and completion fields.
