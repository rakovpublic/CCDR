# Round 10 All Tests Analysis Report

Generated from:

- `outputs/round10_summary.json`
- `outputs/test51_round10_joint_dashboard.json`
- Runner source: `run_all.py` (`RUNNER_VERSION = "v65"`)
- Dashboard/gate source: `ccdr_r10_common.py`

Run window: `2026-05-15T23:06:35Z` to `2026-05-16T02:50:10Z`.

V66 update: report-driven improvements were implemented after the original analysis. The current resumed summary is runner `v66`, with `51/51` tests present and dashboard status `dashboard_positive_current_only_v66`.

## Executive Confirmation Ledger

- Runner completeness is confirmed: `expected_tests = 51`, `n_tests_run = 51`, and every test returned `runner_returncode = 0`.
- The dashboard is strict: readiness, coverage, and compatibility statuses do not inflate confirmation counts.
- Current non-SM confirm-like results: 4 tests.
- SM constant consistency confirm-like checks: 5 tests.
- Coverage confirmation: 1 test, explicitly coverage only and not a detection.
- Blocked or gate-failed tests: 6 tests.
- Ready or compatible tests: 34 tests.
- Dashboard aggregation test: 1 test.

## Confirms Available Now

These are the strongest present confirms in the run output.

1. `R10-T10` / `P38`: void morphology robust-confirm null hardening.
   - Status: `void_morphology_artifact_backed_confirm_like_v54`.
   - Gate: `p38_void_measurement_gate_v54` passed with no missing fields.
   - Evidence: 2 catalogue families, 164362 voids, source hashes present, leave-one-catalogue positivity, and radius-preserving spatial null support.
   - Improvement to publication-grade: add mock-catalogue comparison and an effect-size table, not only p-values.

2. `R10-T12` / `P36/local a0`: SPARC robust local RAR/a0 bootstrap test.
   - Status: `robust_confirm_like`.
   - Evidence: 175 galaxies, 3389 points, best `a0 = 9.5499e-11 m/s^2`, bootstrap 68% CI `[8.9125e-11, 1.0233e-10]`, leave-one-galaxy-out span `[9.3325e-11, 9.7724e-11]`, `log10_rms_dex = 0.1953`.
   - Improvement to publication-grade: add independent external rotation-curve/systematics replication and make the stellar mass-to-light assumptions a controlled systematic sweep.

3. `R10-T13` / `P36/high-z a0`: high-z clean physical-radius claim subset.
   - Status: `highz_a0_clean_claim_confirm_like_v66`.
   - Gate: `p36_highz_clean_claim_gate_v66` passed with no missing fields.
   - Evidence: 1904 clean physical-radius claim rows, source counts KGES 235, KROSS 289, SAMI 235, UNKNOWN_HIGHZ 1145, median clean-claim acceleration `4.6307e-10 m/s^2`, and source bootstrap/leave-one-source checks above local a0.
   - Limit: 30.5% tiny/proxy/invalid rows were quarantined outside the claim subset; the claim is for the clean physical-radius subset only.

4. `R10-T14` / `P36`: cross-catalogue high-z clean physical-radius claim subset.
   - Status: `highz_a0_clean_claim_confirm_like_v66`.
   - Gate: same v66 clean-claim gate, passed with no missing fields.
   - Evidence matches the clean subset above because the shared v65 high-z artifact is reused in quick validation.

5. `R10-SMD01` through `R10-SMD05`: SM constant consistency checks.
   - Status: `smd_constant_consistency_confirm_like`.
   - Interpretation: useful numerical consistency checks, but the dashboard correctly labels them `not_claim_grade` because they test constants/inventory consistency, not a full derivation pipeline.

6. `R10-T26` / `LZ`: explicit-unit coverage parser.
   - Status: `mass_window_coverage_confirmed`.
   - Evidence: coverage verification reports 10 rows in the predicted 500-3000 GeV window.
   - Limit: coverage only, not a dark-matter detection; units are still not fully verified.

## Dashboard Counts

| Bucket | Count | Meaning |
| --- | ---: | --- |
| `nonSM_confirm_like` | 4 | Best scientific confirm-like results in the run. |
| `SM_constant_consistency` | 5 | Constant-level consistency checks, not claim-grade confirms. |
| `coverage_confirmed` | 1 | Coverage/measurement-window confirmation, not detection. |
| `ready_or_compatible` | 34 | Positive, compatible, or readiness results that should not be called confirms. |
| `blocked_or_gate_failed` | 6 | Tests that need a missing data product, likelihood, null, mask, or control gate. |
| `dashboard` | 1 | Aggregation test with strict current-run semantics. |

## All Test Outcomes

| No. | Test | Prediction | Bucket | Status | Confirmation read |
| ---: | --- | --- | --- | --- | --- |
| 1 | `R10-T01` | P39 Pantheon+ plus DESI DR2 BAO | ready_or_compatible | `positive_compatible` | Positive-compatible only; needs full covariance, systematics splits, and stronger Delta chi2. |
| 2 | `R10-T02` | P1/P39 low-z systematic plus BAO | ready_or_compatible | `positive_compatible` | Same P39 diagnostic hook; not confirm-like. |
| 3 | `R10-T03` | P39 DESI DR2 BAO grid | ready_or_compatible | `positive_compatible` | Public BAO vector/covariance parsed; Delta chi2 about 1.21 is not enough for confirm. |
| 4 | `R10-T04` | P30 frozen diagnostic tension workflow | blocked_or_gate_failed | `density_kappa_same_mask_route_blocked_v65` | Blocked by curl/control tension and route-sign conflict. |
| 5 | `R10-T05` | P30 Planck exact-kappa route guard | ready_or_compatible | `density_kappa_positive_ready` | Positive-ready only; downstream P30 controls still block confirm. |
| 6 | `R10-T06` | P30 Euclid object-coordinate sample | ready_or_compatible | `euclid_mer_catalogue_sample_positive_ready` | Positive-ready object-coordinate sample; not a confirm. |
| 7 | `R10-T07` | P33 density-BAO covariance/null scaffold | blocked_or_gate_failed | `p33_density_bao_alpha_measurement_required_v65` | Blocked by missing DESI LSS RA/DEC/Z alpha measurement and nulls. |
| 8 | `R10-T08` | P35 BAO harmonic-comb proxy | ready_or_compatible | `harmonic_proxy_positive_ready` | Readiness positive; full power-spectrum parser still required. |
| 9 | `R10-T09` | P3 filament orientation endpoint | blocked_or_gate_failed | `endpoint_data_limited` | Data-limited endpoint case; needs exact endpoint/node-pair public metadata. |
| 10 | `R10-T10` | P38 void morphology | nonSM_confirm_like | `void_morphology_artifact_backed_confirm_like_v54` | Confirm-like. Strongest P38 result, with publication-grade effect-size work still needed. |
| 11 | `R10-T11` | CL4 P3+P38 bridge | ready_or_compatible | `partial_positive_bridge` | Bridge positive because one side is strong; not an independent confirm. |
| 12 | `R10-T12` | P36 local a0/SPARC | nonSM_confirm_like | `robust_confirm_like` | Confirm-like. Strong SPARC local RAR/a0 result. |
| 13 | `R10-T13` | P36 high-z a0 | nonSM_confirm_like | `highz_a0_clean_claim_confirm_like_v66` | Confirm-like after v66 clean physical-radius subset gate. |
| 14 | `R10-T14` | P36 cross-catalogue high-z a0 | nonSM_confirm_like | `highz_a0_clean_claim_confirm_like_v66` | Confirm-like after v66 clean physical-radius subset gate. |
| 15 | `R10-T15` | P29 growth live/frozen | ready_or_compatible | `consistent_bound_only` | Compatible-bound proxy only. |
| 16 | `R10-T16` | P8/P8c NANOGrav kappa-sky | ready_or_compatible | `pta_density_cross_positive_ready` | Positive-ready; weighted statistic is missing. |
| 17 | `R10-T17` | CL2 residual/TOA-weighted kappa | ready_or_compatible | `pta_density_cross_positive_compatible` | Compatible, but confirm gate missing residual/TOA weights plus kappa samples. |
| 18 | `R10-T18` | P32 GWOSC metadata ringdown | ready_or_compatible | `ringdown_metadata_positive_ready` | Metadata-ready only. |
| 19 | `R10-T19` | P32 strain residual/ringdown scaffold | blocked_or_gate_failed | `ringdown_strain_analysis_required` | Needs detector split, injection null, leave-one-event-out, and Delta chi2 gate. |
| 20 | `R10-T20` | No-FTL GW170817 | ready_or_compatible | `consistent_bound_only` | Consistent bound only. |
| 21 | `R10-T21` | P40 BK18 B-mode | blocked_or_gate_failed | `p40_bb_likelihood_required` | Needs BB bandpowers, covariance, and template-amplitude likelihood. |
| 22 | `R10-T22` | P40 Planck/BK18 cross-bound | blocked_or_gate_failed | `p40_bb_likelihood_required` | Same P40 likelihood gap. |
| 23 | `R10-T23` | P28 FIRAS spectral distortion | ready_or_compatible | `consistent_bound_only` | Bound-compatible toy/template check. |
| 24 | `R10-T24` | P28 Planck y-map plus FIRAS | ready_or_compatible | `consistent_bound_only` | Cross-bound readiness only. |
| 25 | `R10-T25` | XENONnT mass-window coverage | ready_or_compatible | `mass_window_quantified_coverage_positive_ready` | Coverage/readiness positive; not detection. |
| 26 | `R10-T26` | LZ mass-window coverage | coverage_confirmed | `mass_window_coverage_confirmed` | Coverage confirmed, explicitly not detection. |
| 27 | `R10-T27` | PandaX mass-window | ready_or_compatible | `mass_window_quantified_positive_ready` | Quantified positive-ready only. |
| 28 | `R10-T28` | P27 direct-detection sensitivity | ready_or_compatible | `sensitivity_positive_ready` | Sensitivity-ready; public products are limits, not peak detection. |
| 29 | `R10-T29` | P37 phase-space drift | ready_or_compatible | `event_level_ready_not_detection_confirmed` | Event-level readiness only. |
| 30 | `R10-T30` | P5 QGP/KSS | ready_or_compatible | `kss_proxy_bound_positive_schema_backed` | Schema-backed bound-positive; not claim-grade. |
| 31 | `R10-T31` | P41 supplementary archive parser | ready_or_compatible | `p41_q2_likelihood_gate_ready` | Gate-ready but missing q2 likelihood rows and CP null. |
| 32 | `R10-T32` | P41 control supplementary archive parser | ready_or_compatible | `p41_q2_likelihood_gate_ready` | Same P41 likelihood gap. |
| 33 | `R10-T33` | P9 MET/DY/HH HEPData | ready_or_compatible | `hepdata_schema_positive_ready` | Units/schema readiness positive. |
| 34 | `R10-SMD01` | SM-D1 alpha inverse | SM_constant_consistency | `smd_constant_consistency_confirm_like` | Constant consistency, not claim-grade. |
| 35 | `R10-SMD02` | SM-D2 alpha_s(mZ) | SM_constant_consistency | `smd_constant_consistency_confirm_like` | Constant consistency, not claim-grade. |
| 36 | `R10-SMD03` | SM-D3 weak mixing angle | SM_constant_consistency | `smd_constant_consistency_confirm_like` | Constant consistency, not claim-grade. |
| 37 | `R10-SMD04` | SM-D4 Higgs mass | SM_constant_consistency | `smd_constant_consistency_confirm_like` | Constant consistency, not claim-grade. |
| 38 | `R10-SMD05` | SM-D5 Koide charged leptons | SM_constant_consistency | `smd_constant_consistency_confirm_like` | Constant consistency, not claim-grade. |
| 39 | `R10-SMD06` | SM-D6 fermion masses | ready_or_compatible | `consistent_constant_check` | Inventory consistency only. |
| 40 | `R10-SMD07` | SM-D7 CKM | ready_or_compatible | `consistent_constant_check` | Inventory consistency only. |
| 41 | `R10-SMD08` | SM-D8 CKM CP/Jarlskog | ready_or_compatible | `consistent_constant_check` | Inventory consistency only. |
| 42 | `R10-SMD09` | SM-D9 gauge group | ready_or_compatible | `structural_consistency_positive` | Structural consistency, not statistical dataset confirm. |
| 43 | `R10-SMD10` | SM-D10 neutron EDM/strong CP | ready_or_compatible | `consistent_constant_check` | Bound/constant consistency only. |
| 44 | `R10-DC01` | Dark-Cone lensing anticipation | ready_or_compatible | `branch_survival_positive` | Branch-survival/readiness positive. |
| 45 | `R10-DC02` | Dark-Cone cosmic web | ready_or_compatible | `partial_positive_bridge` | Bridge positive, not independent confirm. |
| 46 | `R10-DC03` | Dark-Cone halo sharpness | ready_or_compatible | `branch_survival_positive` | Data-readiness branch survival. |
| 47 | `R10-DCN01` | DCN/AQN microlensing | ready_or_compatible | `dcn_allowed_window_quantified_positive` | Digitized-window positive-ready. |
| 48 | `R10-DCN02` | DCN/AQN macro impact | ready_or_compatible | `dcn_allowed_window_quantified_positive` | Digitized-window positive-ready. |
| 49 | `R10-CL05` | CL5 P39+P40 joint | ready_or_compatible | `partial_positive_bridge` | Joint positive-compatible, not confirm. |
| 50 | `R10-CL06` | CL6 P41+P40 joint | ready_or_compatible | `partial_positive_bridge` | Joint positive, not confirm. |
| 51 | `R10-DASH` | dashboard | dashboard | `dashboard_positive_current_only_v65` | Aggregator confirms strict current-run bucket semantics. |

## Highest-Value Improvement Plan

1. P36 high-z (`R10-T13`, `R10-T14`): implemented in v66.
   - V66 quarantines tiny/proxy/invalid rows outside the claim subset instead of letting them veto explicitly physical-radius rows.
   - Current clean claim rows: 1904, with median clean-claim acceleration `4.6307e-10 m/s^2`.
   - Source bootstrap and leave-one-source checks both pass above local a0.

2. P30 (`R10-T04`): replace the finite-footprint proxy with an official or persisted ACT mask path.
   - `p30_same_mask_recompute_gate_v65` reports `curl_abs_over_science_abs = 1.7469`.
   - Missing fields: `curl_abs_le_half_science_abs_after_patch_reject`, `redshift_density_residualization_still_required`, and `enough_unrejected_patches`.
   - There is a route sign conflict: SDSS delta is positive (`0.0789`) while Euclid delta is negative (`-0.1898`).
   - Next action: propagate the same mask object into science, curl, variants, nulls, and patch rejection before sign inspection; then rerun residualization.

3. P33 (`R10-T07`): solve exact DESI LSS access.
   - The v65 alpha autofit attempted exact DESI LSS clustering/random files but got HTTP 401 responses.
   - Missing fields: `alpha_high_density`, `alpha_low_density`, `delta_alpha`, covariance/bootstrap fit, Delta alpha sigma >= 2, density-label shuffle p <= 0.05, and sky-shuffle p <= 0.05.
   - Next action: update to accessible public DR2/DR1 LSS catalogue paths or mirror/cached public products with RA/DEC/Z plus randoms.

4. PTA/CL2 (`R10-T17`): build the residual/TOA-weighted kappa join.
   - v65 gate missing fields: `weighted_statistic`, `sky_shuffle_p_le_0p05`, `predeclared_sign`, and `top_weight_removal_stable`.
   - Next action: sample ACT/Planck kappa at pulsar coordinates and join it to public residual or TOA-weight rows with coordinate/source hashes.

5. P32 (`R10-T19`): finish the publication-grade strain likelihood path.
   - Missing fields: `injection_null_passed`, `detector_split_passed`, `leave_one_event_out_stable`, and `delta_chi2_gr_minus_ccdr_ge_4`.
   - Next action: add `detector_fits` and `injection_nulls` to the public strain artifact through the code path, then rerun the detector split.

6. P40 (`R10-T21`, `R10-T22`): load actual BB bandpowers and covariance.
   - Missing fields: `bb_bandpowers_loaded`, `covariance_loaded`, `template_amplitude`, and `template_amplitude_sigma`.
   - Next action: write a dedicated BK18/Planck bandpower table loader and a minimal template-amplitude likelihood.

7. P41 (`R10-T31`, `R10-T32`): turn structured parser readiness into a numeric likelihood.
   - Missing fields: `q2_value_error_rows_loaded`, `delta_chi2_sm_minus_wilson_ge_9`, `cp_null_passed`, and `observable_bin_jackknife_stable`.
   - Next action: extract q2/value/error rows into `measurements/p41_q2_wilson_likelihood_v64_AUTO_PUBLIC.json` or newer, then fit SM vs Wilson and run CP/jackknife controls.

8. P3 (`R10-T09`): resolve exact endpoint metadata.
   - Current status is `endpoint_data_limited`.
   - Next action: find a public endpoint/node-pair table with explicit orientation fields, or keep it data-limited without promotion.

## Runtime/Engineering Notes

- The all-test run completed, but several scripts are very slow: T14 about 6136 s, T04 about 4763 s, T13 about 1134 s, T17 about 478 s, T16 about 441 s, and T12 about 264 s.
- Add narrower cache checkpoints and resumable sub-artifacts for the slow parsers, especially P36 high-z and P30.
- Keep the dashboard policy strict. The current v65 dashboard correctly prevents ready/compatible, coverage, and constant-inventory statuses from being overcounted as scientific confirms.
- Gate implementation areas in `ccdr_r10_common.py`:
  - P36 high-z gate: around `p36_highz_large_radius_gate_v65`.
  - P30 same-mask gate: around `p30_same_mask_recompute_gate_v65`.
  - P33 alpha gate: around `p33_alpha_measurement_gate_v65`.
  - PTA kappa gate: around `pta_weighted_kappa_residual_gate_v65`.
  - P32 strain gate: around `p32_strain_likelihood_gate_v65`.
  - P40/P41 likelihood gates: around `p40_bb_likelihood_gate_v64` and `p41_q2_wilson_likelihood_gate_v64`.

## Bottom Line

The run is technically healthy and complete. After v66, the confirm picture is stricter and better: 4 non-SM confirm-like results, 5 SM constant consistency checks, and 1 coverage confirmation. P36 high-z has been promoted via a clean physical-radius claim subset; P30, P33, PTA/CL2, P32, P40, and P41 remain gated by specific missing controls or data products rather than dashboard relabeling.
