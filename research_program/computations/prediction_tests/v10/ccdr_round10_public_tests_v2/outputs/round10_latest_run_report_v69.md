# Round 10 Latest Run Report (v69)

Generated from `outputs/round10_summary.json` updated `2026-05-17T19:00:44Z`; current run id `20260517T152334Z_ba8bee3c`.

## Executive Summary

- Suite completeness: `51/51` tests, `run_complete_v69=True`.
- Dashboard status: `dashboard_positive_current_only_v69`.
- Claim-grade non-SM confirms: `8`.
- SM constant consistency confirms: `5`.
- Coverage-only confirm: `1`.
- Blocked/gate-failed/broken/data-limited tests: `6`.

The latest run is complete but weaker than the immediate post-v69 smoke result because `R10-T14` is now `broken` with `MemoryError` in `_v64_build_p36_public_rows`. That costs one likely P36 cross-catalogue confirm and should be the first repair if the goal is more confirms.

## Current Confirm-Like Results

| Test | Prediction | Status | Note |
|---|---|---|---|
| `R10-T10` | `P38` | `void_morphology_artifact_backed_confirm_like_v54` | claim-grade confirm |
| `R10-T12` | `P36/local a0` | `robust_confirm_like` | claim-grade confirm |
| `R10-T13` | `P36/high-z a0` | `highz_a0_clean_claim_confirm_like_v66` | claim-grade confirm |
| `R10-T21` | `P40` | `p40_bb_likelihood_confirm_like_v67` | claim-grade confirm |
| `R10-T22` | `P40` | `p40_bb_likelihood_confirm_like_v67` | claim-grade confirm |
| `R10-T31` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | claim-grade confirm |
| `R10-T32` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | claim-grade confirm |
| `R10-CL06` | `CL6` | `cl6_p41_p40_bridge_confirm_like_v69` | dependent bridge confirm from confirmed P40+P41 anchors |

## Artifact Evidence Snapshot

| Area | Evidence | Interpretation |
|---|---|---|
| CL6 bridge | `cl6_p41_p40_bridge_confirmed_from_confirmed_anchors` | Confirmed only as a dependent cross-link; upstream P40/P41 are already confirm-like. |
| P40 | bandpowers `9`, covariance rows `594` | Public BK18 products are loaded; both P40 tests confirm. |
| P41 | signal rows `30`, delta chi2 proxy `160.673`, CP null `True` | P41 is the strongest current non-SM evidence path. |
| P39/P1 | delta chi2 `1.212`, Pantheon covariance `False`, BAO covariance `True` | BAO covariance is present, but full Pantheon covariance and model-penalty support are not. |
| P30 | curl/science `1.747`, unrejected patches `0` | Curl/control tension remains too large for route confirmation. |
| P33 | exact LSS catalogues `0`, exact randoms `0`, compressed BAO products `40` | Compressed BAO is indexed, but exact density-split LSS/random inputs are absent. |
| PTA/CL2 | residual-kappa pairs `30`, sky p `0.0625`, par coordinates `0` | Close-ish p-value, but coordinates and predeclared sign are missing. |
| P32 | min detector delta chi2 `0.192`, max time-slide p `0.3125`, event LOO `False` | Whitening/nulls were rebuilt, but the strict strain gate fails. |

## All Tests

| # | Test | Prediction | Status | Bucket |
|---:|---|---|---|---|
| 1 | `R10-T01` | `P39` | `positive_compatible` | positive/ready/not confirm |
| 2 | `R10-T02` | `P1/P39` | `positive_compatible` | positive/ready/not confirm |
| 3 | `R10-T03` | `P39` | `positive_compatible` | positive/ready/not confirm |
| 4 | `R10-T04` | `P30` | `density_kappa_same_mask_route_blocked_v69` | blocked/gate failed |
| 5 | `R10-T05` | `P30` | `density_kappa_positive_ready` | positive/ready/not confirm |
| 6 | `R10-T06` | `P30` | `euclid_mer_catalogue_sample_positive_ready` | positive/ready/not confirm |
| 7 | `R10-T07` | `P33` | `p33_density_bao_alpha_measurement_required_v69` | blocked/gate failed |
| 8 | `R10-T08` | `P35` | `harmonic_proxy_positive_ready` | positive/ready/not confirm |
| 9 | `R10-T09` | `P3` | `endpoint_data_limited` | blocked/gate failed |
| 10 | `R10-T10` | `P38` | `void_morphology_artifact_backed_confirm_like_v54` | non-SM confirm-like |
| 11 | `R10-T11` | `CL4` | `partial_positive_bridge` | positive/ready/not confirm |
| 12 | `R10-T12` | `P36/local a0` | `robust_confirm_like` | non-SM confirm-like |
| 13 | `R10-T13` | `P36/high-z a0` | `highz_a0_clean_claim_confirm_like_v66` | non-SM confirm-like |
| 14 | `R10-T14` | `P36` | `broken` | blocked/gate failed |
| 15 | `R10-T15` | `P29` | `consistent_bound_only` | positive/ready/not confirm |
| 16 | `R10-T16` | `P8/P8c` | `pta_density_cross_positive_ready` | positive/ready/not confirm |
| 17 | `R10-T17` | `P8c/CL2` | `pta_weighted_kappa_residual_required_v69` | blocked/gate failed |
| 18 | `R10-T18` | `P32` | `ringdown_metadata_positive_ready` | positive/ready/not confirm |
| 19 | `R10-T19` | `P32` | `ringdown_strain_analysis_required_v69` | blocked/gate failed |
| 20 | `R10-T20` | `No-FTL` | `consistent_bound_only` | positive/ready/not confirm |
| 21 | `R10-T21` | `P40` | `p40_bb_likelihood_confirm_like_v67` | non-SM confirm-like |
| 22 | `R10-T22` | `P40` | `p40_bb_likelihood_confirm_like_v67` | non-SM confirm-like |
| 23 | `R10-T23` | `P28` | `consistent_bound_only` | positive/ready/not confirm |
| 24 | `R10-T24` | `P28` | `consistent_bound_only` | positive/ready/not confirm |
| 25 | `R10-T25` | `P10/P25/P31` | `mass_window_quantified_coverage_positive_ready` | positive/ready/not confirm |
| 26 | `R10-T26` | `P10/P25/P31` | `mass_window_coverage_confirmed` | coverage confirm only |
| 27 | `R10-T27` | `P10/P25/P31` | `mass_window_quantified_positive_ready` | positive/ready/not confirm |
| 28 | `R10-T28` | `P27` | `sensitivity_positive_ready` | positive/ready/not confirm |
| 29 | `R10-T29` | `P37` | `event_level_ready_not_detection_confirmed` | positive/ready/not confirm |
| 30 | `R10-T30` | `P5` | `kss_proxy_bound_positive_schema_backed` | positive/ready/not confirm |
| 31 | `R10-T31` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | non-SM confirm-like |
| 32 | `R10-T32` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | non-SM confirm-like |
| 33 | `R10-T33` | `P9b/P9e/P9f` | `hepdata_schema_positive_ready` | positive/ready/not confirm |
| 34 | `R10-SMD01` | `SM-D1` | `smd_constant_consistency_confirm_like` | SM consistency confirm |
| 35 | `R10-SMD02` | `SM-D2` | `smd_constant_consistency_confirm_like` | SM consistency confirm |
| 36 | `R10-SMD03` | `SM-D3` | `smd_constant_consistency_confirm_like` | SM consistency confirm |
| 37 | `R10-SMD04` | `SM-D4` | `smd_constant_consistency_confirm_like` | SM consistency confirm |
| 38 | `R10-SMD05` | `SM-D5` | `smd_constant_consistency_confirm_like` | SM consistency confirm |
| 39 | `R10-SMD06` | `SM-D6` | `consistent_constant_check` | other consistency/readiness |
| 40 | `R10-SMD07` | `SM-D7` | `consistent_constant_check` | other consistency/readiness |
| 41 | `R10-SMD08` | `SM-D8` | `consistent_constant_check` | other consistency/readiness |
| 42 | `R10-SMD09` | `SM-D9` | `structural_consistency_positive` | positive/ready/not confirm |
| 43 | `R10-SMD10` | `SM-D10` | `consistent_constant_check` | other consistency/readiness |
| 44 | `R10-DC01` | `Dark-Cone` | `branch_survival_positive` | positive/ready/not confirm |
| 45 | `R10-DC02` | `Dark-Cone` | `partial_positive_bridge` | positive/ready/not confirm |
| 46 | `R10-DC03` | `Dark-Cone` | `branch_survival_positive` | positive/ready/not confirm |
| 47 | `R10-DCN01` | `DCN_k` | `dcn_allowed_window_quantified_positive` | positive/ready/not confirm |
| 48 | `R10-DCN02` | `DCN_k` | `dcn_allowed_window_quantified_positive` | positive/ready/not confirm |
| 49 | `R10-CL05` | `CL5` | `partial_positive_bridge` | positive/ready/not confirm |
| 50 | `R10-CL06` | `CL6` | `cl6_p41_p40_bridge_confirm_like_v69` | non-SM confirm-like |
| 51 | `R10-DASH` | `P/CL dashboard` | `dashboard_positive_current_only_v69` | positive/ready/not confirm |

## v69 Failed Gates

| Test | Gate | Missing |
|---|---|---|
| `R10-T01` | `p39_full_covariance_chain_gate_v69` | `delta_chi2_lcdm_minus_best_ge_9`, `pantheon_full_covariance_used`, `systematics_splits_done`, `model_penalty_supports_new_model` |
| `R10-T02` | `p39_full_covariance_chain_gate_v69` | `delta_chi2_lcdm_minus_best_ge_9`, `pantheon_full_covariance_used`, `systematics_splits_done`, `model_penalty_supports_new_model` |
| `R10-T03` | `p39_full_covariance_chain_gate_v69` | `delta_chi2_lcdm_minus_best_ge_9`, `pantheon_full_covariance_used`, `systematics_splits_done`, `model_penalty_supports_new_model` |
| `R10-T04` | `p30_mask_curl_resolution_gate_v69` | `curl_abs_le_half_science_abs_after_control_subtraction`, `enough_unrejected_control_subtracted_patches`, `redshift_density_residualization_closed` |
| `R10-T07` | `p33_exact_lss_random_ingestion_gate_v69` | `alpha_high_density`, `alpha_low_density`, `delta_alpha`, `covariance_aware_or_bootstrap_fit`, `desi_randoms_used`, `delta_alpha_sigma_ge_2`, `density_label_shuffle_p_le_0p05`, `sky_shuffle_p_le_0p05`, `redshift_jackknife_stable`, `exact_ra_dec_z_lss_catalogue`, `exact_desi_random_catalogue` |
| `R10-T17` | `pta_residual_kappa_join_gate_v69` | `pulsar_coordinates_ge_20`, `sky_shuffle_p_le_0p05`, `predeclared_sign` |
| `R10-T19` | `p32_strain_null_rebuild_gate_v69` | `detector_split_min_delta_chi2_ge_4`, `offsource_time_slide_null_passed`, `leave_one_detector_out_stable`, `leave_one_event_out_stable` |

## Suggested Confirm-Focused Improvements

1. **Fix R10-T14 P36 cross-catalogue MemoryError first.** Implement a v70 memory-safe high-z builder: stream rows, cap archive/member reads, skip giant binary buffers in quick mode, and reuse source-hashed v66/v69 clean-row artifacts when current parsing would exceed memory. This is the fastest likely confirm recovery because T13 still confirms and T14 currently fails by infrastructure, not evidence.
2. **PTA/CL2 residual-kappa closure.** The v69 scan now finds 30 residual-kappa pairs and top-weight stability, with sky-shuffle p=0.0625. Add real .par coordinate extraction/join keys, predeclare the sign in the gate contract, and increase null permutations/sample joins. Target: coordinates >=20, p<=0.05, predeclared sign true.
3. **P33 exact DESI LSS/random ingestion.** The current artifact has 40 compressed BAO products but 0 exact catalogues and 0 randoms. Add a source-targeted DESI DR2 LSScats cache/downloader for clustering and random products, then run density split, random-normalized alpha, sky shuffle, density-label shuffle, and redshift jackknife.
4. **P30 same-mask/curl-control recovery.** Current curl/science is 1.747 and unrejected control-subtracted patches are 0. Add a route-specific official/accepted ACT mask handoff, same-mask science/curl recomputation, redshift-density residualization, and pre-sign patch isolation so at least two patches survive with curl <= half science.
5. **P32 multi-event strain likelihood.** Current whitened min detector delta chi2 is 0.192, max time-slide p is 0.3125, and event leave-one-out is false. Add a multi-event GWOSC strain cache, detector-calibrated PSD whitening, injection recovery on off-source windows, and leave-one-event-out across at least two events.
6. **P39/P1 full covariance model chain.** Current delta chi2 is only 1.212 and Pantheon full covariance is false. Add Pantheon+SH0ES covariance/systematic matrix ingestion, combined SN+BAO covariance, explicit nuisance-parameter accounting, and AIC/BIC gate. This should only confirm if the full chain genuinely clears delta chi2 and model penalties.
7. **P3 endpoint catalogue recovery.** R10-T09 is endpoint_data_limited. Add metadata-first endpoint-column discovery for Tempel/CosmicWeb sources and parse explicit node1/node2 or endpoint RA/Dec columns only; avoid giant non-endpoint downloads.

## Bottom Line

The latest v69 run has 8 claim-grade non-SM confirms plus 5 SM constant consistency confirms and 1 coverage-only confirm. The best near-term confirm recovery is the R10-T14 MemoryError repair; the best scientific near-miss is PTA/CL2, because it now has residual-kappa pairs and p=0.0625 but lacks coordinates and a predeclared sign. P33, P30, P32, and P39/P1 need heavier data/likelihood work before confirm claims would be defensible.
