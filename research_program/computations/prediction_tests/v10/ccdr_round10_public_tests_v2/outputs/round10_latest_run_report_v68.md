# Round 10 Latest Run Report v68

Generated from: outputs/round10_summary.json and outputs/test51_round10_joint_dashboard.json
Run updated UTC: 2026-05-16T23:17:15Z
Runner: v68; complete: 51/51; run_complete_v68: true

## Executive Snapshot

- Non-SM confirm-like rows in dashboard: 8.
- SM constant consistency rows: 5.
- Coverage-confirmed rows: 1.
- Blocked or gate-failed rows in dashboard: 5.
- v68 artifacts indexed by dashboard: 8.

## Confirm Ledger

| Test | Prediction | Status |
|---|---|---|
| R10-T10 | P38 | void_morphology_artifact_backed_confirm_like_v54 |
| R10-T12 | P36/local a0 | robust_confirm_like |
| R10-T13 | P36/high-z a0 | highz_a0_clean_claim_confirm_like_v66 |
| R10-T14 | P36 | highz_a0_clean_claim_confirm_like_v66 |
| R10-T21 | P40 | p40_bb_likelihood_confirm_like_v67 |
| R10-T22 | P40 | p40_bb_likelihood_confirm_like_v67 |
| R10-T31 | P41 | p41_q2_wilson_likelihood_confirm_like_v68 |
| R10-T32 | P41 | p41_q2_wilson_likelihood_confirm_like_v68 |

## Status Counts

| Status | Count |
|---|---:|
| branch_survival_positive | 2 |
| consistent_bound_only | 4 |
| consistent_constant_check | 4 |
| dashboard_positive_current_only_v68 | 1 |
| dcn_allowed_window_quantified_positive | 2 |
| density_kappa_positive_ready | 1 |
| density_kappa_same_mask_route_blocked_v68 | 1 |
| endpoint_data_limited | 1 |
| euclid_mer_catalogue_sample_positive_ready | 1 |
| event_level_ready_not_detection_confirmed | 1 |
| harmonic_proxy_positive_ready | 1 |
| hepdata_schema_positive_ready | 1 |
| highz_a0_clean_claim_confirm_like_v66 | 2 |
| kss_proxy_bound_positive_schema_backed | 1 |
| mass_window_coverage_confirmed | 1 |
| mass_window_quantified_coverage_positive_ready | 1 |
| mass_window_quantified_positive_ready | 1 |
| p33_density_bao_alpha_measurement_required_v68 | 1 |
| p40_bb_likelihood_confirm_like_v67 | 2 |
| p41_q2_wilson_likelihood_confirm_like_v68 | 2 |
| partial_positive_bridge | 4 |
| positive_compatible | 3 |
| pta_density_cross_positive_ready | 1 |
| pta_weighted_kappa_residual_required_v68 | 1 |
| ringdown_metadata_positive_ready | 1 |
| ringdown_strain_analysis_required_v68 | 1 |
| robust_confirm_like | 1 |
| sensitivity_positive_ready | 1 |
| smd_constant_consistency_confirm_like | 5 |
| structural_consistency_positive | 1 |
| void_morphology_artifact_backed_confirm_like_v54 | 1 |

## v68 Artifact Evidence

| Area | Status | Key evidence | Current blocker |
|---|---|---|---|
| P41 | lhcb_supplementary_q2_tables_fit_built | 30 signal q2 rows; delta chi2 proxy 160.673; CP/sign flags pass. | Already confirm-like for v68 proxy; publication-grade global flavio/EOS likelihood still useful. |
| P32 | h1_l1_text_strain_likelihood_rebuilt | H1/L1 rebuilt; min detector delta 16.119. | Pre-peak/null max 24.299 exceeds signal gate; leave-one-event-out absent. |
| P33 | compressed_bao_observable_index_built | 40 compressed BAO products indexed. | Exact LSS catalogues 0; randoms 0. |
| PTA/CL2 | pta_residual_kappa_pairs_absent_after_v68_scan | Residual-kappa pairs 0. | Pulsar coordinates/residual weights/kappa samples not joined. |
| P30 | mask_control_resolution_recomputed | Control-subtracted unrejected patches 0; curl/science 1.747. | Official/accepted mask absent; curl and residualization fail. |
| P39/P1 | pantheon_bao_covariance_model_penalty_audit_built | BAO covariance used; delta chi2 1.212; delta AIC 0.788; delta BIC 1.353. | Pantheon full covariance false; systematics absent; model penalty not supportive. |

## Failed v68 Gates

| Test | Prediction | Gate | Pass | Missing |
|---|---|---|---|---|
| R10-T01 | P39 | p39_likelihood_gate_v68 | no | delta_chi2_lcdm_minus_best_ge_9; pantheon_full_covariance_used; systematics_splits_done; model_penalty_supports_new_model |
| R10-T02 | P1/P39 | p39_likelihood_gate_v68 | no | delta_chi2_lcdm_minus_best_ge_9; pantheon_full_covariance_used; systematics_splits_done; model_penalty_supports_new_model |
| R10-T03 | P39 | p39_likelihood_gate_v68 | no | delta_chi2_lcdm_minus_best_ge_9; pantheon_full_covariance_used; systematics_splits_done; model_penalty_supports_new_model |
| R10-T04 | P30 | p30_confirm_recovery_gate_v68 | no | official_or_accepted_act_mask; curl_abs_le_half_science_abs_after_control_subtraction; enough_unrejected_control_subtracted_patches; redshift_density_residualization_still_required |
| R10-T07 | P33 | p33_alpha_measurement_gate_v68 | no | alpha_high_density; alpha_low_density; delta_alpha; covariance_aware_or_bootstrap_fit; desi_randoms_used; delta_alpha_sigma_ge_2; density_label_shuffle_p_le_0p05; sky_shuffle_p_le_0p05; redshift_jackknife_stable; exact_ra_dec_z_lss_catalogue; exact_desi_random_catalogue |
| R10-T17 | P8c/CL2 | pta_weighted_kappa_residual_gate_v68 | no | pulsar_coordinates_hashed; weighted_statistic; sky_shuffle_p_le_0p05; top_weight_removal_stable; predeclared_sign |
| R10-T19 | P32 | p32_strain_likelihood_gate_v68 | no | injection_null_passed; leave_one_event_out_stable |

## All Tests

| # | Test | Prediction | Class | Status | Next confirm move |
|---:|---|---|---|---|---|
| 1 | R10-T01 | P39 | ready/compatible | positive_compatible | Needs full covariance/systematics and stronger model penalty; current delta chi2 is 1.21. |
| 2 | R10-T02 | P1/P39 | ready/compatible | positive_compatible | Needs full covariance/systematics and stronger model penalty; current delta chi2 is 1.21. |
| 3 | R10-T03 | P39 | ready/compatible | positive_compatible | Needs full covariance/systematics and stronger model penalty; current delta chi2 is 1.21. |
| 4 | R10-T04 | P30 | blocked/gated | density_kappa_same_mask_route_blocked_v68 | Needs official/accepted ACT mask, curl below threshold, residualization resolved. |
| 5 | R10-T05 | P30 | ready/compatible | density_kappa_positive_ready | Positive/readiness result; not claim-grade under current policy. |
| 6 | R10-T06 | P30 | ready/compatible | euclid_mer_catalogue_sample_positive_ready | Positive/readiness result; not claim-grade under current policy. |
| 7 | R10-T07 | P33 | blocked/gated | p33_density_bao_alpha_measurement_required_v68 | Needs exact DESI RA/DEC/Z LSS plus randoms; compressed BAO alone is not enough. |
| 8 | R10-T08 | P35 | ready/compatible | harmonic_proxy_positive_ready | Positive/readiness result; not claim-grade under current policy. |
| 9 | R10-T09 | P3 | blocked/gated | endpoint_data_limited | Needs exact endpoint catalogue and redshift/null controls. |
| 10 | R10-T10 | P38 | nonSM confirm | void_morphology_artifact_backed_confirm_like_v54 | Already claim-grade in dashboard. |
| 11 | R10-T11 | CL4 | ready/compatible | partial_positive_bridge | Bridge remains partial until both anchors and independence/null gates pass. |
| 12 | R10-T12 | P36/local a0 | nonSM confirm | robust_confirm_like | Already claim-grade in dashboard. |
| 13 | R10-T13 | P36/high-z a0 | nonSM confirm | highz_a0_clean_claim_confirm_like_v66 | Already claim-grade in dashboard. |
| 14 | R10-T14 | P36 | nonSM confirm | highz_a0_clean_claim_confirm_like_v66 | Already claim-grade in dashboard. |
| 15 | R10-T15 | P29 | bound/check | consistent_bound_only | Positive/readiness result; not claim-grade under current policy. |
| 16 | R10-T16 | P8/P8c | ready/compatible | pta_density_cross_positive_ready | Positive/readiness result; not claim-grade under current policy. |
| 17 | R10-T17 | P8c/CL2 | blocked/gated | pta_weighted_kappa_residual_required_v68 | Needs pulsar coordinates, residual/TOA weights, and finite kappa samples joined. |
| 18 | R10-T18 | P32 | ready/compatible | ringdown_metadata_positive_ready | Positive/readiness result; not claim-grade under current policy. |
| 19 | R10-T19 | P32 | blocked/gated | ringdown_strain_analysis_required_v68 | H1/L1 fits exist, but injection/pre-peak null and event leave-one-out fail. |
| 20 | R10-T20 | No-FTL | bound/check | consistent_bound_only | Positive/readiness result; not claim-grade under current policy. |
| 21 | R10-T21 | P40 | nonSM confirm | p40_bb_likelihood_confirm_like_v67 | Already claim-grade in dashboard. |
| 22 | R10-T22 | P40 | nonSM confirm | p40_bb_likelihood_confirm_like_v67 | Already claim-grade in dashboard. |
| 23 | R10-T23 | P28 | bound/check | consistent_bound_only | Positive/readiness result; not claim-grade under current policy. |
| 24 | R10-T24 | P28 | bound/check | consistent_bound_only | Positive/readiness result; not claim-grade under current policy. |
| 25 | R10-T25 | P10/P25/P31 | ready/compatible | mass_window_quantified_coverage_positive_ready | Readiness/coverage only; event-level likelihood needed for detection-style claims. |
| 26 | R10-T26 | P10/P25/P31 | coverage | mass_window_coverage_confirmed | Coverage confirmed; detection claim remains disabled. |
| 27 | R10-T27 | P10/P25/P31 | ready/compatible | mass_window_quantified_positive_ready | Readiness/coverage only; event-level likelihood needed for detection-style claims. |
| 28 | R10-T28 | P27 | ready/compatible | sensitivity_positive_ready | Readiness/coverage only; event-level likelihood needed for detection-style claims. |
| 29 | R10-T29 | P37 | ready/compatible | event_level_ready_not_detection_confirmed | Readiness/coverage only; event-level likelihood needed for detection-style claims. |
| 30 | R10-T30 | P5 | ready/compatible | kss_proxy_bound_positive_schema_backed | Positive/readiness result; not claim-grade under current policy. |
| 31 | R10-T31 | P41 | nonSM confirm | p41_q2_wilson_likelihood_confirm_like_v68 | Already claim-grade in dashboard. |
| 32 | R10-T32 | P41 | nonSM confirm | p41_q2_wilson_likelihood_confirm_like_v68 | Already claim-grade in dashboard. |
| 33 | R10-T33 | P9b/P9e/P9f | ready/compatible | hepdata_schema_positive_ready | Positive/readiness result; not claim-grade under current policy. |
| 34 | R10-SMD01 | SM-D1 | SM consistency | smd_constant_consistency_confirm_like | SM constant consistency, not non-SM discovery evidence. |
| 35 | R10-SMD02 | SM-D2 | SM consistency | smd_constant_consistency_confirm_like | SM constant consistency, not non-SM discovery evidence. |
| 36 | R10-SMD03 | SM-D3 | SM consistency | smd_constant_consistency_confirm_like | SM constant consistency, not non-SM discovery evidence. |
| 37 | R10-SMD04 | SM-D4 | SM consistency | smd_constant_consistency_confirm_like | SM constant consistency, not non-SM discovery evidence. |
| 38 | R10-SMD05 | SM-D5 | SM consistency | smd_constant_consistency_confirm_like | SM constant consistency, not non-SM discovery evidence. |
| 39 | R10-SMD06 | SM-D6 | bound/check | consistent_constant_check | Positive/readiness result; not claim-grade under current policy. |
| 40 | R10-SMD07 | SM-D7 | bound/check | consistent_constant_check | Positive/readiness result; not claim-grade under current policy. |
| 41 | R10-SMD08 | SM-D8 | bound/check | consistent_constant_check | Positive/readiness result; not claim-grade under current policy. |
| 42 | R10-SMD09 | SM-D9 | ready/compatible | structural_consistency_positive | Positive/readiness result; not claim-grade under current policy. |
| 43 | R10-SMD10 | SM-D10 | bound/check | consistent_constant_check | Positive/readiness result; not claim-grade under current policy. |
| 44 | R10-DC01 | Dark-Cone | ready/compatible | branch_survival_positive | Positive/readiness result; not claim-grade under current policy. |
| 45 | R10-DC02 | Dark-Cone | ready/compatible | partial_positive_bridge | Bridge remains partial until both anchors and independence/null gates pass. |
| 46 | R10-DC03 | Dark-Cone | ready/compatible | branch_survival_positive | Positive/readiness result; not claim-grade under current policy. |
| 47 | R10-DCN01 | DCN_k | ready/compatible | dcn_allowed_window_quantified_positive | Positive/readiness result; not claim-grade under current policy. |
| 48 | R10-DCN02 | DCN_k | ready/compatible | dcn_allowed_window_quantified_positive | Positive/readiness result; not claim-grade under current policy. |
| 49 | R10-CL05 | CL5 | ready/compatible | partial_positive_bridge | Bridge remains partial until both anchors and independence/null gates pass. |
| 50 | R10-CL06 | CL6 | ready/compatible | partial_positive_bridge | Candidate for bridge confirm from existing P40+P41 confirms, with independence policy. |
| 51 | R10-DASH | P/CL dashboard | ready/compatible | dashboard_positive_current_only_v68 | Positive/readiness result; not claim-grade under current policy. |

## Six Confirm-Focused Improvements

1. CL6 bridge confirm gate from P40 plus P41. Current CL6 is partial_positive_bridge while both anchors now confirm (P40 x2 and P41 x2). Add a bridge-level gate that requires independent upstream artifacts, no shared manual fill, and explicit statement that this is a bridge confirm, not a new independent physics confirmation. Expected yield: +1 bridge confirm if policy accepts anchor-combination evidence.
2. P32 strain null rebuild. Current H1/L1 fits are real and strong enough on detector delta (min 16.12), but the pre-peak null is larger (24.30) and event leave-one-out is absent. Implement whitening/PSD estimation, off-source injections, time-slide nulls, and add 2-3 more high-SNR GWOSC events for leave-one-event-out. Expected yield: +1 only if null suppression holds after whitening.
3. P30 official mask and curl-control closure. Current same-mask route is blocked by no official/accepted ACT mask, zero clean control-subtracted patches, curl/science 1.747, and residualization. Extract or define accepted ACT DR6 finite-support mask, rerun patch protocol with edge exclusion, and residualize density/redshift before sign testing. Expected yield: +1 if curl/science falls below 0.5 and at least two patches survive.
4. P33 exact DESI LSS/random ingestion. v68 indexed 40 compressed BAO products but found zero exact LSS and zero random catalogues. Add a robust DESI/SDSS LSS mirror loader for clustering plus random RA/DEC/Z, then run density-split BAO alpha with covariance, sky shuffle, label shuffle, and redshift jackknife. Expected yield: +1 if delta alpha significance is >=2 sigma with p<=0.05 nulls.
5. PTA/CL2 residual-kappa join. v68 found zero residual-kappa pairs. Load NANOGrav pulsar positions and residual/TOA weights from public 15-year products, sample ACT/Planck kappa at pulsar coordinates, and run top-weight removal plus sky shuffle. Expected yield: +1 if weighted statistic survives p<=0.05 and sign is predeclared.
6. P39/P1 full covariance cosmology chain. Current delta chi2 is 1.21 with positive delta AIC/BIC and missing Pantheon full covariance/systematics. Add Pantheon+ covariance/systematics files, DESI covariance blocks, and a proper LCDM vs CCDR parameter-chain comparison with AIC/BIC/Bayes factor. Expected yield: low unless the real delta chi2 rises toward >=9 after the full chain.

## Bottom Line

The latest run is healthy and complete. The strongest new confirm is P41: the v68 table-based Wilson proxy passes both P41 tests. The most realistic next extra confirm is CL6 as a bridge-level confirm from already-confirmed P40 and P41 anchors. The most scientifically meaningful next confirm attempt is P32, because it already has real H1/L1 signal fits, but the null test currently blocks it.
