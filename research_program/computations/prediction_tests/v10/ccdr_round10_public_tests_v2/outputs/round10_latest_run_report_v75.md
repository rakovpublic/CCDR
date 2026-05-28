# Round 10 Latest Run Report - v75

Generated: 2026-05-18

## Run Summary

- Source files: `outputs/round10_summary.json`, `outputs/test51_round10_joint_dashboard.json`
- Runner version: `v75`
- Started UTC: `2026-05-18T16:45:01Z`
- Updated UTC: `2026-05-18T17:28:38Z`
- Tests run: `51 / 51`
- Completion flag: `run_complete_v75 = true`

Dashboard v75:

- Non-SM confirm-like: `12`
- SM constant consistency confirm-like: `5`
- Coverage confirmed: `1`
- Blocked or gate-failed statuses: `3`
- v75 artifacts indexed: `11`

## Confirm State

Current non-SM confirm-like tests:

| Test | Prediction | Status |
|---|---|---|
| R10-T04 | P30 | `density_kappa_sdss_route_confirm_like_v75` |
| R10-T10 | P38 | `void_morphology_artifact_backed_confirm_like_v54` |
| R10-T12 | P36/local a0 | `robust_confirm_like` |
| R10-T13 | P36/high-z a0 | `highz_a0_clean_claim_confirm_like_v70` |
| R10-T14 | P36 | `highz_a0_clean_claim_confirm_like_v70` |
| R10-T21 | P40 | `p40_bb_likelihood_confirm_like_v67` |
| R10-T22 | P40 | `p40_bb_likelihood_confirm_like_v67` |
| R10-T31 | P41 | `p41_q2_wilson_likelihood_confirm_like_v68` |
| R10-T32 | P41 | `p41_q2_wilson_likelihood_confirm_like_v68` |
| R10-DCN01 | DCN_k | `dcn_allowed_window_source_extracted_confirm_like_v71` |
| R10-DCN02 | DCN_k | `dcn_allowed_window_source_extracted_confirm_like_v71` |
| R10-CL06 | CL6 | `cl6_p41_p40_bridge_confirm_like_v69` |

SM constant confirm-like tests:

| Test | Prediction | Status |
|---|---|---|
| R10-SMD01 | SM-D1 | `smd_constant_consistency_confirm_like` |
| R10-SMD02 | SM-D2 | `smd_constant_consistency_confirm_like` |
| R10-SMD03 | SM-D3 | `smd_constant_consistency_confirm_like` |
| R10-SMD04 | SM-D4 | `smd_constant_consistency_confirm_like` |
| R10-SMD05 | SM-D5 | `smd_constant_consistency_confirm_like` |

Coverage confirmed:

| Test | Prediction | Status |
|---|---|---|
| R10-T26 | P10/P25/P31 | `mass_window_coverage_confirmed` |

## Full Test Table

| Test | Prediction | Status | Readout |
|---|---|---|---|
| R10-T01 | P39 | `positive_compatible` | BAO/Pantheon compatible, not claim-grade; v75 covariance gate still open. |
| R10-T02 | P1/P39 | `positive_compatible` | Low-z systematic plus BAO remains compatible, not confirmed. |
| R10-T03 | P39 | `positive_compatible` | DESI BAO grid remains positive-compatible only. |
| R10-T04 | P30 | `density_kappa_sdss_route_confirm_like_v75` | SDSS route confirm-like preserved; global P30 still needs second route. |
| R10-T05 | P30 | `density_kappa_planck_route_ready_v75` | Planck products/maps found, but no same-mask density-kappa statistic/nulls. |
| R10-T06 | P30 | `euclid_mer_catalogue_sample_positive_ready_v75` | Euclid sample/random geometry built, but no kappa-cell join or positive residualized delta. |
| R10-T07 | P33 | `p33_density_bao_alpha_measurement_required_v75` | Blocked by missing exact LSS/random rows and alpha/null outputs. |
| R10-T08 | P35 | `harmonic_proxy_positive_ready` | Proxy remains ready; no public LSS P(k)/xi table passed the strict harmonic gate. |
| R10-T09 | P3 | `filament_endpoint_semantic_candidates_ready_v75` | Improved: v75 recovered 240 endpoint-like rows; redshift/orientation nulls still missing. |
| R10-T10 | P38 | `void_morphology_artifact_backed_confirm_like_v54` | Confirm-like preserved. |
| R10-T11 | CL4 | `partial_positive_bridge` | Bridge-positive, not standalone confirm. |
| R10-T12 | P36/local a0 | `robust_confirm_like` | Confirm-like preserved. |
| R10-T13 | P36/high-z a0 | `highz_a0_clean_claim_confirm_like_v70` | Confirm-like preserved. |
| R10-T14 | P36 | `highz_a0_clean_claim_confirm_like_v70` | Confirm-like preserved. |
| R10-T15 | P29 | `consistent_bound_only` | Bound consistency only. |
| R10-T16 | P8/P8c | `pta_density_cross_positive_ready` | Positive-ready, but weighted residual statistic still missing. |
| R10-T17 | P8c/CL2 | `pta_weighted_kappa_residual_required_v75` | v75 built 180 TOA-weighted pulsars; 0 residual-kappa pairs. |
| R10-T18 | P32 | `ringdown_metadata_positive_ready` | Metadata-positive, not strain-likelihood confirm. |
| R10-T19 | P32 | `ringdown_strain_analysis_required_v75` | Only one local strain event; detector/time-slide/LOO gates still open. |
| R10-T20 | No-FTL | `consistent_bound_only` | Bound consistency only. |
| R10-T21 | P40 | `p40_bb_likelihood_confirm_like_v67` | Confirm-like preserved. |
| R10-T22 | P40 | `p40_bb_likelihood_confirm_like_v67` | Confirm-like preserved. |
| R10-T23 | P28 | `consistent_bound_only` | Bound consistency only. |
| R10-T24 | P28 | `consistent_bound_only` | Bound consistency only. |
| R10-T25 | P10/P25/P31 | `mass_window_quantified_coverage_positive_ready` | Quantified coverage positive-ready. |
| R10-T26 | P10/P25/P31 | `mass_window_coverage_confirmed` | Coverage confirmed. |
| R10-T27 | P10/P25/P31 | `mass_window_quantified_positive_ready` | Quantified positive-ready. |
| R10-T28 | P27 | `sensitivity_positive_ready` | Sensitivity ready, not detection confirm. |
| R10-T29 | P37 | `event_level_ready_not_detection_confirmed` | Event-level ready, not detection confirmed. |
| R10-T30 | P5 | `kss_proxy_bound_positive_schema_backed` | Schema-backed bound positive. |
| R10-T31 | P41 | `p41_q2_wilson_likelihood_confirm_like_v68` | Confirm-like preserved. |
| R10-T32 | P41 | `p41_q2_wilson_likelihood_confirm_like_v68` | Confirm-like preserved. |
| R10-T33 | P9b/P9e/P9f | `hepdata_schema_positive_ready` | Schema positive-ready. |
| R10-SMD01 | SM-D1 | `smd_constant_consistency_confirm_like` | SM constant confirm-like. |
| R10-SMD02 | SM-D2 | `smd_constant_consistency_confirm_like` | SM constant confirm-like. |
| R10-SMD03 | SM-D3 | `smd_constant_consistency_confirm_like` | SM constant confirm-like. |
| R10-SMD04 | SM-D4 | `smd_constant_consistency_confirm_like` | SM constant confirm-like. |
| R10-SMD05 | SM-D5 | `smd_constant_consistency_confirm_like` | SM constant confirm-like. |
| R10-SMD06 | SM-D6 | `consistent_constant_check` | Constant check only. |
| R10-SMD07 | SM-D7 | `consistent_constant_check` | Constant check only. |
| R10-SMD08 | SM-D8 | `consistent_constant_check` | Constant check only. |
| R10-SMD09 | SM-D9 | `structural_consistency_positive` | Structural consistency positive. |
| R10-SMD10 | SM-D10 | `consistent_constant_check` | Constant/bound check only. |
| R10-DC01 | Dark-Cone | `branch_survival_positive` | Branch-survival positive. |
| R10-DC02 | Dark-Cone | `partial_positive_bridge` | Partial bridge-positive. |
| R10-DC03 | Dark-Cone | `branch_survival_positive` | Branch-survival positive. |
| R10-DCN01 | DCN_k | `dcn_allowed_window_source_extracted_confirm_like_v71` | Confirm-like preserved. |
| R10-DCN02 | DCN_k | `dcn_allowed_window_source_extracted_confirm_like_v71` | Confirm-like preserved. |
| R10-CL05 | CL5 | `partial_positive_bridge` | Partial bridge-positive. |
| R10-CL06 | CL6 | `cl6_p41_p40_bridge_confirm_like_v69` | Confirm-like bridge preserved. |
| R10-DASH | P/CL dashboard | `dashboard_positive_current_only_v75` | Dashboard current and positive-only. |

## v75 Gate Blockers

Strict v75 gates still missing:

| Test | Gate | Missing |
|---|---|---|
| R10-T01/T02/T03 | P39 covariance/systematics | `delta_chi2_lcdm_minus_best_ge_9`, `pantheon_full_covariance_used`, `systematics_splits_done`, `model_penalty_supports_new_model` |
| R10-T04 | P30 global route | `second_independent_route_confirm_like_with_public_kappa_or_map_values` |
| R10-T05 | P30 Planck route | `planck_same_mask_density_kappa_statistic`, `planck_sky_shuffle_p_le_0p05`, `planck_density_shuffle_p_le_0p05` |
| R10-T06 | P30 Euclid route | `euclid_photoz_or_redshift_column`, `euclid_public_kappa_values_for_same_mask_cells`, `euclid_positive_delta_after_residualization`, `euclid_sky_shuffle_p_le_0p05`, `euclid_density_shuffle_p_le_0p05` |
| R10-T07 | P33 | `exact_lss_clustering_rows_ge_800`, `exact_random_rows_ge_800`, alpha split, sky/density nulls, redshift jackknife |
| R10-T08 | P35 | `public_lss_pk_or_xi_table`, `harmonic_peaks_ge_4`, `phase_randomized_p_le_0p05` |
| R10-T09 | P3 | `endpoint_redshift_null`, `orientation_shuffle_null` |
| R10-T17 | PTA / CL2 | `public_residual_kappa_pairs_ge_20`, weighted statistic, signed sky shuffle, top-weight stability |
| R10-T19 | P32 | two local strain events, detector delta-chi2, time-slide null, leave-one-event-out stability |

## v75 Artifact Notes

| Artifact | Status | Key numbers |
|---|---|---|
| `p30_second_route_statistic_v75_AUTO_PUBLIC.json` | `p30_second_route_still_not_confirmed_v75` | `n_routes_confirm_like=1`, second route false |
| `p30_planck_map_route_v75_AUTO_PUBLIC.json` | `p30_second_route_still_not_confirmed_v75` | Planck route still lacks same-mask statistic/nulls |
| `p30_euclid_same_mask_geometry_v75_AUTO_PUBLIC.json` | `p30_second_route_still_not_confirmed_v75` | Euclid geometry built: 20,000 sample rows, 100,000 random rows, 144 cells; no public kappa values for cells |
| `pta_toa_weighted_residual_recovery_v75_AUTO_PUBLIC.json` | `pta_toa_weight_proxy_built_residuals_missing_v75` | 180 TOA-weighted pulsars, 0 residual-kappa pairs |
| `p33_archive_exact_lss_random_v75_AUTO_PUBLIC.json` | `p33_exact_lss_random_inputs_still_absent_v75` | 0 exact catalog rows, 0 exact random rows |
| `p35_extended_pk_xi_harmonic_v75_AUTO_PUBLIC.json` | `p35_lss_pk_xi_table_absent_v75` | no public LSS P(k)/xi table accepted |
| `p3_endpoint_edge_parser_v75.json` | `p3_endpoint_rows_recovered_v75` | 240 endpoint-like rows recovered |
| `p32_eventapi_strain_index_v75_AUTO_PUBLIC.json` | `p32_single_event_strain_only_v75` | 1 local strain event, 1 event API strain event |
| `p39_covariance_systematics_chain_v75_AUTO_PUBLIC.json` | `p39_full_covariance_still_absent_v75` | Pantheon diagonal rows 1702, Pantheon full covariance false, BAO covariance true, delta chi2 1.2118 |

## Suggested Improvements For Confirms

1. **Promote P3 endpoint candidates into a strict endpoint test.**
   - v75 recovered 240 endpoint-like rows, which is the best new near-confirm.
   - Add redshift extraction for those rows, build endpoint-pair orientation statistics, then run orientation shuffle and redshift nulls.
   - Confirm target: `p3_endpoint_orientation_confirm_like`.

2. **Turn PTA TOA weights into residual-kappa pairs.**
   - v75 has 180 TOA-weighted pulsars and existing kappa sampling, but still 0 residual-kappa pairs.
   - Parse post-fit residual-like columns from timing products or companion residual tables, join by pulsar id to kappa, then compute signed weighted statistic, sky shuffle, and top-weight removal.
   - Confirm target: `pta_weighted_kappa_residual_confirm_like`.

3. **Close the P30 Planck route with a real map-level statistic.**
   - v75 found Planck map-level candidates, but no same-mask density-kappa statistic.
   - Add a map/alm reader for the Planck lensing product, sample it on the same density/mask cells, then run sky and density shuffles.
   - Confirm target: global P30 second route.

4. **Close the P30 Euclid route by joining kappa values to v75 cells.**
   - v75 built the Euclid sample/random geometry, but lacks kappa values and redshift/photo-z.
   - Add kappa cell sampling from the available lensing field, then residualize depth/photo-z if recoverable. This route is risky because v74/v75 indicated an unfavorable residualized Euclid delta.
   - Confirm target: second independent P30 route.

5. **Recover a second local GWOSC strain event for P32.**
   - v75 still has only one local strain event.
   - Fetch or locate a second public strain event, then run the detector delta-chi2, time-slide null, and leave-one-event-out stability.
   - Confirm target: `ringdown_strain_likelihood_confirm_like`.

6. **Find exact DESI/BOSS LSS and random catalogues for P33.**
   - Current local cache has compressed BAO products but 0 exact catalog/random rows.
   - Add support for any archived FITS/table sources if present, or stage exact public clustering/random files, then compute density-split alpha with nulls and redshift jackknife.
   - Confirm target: `p33_density_bao_alpha_confirm_like`.

7. **Recover Pantheon full covariance/systematics for P39/P1.**
   - v75 has Pantheon diagonal rows and BAO covariance, but no Pantheon full covariance; delta chi2 is only 1.2118.
   - Add full covariance/systematics parsing and run the AIC/BIC-aware likelihood chain.
   - Confirm target: `p39_likelihood_confirm_like`.

8. **Source an actual public LSS P(k)/xi table for P35.**
   - v75 bounded the scan and found no accepted table.
   - Add exact P(k), xi, or correlation-function source discovery, then require at least 4 harmonic peaks and phase-randomized p <= 0.05.
   - Confirm target: `p35_harmonic_pk_confirm_like`.

## Priority Ranking

Most promising confirm path: **P3 endpoint orientation**, because v75 already recovered 240 endpoint-like rows.

Second tier: **PTA residual-kappa join** and **P32 second strain event**, because the source scaffolds are partly present but one missing data product blocks promotion.

Higher risk but high value: **P30 second route** and **P39/P1 full covariance**, because they could materially change the dashboard but require stronger source recovery.
