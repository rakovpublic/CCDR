# Round 10 Latest Run Report - v73

Generated from `outputs/round10_summary.json` and `outputs/test51_round10_joint_dashboard.json`.

- Runner version: `v73`
- Updated UTC: `2026-05-18T08:46:01Z`
- Tests run: `51 / 51`
- Completion flag: `run_complete_v73 = true`
- Dashboard counts: `12` non-SM confirm-like, `5` SM constant-consistency checks, `1` coverage-confirmed, `4` blocked/gate-failed
- v73 artifacts indexed by dashboard: `11`

## Executive Summary

The v73 run is complete and internally consistent. The main gain versus v72 is P30: the SDSS scoped route is now restored as confirm-like while the global P30 claim remains correctly blocked until a second independent route passes.

The four active blockers are still data/product blockers, not runner failures:

- `R10-T07 / P33`: exact LSS clustering and random rows remain absent.
- `R10-T09 / P3`: explicit endpoint/node-pair rows remain absent.
- `R10-T17 / P8c/CL2`: public NANOGrav coordinates are present, but residual-kappa pairs are absent.
- `R10-T19 / P32`: only `GW150914` is available as a strain event in the current cache.

The three P39/P1 tests are `positive_compatible`, not confirm-like, because the publication likelihood chain still lacks full Pantheon covariance/systematics support and has only `Delta chi2 = 1.2117649731708635`, below the `>=9` gate.

## All Test Results

| Test | Prediction | Status | Class |
|---|---|---|---|
| R10-T01 | P39 | `positive_compatible` | Ready/compatible |
| R10-T02 | P1/P39 | `positive_compatible` | Ready/compatible |
| R10-T03 | P39 | `positive_compatible` | Ready/compatible |
| R10-T04 | P30 | `density_kappa_sdss_route_confirm_like_v73` | Confirm-like |
| R10-T05 | P30 | `density_kappa_planck_route_ready_v73` | Ready/compatible |
| R10-T06 | P30 | `euclid_mer_catalogue_sample_positive_ready_v73` | Ready/compatible |
| R10-T07 | P33 | `p33_density_bao_alpha_measurement_required_v73` | Blocked |
| R10-T08 | P35 | `harmonic_proxy_positive_ready` | Ready/compatible |
| R10-T09 | P3 | `endpoint_data_limited` | Blocked |
| R10-T10 | P38 | `void_morphology_artifact_backed_confirm_like_v54` | Confirm-like |
| R10-T11 | CL4 | `partial_positive_bridge` | Partial |
| R10-T12 | P36/local a0 | `robust_confirm_like` | Confirm-like |
| R10-T13 | P36/high-z a0 | `highz_a0_clean_claim_confirm_like_v70` | Confirm-like |
| R10-T14 | P36 | `highz_a0_clean_claim_confirm_like_v70` | Confirm-like |
| R10-T15 | P29 | `consistent_bound_only` | Bound only |
| R10-T16 | P8/P8c | `pta_density_cross_positive_ready` | Ready/compatible |
| R10-T17 | P8c/CL2 | `pta_weighted_kappa_residual_required_v73` | Blocked |
| R10-T18 | P32 | `ringdown_metadata_positive_ready` | Ready/compatible |
| R10-T19 | P32 | `ringdown_strain_analysis_required_v73` | Blocked |
| R10-T20 | No-FTL | `consistent_bound_only` | Bound only |
| R10-T21 | P40 | `p40_bb_likelihood_confirm_like_v67` | Confirm-like |
| R10-T22 | P40 | `p40_bb_likelihood_confirm_like_v67` | Confirm-like |
| R10-T23 | P28 | `consistent_bound_only` | Bound only |
| R10-T24 | P28 | `consistent_bound_only` | Bound only |
| R10-T25 | P10/P25/P31 | `mass_window_quantified_coverage_positive_ready` | Ready/compatible |
| R10-T26 | P10/P25/P31 | `mass_window_coverage_confirmed` | Coverage-confirmed |
| R10-T27 | P10/P25/P31 | `mass_window_quantified_positive_ready` | Ready/compatible |
| R10-T28 | P27 | `sensitivity_positive_ready` | Ready/compatible |
| R10-T29 | P37 | `event_level_ready_not_detection_confirmed` | Ready/compatible |
| R10-T30 | P5 | `kss_proxy_bound_positive_schema_backed` | Schema-backed |
| R10-T31 | P41 | `p41_q2_wilson_likelihood_confirm_like_v68` | Confirm-like |
| R10-T32 | P41 | `p41_q2_wilson_likelihood_confirm_like_v68` | Confirm-like |
| R10-T33 | P9b/P9e/P9f | `hepdata_schema_positive_ready` | Ready/compatible |
| R10-SMD01 | SM-D1 | `smd_constant_consistency_confirm_like` | SM consistency |
| R10-SMD02 | SM-D2 | `smd_constant_consistency_confirm_like` | SM consistency |
| R10-SMD03 | SM-D3 | `smd_constant_consistency_confirm_like` | SM consistency |
| R10-SMD04 | SM-D4 | `smd_constant_consistency_confirm_like` | SM consistency |
| R10-SMD05 | SM-D5 | `smd_constant_consistency_confirm_like` | SM consistency |
| R10-SMD06 | SM-D6 | `consistent_constant_check` | Constant check |
| R10-SMD07 | SM-D7 | `consistent_constant_check` | Constant check |
| R10-SMD08 | SM-D8 | `consistent_constant_check` | Constant check |
| R10-SMD09 | SM-D9 | `structural_consistency_positive` | Structural positive |
| R10-SMD10 | SM-D10 | `consistent_constant_check` | Constant check |
| R10-DC01 | Dark-Cone | `branch_survival_positive` | Positive |
| R10-DC02 | Dark-Cone | `partial_positive_bridge` | Partial |
| R10-DC03 | Dark-Cone | `branch_survival_positive` | Positive |
| R10-DCN01 | DCN_k | `dcn_allowed_window_source_extracted_confirm_like_v71` | Confirm-like |
| R10-DCN02 | DCN_k | `dcn_allowed_window_source_extracted_confirm_like_v71` | Confirm-like |
| R10-CL05 | CL5 | `partial_positive_bridge` | Partial |
| R10-CL06 | CL6 | `cl6_p41_p40_bridge_confirm_like_v69` | Confirm-like |
| R10-DASH | P/CL dashboard | `dashboard_positive_current_only_v73` | Dashboard |

## Current Confirm-Like Tests

Dashboard v73 counts these 12 non-SM confirm-like results:

- `R10-T04 / P30`: `density_kappa_sdss_route_confirm_like_v73`
- `R10-T10 / P38`: `void_morphology_artifact_backed_confirm_like_v54`
- `R10-T12 / P36/local a0`: `robust_confirm_like`
- `R10-T13 / P36/high-z a0`: `highz_a0_clean_claim_confirm_like_v70`
- `R10-T14 / P36`: `highz_a0_clean_claim_confirm_like_v70`
- `R10-T21 / P40`: `p40_bb_likelihood_confirm_like_v67`
- `R10-T22 / P40`: `p40_bb_likelihood_confirm_like_v67`
- `R10-T31 / P41`: `p41_q2_wilson_likelihood_confirm_like_v68`
- `R10-T32 / P41`: `p41_q2_wilson_likelihood_confirm_like_v68`
- `R10-DCN01 / DCN_k`: `dcn_allowed_window_source_extracted_confirm_like_v71`
- `R10-DCN02 / DCN_k`: `dcn_allowed_window_source_extracted_confirm_like_v71`
- `R10-CL06 / CL6`: `cl6_p41_p40_bridge_confirm_like_v69`

## v73 Artifact Evidence

| Artifact | Status | Key evidence |
|---|---|---|
| `measurements/p30_sdss_route_repair_matrix_v73_AUTO_PUBLIC.json` | `p30_sdss_route_repaired_confirm_like_v73` | `sdss_scoped_confirm_like=true`, `n_routes_confirm_like=1` |
| `measurements/pta_public_residual_kappa_source_scan_v73_AUTO_PUBLIC.json` | `pta_public_residual_kappa_pairs_still_absent_v73` | `n_public_par_coordinates=98`, `n_public_residual_kappa_pairs=0` |
| `measurements/p33_exact_lss_random_source_scan_v73_AUTO_PUBLIC.json` | `p33_exact_lss_random_inputs_still_absent_v73` | `n_exact_catalog_rows=0`, `n_exact_random_rows=0` |
| `measurements/p35_lss_pk_xi_source_scan_v73_AUTO_PUBLIC.json` | `p35_lss_pk_or_xi_table_absent_v73` | no public LSS P(k)/xi table found |
| `outputs/p3_endpoint_semantic_source_scan_v73.json` | `p3_endpoint_semantics_still_absent_v73` | `n_strict_endpoint_rows=0` |
| `measurements/p32_multi_event_strain_source_scan_v73_AUTO_PUBLIC.json` | `p32_single_event_strain_only_v73` | `event_ids_seen=["GW150914"]` |
| `measurements/p39_full_covariance_source_scan_v73_AUTO_PUBLIC.json` | `p39_full_covariance_source_audit_built_v73` | `Delta chi2=1.2117649731708635`, `pantheon_full_covariance_used=false`, `bao_full_covariance_used=true` |

## Main Failed v73 Gates

- P39/P1 (`R10-T01` to `R10-T03`): missing `delta_chi2_lcdm_minus_best_ge_9`, `pantheon_full_covariance_used`, `systematics_splits_done`, `model_penalty_supports_new_model`.
- P30 global (`R10-T04`): missing `second_independent_route_confirm_like`.
- P30 Planck (`R10-T05`): missing `planck_same_mask_statistic`, `planck_sky_shuffle_p_le_0p05`, `planck_density_shuffle_p_le_0p05`.
- P30 Euclid (`R10-T06`): missing `euclid_photoz_depth_randoms`, `euclid_same_mask_statistic`, `euclid_sky_shuffle_p_le_0p05`, `euclid_density_shuffle_p_le_0p05`.
- P33 (`R10-T07`): missing exact LSS rows, exact random rows, alpha high/low/delta, sky null, label null, redshift jackknife.
- P35 (`R10-T08`): missing public LSS P(k)/xi table, `harmonic_peaks_ge_4`, `phase_randomized_p_le_0p05`.
- P3 (`R10-T09`): missing explicit endpoint/node-pair rows, endpoint-redshift null, orientation-shuffle null.
- PTA/CL2 (`R10-T17`): missing public residual-kappa pairs, positive weighted statistic, signed shuffle p<=0.05, top-weight stability.
- P32 (`R10-T19`): missing two distinct public strain events, detector delta chi2>=4, time-slide p<=0.05, leave-one-event-out stability.

## Suggested Improvements For More Confirms

1. P30 second independent route confirmer.
   Build an executable Planck and/or Euclid same-mask route that mirrors the SDSS route contract. Confirm target: promote global P30 only if a second route passes the same-mask statistic, sky shuffle p<=0.05, density shuffle p<=0.05, and route sign stability.

2. PTA residual-kappa join builder.
   Use the 98 public NANOGrav par coordinates now found by v73, parse post-fit residual or TOA residual products, sample ACT/Planck kappa at pulsar positions, and build a source-hashed residual-kappa table. Confirm target: >=20 public residual-kappa pairs, positive weighted statistic, one-sided signed shuffle p<=0.05, top-weight stability.

3. P33 exact DESI/BOSS LSS plus random ingestion.
   Add a targeted LSScats/random parser for DESI/BOSS/eBOSS clustering products rather than relying on compressed BAO summaries. Confirm target: >=800 clustering rows, >=800 random rows, alpha high/low/delta, sky shuffle p<=0.05, density-label shuffle p<=0.05, redshift jackknife stable.

4. P35 public P(k)/xi harmonic parser.
   Ingest public DESI/BOSS/eBOSS correlation-function or power-spectrum tables with source hashes and covariance where available. Confirm target: real LSS P(k)/xi table, >=4 harmonic peaks, phase-randomized p<=0.05.

5. P3 endpoint catalogue recovery.
   Target explicit filament endpoint catalogues, especially Tempel/VizieR/CosmicWeb-style node-pair tables, and parse endpoint redshift columns. Confirm target: strict endpoint/node-pair rows, endpoint-redshift null, orientation-shuffle null.

6. P32 multi-event strain likelihood.
   Add at least one more public strain event beyond `GW150914`, then run the same detector-split ringdown likelihood with time slides and leave-one-event-out stability. Confirm target: >=2 distinct public strain events, min detector delta chi2>=4, time-slide p<=0.05, event LOO stable.

7. P39/P1 publication likelihood chain.
   Add Pantheon full covariance/systematics and rerun the combined SN+BAO model comparison with AIC/BIC accounting. Confirm target: full Pantheon covariance, systematics splits, BAO covariance, Delta chi2>=9, AIC/BIC support for the new model.

## Lower-Priority Confirm Opportunities

These are positive or ready but need stricter evidence before they should be counted as confirms:

- `R10-T25`, `R10-T27`, `R10-T28`, `R10-T29`: direct-detection/event-level routes need likelihood-level products, not only coverage/sensitivity readiness.
- `R10-T11`, `R10-DC02`, `R10-CL05`: bridge tests should only promote if their upstream anchors are independently confirm-like and the bridge artifact proves non-duplication.
- `R10-T30`, `R10-T33`: HEPData/schema-backed positives need numerical likelihood or residual tests before confirmation.
- `R10-SMD06` to `R10-SMD10`: current results are constant/structural checks, not derivation confirms.

## Bottom Line

v73 is a good confirmation-accounting run: one important scoped confirm was recovered for P30, and no readiness result was over-counted. The fastest next confirm path is probably P30 global, because SDSS is already confirmed and only one independent route is missing. The next best scientific paths are PTA residual-kappa and P32 multi-event strain, but both require actual missing public data products rather than only parser changes.
