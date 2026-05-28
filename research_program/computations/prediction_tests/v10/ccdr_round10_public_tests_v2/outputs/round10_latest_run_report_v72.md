# Round 10 Latest Run Report (v72)

Generated from `outputs/round10_summary.json` updated `2026-05-18T03:04:27Z` (`2026-05-18T06:04:27+03:00` local Kyiv time).

## Executive Summary

- Suite completeness: `51/51` tests, `run_complete_v72=True`.
- Runner version: `v72`.
- Dashboard status: `dashboard_positive_current_only_v72`.
- Claim-grade non-SM confirm-like results: `11`.
- SM constant consistency confirms: `5`.
- Coverage-only confirm: `1`.
- Active blocked/data-limited tests in the dashboard: `5`.
- v72 is stricter than v71: P30/SDSS and PTA/CL2 are no longer counted as confirm-like because the current public-only gates fail.

The latest run is complete and more conservative. The strongest current non-SM confirmations are P38, P36, P40, P41, DCN/AQN, and the CL6 bridge. The main confirm blockers are P30 global route closure, P33 exact density-split BAO, P3 endpoint semantics, PTA public residual-kappa rows, and P32 multi-event strain.

## Current Confirm-Like Results

| Test | Prediction | Status | Note |
|---|---|---|---|
| `R10-T10` | `P38` | `void_morphology_artifact_backed_confirm_like_v54` | Void morphology remains artifact-backed confirm-like. |
| `R10-T12` | `P36/local a0` | `robust_confirm_like` | SPARC local RAR/a0 bootstrap path remains robust. |
| `R10-T13` | `P36/high-z a0` | `highz_a0_clean_claim_confirm_like_v70` | High-z clean-claim path remains confirm-like. |
| `R10-T14` | `P36` | `highz_a0_clean_claim_confirm_like_v70` | Cross-catalogue high-z a0 path remains confirm-like. |
| `R10-T21` | `P40` | `p40_bb_likelihood_confirm_like_v67` | BK18 bandpower/covariance path remains confirm-like. |
| `R10-T22` | `P40` | `p40_bb_likelihood_confirm_like_v67` | Planck/BK18 cross-bound path remains confirm-like. |
| `R10-T31` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | LHCb q2/Wilson likelihood proxy remains confirm-like. |
| `R10-T32` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | Control supplementary q2/Wilson path remains confirm-like. |
| `R10-DCN01` | `DCN_k` | `dcn_allowed_window_source_extracted_confirm_like_v71` | Source-extracted quantified window remains confirm-like. |
| `R10-DCN02` | `DCN_k` | `dcn_allowed_window_source_extracted_confirm_like_v71` | Source-extracted quantified window remains confirm-like. |
| `R10-CL06` | `CL6` | `cl6_p41_p40_bridge_confirm_like_v69` | Dependent bridge from confirmed P40 and P41 anchors. |

## Evidence Snapshot

| Area | Evidence | Interpretation |
|---|---|---|
| Dashboard | `n_nonSM_confirm_like=11`, `n_SM_constant_consistency=5`, `n_coverage_confirmed=1`, `n_blocked_or_gate_failed=5` | v72 has fewer confirms because it enforces stricter source/public-input separation. |
| P30 | `p30_cross_route_closure_matrix_v72` has `n_routes_confirm_like=0`; missing `sdss_route_confirm_like` and `second_independent_route_confirm_like` | P30 is now blocked globally and no longer counted as scoped SDSS confirm in the latest recompute. |
| P33 | `n_exact_catalog_rows=0`, `n_exact_random_rows=0`, `n_compressed_bao_products=40` | BAO products exist, but exact clustering/random inputs for density-split alpha are absent. |
| P35 | `n_compressed_bao_products=40`, status `p35_lss_pk_or_xi_table_absent` | Compressed BAO coverage is ready; no LSS P(k)/xi table is available for claim-grade harmonic evidence. |
| P3 | `n_strict_endpoint_rows=0`, status `p3_endpoint_semantics_still_absent` | No explicit endpoint/node-pair rows, so endpoint-redshift and orientation nulls cannot run. |
| PTA/CL2 | `0` public residual-kappa pairs, `44` measurement/output echo rows excluded | v72 correctly blocks PTA/CL2; previous near-miss was not supported by public residual-kappa source rows. |
| P32 | distinct event IDs: `GW150914`; min detector delta chi2 `0.1924557`; max time-slide p `0.3125` | Single-event strain only; multi-event and null thresholds fail. |
| P39/P1 | `1702` Pantheon diagonal rows, BAO covariance used, Pantheon full covariance false, delta chi2 `1.2117649732` | Positive-compatible only; far below publication-grade likelihood confirmation. |
| DCN/AQN | DCN01: `23` quantified, `9` constraint-like, `5` allowed/detection-like claims. DCN02: `15`, `6`, `8` | DCN/AQN remains source-extracted confirm-like. |

## All Tests

| # | Test | Prediction | Status | Bucket |
|---:|---|---|---|---|
| 1 | `R10-T01` | `P39` | `positive_compatible` | positive/ready; publication likelihood gate unmet |
| 2 | `R10-T02` | `P1/P39` | `positive_compatible` | positive/ready; publication likelihood gate unmet |
| 3 | `R10-T03` | `P39` | `positive_compatible` | positive/ready; BAO likelihood not claim-grade |
| 4 | `R10-T04` | `P30` | `density_kappa_global_route_blocked_v72` | blocked; SDSS/global route gate fails |
| 5 | `R10-T05` | `P30` | `density_kappa_planck_route_ready_v72` | positive/ready; Planck route not confirm |
| 6 | `R10-T06` | `P30` | `euclid_mer_catalogue_sample_positive_ready_v72` | positive/ready; Euclid route not confirm |
| 7 | `R10-T07` | `P33` | `p33_density_bao_alpha_measurement_required_v72` | blocked/data-limited |
| 8 | `R10-T08` | `P35` | `harmonic_proxy_positive_ready` | positive/ready; LSS P(k)/xi absent |
| 9 | `R10-T09` | `P3` | `endpoint_data_limited` | blocked/data-limited |
| 10 | `R10-T10` | `P38` | `void_morphology_artifact_backed_confirm_like_v54` | non-SM confirm-like |
| 11 | `R10-T11` | `CL4` | `partial_positive_bridge` | bridge positive, not confirm |
| 12 | `R10-T12` | `P36/local a0` | `robust_confirm_like` | non-SM confirm-like |
| 13 | `R10-T13` | `P36/high-z a0` | `highz_a0_clean_claim_confirm_like_v70` | non-SM confirm-like |
| 14 | `R10-T14` | `P36` | `highz_a0_clean_claim_confirm_like_v70` | non-SM confirm-like |
| 15 | `R10-T15` | `P29` | `consistent_bound_only` | bound consistency only |
| 16 | `R10-T16` | `P8/P8c` | `pta_density_cross_positive_ready` | positive/ready; supports PTA path |
| 17 | `R10-T17` | `P8c/CL2` | `pta_weighted_kappa_residual_required_v72` | blocked; public residual-kappa rows absent |
| 18 | `R10-T18` | `P32` | `ringdown_metadata_positive_ready` | metadata ready only |
| 19 | `R10-T19` | `P32` | `ringdown_strain_analysis_required_v72` | blocked/data-limited |
| 20 | `R10-T20` | `No-FTL` | `consistent_bound_only` | bound consistency only |
| 21 | `R10-T21` | `P40` | `p40_bb_likelihood_confirm_like_v67` | non-SM confirm-like |
| 22 | `R10-T22` | `P40` | `p40_bb_likelihood_confirm_like_v67` | non-SM confirm-like |
| 23 | `R10-T23` | `P28` | `consistent_bound_only` | bound consistency only |
| 24 | `R10-T24` | `P28` | `consistent_bound_only` | bound consistency only |
| 25 | `R10-T25` | `P10/P25/P31` | `mass_window_quantified_coverage_positive_ready` | coverage positive-ready |
| 26 | `R10-T26` | `P10/P25/P31` | `mass_window_coverage_confirmed` | coverage confirm only |
| 27 | `R10-T27` | `P10/P25/P31` | `mass_window_quantified_positive_ready` | positive/ready |
| 28 | `R10-T28` | `P27` | `sensitivity_positive_ready` | positive/ready |
| 29 | `R10-T29` | `P37` | `event_level_ready_not_detection_confirmed` | event-level readiness, not detection |
| 30 | `R10-T30` | `P5` | `kss_proxy_bound_positive_schema_backed` | schema-backed positive-ready |
| 31 | `R10-T31` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | non-SM confirm-like |
| 32 | `R10-T32` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | non-SM confirm-like |
| 33 | `R10-T33` | `P9b/P9e/P9f` | `hepdata_schema_positive_ready` | schema-backed positive-ready |
| 34 | `R10-SMD01` | `SM-D1` | `smd_constant_consistency_confirm_like` | SM consistency confirm |
| 35 | `R10-SMD02` | `SM-D2` | `smd_constant_consistency_confirm_like` | SM consistency confirm |
| 36 | `R10-SMD03` | `SM-D3` | `smd_constant_consistency_confirm_like` | SM consistency confirm |
| 37 | `R10-SMD04` | `SM-D4` | `smd_constant_consistency_confirm_like` | SM consistency confirm |
| 38 | `R10-SMD05` | `SM-D5` | `smd_constant_consistency_confirm_like` | SM consistency confirm |
| 39 | `R10-SMD06` | `SM-D6` | `consistent_constant_check` | SM consistency check |
| 40 | `R10-SMD07` | `SM-D7` | `consistent_constant_check` | SM consistency check |
| 41 | `R10-SMD08` | `SM-D8` | `consistent_constant_check` | SM consistency check |
| 42 | `R10-SMD09` | `SM-D9` | `structural_consistency_positive` | structural consistency positive |
| 43 | `R10-SMD10` | `SM-D10` | `consistent_constant_check` | SM consistency check |
| 44 | `R10-DC01` | `Dark-Cone` | `branch_survival_positive` | branch survival positive |
| 45 | `R10-DC02` | `Dark-Cone` | `partial_positive_bridge` | bridge positive, not confirm |
| 46 | `R10-DC03` | `Dark-Cone` | `branch_survival_positive` | branch survival positive |
| 47 | `R10-DCN01` | `DCN_k` | `dcn_allowed_window_source_extracted_confirm_like_v71` | non-SM confirm-like |
| 48 | `R10-DCN02` | `DCN_k` | `dcn_allowed_window_source_extracted_confirm_like_v71` | non-SM confirm-like |
| 49 | `R10-CL05` | `CL5` | `partial_positive_bridge` | bridge positive, not confirm |
| 50 | `R10-CL06` | `CL6` | `cl6_p41_p40_bridge_confirm_like_v69` | non-SM confirm-like |
| 51 | `R10-DASH` | `P/CL dashboard` | `dashboard_positive_current_only_v72` | dashboard/control |

## Active Confirm Blockers

| Test(s) | Gate / Artifact | Missing / Required |
|---|---|---|
| `R10-T04` | `p30_cross_route_closure_gate_v72` | `sdss_route_confirm_like`, `second_independent_route_confirm_like`. Current matrix has `n_routes_confirm_like=0`. |
| `R10-T07` | `p33_exact_density_split_gate_v72` | Exact LSS clustering rows, exact random rows, alpha high/low/delta, sky p, density-label p, redshift jackknife. |
| `R10-T08` | `p35_lss_pk_harmonic_gate_v72` | Public LSS P(k)/xi table, at least four harmonic peaks, phase-randomized p `<=0.05`. |
| `R10-T09` | `p3_endpoint_semantic_gate_v72` | Explicit endpoint/node-pair rows, endpoint-redshift null, orientation-shuffle null. |
| `R10-T17` | `pta_residual_kappa_public_only_gate_v72` | At least 20 public residual-kappa pairs, positive public statistic, one-sided p `<=0.05`, top-weight stability. |
| `R10-T19` | `p32_multi_event_strain_gate_v72` | Two distinct public strain events, min detector delta chi2 `>=4`, time-slide p `<=0.05`, leave-one-event-out stability. |
| `R10-T01`-`R10-T03` | `p39_publication_likelihood_gate_v72` | Delta chi2 `>=9`, Pantheon full covariance, systematics splits, model-penalty support. |

## Suggested Confirm-Focused Improvements

1. **P30 SDSS route repair plus global split.** First recover the scoped SDSS route by fixing the same-split variant gate; then keep global promotion separate until Planck or Euclid independently passes same-mask/null controls. Target confirms: `R10-T04`, then global P30.
2. **PTA/CL2 public residual-kappa extraction.** Replace the echo-only residual rows with true public NANOGrav residual/TOA-weight rows joined to public kappa samples, then rerun signed sky shuffles and top-weight jackknife. Target confirm: `R10-T17`.
3. **P33 exact density-split BAO.** Load exact DESI/SDSS clustering and random catalogues, compute random-normalized density splits, fit alpha high/low with covariance, and run sky/density-label/redshift nulls. Target confirm: `R10-T07`.
4. **P35 LSS P(k)/xi harmonic parser.** Ingest true LSS power-spectrum or correlation-function tables, subtract broadband, scan the predeclared comb statistic, and run phase-randomized and redshift-bin nulls. Target confirm: `R10-T08`.
5. **P3 endpoint semantic recovery.** Add source-specific endpoint/node-pair parsers for candidate catalogues while preserving the giant-download guard; run endpoint-redshift and orientation-shuffle nulls. Target confirm: `R10-T09`, with downstream `CL4` support.
6. **P32 multi-event strain likelihood.** Cache at least one additional GWOSC event beyond GW150914, then rerun PSD whitening, detector split, off-source injections/time slides, and leave-one-event-out stability. Target confirm: `R10-T19`.
7. **P39/P1 full publication likelihood.** Ingest Pantheon+ full covariance/systematics, combine with BAO covariance using nuisance parameters, and require delta chi2 `>=9` after AIC/BIC penalty. Target confirms: `R10-T01`, `R10-T02`, `R10-T03`, and stronger `CL5`.

## Confirm Priority

The highest priority is P30 scoped-route repair because it lost confirm-like status under v72 and has a clear missing gate. Next are PTA/CL2 and P33: both have precise missing public inputs. P35 and P3 are medium-heavy parser work. P32 and P39/P1 are heavier likelihood/data-ingestion tasks and should remain strictly blocked until the public data requirements truly pass.
