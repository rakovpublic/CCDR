# Round 10 Latest Run Report (v71)

Generated from `outputs/round10_summary.json` updated `2026-05-18T00:18:07Z` (`2026-05-18T03:18:07+03:00` local Kyiv time).

## Executive Summary

- Suite completeness: `51/51` tests, `run_complete_v71=True`.
- Runner version: `v71`.
- Dashboard status: `dashboard_positive_current_only_v71`.
- Claim-grade non-SM confirm-like results: `12`.
- SM constant consistency confirms: `5`.
- Coverage-only confirm: `1`.
- Active blocked/data-limited tests in the dashboard: `4`.
- Important latest-run change: `R10-T17` is no longer confirm-like in the latest aggregate. The v70 signed PTA residual-kappa gate is a near-miss with one-sided signed sky-shuffle `p=0.0579427083`, above the required `p<=0.05`.

The v71 run is complete and conservative. It adds source-hashed DCN/AQN confirm-like support and keeps the new v71 gates strict: P33, P3, PTA/CL2, and P32 strain are not promoted until their exact input/null requirements pass. P39/P1, P30 Planck/Euclid, and P35 remain positive/ready rather than confirm-like under the current evidence rules.

## Current Confirm-Like Results

| Test | Prediction | Status | Note |
|---|---|---|---|
| `R10-T04` | `P30` | `density_kappa_sdss_route_confirm_like_v71` | SDSS route confirm-like only; global route closure is still not met. |
| `R10-T10` | `P38` | `void_morphology_artifact_backed_confirm_like_v54` | Void morphology confirm-like with artifact-backed null hardening. |
| `R10-T12` | `P36/local a0` | `robust_confirm_like` | SPARC local RAR/a0 bootstrap path remains robust. |
| `R10-T13` | `P36/high-z a0` | `highz_a0_clean_claim_confirm_like_v70` | High-z clean-claim path remains confirm-like. |
| `R10-T14` | `P36` | `highz_a0_clean_claim_confirm_like_v70` | Cross-catalogue high-z a0 path remains confirm-like. |
| `R10-T21` | `P40` | `p40_bb_likelihood_confirm_like_v67` | BK18 bandpower/covariance path remains confirm-like. |
| `R10-T22` | `P40` | `p40_bb_likelihood_confirm_like_v67` | Planck/BK18 cross-bound path remains confirm-like. |
| `R10-T31` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | LHCb q2/Wilson likelihood proxy remains confirm-like. |
| `R10-T32` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | Control supplementary q2/Wilson path remains confirm-like. |
| `R10-DCN01` | `DCN_k` | `dcn_allowed_window_source_extracted_confirm_like_v71` | Source-text quantified-window extraction now supports confirm-like status. |
| `R10-DCN02` | `DCN_k` | `dcn_allowed_window_source_extracted_confirm_like_v71` | Source-text quantified-window extraction now supports confirm-like status. |
| `R10-CL06` | `CL6` | `cl6_p41_p40_bridge_confirm_like_v69` | Dependent bridge from confirmed P40 and P41 anchors. |

## Evidence Snapshot

| Area | Evidence | Interpretation |
|---|---|---|
| Dashboard | `n_nonSM_confirm_like=12`, `n_SM_constant_consistency=5`, `n_coverage_confirmed=1`, `n_blocked_or_gate_failed=4` | Latest aggregate is complete but slightly weaker than the earlier v71 target because PTA/CL2 missed the strict p-value threshold. |
| P30 | `p30_global_route_closure_v71_AUTO_PUBLIC.json` status `p30_global_route_still_scoped_to_sdss` | `R10-T04` is confirm-like only for the SDSS route; Planck and Euclid remain ready/supportive, not global confirmation. |
| P33 | `n_exact_catalog_rows=0`, `n_exact_random_rows=0`, `n_compressed_bao_products=40` | Compressed BAO products exist, but exact density-split LSS catalogues/randoms are absent. |
| P35 | `n_compressed_bao_products=40`, `best_power_spectrum_candidate=null`, status `harmonic_pk_exact_spectrum_absent` | The v71 scope guard correctly rejects unrelated Planck spectra; no real LSS P(k) table is available yet. |
| P3 | `n_strict_endpoint_rows=0`, status `strict_endpoint_semantics_absent` | No explicit endpoint/node-pair semantic rows are available; endpoint nulls cannot run. |
| PTA/CL2 | `318` full coordinates, `32` residual-kappa pairs, statistic `1857.6203125`, one-sided p `0.0579427083`, top-weight stable | This is close but not confirm-like; the strict `p<=0.05` gate correctly blocks promotion. |
| P32 | event IDs seen: `GW150914`; min detector delta chi2 `0.1924557`; max time-slide p `0.3125` | One event is indexed, but multi-event strain and null requirements are not met. |
| P39/P1 | `1702` Pantheon diagonal rows, `delta_chi2_lcdm_minus_best=1.2117649732` | Positive-compatible only; far below the confirm threshold and missing full covariance/systematics support. |
| DCN/AQN | DCN01 has `23` quantified claims, `9` constraint-like, `5` allowed/detection-like; DCN02 has `15`, `6`, and `8` | v71 converted both DCN/AQN tests to source-extracted confirm-like evidence. |

## All Tests

| # | Test | Prediction | Status | Bucket |
|---:|---|---|---|---|
| 1 | `R10-T01` | `P39` | `positive_compatible` | positive/ready; P39 covariance/penalty gate unmet |
| 2 | `R10-T02` | `P1/P39` | `positive_compatible` | positive/ready; P39/P1 covariance/penalty gate unmet |
| 3 | `R10-T03` | `P39` | `positive_compatible` | positive/ready; BAO likelihood not claim-grade |
| 4 | `R10-T04` | `P30` | `density_kappa_sdss_route_confirm_like_v71` | non-SM confirm-like |
| 5 | `R10-T05` | `P30` | `density_kappa_planck_route_ready_v71` | positive/ready; global-route support only |
| 6 | `R10-T06` | `P30` | `euclid_mer_catalogue_sample_positive_ready_v71` | positive/ready; global-route support only |
| 7 | `R10-T07` | `P33` | `p33_density_bao_alpha_measurement_required_v71` | blocked/data-limited |
| 8 | `R10-T08` | `P35` | `harmonic_proxy_positive_ready` | positive/ready; P(k) absent |
| 9 | `R10-T09` | `P3` | `endpoint_data_limited` | blocked/data-limited |
| 10 | `R10-T10` | `P38` | `void_morphology_artifact_backed_confirm_like_v54` | non-SM confirm-like |
| 11 | `R10-T11` | `CL4` | `partial_positive_bridge` | bridge positive, not confirm |
| 12 | `R10-T12` | `P36/local a0` | `robust_confirm_like` | non-SM confirm-like |
| 13 | `R10-T13` | `P36/high-z a0` | `highz_a0_clean_claim_confirm_like_v70` | non-SM confirm-like |
| 14 | `R10-T14` | `P36` | `highz_a0_clean_claim_confirm_like_v70` | non-SM confirm-like |
| 15 | `R10-T15` | `P29` | `consistent_bound_only` | bound consistency only |
| 16 | `R10-T16` | `P8/P8c` | `pta_density_cross_positive_ready` | positive/ready; supports PTA path |
| 17 | `R10-T17` | `P8c/CL2` | `pta_weighted_kappa_residual_required_v70` | blocked; near-miss signed p-value |
| 18 | `R10-T18` | `P32` | `ringdown_metadata_positive_ready` | metadata ready only |
| 19 | `R10-T19` | `P32` | `ringdown_strain_analysis_required_v71` | blocked/data-limited |
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
| 51 | `R10-DASH` | `P/CL dashboard` | `dashboard_positive_current_only_v71` | dashboard/control |

## Active Confirm Blockers

| Test(s) | Gate / Artifact | Missing / Required |
|---|---|---|
| `R10-T07` | `p33_density_split_alpha_v71_AUTO_PUBLIC.json` | Exact DESI/SDSS clustering rows, random rows, random-normalized density split, alpha fit, sky shuffle, density-label shuffle, redshift jackknife. |
| `R10-T09` | `p3_endpoint_recovery_v71.json` | Explicit endpoint or node-pair rows, endpoint-redshift null, orientation-shuffle null. |
| `R10-T17` | `pta_residual_kappa_signed_gate_v70_AUTO_PUBLIC.json` | Signed one-sided sky-shuffle p must be `<=0.05`; latest value is `0.0579427083`. |
| `R10-T19` | `p32_multi_event_strain_likelihood_v71_AUTO_PUBLIC.json` | At least two public strain events, detector delta chi2 `>=4`, time-slide p `<=0.05`, leave-one-event-out stability. |
| `R10-T01`-`R10-T03` | `p39_full_covariance_penalty_chain_v71_AUTO_PUBLIC.json` | Pantheon full covariance/systematics, model penalty support, and `delta_chi2_lcdm_minus_best>=9`; latest value is `1.2117649732`. |
| `R10-T05`/`R10-T06` | `p30_global_route_closure_v71_AUTO_PUBLIC.json` | Independent Planck/Euclid route closure with same-mask statistic/nulls before global P30 promotion. |
| `R10-T08` | `p35_harmonic_pk_parser_v71_AUTO_PUBLIC.json` | Real LSS P(k) or harmonic power-spectrum table, at least four harmonic peaks, phase-randomized p `<=0.05`. |

## Suggested Confirm-Focused Improvements

1. **Recover PTA/CL2 signed residual-kappa confirm.** This is the closest target: keep the predeclared positive sign, but improve the public residual-kappa join, pair weighting, sky shuffle count, and jackknife audit until the one-sided p-value is reproducibly `<=0.05`. Target confirm: `R10-T17`.
2. **Build exact P33 DESI/SDSS density-split BAO alpha measurement.** Load exact clustering and random catalogues, compute random-normalized density splits, fit alpha high/low with covariance/bootstrap, and require sky/density-label/redshift nulls. Target confirm: `R10-T07`.
3. **Close P30 beyond the SDSS route.** Preserve `R10-T04` as scoped SDSS confirm-like, then add independent Planck and Euclid same-mask statistics, curl/random controls, and residualization so the global P30 claim can promote cleanly. Target confirms: `R10-T05`, `R10-T06`, and global P30.
4. **Upgrade P35 from BAO coverage to real harmonic/P(k) evidence.** Ingest public DESI/BOSS/eBOSS LSS P(k) or correlation-function tables, subtract broadband shape, compute the predeclared comb statistic, and run phase-randomized/redshift-bin jackknife nulls. Target confirm: `R10-T08`.
5. **Recover P3 endpoint semantics.** Add source-specific parsers for explicit endpoint/node-pair columns, avoid giant non-endpoint downloads, then run endpoint-redshift and orientation-shuffle nulls. Target confirm: `R10-T09`, with downstream support for `CL4`.
6. **Make P32 strain multi-event and null-complete.** Add at least one more GWOSC strain event beyond GW150914, rerun PSD whitening, detector split, off-source injections/time slides, and leave-one-event-out stability. Target confirm: `R10-T19`.
7. **Make P39/P1 publication-grade.** Ingest Pantheon+ full covariance/systematics, combine with BAO using nuisance-parameter fits, and require `delta_chi2>=9` after model penalty. Target confirms: `R10-T01`, `R10-T02`, `R10-T03`, and stronger `CL5`.

## Confirm Priority

Highest near-term path is `R10-T17`, because it is a strict-threshold near-miss rather than a missing-data problem. Next best are `R10-T07` and `R10-T08`, where the current compressed BAO coverage is useful but the exact LSS/P(k) inputs decide whether confirmation is possible. `R10-T09`, `R10-T19`, and `R10-T01`-`R10-T03` are heavier lifts because they need new semantic input recovery or full likelihood machinery rather than only gate refinement.
