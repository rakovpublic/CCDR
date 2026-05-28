# Round 10 Latest Run Report (v70)

Generated from `outputs/round10_summary.json` updated `2026-05-17T21:37:19Z` (`2026-05-18T00:37:19+03:00` local Kyiv time). Current run id: `20260517T195329Z_67d42547`.

## Executive Summary

- Suite completeness: `51/51` tests, `run_complete_v70=True`.
- Runner version: `v70`.
- Dashboard status: `dashboard_positive_current_only_v70`.
- Claim-grade non-SM confirm-like results: `11`.
- SM constant consistency confirms: `5`.
- Coverage-only confirm: `1`.
- Active blocked/data-limited tests in the dashboard: `3`.
- Active v70 confirm blockers outside the dashboard blocked bucket: P39/P1 full covariance/model-penalty gate remains unmet for `R10-T01`, `R10-T02`, and `R10-T03`.

The v70 run is materially stronger than v69. The former `R10-T14` memory failure is repaired, `R10-T04` now has a scoped SDSS-route P30 confirm, and `R10-T17` now has a signed PTA residual-kappa confirm-like result. The strict no-overclaim gates still correctly prevent promotion for P33, P3, P32 strain, and full P39/P1 publication-grade likelihood.

## Current Confirm-Like Results

| Test | Prediction | Status | Note |
|---|---|---|---|
| `R10-T04` | `P30` | `density_kappa_sdss_route_confirm_like_v70` | SDSS route only; global P30 still needs independent route closure. |
| `R10-T10` | `P38` | `void_morphology_artifact_backed_confirm_like_v54` | Void morphology confirm-like with artifact-backed null hardening. |
| `R10-T12` | `P36/local a0` | `robust_confirm_like` | SPARC local RAR/a0 bootstrap and leave-one-galaxy-out stable. |
| `R10-T13` | `P36/high-z a0` | `highz_a0_clean_claim_confirm_like_v70` | Memory-safe v70 clean-claim high-z gate. |
| `R10-T14` | `P36` | `highz_a0_clean_claim_confirm_like_v70` | Recovered from v69 `MemoryError`; uses source-hashed clean rows. |
| `R10-T17` | `P8c/CL2` | `pta_weighted_kappa_residual_confirm_like_v70` | Signed residual-kappa gate passes. |
| `R10-T21` | `P40` | `p40_bb_likelihood_confirm_like_v67` | BK18 bandpower/covariance path loaded. |
| `R10-T22` | `P40` | `p40_bb_likelihood_confirm_like_v67` | Planck/BK18 cross-bound path loaded. |
| `R10-T31` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | LHCb supplementary q2/Wilson likelihood proxy passes. |
| `R10-T32` | `P41` | `p41_q2_wilson_likelihood_confirm_like_v68` | Control supplementary q2/Wilson path passes. |
| `R10-CL06` | `CL6` | `cl6_p41_p40_bridge_confirm_like_v69` | Dependent bridge from confirmed P40 and P41 anchors. |

## Evidence Snapshot

| Area | Evidence | Interpretation |
|---|---|---|
| P30 SDSS route | `p30_sdss_route_confirm_v70_AUTO_PUBLIC.json` status `sdss_route_confirm_gate_passed` | Confirm-like only for the SDSS route; global P30 still needs Planck/Euclid route closure and residualization. |
| P36 high-z | `1904` clean-claim rows, median clean acceleration `4.6307e-10 m/s^2` | v70 avoids the old full-source memory path and recovers both high-z P36 tests. |
| PTA/CL2 | `31` residual-kappa pairs, one-sided signed p `0.0423177083`, weighted statistic `1894.9629`, top-weight stable | This is the most important v70 scientific recovery. |
| P40 | `9` bandpower rows, `594` covariance rows | Public BK18 products are loaded; both P40 tests remain confirm-like. |
| P41 | `30` signal rows, CP null passed, strong proxy fit | P41 remains one of the strongest current non-SM evidence paths. |
| CL6 bridge | P40 and P41 both confirmed upstream | CL6 is valid as a dependent bridge, not an independent new measurement. |
| P39/P1 | Missing `delta_chi2_lcdm_minus_best_ge_9`, Pantheon full covariance, systematics splits, model-penalty support | Positive-compatible only; do not claim publication-grade P39/P1 confirmation yet. |
| P33 | Exact DESI LSS clustering catalogues and randoms absent | Needs real density-split BAO alpha measurement before confirmation. |
| P3 | No explicit endpoint/node-pair rows found | Data-limited until endpoint columns and nulls are recovered. |
| P32 strain | Multi-event public strain inputs absent; strict strain/null requirements unmet | Metadata is ready, but strain-level confirmation is still blocked. |

## All Tests

| # | Test | Prediction | Status | Bucket |
|---:|---|---|---|---|
| 1 | `R10-T01` | `P39` | `positive_compatible` | positive/ready; active P39 v70 gate unmet |
| 2 | `R10-T02` | `P1/P39` | `positive_compatible` | positive/ready; active P39 v70 gate unmet |
| 3 | `R10-T03` | `P39` | `positive_compatible` | positive/ready; active P39 v70 gate unmet |
| 4 | `R10-T04` | `P30` | `density_kappa_sdss_route_confirm_like_v70` | non-SM confirm-like |
| 5 | `R10-T05` | `P30` | `density_kappa_positive_ready` | positive/ready; possible global P30 support |
| 6 | `R10-T06` | `P30` | `euclid_mer_catalogue_sample_positive_ready` | positive/ready; possible global P30 support |
| 7 | `R10-T07` | `P33` | `p33_density_bao_alpha_measurement_required_v70` | blocked/data-limited |
| 8 | `R10-T08` | `P35` | `harmonic_proxy_positive_ready` | positive/ready |
| 9 | `R10-T09` | `P3` | `endpoint_data_limited` | blocked/data-limited |
| 10 | `R10-T10` | `P38` | `void_morphology_artifact_backed_confirm_like_v54` | non-SM confirm-like |
| 11 | `R10-T11` | `CL4` | `partial_positive_bridge` | bridge positive, not confirm |
| 12 | `R10-T12` | `P36/local a0` | `robust_confirm_like` | non-SM confirm-like |
| 13 | `R10-T13` | `P36/high-z a0` | `highz_a0_clean_claim_confirm_like_v70` | non-SM confirm-like |
| 14 | `R10-T14` | `P36` | `highz_a0_clean_claim_confirm_like_v70` | non-SM confirm-like |
| 15 | `R10-T15` | `P29` | `consistent_bound_only` | bound consistency only |
| 16 | `R10-T16` | `P8/P8c` | `pta_density_cross_positive_ready` | positive/ready; supports CL2 |
| 17 | `R10-T17` | `P8c/CL2` | `pta_weighted_kappa_residual_confirm_like_v70` | non-SM confirm-like |
| 18 | `R10-T18` | `P32` | `ringdown_metadata_positive_ready` | metadata ready only |
| 19 | `R10-T19` | `P32` | `ringdown_strain_analysis_required_v70` | blocked/data-limited |
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
| 47 | `R10-DCN01` | `DCN_k` | `dcn_digitized_window_positive_ready` | positive/ready |
| 48 | `R10-DCN02` | `DCN_k` | `dcn_allowed_window_quantified_positive` | positive/ready |
| 49 | `R10-CL05` | `CL5` | `partial_positive_bridge` | bridge positive, not confirm |
| 50 | `R10-CL06` | `CL6` | `cl6_p41_p40_bridge_confirm_like_v69` | non-SM confirm-like |
| 51 | `R10-DASH` | `P/CL dashboard` | `dashboard_positive_current_only_v70` | dashboard/control |

## Active v70 Gate Blockers

| Test(s) | Gate | Missing / Required |
|---|---|---|
| `R10-T01`, `R10-T02`, `R10-T03` | `p39_full_covariance_chain_gate_v70` | `delta_chi2_lcdm_minus_best_ge_9`, `pantheon_full_covariance_used`, `systematics_splits_done`, `model_penalty_supports_new_model` |
| `R10-T07` | `p33_desi_lss_random_manifest_gate_v70` | exact DESI LSS clustering catalogues, exact DESI random catalogues, random-normalized density-split alpha, sky shuffle p <= 0.05, density-label shuffle p <= 0.05, redshift jackknife stability |
| `R10-T09` | `p3_endpoint_recovery_gate_v70` | explicit endpoint or node-pair rows, endpoint redshift null, orientation shuffle null |
| `R10-T19` | `p32_multi_event_strain_gate_v70` | at least two public strain events, detector delta chi2 >= 4, time-slide p <= 0.05, leave-one-event-out stability |

Historical failed gates for P30, PTA/CL2, P40, and P41 remain in the dashboard audit trail, but the latest active gates for those paths now pass or are superseded by later confirm-like artifacts.

## Suggested Confirm-Focused Improvements

1. **P33 exact DESI density-split BAO alpha pipeline.** Build a targeted LSScats clustering/random cache and require exact RA/Dec/z rows, random-normalized density split, covariance/bootstrap alpha fit, sky shuffle, density-label shuffle, and redshift jackknife. Target confirm: `R10-T07`.
2. **P30 global-route closure.** Keep the v70 SDSS route confirm, but add an independent Planck/Euclid route gate that residualizes redshift-density structure and applies same-mask/curl controls. Target confirms: promote `R10-T05`/`R10-T06` from positive-ready into a global P30 cross-route confirm without weakening the current claim scope.
3. **P35 full harmonic/P(k) parser.** Upgrade the harmonic-comb proxy into a real BAO phase or P(k) measurement using public DESI/BAO products with redshift-bin jackknife and phase-randomized nulls. Target confirm: `R10-T08`.
4. **P3 endpoint recovery with strict semantic columns.** Add source-specific parsers for explicit node1/node2 or endpoint RA/Dec/z columns, plus endpoint-redshift and orientation-shuffle nulls. Target confirm: `R10-T09`, and downstream support for `CL4`.
5. **P32 multi-event strain likelihood.** Cache at least two GWOSC public strain events, run detector-calibrated PSD whitening, off-source injections/time slides, detector split, and leave-one-event-out stability. Target confirm: `R10-T19`.
6. **P39/P1 full covariance plus model-penalty chain.** Ingest Pantheon+SH0ES covariance/systematics, fit combined SN+BAO with nuisance parameters, and require delta chi2 >= 9 after AIC/BIC or similar penalty. Target confirms: `R10-T01`, `R10-T02`, `R10-T03`, and downstream `CL5`.
7. **DCN/AQN exact curve extraction.** Replace digitized-window readiness with source-table or reproducible curve extraction for the allowed-window calculation, then add exclusion/allowed-region overlap tests and null/control checks. Target confirms: `R10-DCN01` and `R10-DCN02`.

## Confirm Priority

Highest near-term candidates are P30 global-route closure, P35 full harmonic measurement, P3 endpoint recovery, and P33 exact LSS/random ingestion. P32 and P39/P1 are important but heavier: the current v70 measurements are correctly cautious, and they should only convert to confirms if the stricter likelihood/null gates genuinely pass. DCN/AQN is a good medium-effort route because it is already quantified-positive and mainly needs reproducible exact extraction plus stronger controls.
