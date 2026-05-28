# Round 10 Latest Run Report (v67)

Generated from `outputs\round10_summary.json` and `outputs\test51_round10_joint_dashboard.json` on 2026-05-16 21:42:20 +03:00.

## Executive summary

- Runner: `v67`; current run id: `20260516T081213Z_b27bd6b7`; run complete v67: `True`.
- Tests run: 51/51; missing tests: 0.
- Dashboard counts: non-SM confirm-like 6, SM constant consistency 5, coverage-only 1, blocked/gate-failed 7.
- Latest new confirm recovery remains P40: BK18 bandpowers and covariance loaded for R10-T21/T22.
- Best near-confirm candidate is P41: q2 fit rows, CP/sign gates, and jackknife are present; delta chi2 is 7.475, below the >=9 threshold.

## Bucket counts

| Bucket | Count |
|---|---:|
| Positive / not confirm | 32 |
| Confirm blocker | 7 |
| Non-SM confirm-like | 6 |
| SM constant consistency | 5 |
| Coverage only | 1 |

## Confirm-like tests

| Test | Prediction | Status | Notes |
|---|---|---|---|
| R10-T10 | P38 | `void_morphology_artifact_backed_confirm_like_v54` | Already confirm-like; harden by adding independent void catalogue rerun and source-file hash freeze. |
| R10-T12 | P36/local a0 | `robust_confirm_like` | Already confirm-like; harden with external SPARC mirror/hash freeze and alternate acceleration estimator. |
| R10-T13 | P36/high-z a0 | `highz_a0_clean_claim_confirm_like_v66` | Already confirm-like on clean high-z claim subset; next hardening is independent source-2/3 parser and stricter tiny-row quarantine audit. |
| R10-T14 | P36 | `highz_a0_clean_claim_confirm_like_v66` | Already confirm-like on clean high-z claim subset; next hardening is independent source-2/3 parser and stricter tiny-row quarantine audit. |
| R10-T21 | P40 | `p40_bb_likelihood_confirm_like_v67` | Already v67 confirm-like via BK18 bandpower+covariance parsing; harden with Planck cross-check and foreground-control sensitivity sweep. |
| R10-T22 | P40 | `p40_bb_likelihood_confirm_like_v67` | Already v67 confirm-like via BK18 bandpower+covariance parsing; harden with Planck cross-check and foreground-control sensitivity sweep. |

## v67 strict gate evidence

- P30 R10-T04: curl/science ratio 1.747; unrejected v67 patches 0; residualization required: True.
- P33 R10-T07: exact LSS status `compressed_bao_products_loaded`; compressed BAO mean/cov rows indexed: 40; still missing exact RA/DEC/Z random catalogues.
- PTA/CL2 R10-T17: join-like rows 0; residual-kappa pairs 0; weighted statistic unavailable.
- P32 R10-T19: cached detectors `H1, L1`; likelihood rows 1; delta chi2 2.2E-05; null gates still false.
- P40 R10-T21/T22: bandpower rows 9; covariance rows 594; template amplitude 0.972408; sigma 1.434283.
- P41 R10-T31/T32: q2 fit rows 9; CP null True; sign basis True; jackknife stable True; delta chi2 7.475 < 9.

## Highest-payoff confirm improvements

1. P41 fitter upgrade: improve structured q2/value/error extraction from supplementary material and use observable-aware SM/Wilson predictions. Current delta chi2 is 7.475, so this is the closest confirm target.
2. P33 exact DESI LSS ingestion: place/load public clustering and random FITS with RA/DEC/Z/weights. The compressed BAO cache is now indexed, but confirm needs density-split alpha with randoms and shuffles.
3. P32 strain likelihood rebuild: run H1 and L1 detector-specific fits from cached GWOSC files, then injection-null and leave-one-event-out controls. Inputs are present; likelihood rows are not yet two-detector.
4. PTA residual-kappa join: build a real residual/TOA-weighted pulsar table joined to kappa samples. Current v67 scan found zero residual-kappa pairs.
5. P30 mask/control resolution: accept/extract official ACT mask or equivalence appendix, residualize route disagreement, and reduce curl leakage. Current stricter rule leaves zero clean patches.
6. P39/P1 cosmology likelihood: full covariance plus systematics splits and model-penalty gate; current delta chi2 is about 1.21, below confirm threshold.

## All tests

| # | Test | Prediction | Status | Bucket | Confirm-focused next action |
|---:|---|---|---|---|---|
| 1 | R10-T01 | P39 | `positive_compatible` | Positive / not confirm | Upgrade P39/P1 from diagnostic to confirm: full SN+BAO covariance, systematics splits, model penalty, and delta chi2 >= 9. |
| 2 | R10-T02 | P1/P39 | `positive_compatible` | Positive / not confirm | Upgrade P39/P1 from diagnostic to confirm: full SN+BAO covariance, systematics splits, model penalty, and delta chi2 >= 9. |
| 3 | R10-T03 | P39 | `positive_compatible` | Positive / not confirm | Upgrade P39/P1 from diagnostic to confirm: full SN+BAO covariance, systematics splits, model penalty, and delta chi2 >= 9. |
| 4 | R10-T04 | P30 | `density_kappa_same_mask_route_blocked_v67` | Confirm blocker | Resolve P30 controls: official/accepted ACT mask, residualize redshift-density route disagreement, reduce curl/science <= 0.5, keep >=2 clean patches. |
| 5 | R10-T05 | P30 | `density_kappa_positive_ready` | Positive / not confirm | Merge into P30 confirm route only after same-mask/curl/residualization gates from R10-T04 pass. |
| 6 | R10-T06 | P30 | `euclid_mer_catalogue_sample_positive_ready` | Positive / not confirm | Merge into P30 confirm route only after same-mask/curl/residualization gates from R10-T04 pass. |
| 7 | R10-T07 | P33 | `p33_density_bao_alpha_measurement_required_v67` | Confirm blocker | Load exact DESI LSS clustering/random RA/DEC/Z catalogues; compressed BAO is indexed but insufficient for density-split alpha confirm. |
| 8 | R10-T08 | P35 | `harmonic_proxy_positive_ready` | Positive / not confirm | Turn harmonic proxy into fit: covariance-aware BAO/Cl likelihood, predeclared comb statistic, phase-scramble/null controls. |
| 9 | R10-T09 | P3 | `endpoint_data_limited` | Confirm blocker | Find explicit filament endpoint catalogue semantics plus endpoint/redshift nulls; generic numeric rows are not enough. |
| 10 | R10-T10 | P38 | `void_morphology_artifact_backed_confirm_like_v54` | Non-SM confirm-like | Already confirm-like; harden by adding independent void catalogue rerun and source-file hash freeze. |
| 11 | R10-T11 | CL4 | `partial_positive_bridge` | Positive / not confirm | Promote bridge only after both P3 endpoint and P38 void gates are confirm-grade in the same run. |
| 12 | R10-T12 | P36/local a0 | `robust_confirm_like` | Non-SM confirm-like | Already confirm-like; harden with external SPARC mirror/hash freeze and alternate acceleration estimator. |
| 13 | R10-T13 | P36/high-z a0 | `highz_a0_clean_claim_confirm_like_v66` | Non-SM confirm-like | Already confirm-like on clean high-z claim subset; next hardening is independent source-2/3 parser and stricter tiny-row quarantine audit. |
| 14 | R10-T14 | P36 | `highz_a0_clean_claim_confirm_like_v66` | Non-SM confirm-like | Already confirm-like on clean high-z claim subset; next hardening is independent source-2/3 parser and stricter tiny-row quarantine audit. |
| 15 | R10-T15 | P29 | `consistent_bound_only` | Positive / not confirm | Bound only: add full growth f_sigma8 likelihood, covariance, live/frozen split, and delta chi2/model-penalty gate. |
| 16 | R10-T16 | P8/P8c | `pta_density_cross_positive_ready` | Positive / not confirm | Positive-ready PTA sky route; needs residual/TOA-weighted statistic, sky shuffle p <= 0.05, and predeclared sign. |
| 17 | R10-T17 | P8c/CL2 | `pta_weighted_kappa_residual_required_v67` | Confirm blocker | Build actual pulsar residual/TOA-kappa join table; current v67 found zero residual-kappa pairs. |
| 18 | R10-T18 | P32 | `ringdown_metadata_positive_ready` | Positive / not confirm | Metadata only: connect GWOSC event metadata to strain-level likelihood/null gates. |
| 19 | R10-T19 | P32 | `ringdown_strain_analysis_required_v67` | Confirm blocker | Run detector-specific H1+L1 strain likelihood fits, injection nulls, leave-one-event-out, and delta chi2 >= 4. |
| 20 | R10-T20 | No-FTL | `consistent_bound_only` | Positive / not confirm | Luminal bound only; confirm would require a new non-luminal predicted effect with event-level likelihood, not just consistency. |
| 21 | R10-T21 | P40 | `p40_bb_likelihood_confirm_like_v67` | Non-SM confirm-like | Already v67 confirm-like via BK18 bandpower+covariance parsing; harden with Planck cross-check and foreground-control sensitivity sweep. |
| 22 | R10-T22 | P40 | `p40_bb_likelihood_confirm_like_v67` | Non-SM confirm-like | Already v67 confirm-like via BK18 bandpower+covariance parsing; harden with Planck cross-check and foreground-control sensitivity sweep. |
| 23 | R10-T23 | P28 | `consistent_bound_only` | Positive / not confirm | Bound only: add full FIRAS/Planck y likelihood, covariance, foreground controls, and predeclared distortion template. |
| 24 | R10-T24 | P28 | `consistent_bound_only` | Positive / not confirm | Bound only: add full FIRAS/Planck y likelihood, covariance, foreground controls, and predeclared distortion template. |
| 25 | R10-T25 | P10/P25/P31 | `mass_window_quantified_coverage_positive_ready` | Positive / not confirm | Direct-detection/phase space: coverage is not detection; needs event-level likelihood, predicted spectrum/window, and detector-specific nulls. |
| 26 | R10-T26 | P10/P25/P31 | `mass_window_coverage_confirmed` | Coverage only | Direct-detection/phase space: coverage is not detection; needs event-level likelihood, predicted spectrum/window, and detector-specific nulls. |
| 27 | R10-T27 | P10/P25/P31 | `mass_window_quantified_positive_ready` | Positive / not confirm | Direct-detection/phase space: coverage is not detection; needs event-level likelihood, predicted spectrum/window, and detector-specific nulls. |
| 28 | R10-T28 | P27 | `sensitivity_positive_ready` | Positive / not confirm | Direct-detection/phase space: coverage is not detection; needs event-level likelihood, predicted spectrum/window, and detector-specific nulls. |
| 29 | R10-T29 | P37 | `event_level_ready_not_detection_confirmed` | Positive / not confirm | Direct-detection/phase space: coverage is not detection; needs event-level likelihood, predicted spectrum/window, and detector-specific nulls. |
| 30 | R10-T30 | P5 | `kss_proxy_bound_positive_schema_backed` | Positive / not confirm | KSS proxy bound: needs HEPData numeric observable+error rows, full covariance or bootstrap, and model-vs-SM likelihood. |
| 31 | R10-T31 | P41 | `p41_q2_wilson_likelihood_required_v67` | Confirm blocker | Near confirm: P41 v67 has q2 fit, CP/sign, stable jackknife; only delta chi2 >= 9 is missing (current 7.475). Improve row extraction/fitter. |
| 32 | R10-T32 | P41 | `p41_q2_wilson_likelihood_required_v67` | Confirm blocker | Near confirm: P41 v67 has q2 fit, CP/sign, stable jackknife; only delta chi2 >= 9 is missing (current 7.475). Improve row extraction/fitter. |
| 33 | R10-T33 | P9b/P9e/P9f | `hepdata_schema_positive_ready` | Positive / not confirm | HEPData schema positive: needs units-verified numeric likelihood rows and model-penalty gate. |
| 34 | R10-SMD01 | SM-D1 | `smd_constant_consistency_confirm_like` | SM constant consistency | SM constant consistency only; confirm-grade CCDR derivation needs preregistered no-target-fit derivation with uncertainty and residual sigma. |
| 35 | R10-SMD02 | SM-D2 | `smd_constant_consistency_confirm_like` | SM constant consistency | SM constant consistency only; confirm-grade CCDR derivation needs preregistered no-target-fit derivation with uncertainty and residual sigma. |
| 36 | R10-SMD03 | SM-D3 | `smd_constant_consistency_confirm_like` | SM constant consistency | SM constant consistency only; confirm-grade CCDR derivation needs preregistered no-target-fit derivation with uncertainty and residual sigma. |
| 37 | R10-SMD04 | SM-D4 | `smd_constant_consistency_confirm_like` | SM constant consistency | SM constant consistency only; confirm-grade CCDR derivation needs preregistered no-target-fit derivation with uncertainty and residual sigma. |
| 38 | R10-SMD05 | SM-D5 | `smd_constant_consistency_confirm_like` | SM constant consistency | SM constant consistency only; confirm-grade CCDR derivation needs preregistered no-target-fit derivation with uncertainty and residual sigma. |
| 39 | R10-SMD06 | SM-D6 | `consistent_constant_check` | Positive / not confirm | Consistency check only; add structured preregistered derivation package and anti-postdiction marker. |
| 40 | R10-SMD07 | SM-D7 | `consistent_constant_check` | Positive / not confirm | Consistency check only; add structured preregistered derivation package and anti-postdiction marker. |
| 41 | R10-SMD08 | SM-D8 | `consistent_constant_check` | Positive / not confirm | Consistency check only; add structured preregistered derivation package and anti-postdiction marker. |
| 42 | R10-SMD09 | SM-D9 | `structural_consistency_positive` | Positive / not confirm | Consistency check only; add structured preregistered derivation package and anti-postdiction marker. |
| 43 | R10-SMD10 | SM-D10 | `consistent_constant_check` | Positive / not confirm | Consistency check only; add structured preregistered derivation package and anti-postdiction marker. |
| 44 | R10-DC01 | Dark-Cone | `branch_survival_positive` | Positive / not confirm | Branch-survival/bridge only; needs predeclared observable, public data likelihood, nulls, and falsification threshold. |
| 45 | R10-DC02 | Dark-Cone | `partial_positive_bridge` | Positive / not confirm | Branch-survival/bridge only; needs predeclared observable, public data likelihood, nulls, and falsification threshold. |
| 46 | R10-DC03 | Dark-Cone | `branch_survival_positive` | Positive / not confirm | Branch-survival/bridge only; needs predeclared observable, public data likelihood, nulls, and falsification threshold. |
| 47 | R10-DCN01 | DCN_k | `dcn_allowed_window_quantified_positive` | Positive / not confirm | Allowed-window positive only; needs event-rate likelihood and exposure/systematics nulls. |
| 48 | R10-DCN02 | DCN_k | `dcn_allowed_window_quantified_positive` | Positive / not confirm | Allowed-window positive only; needs event-rate likelihood and exposure/systematics nulls. |
| 49 | R10-CL05 | CL5 | `partial_positive_bridge` | Positive / not confirm | Joint bridge: promote only after both component predictions are confirm-like under one joint covariance/gate. |
| 50 | R10-CL06 | CL6 | `partial_positive_bridge` | Positive / not confirm | Joint bridge: promote only after both component predictions are confirm-like under one joint covariance/gate. |
| 51 | R10-DASH | P/CL dashboard | `dashboard_positive_current_only_v67` | Positive / not confirm | Dashboard only; keep strict bucket semantics and add regression checks for confirm counts. |

## Raw status counts

| Status | Count |
|---|---:|
| `smd_constant_consistency_confirm_like` | 5 |
| `consistent_constant_check` | 4 |
| `partial_positive_bridge` | 4 |
| `consistent_bound_only` | 4 |
| `positive_compatible` | 3 |
| `p41_q2_wilson_likelihood_required_v67` | 2 |
| `highz_a0_clean_claim_confirm_like_v66` | 2 |
| `p40_bb_likelihood_confirm_like_v67` | 2 |
| `dcn_allowed_window_quantified_positive` | 2 |
| `branch_survival_positive` | 2 |
| `sensitivity_positive_ready` | 1 |
| `mass_window_quantified_positive_ready` | 1 |
| `mass_window_coverage_confirmed` | 1 |
| `event_level_ready_not_detection_confirmed` | 1 |
| `hepdata_schema_positive_ready` | 1 |
| `structural_consistency_positive` | 1 |
| `kss_proxy_bound_positive_schema_backed` | 1 |
| `dashboard_positive_current_only_v67` | 1 |
| `p33_density_bao_alpha_measurement_required_v67` | 1 |
| `harmonic_proxy_positive_ready` | 1 |
| `endpoint_data_limited` | 1 |
| `density_kappa_same_mask_route_blocked_v67` | 1 |
| `density_kappa_positive_ready` | 1 |
| `euclid_mer_catalogue_sample_positive_ready` | 1 |
| `void_morphology_artifact_backed_confirm_like_v54` | 1 |
| `ringdown_metadata_positive_ready` | 1 |
| `ringdown_strain_analysis_required_v67` | 1 |
| `mass_window_quantified_coverage_positive_ready` | 1 |
| `robust_confirm_like` | 1 |
| `pta_density_cross_positive_ready` | 1 |
| `pta_weighted_kappa_residual_required_v67` | 1 |
