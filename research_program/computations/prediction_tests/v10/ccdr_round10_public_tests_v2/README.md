
# CCDR Round-10 Public Tests v2

Generated: 2026-05-03T20:39:31Z

This bundle contains 51 Python tests for CCDR v7.6 / Synthesis v3.6 public-data auditing.

## Goals

- Download public data automatically where endpoints are available.
- Never terminate without JSON: every test goes through `safe_json_main`.
- Use `data_limited` or `readiness_only` honestly when public data are too large, unavailable, or require a specialised parser.
- Keep SM-D tests separate from the main P1-P41 prediction list.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_all.py
```

Large products are disabled by default:

```bash
python run_all.py --allow-large --max-mb 5000
```

Filter by prediction or filename:

```bash
python run_all.py --only P41
python run_all.py --only SMD
python run_all.py --only pantheon
```

Outputs are written to:

```text
outputs/*.json
outputs/round10_summary.json
```

## Status semantics

- `partial`: a real public-data parser ran and produced a preliminary statistic.
- `diagnostic`: a real parser ran, but the statistic is intentionally not a final falsification statistic.
- `readiness_only`: public source exists/reachable; event-level or map-level parser is not yet decisive.
- `data_limited`: public endpoint unavailable, too large without `--allow-large`, or layout changed.
- `broken`: unexpected bug; should be fixed.

## Notes

This is a Round-10 starting bundle. The highest-priority upgrades are:
1. Replace inventory tests with exact row-schema parsers for DESI DR2 BAO and HEPData P41 tables.
2. Add ACT/Planck/Euclid map samplers behind `--allow-large`.
3. Add a true SPARC RAR/a0 parser using baryonic columns and galaxy metadata.
4. Add VAST/filament catalogue parsers for P3/P38 instead of metadata checks.
5. Add a dashboard that ingests all JSON outputs and classifies confirmed/plausible/null/falsified.

## Round-10 v22 patch

This bundle includes `ROUND10_V22_PATCH_NOTES.md`.  v22 adds confirm-oriented hardening gates for P36 high-z a0, P30 density-kappa, P38 void morphology, P8/P8c NANOGrav, P33 density-BAO, P39 likelihood evidence, P32 ringdown, P41 B-anomaly tables, direct-detection unit coverage, and HEPData schema validation.

`run_all.py` now supports `--script-timeout` to prevent a single public endpoint from hanging the whole suite.


## Round-10 v23 note

The v23 patch overrides the v22 runner names in `ccdr_r10_common.py`, so existing commands still work. It adds stricter confirm-hardening gates for P36, P38, P30, P8, P33, P39, P32, P41, direct detection, and HEPData. See `ROUND10_V23_PATCH_NOTES.md`.


## v24 confirm-first hardening

This bundle adds v24 gates for stricter confirmation claims: P36 object-level V^2/R, P30 mask/random/estimator harmonization, P33 measured-density BAO, P41 q² likelihood audit, P32 strain plan, HEPData endpoint schema, direct-detection coverage policy, and SM-D derivation gates. Run `python run_all.py --quick --script-timeout 60` for a smoke test; run targeted tests without `--quick` and with `--allow-large` for public-data scans.


## v25 confirm-first hardening

This bundle adds v25 instrumentation for P30 mask/random/variant route separation, P36 high-z object-level V²/R gates, P33 measured density-BAO alpha result schema, P41 q²-likelihood major-claim gate, P32 strain-level manifest, HEPData endpoint-only schema policy, direct-detection coverage-only policy, and SM-D derivation-mode gates.

Run smoke test:

```powershell
python run_all.py --quick --script-timeout 60 --timeout 5 --max-mb 1
```

Best confirm-target runs:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 7200
python run_all.py --only P36 --allow-large --max-mb 5000 --script-timeout 1200
python run_all.py --only P33 --allow-large --max-mb 5000 --script-timeout 1200
python run_all.py --only P41 --allow-large --max-mb 5000 --script-timeout 1200
```

## v26 confirm-execution hardening

This bundle includes v26 gates for the ten latest confirmation blockers:
P30 ACT mask/random/variant/route separation, P36 high-z strict object V²/R parsing and image-FITS rejection, P33 measured density-split BAO alpha, P41 CP-control + Wilson/SM major-claim gate, P32 strain-level execution gate, and SM-D derivation gates.

Recommended target runs:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 7200
python run_all.py --only P36 --allow-large --max-mb 5000 --script-timeout 1200
python run_all.py --only P33 --allow-large --max-mb 5000 --script-timeout 1200
python run_all.py --only P41 --allow-large --max-mb 5000 --script-timeout 1200
python run_all.py --only P32 --allow-large --max-mb 5000 --script-timeout 1200
```

Smoke checks:

```powershell
python tests/test04_p30_act_dr6_lensing_inventory.py --quick --max-mb 1 --timeout 5
python tests/test13_p36_kmos3d_inventory.py --quick --max-mb 1 --timeout 5
python tests/test31_p41_lhcb_bsll_inventory.py --quick --max-mb 1 --timeout 5
```

## v27 confirm-execution patch

This bundle adds v27 operational artifacts for the ten latest confirmation blockers:
P30 empirical mask/random catalogues/same-split variants/route-specific claims, P36 high-z object acceleration table, P33 measured density-split BAO alpha schema, P41 q²/CP rows and Wilson/SM gate, P32 strain execution manifest, and SM-D derivation schemas.

Recommended runs:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 7200
python run_all.py --only P36 --allow-large --max-mb 5000 --script-timeout 1200
python run_all.py --only P33 --allow-large --max-mb 5000 --script-timeout 1200
python run_all.py --only P41 --allow-large --max-mb 5000 --script-timeout 1200
python run_all.py --only P32 --allow-large --max-mb 5000 --script-timeout 2400
```

Smoke test:

```powershell
python run_all.py --quick --script-timeout 90 --timeout 5 --max-mb 1
```


## v28 confirm-execution patch

Adds measured-artifact consumers and stricter confirm gates for P30, P36 high-z, P33, P41, P32, and SM-D. Run targeted heavy tests with `--allow-large` to allow P30 random-normalized kappa sampling.


## v29 confirm-execution hardening

v29 adds same-split ACT variant evaluation for the P30-SDSS random-normalized route, a curl-control gate, route-specific P30-SDSS promotion policy, source-specific P36 high-z object parsers, P41 Wilson/SM likelihood consumers, P33 measured-alpha consumers, and P32 strain-fit product gates. Run P30 with `--allow-large` to recompute the same-split variant matrix.

## Round-10 v30 confirm-target patch

v30 implements the ten follow-up improvements from the v29 analysis:

- quantitative P30 curl diagnostics (`p30_curl_diagnostics_v30`), including curl/science ratio and non-significance rule;
- thresholded P30 curl pass rule (`abs(curl_delta) < 0.60 * median_abs_science_delta` plus non-extreme curl p-value);
- explicit `P30-SDSS_route_near_confirm_candidate` bucket separate from global P30;
- P30-SDSS bootstrap CI/effect-size artifact (`outputs/p30_sdss_bootstrap_ci_v30.json`);
- Euclid quarantine/repair policy until photo-z/depth/quality cuts and field randoms are implemented;
- stricter P36 source-specific object acceleration parser and `outputs/p36_highz_object_acceleration_rows_v30.json`;
- P41 q2/value/error likelihood consumer and simple Wilson/SM proxy fitter (`outputs/p41_wilson_sm_likelihood_v30.json`);
- P33 measured BAO alpha consumer (`outputs/p33_density_split_alpha_result_v30.json`);
- P32 minimal strain-run manifest and strict strain-result gate (`outputs/p32_minimal_strain_run_manifest_v30.json`);
- dashboard v30 instrumentation with near-confirm and blocked-gate summaries.

Recommended next confirm run:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Inspect:

```text
outputs/p30_curl_diagnostics_v30.json
outputs/p30_sdss_bootstrap_ci_v30.json
outputs/test04_p30_act_dr6_lensing_inventory.json
```


## v31 confirm-target patch

Run the most important target:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Inspect:

```text
outputs/p30_curl_diagnostics_v31.json
outputs/p30_sdss_bootstrap_ci_v30.json
outputs/test04_p30_act_dr6_lensing_inventory.json
```

v31 fixes the curl-diagnostic schema bug from v30 by reading `variant_results` from the v29 same-split artifact. It keeps P30-SDSS route confirmation separate from global P30.


## v32 confirm-target patch

Adds P30 curl-remediation diagnostics, curl permutation/rotation controls, science/curl ratio uncertainty, curl-subtracted diagnostics, frequency/systematic family splits, and keeps P30-SDSS route-specific promotion separate from global P30. Also adds stricter P36/P41/P33/P32 product contracts. See `ROUND10_V32_PATCH_NOTES.md`.


## v33 confirm-target patch

This patch focuses on the P30-SDSS curl blocker. New outputs include:

- `outputs/p30_curl_clean_route_v33.json`
- `p30_sdss_route_confirm_gate_v33` in the P30 test JSON
- `dashboard_v33` in the dashboard JSON

Recommended P30 run:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Inspect:

```text
outputs/p30_curl_clean_route_v33.json
outputs/p30_curl_remediation_v32.json
outputs/p30_curl_diagnostics_v31.json
outputs/test04_p30_act_dr6_lensing_inventory.json
```

P30-SDSS remains route-specific. Global P30 still requires a second independent repaired route.

## v34 confirm-target patch

v34 focuses on the nearest new confirmation route: **P30-SDSS-core**.
It adds a heavy-mode curl-projected residual test for the baseline/f150/tonly core family, paired bootstrap science-vs-curl, residual-curl null, and RA/Dec patch jackknife. The route remains separate from global P30.

Recommended run:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Inspect:

```text
outputs/p30_sdss_core_curl_projected_residual_v34.json
outputs/test04_p30_act_dr6_lensing_inventory.json
```

## v35 confirm-target patch

v35 focuses on the narrow blocker identified by the v34 run: **P30-SDSS-core needs a reliable curl-projected residual test**.

Main changes:

1. P30 low-memory curl-projected residual test using sequential low-Nside ACT map sampling.
2. P30 always writes `outputs/p30_sdss_core_curl_projected_residual_v35.json`, even on failure.
3. P30 residual fields now include beta/projection/residual deltas, residual p-values, paired bootstrap, residual-curl null, and patch jackknife.
4. P30 route-specific promotion remains `P30-SDSS-core` only; global P30 still needs a second independent route.
5. P3 endpoint prefilter is now a hard skip when no endpoint/node-pair columns are detected.
6. P36 high-z, P41, P33, and P32 contracts remain strict measurement gates.

Recommended target run:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Optional lower memory override:

```powershell
$env:CCDR_P30_PROJECTION_NSIDE="64"
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Inspect:

```text
outputs/p30_sdss_core_curl_projected_residual_v35.json
outputs/test04_p30_act_dr6_lensing_inventory.json
```

## v36 confirm-target patch

v36 adds a P30 projection-consistency layer. It prevents the curl-projected residual result from being used as physics evidence unless the raw pre-projection deltas reproduce the canonical v29/v34 same-split route.

Recommended run:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Inspect:

```text
outputs/p30_projection_consistency_audit_v36.json
outputs/p30_canonical_split_manifest_v36.json
outputs/p30_map_sign_convention_manifest_v36.json
outputs/p30_nside_raw_delta_comparison_v36.json
outputs/p30_pixel_level_sample_audit_v36.json
outputs/test04_p30_act_dr6_lensing_inventory.json
```

If `raw_projection_consistency_pass` is false, the correct label is projection-pipeline mismatch, not a CCDR/P30 physics claim.


## v37 confirm hardening

The v37 patch repairs the P30 canonical split manifest crash and forces all P30 curl-projected residual interpretation to use canonical v28/v29 SDSS labels. Projection outputs are claim-usable only if raw pre-projection deltas reproduce the canonical same-split route. It also adds v37 dashboard accounting and preserves strict P3/P36/P41/P33/P32 gates.

Recommended P30 run:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Inspect:

```text
outputs/p30_canonical_split_manifest_v37.json
outputs/p30_projection_consistency_audit_v37.json
outputs/p30_map_sign_convention_manifest_v37.json
outputs/p30_nside_raw_delta_comparison_v37.json
outputs/p30_pixel_level_sample_audit_v37.json
outputs/test04_p30_act_dr6_lensing_inventory.json
```

## Round-10 v38 patch

v38 adds a row/order/label sampler audit for P30. It preserves the positive P30-SDSS random-normalized route separately, but disables curl-projection residual claims unless the projection sampler reproduces canonical v29/v34 raw deltas.

Recommended run:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Inspect:

```text
outputs/p30_row_by_row_sampler_audit_v38.json
outputs/p30_high_low_inversion_test_v38.json
outputs/p30_object_order_lock_v38.json
outputs/p30_direct_pixelfunc_comparison_v38.json
outputs/test04_p30_act_dr6_lensing_inventory.json
```

## v39 confirm-hardening patch

v39 targets the P30 projection H/L inversion suspected in v38. It adds dual-sign raw deltas, manifest-only projection rows, an orientation fix test, and a fixed curl-projection path that runs only after raw deltas reproduce the canonical same-split route.

Recommended P30 run:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Inspect:

```text
outputs/p30_projection_label_fix_test_v39.json
outputs/p30_fixed_curl_projection_v39.json
outputs/test04_p30_act_dr6_lensing_inventory.json
```


## v40 confirm-target patch

Adds P30-SDSS-core curl-residual separation after the v39 H/L orientation fix. The v40 path caches sampled low-Nside ACT values, fits science-vs-curl residuals, performs patch-level paired bootstrap, tests residual curl-null, and keeps promotion route-specific only. P36/P41/P33/P32 strict measurement gates remain blocked until real object/likelihood/alpha/strain products exist.

## v41 measurement-first confirm hardening

v41 focuses on converting blocked contracts into real measurement consumers:
P36 high-z source-specific object rows, P41 Wilson/SM likelihood rows, P33 density-split BAO alpha rows, and P32 one-event GW150914 strain products. It also separates P30-SDSS-core route confirmation from global P30, adds patch-based residual curl-null accounting, leave-one-patch-out stability, and a persistent empirical ACT finite-support mask product.

Recommended targeted runs:

```powershell
python run_all.py --only P36 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P41 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P33 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P32 --allow-large --max-mb 5000 --script-timeout 2400
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Important v41 outputs:

- `outputs/p36_highz_real_object_catalogue_rows_v41.json`
- `outputs/p41_wilson_sm_likelihood_rows_v41.json`
- `outputs/p33_density_split_alpha_measurement_v41.json`
- `outputs/p32_gw150914_one_event_strain_result_v41.json`
- `outputs/p30_patch_based_residual_curl_null_v41.json`
- `outputs/p30_empirical_finite_support_mask_product_v41.json`

## v42 audit-first confirm hardening

v42 focuses on protecting and extending the v41 results:

- P36 high-z now writes `outputs/p36_highz_objectlevel_audit_v42.json` with accepted rows, sources, unit provenance, source medians, leave-one-source checks, and a source-bootstrap ratio-to-local-a0 CI.
- P30 now writes `outputs/p30_explicit_patch_rows_v42.json` and requires explicit sky-patch rows (`n_patch_rows >= 6`) before judging residual curl-null. Object-level curl p-values are diagnostic only.
- P30 empirical ACT mask metadata is persisted as `outputs/p30_empirical_act_mask_product_v42.json`; it remains diagnostic-equivalent unless an official ACT mask or mask-equivalence appendix is available.
- P41 now writes `outputs/p41_wilson_sm_likelihood_v42.json`, including a numerical SM-vs-Wilson comparison and a conservative one-parameter Wilson-shift fallback when explicit Wilson columns are absent.
- P33 now writes `outputs/p33_density_split_alpha_measurement_v42.json`, consuming any real alpha_high/alpha_low/delta_alpha products and retaining strict covariance/null gates.
- P32 now writes `outputs/p32_gw150914_one_event_strain_result_v42.json`, consuming measured strain/PSD/GR/CCDR residual/injection/detector-split products and refusing metadata-only promotion.
- The dashboard includes `dashboard_v42` with blocked-gate accounting.

Recommended targeted runs:

```powershell
python run_all.py --only P36 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P41 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P33 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P32 --allow-large --max-mb 5000 --script-timeout 2400
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Claim policy remains strict: P36 high-z requires the v42 audit gate; P30 can only promote route-specifically; P41/P33/P32 require measured products.

## v43 confirm-target hardening

v43 implements the next 10 improvements from the v42 report:

- P36 high-z: second-source discovery, strict source audit, source bootstrap, and leave-one-source hard gate.
- P30: bad-patch isolation, leave-one-patch-out stability, one-patch quarantine diagnostic, and mask-consistency gate.
- P41: explicit Wilson/SM coefficient-fit layer and CP-control split.
- P33: stricter measured alpha_high/alpha_low density-split consumer.
- P32: optional GW150914 GWOSC strain download/product gate plus strict measured-product requirements.
- P3: endpoint hard-skip preserved.
- Dashboard: v43 strict dashboard separation and blocked-gate accounting.

Recommended targeted runs:

```powershell
python run_all.py --only P36 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
python run_all.py --only P41 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P33 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P32 --allow-large --max-mb 5000 --script-timeout 2400
```

Inspect:

```text
outputs/p36_highz_second_source_audit_v43.json
outputs/p30_bad_patch_isolation_v43.json
outputs/p30_mask_consistency_v43.json
outputs/p41_wilson_coefficient_fit_v43.json
outputs/p33_density_split_alpha_measurement_v43.json
outputs/p32_gw150914_one_event_strain_result_v43.json
```

## v44 confirm-target hardening

v44 adds strict second-source, patch/mask, and measured-product consumers:

- `outputs/p36_highz_second_source_robustness_v44.json`
- `outputs/p36_highz_source_audit_v44.csv`
- `outputs/p30_predeclared_patch_protocol_v44.json`
- `outputs/p30_bad_patch_investigation_v44.json`
- `outputs/p30_mask_equivalence_appendix_v44.json`
- `outputs/p41_wilson_coefficient_fit_v44.json`
- `outputs/p33_density_split_alpha_measurement_v44.json`
- `outputs/p32_gw150914_one_event_strain_result_v44.json`

Recommended targeted runs:

```powershell
python run_all.py --only P36 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
python run_all.py --only P41 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P33 --allow-large --max-mb 5000 --script-timeout 1800
python run_all.py --only P32 --allow-large --max-mb 5000 --script-timeout 2400
```

P30 mask appendix: to make the empirical ACT finite-support mask publication-equivalent, add `outputs/p30_mask_equivalence_appendix_accepted.json` with:

```json
{
  "mask_equivalence_accepted": true,
  "same_mask_all_variants": true,
  "nulls_use_same_mask": true,
  "edge_exclusion_rule": "documented predeclared rule",
  "finite_support_definition": "documented finite-pixel support definition",
  "variant_consistency_table": "path or embedded table"
}
```


## v45 confirm-target hardening

v45 adds publication-audit outputs and stricter promotion guards:

- `outputs/p36_publication_audit_appendix_v45.json` and `.csv`
- `outputs/p30_mask_equivalence_validator_v45.json`
- `outputs/p30_predeclared_patch_protocol_v45.json`
- `outputs/p30_bad_patch_followup_table_v45.json`
- `outputs/p41_q2_likelihood_table_v45.json`
- `outputs/p33_density_split_alpha_measurement_v45.json`
- `outputs/p32_gwosc_endpoint_resolution_v45.json`

Dashboard v45 separates: (A) non-SM publication confirmations, (B) SM constant checks, and (C) near-confirm/readiness targets.

## v46 confirm-target hardening

v46 adds an override layer on top of v45 with stricter promotion gates:

- **P30**: writes `p30_mask_equivalence_acceptance_v46.json`, `p30_mask_equivalence_appendix_candidate_v46.json`, and `p30_frozen_patch_protocol_v46.json`. P30 cannot promote unless an official ACT mask exists or the user supplies `outputs/p30_mask_equivalence_appendix_accepted.json` with all required fields, and a fresh rerun manifest matches the frozen patch protocol hash.
- **P36**: writes `p36_publication_audit_appendix_v46.json/.csv` and performs a blinded acceleration recomputation from `Vrot_km_s` and `radius_kpc`.
- **P41**: writes `p41_q2_wilson_likelihood_v46.json` and template CSV; promotion requires measured q²/value/error rows, SM and Wilson predictions, Δχ²≥9, and CP controls null/weak.
- **P33**: writes `p33_density_split_alpha_measurement_v46.json` and template CSV; promotion requires measured alpha_high/alpha_low with covariance and density/sky/redshift nulls.
- **P32**: writes `p32_gwosc_endpoint_resolution_v46.json`; strain download is still not enough without PSD, GR fit, CCDR residual fit, injection nulls, and detector split.
- **SMD1–SMD5**: writes/uses `smd_ccdr_predictions_template_v46.json`; constant checks are not full SM derivations until prediction, uncertainty, residual sigma, independent derivation, and cross-prediction gates pass.
- **Dashboard**: adds `dashboard_v46` with three explicit buckets: non-SM publication confirms, SM constant-level confirms, and near-confirm/readiness targets.

## v47 P36 high-z executable hardening

v47 fixes the T13/T14 failure mode where the inherited high-z catalogue path could time out or return `broken`. The high-z test kind now uses a bounded executable guard:

- `test13_p36_kmos3d_inventory.py` writes structured JSON on every run.
- `test14_p36_highz_a0_cross_catalogue_inventory.py` writes structured JSON on every run.
- Missing or unavailable object catalogues return `highz_object_catalogue_data_limited_v47`, not `broken`.
- Existing local audit/catalogue products are consumed if present.
- Promotion to `highz_a0_objectlevel_publication_confirm_like` requires the executable T13/T14 v47 gates to pass.

Important outputs:

```text
outputs/p36_t13_executable_guard_v47.json
outputs/p36_t14_executable_guard_v47.json
outputs/p36_highz_executable_guard_v47_R10-T13.csv
outputs/p36_highz_executable_guard_v47_R10-T14.csv
```

Recommended check:

```powershell
python run_all.py --only P36 --quick --script-timeout 60
```

Claim policy: P36 local a0 remains separate; P36 high-z must not be claimed from a run where T13/T14 are `broken`, `not_run`, or `data_limited`.
