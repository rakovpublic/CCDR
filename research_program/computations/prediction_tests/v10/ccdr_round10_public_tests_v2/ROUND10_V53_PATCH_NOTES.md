# Round-10 v53 confirm-artifact auto-build patch

v53 is built on v52. It keeps strict promotion gates, but adds automatic *provisional* measurement-artifact builders where already-downloaded public/cache data are available. Template artifacts still do not count as evidence.

## Implemented improvements

1. **P38 artifact auto-build**: creates `measurements/p38_void_measurement_v53_auto.json` from cached/public void catalogue-like files and source hashes; promotes only if the full v53 gate passes.
2. **P36 high-z raw-row alias parser**: accepts broader column aliases for `Vrot`, `R_kpc`, redshift, source group, and raw source file.
3. **P36 rejected-row diagnostics**: writes `outputs/p36_highz_rejected_rows_v53.json` to show exactly why raw high-z rows were not usable.
4. **P36 large-radius reanalysis**: writes `outputs/p36_highz_radius_quality_reanalysis_v53_<TEST>.json` with source counts, radius cuts, medians, and row previews.
5. **P30 protocol candidate**: writes `inputs/p30_patch_protocol_v53_FILL_AND_RENAME.json`; it must be filled/activated before a claim run.
6. **P30 route-specific gate**: separates route confirm from global confirm and requires protocol + same-mask/curl/variant conditions.
7. **P33 fillable alpha-measurement contract**: writes `measurements/p33_alpha_measurement_v53_FILL.json` with exact fields needed for promotion.
8. **PTA `.par` coordinate audit**: scans cached NANOGrav `.par` files, parses RAJ/DECJ, hashes sources, and writes `outputs/pta_par_coordinate_audit_v53.json`.
9. **P32/P40/P41 generic likelihood gates**: ingests non-template likelihood artifacts and reports artifact class, missing fields, and source-hash status.
10. **v53 dashboard artifact index**: separates filled artifacts from templates and prioritizes the fastest confirm-recovery paths.

## Claim policy

- `*_confirm_like*` statuses remain the only scientific confirms.
- `coverage_confirmed` is exposure/coverage only.
- Templates and `_FILL` artifacts do not count.
- Auto-built artifacts are provenance records; they promote only if all strict gate fields pass.

## Fastest confirm-recovery files

- `inputs/p36_highz_object_rows_raw.csv/json`
- `measurements/p38_void_measurement_v53_auto.json` or curated `measurements/p38_void_measurement_v53.json`
- `inputs/p30_patch_protocol_v53.json`
- `measurements/p30_independent_route_v53.json`
- `measurements/p33_alpha_measurement_v53.json`
- `measurements/pta_weighted_kappa_residual_v53.json`
