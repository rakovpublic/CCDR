# CCDR Tier-A v9.6 quality patch

This overlay implements the 8 requested improvements for the Tier-A v9.5 large run.

## Apply

Copy/extract these files into the root of `ccdr_tierA_public_tests`, then run:

```powershell
python apply_tiera_v9_6_quality_patch.py --apply
```

## Run supplemental quality diagnostics

```powershell
python run_tiera_v96_quality.py --cache .cache --outdir out_v9_6_quality --allow-large
```

## Improvements implemented

1. Fixes T03 `_pantheon_columns` runtime error.
2. Adds robust kappa-map product loader/classifier for T04/T05/T16; rejects ALM-only products unless `healpy` is available for `alm2map`.
3. Adds Euclid depth/magnitude/quality proxy controls for T06/T07.
4. Adds VizieR/CDS parsers for filament catalogues in T08.
5. Adds helpers for moving T02 toward joint BAO+SN likelihood screens.
6. Adds source seeds and readers for T21/T23 covariance/bandpower products.
7. Adds posterior/chain artifact readers for T15/T17/T24.
8. Adds eta/s posterior-table discovery seeds and strict metadata-vs-physical-data typing for T25.

No manual rows are required. All data are discovered/downloaded automatically by scripts.
