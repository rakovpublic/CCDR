# Test 08 — Dark-Matter Mass-Peak Counting from Direct Detection

## Summary

- Analysis mode: **stacked public-data mass-space residual scan**
- Loaded sources: **1**
- Skipped sources: **7**
- Significant peaks found: **0**
- Inferred N: **None**
- v6 cascade pass: **False**
- v5 single-step pass: **False**

## Peak summary

- No ≥3σ proxy peaks in the combined public-data mass scan.

## Consecutive ratios

- Fewer than 2 peaks, so no ratio test can fire.

## Loaded curves

- XENONnT / XENONnT_light_wimp_repo/light_wimp_data_release-master\light_wimp_data_release\limits\XENONnT_2024_SI.csv: 10 points, mass range 3–12 GeV

## Skipped / partial sources

- LZ_4.2tyr_SI: File is not a zip file
- LZ_5.7tyr_light_SI: File is not a zip file
- PandaX4T_first_analysis_xlsx: downloaded but no usable mass-limit curve found
- PandaX4T_first_analysis_eff_root: downloaded ROOT efficiency file; not converted to a mass-limit curve in mass-space scan mode
- PandaX4T_lightDM_run0: downloaded event-level table; not directly used in mass-space scan mode
- PandaX4T_lightDM_run1: downloaded event-level table; not directly used in mass-space scan mode
- PandaX4T_run0_S2only_zip: 404 Client Error: Not Found for url: https://static.pandax.sjtu.edu.cn/download/data-share/p4-s2-only/s2only_data_release.zip

## Caveats

- This is a public-data proxy analysis in DM-mass space, not the collaborations' full joint unbinned recoil likelihood.
- Peak significances are derived from a stacked excess-score scan built from public observed/expected curves or detrended public limit curves.
- If a source did not expose a directly machine-readable SI-WIMP curve, it was downloaded and logged as skipped or partial rather than silently ignored.

## Diagnostics

```json
{
  "noise": 0.001079074752978618,
  "best_k": 0,
  "aic": {},
  "bic": {}
}
```