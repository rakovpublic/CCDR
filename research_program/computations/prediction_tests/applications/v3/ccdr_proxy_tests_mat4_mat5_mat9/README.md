# CCDR lean public materials/phonon proxy tests

This V6 bundle intentionally drops the fusion FR3/FR6 and QC5 tests from the batch.
It runs only three materials/phonon literature-proxy tests:

- `test02_mat5_kibble_zurek_gb_density.py` — Kibble-Zurek-like cooling-rate / microstructure scaling.
- `test03_mat4_bi2te3_gb_zt_gain.py` — 30°/30 V Bi2Te3-family grain-boundary ZT proxy.
- `test05_mat9_moire_twist_phonon_bandgap.py` — moire/twist-angle phonon or bandgap tunability.

Run all tests:

```powershell
py .\run_all_public_proxy_tests.py --outdir out_public_tests --force
```

Run one test:

```powershell
py .\test02_mat5_kibble_zurek_gb_density.py --outdir out_public_tests --force
py .\test03_mat4_bi2te3_gb_zt_gain.py --outdir out_public_tests --force
py .\test05_mat9_moire_twist_phonon_bandgap.py --outdir out_public_tests --force
```

All data are downloaded by the scripts from public URLs. The scripts are deliberately conservative: they should emit `partial_*` or `data_limited` rather than decisive confirm/falsify when the public extraction is qualitative, confounded, or not machine-readable.

## V6 changes

- FR3/FR6 and QC5 are removed from the runner and from the zip.
- MAT4 rejects garbled supplement baselines, reports the missing baseline inputs, and computes the target baseline ZT window implied by the parsed 30V ZT and the 1.1–1.3 predicted ratio.
- MAT5 rejects weak inequality/generic threshold prose, preserves same-composition grouping, discovers public supplementary/source-data CSV/XLSX/ZIP links, and adds leave-one-out stability plus near-miss diagnostics for promising n=4 groups.
- MAT9 adds inline-string XLSX parsing, TSV/JSON-like source-data handling, stricter source-data blocker reporting, and keeps artifact rejection for LaTeX/affiliation/figure garbage.
