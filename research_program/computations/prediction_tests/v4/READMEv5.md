# CCDR v7 / Synthesis v3 public-data six-test battery

This bundle implements the six tests described in the user-provided test plan for **CCDR v7** and **Synthesis v3**.

## Included tests

1. `test01_p30_density_correlated_variation.py`
   - P30 density-correlated spatial variation in Ω_DM/Ω_B
   - now forces **public Euclid Q1** through **IRSA TAP** instead of defaulting to the earlier SDSS fallback
   - includes a **spatial variance map** of κ residuals versus local density contrast on ~Mpc scales
   - includes a **joint Test 01 + Test 02** local-density crystal-ordering fit

2. `test02_filament_alignment_euclid_replication.py`
   - independent replication of filament alignment (legacy P3 stress test)
   - now forces **public Euclid Q1** through **IRSA TAP**
   - adds a **density-stratified null control** that shuffles axes only within density bins

3. `test03_p29_multi_redshift_growth_proxy.py`
   - P29 time-varying DM abundance precursor via public multi-redshift growth points
   - keeps the transparent compressed full-shape/RSD proxy
   - adds a **joint Test 03 + Test 05** asynchronous-stage fit

4. `test04_p31_dm_peak_drift_latest_releases.py`
   - P31 drifting DM mass-peak audit using the newest public direct-detection HEPData records discoverable at runtime

5. `test05_highz_a0_precursor.py`
   - local a0 -> Milgrom retreat extended to a high-z precursor proxy
   - uses public machine-readable local SPARC data plus a public high-z ionised-gas kinematics proxy sample
   - includes the same **joint Test 03 + Test 05** asynchronous-stage fit

6. `test06_cern_run3_quick_check.py`
   - quick Run 3 HEPData screening for remaining CERN-side signatures (P9b/d/e/f)

## Notes

- All scripts download data from public sources at runtime.
- Several of the v7/v3 claims still do **not** have perfect public machine-readable products. Where that is true, the scripts are explicit that they are **proxy / screening / readiness** implementations rather than collaboration-grade final analyses.
- The Euclid Q1 public archive path now targets the public **IRSA TAP** service, in line with the March 19, 2025 Q1 release documentation.
- The helper file now includes lightweight utilities for:
  - Euclid Q1 IRSA TAP access
  - local-density estimation on ~Mpc scales
  - density-stratified null permutations
  - shared density-ordering fits
  - shared async-stage fits for Tests 03 and 05

## Basic usage

```bash
python test01_p30_density_correlated_variation.py
python test02_filament_alignment_euclid_replication.py
python test03_p29_multi_redshift_growth_proxy.py
python test04_p31_dm_peak_drift_latest_releases.py
python test05_highz_a0_precursor.py
python test06_cern_run3_quick_check.py
```

Each script writes a JSON summary into its own output directory. Several of the updated scripts also write machine-readable CSV side products for the new map / joint-fit layers.
