# CCDR v7.3 / Synthesis v3.3 public-data twelve-test battery

This bundle implements a **twelve-test public-data battery** aligned to the current CCDR v7.3 and Extended Crystallographic Synthesis v3.3 documents.

## Included tests

1. `test01_p30_non_euclid_lensing_replication_v73.py`
   - P30 replication with Euclid-Q1-like galaxy density against ACT/Planck public lensing access paths.
   - Adds **density-stratified reducing-volume nulls** to test cross-patch DM-composition variation under the new dimension-origin prior.

2. `test02_p30_euclid_systematics_audit_v73.py`
   - Euclid-Q1 internal nuisance/split audit of the density–κ signal.
   - Reports RA/Dec/redshift/threshold asymmetries explicitly as possible **patchiness / reducing-volume signatures**, not only as survey systematics.

3. `test03_highz_a0_sparc_kmos3d_v73.py`
   - Local SPARC anchor + public high-z KMOS3D-like proxy for `a0(z)`.
   - Outputs the **three-point MOND-sequence ν extractor** for new P36 / CL1 and compares it to the SN-based ν band.

4. `test04_nu_public_redrive_v73.py`
   - Compact Pantheon+ + DESI DR2 + Planck PR3 ν re-drive proxy.
   - Adds the v7.3 **CL1–CL3 triangle** (`ν_SN`, `ν_MOND`, time-crystal `Q` proxy).

5. `test05_p3_euclid_filament_ordering_v73.py`
   - Transparent-finder filament ordering proxy.
   - Explicitly checks whether the amplitude depends on density in the direction expected from the reducing-volume prior.

6. `test06_p29_multi_redshift_growth_v73.py`
   - Public DESI full-shape/RSD proxy for ongoing DM abundance growth.
   - Adds a **live/frozen decomposition** instead of a strict binary interpretation.

7. `test07_p36_mond_sequence_nu_extractor.py`
   - New standalone implementation of the immediate three-point `a0(z)` fit.
   - Feeds **CL1** and tests whether the low-z SN systematic interpretation is reinforced.

8. `test08_p33_density_correlated_bao.py`
   - New density-stratified BAO sister-test to P30.
   - Uses public DESI DR2 summary tables plus the same density binning logic as the κ pipeline.

9. `test09_cl2_p8c_reducing_volume_cross.py`
   - New spatial cross-correlation proxy between a NANOGrav-like PTA×cosmic-web field and the Euclid-Q1-like density–κ field.
   - Tests whether the old P8c wrong-sign can be reinterpreted as a **reducing-volume signature** under CL2.

10. `test10_p38_void_wall_cauchy_tail.py`
    - New stacked void-wall transverse-density proxy.
    - Measures whether the transverse tails are more Cauchy-like than Gaussian (`k4 > 4`), as expected for P38.

11. `test11_hardcore_window_readiness_v73.py`
    - Updated hard-core readiness tracker for the `0.5–3 TeV` window.
    - Distinguishes **mere public overlap** from actual **peak-resolution readiness**, and reports example `E[n_peaks]` values under the v7.3 probability-weighted prior.

12. `test12_p37_phase_space_drift_proxy.py`
    - New subthreshold galactic-orbit phase proxy for DM mass drift (`dm/m ~ ν H0`).
    - Uses a Gaia-like rotation-curve proxy and simulated detector populations to quantify opposite-phase spectral offsets.

## What changed versus the older battery

- The **dimension-origin prior** is now part of the interpretation layer throughout the bundle.
- Tests that previously treated every asymmetry or wrong-sign result as a failure now separate:
  - likely survey/systematics behavior,
  - physically allowed **cross-patch variation**,
  - and simple lack of sensitivity.
- The local SPARC anchor no longer silently returns `n_points = 0` in proxy mode.
- Downloads are **streamed to disk** to avoid the large in-memory failures that affected earlier bundles.
- The SDSS parser explicitly strips `#Table1` / `#table1` style prefixes that previously broke SkyServer CSV handling.
- The Pantheon-style parser logic avoids deprecated `delim_whitespace=` usage.

## Important interpretation note

These scripts are **public-data screening / proxy tools**. They are designed to auto-download public products when possible and to fall back to deterministic, documented proxy data when collaboration-grade machine-readable products are unavailable in the runtime environment.

They are therefore useful for:
- sign checks,
- null-control design,
- pipeline debugging,
- and deciding which claims deserve a heavier follow-up.

They are **not** a substitute for a survey collaboration likelihood or a detector collaboration event-level analysis.

## Usage

Run a single test:

```bash
python test01_p30_non_euclid_lensing_replication_v73.py
```

Run the full bundle:

```bash
python run_all_v73.py
```

Outputs are JSON so they are easy to diff across iterations.
