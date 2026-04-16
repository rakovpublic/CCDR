# CCDR v6.2 / Synthesis v2.2 round-3 public-data tests

This bundle implements eight separate Python tests aligned with the live claims in
CCDR v6.2 and Synthesis v2.2. Every script downloads its inputs from public
sources at runtime and writes a JSON summary to its output directory.

These are **public-data falsifiability tools**, not exact reproductions of full
collaboration pipelines. Where an official likelihood, catalogue constructor, or
external binary package is not publicly packaged in a lightweight way, the
script uses a transparent proxy and says so explicitly in its docstring and
output notes.

## Files

- `test01_nu_redshift_tomography.py`
- `test02_pantheon_nuisance_audit.py`
- `test03_filament_public_replication.py`
- `test04_filament_dual_finder_consistency.py`
- `test05_wz_single_peak_coherence.py`
- `test06_local_a0_subset_hierarchical.py`
- `test07_koide_common_scale_scan.py`
- `test08_dm_hard_core_readiness.py`
- `_common_public_data.py`

## Suggested environment

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Example usage

```bash
python test01_nu_redshift_tomography.py --outdir out01
python test02_pantheon_nuisance_audit.py --outdir out02
python test03_filament_public_replication.py --outdir out03
python test04_filament_dual_finder_consistency.py --outdir out04
python test05_wz_single_peak_coherence.py --outdir out05
python test06_local_a0_subset_hierarchical.py --outdir out06
python test07_koide_common_scale_scan.py --outdir out07
python test08_dm_hard_core_readiness.py --outdir out08
```

## Notes

- Tests 1, 2, and 5 use compact public-data RVM-like fits built on Pantheon+,
  DESI DR2 BAO, and a Planck PR3-derived `r_drag` prior.
- Tests 3 and 4 use public SDSS SkyServer spectroscopy with transparent local
  filament estimators rather than external catalogue builders like DisPerSE.
- Test 6 uses the public SPARC machine-readable tables and a lightweight
  hierarchical-style shrinkage model.
- Test 7 parses public PDG 2024 summary PDFs and performs a one-loop running
  scan as a transparent stress test of the common-scale Koide reading.
- Test 8 is a readiness and coverage audit for the CCDR hard-core direct-
  detection claim, not a hard-core falsification in itself.
