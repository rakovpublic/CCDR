# CCDR/Synthesis v7.5/v3.5 Tier-A public-data tests

This bundle implements 25 Tier-A prediction tests. Every script is designed to download all required data automatically from public sources into a local cache. No manual file passing is required.

Some public products are very large or have access endpoints that change over time. The suite handles those cases by:

- using public APIs where possible: Zenodo, GitHub raw/API, IRSA TAP, GWOSC event API;
- writing a structured JSON result instead of crashing when a public endpoint changes;
- marking such cases as `data_limited`, `missing_optional_dependency`, or `large_download_not_enabled`;
- requiring `--allow-large` for products that can be hundreds of MB to many GB, e.g. ACT/Planck maps, Euclid catalog chunks, NANOGrav timing archives, GW strain.


## v9.2 fixes after first Windows run

This patched bundle fixes issues found in the supplied PowerShell output:

- GitHub data-release repositories now fall back from `main` to `master`; this fixes CobayaSampler `bao_data` and XENONnT `light_wimp_data_release` 404s.
- KMOS/KROSS high-z scripts now prefer exact `Z`, `VC`/`V22`, and `R_IM`/`RD_RPSF` columns instead of loose substring matching.
- VAST void catalog parsing now handles headerless numeric `.dat/.txt` tables and avoids using the first data row as column names.
- eROSITA cluster scripts now extract `.fits.tgz` archives before parsing FITS tables.
- Euclid patch-spread and filament-orientation scripts are now field-aware, avoiding false statistics from combining widely separated Q1 pointings as one connected field.
- FIRAS and Planck spectrum scripts force headerless numeric parsing when needed.
- BICEP/Keck BK18 B-mode script uses the direct public bandpower text file, avoiding the SSL-hostname failure from the old landing page.
- QGP KSS script tries the exact HEPData record `ins1666817` before generic HEPData search pages.
- The Pantheon+ SN-only ν proxy is now conservatively bounded and warns when it hits the bound; it remains a diagnostic, not a full RVM likelihood.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run all tests

```powershell
python run_all_tierA.py --outdir out_tierA --cache .cache
```

For large public products:

```powershell
python run_all_tierA.py --outdir out_tierA_large --cache .cache --allow-large
```

Run one test:

```powershell
python tests/test11_sparc_local_a0_anchor.py --outdir out_test11 --cache .cache
```

## Output format

Each test writes a JSON file with:

- `test_id`, `prediction_ids`, `status`
- `data_sources`, including URLs actually attempted or used
- `metrics`
- `falsification_logic`
- `notes` and `warnings`

Statuses are deliberately conservative:

- `confirm_like`: statistic has predicted sign/scale and survives internal nulls.
- `suggestive`: sign/scale is interesting but not decisive.
- `null`: predicted signal absent under implemented protocol.
- `data_limited`: public products were unavailable, too large without `--allow-large`, or did not expose enough columns for the intended statistic.
- `broken`: code exception or public endpoint failure not recoverable by fallbacks.

## Important limitations

These scripts are public-data screening tests, not collaboration-grade analyses. They avoid specialized software by design. For example:

- NANOGrav timing residuals are not refit with TEMPO2/enterprise; scripts use available public archives/posteriors and mark full residual analyses as data-limited.
- ACT/Planck/Euclid map-level tests use optional Astropy FITS/WCS and coarse sampling, not HEALPix pipelines.
- GWOSC ringdown uses a simple damped-sinusoid fit, not a full LALInference/Bilby analysis.
- Cosmology scripts use a lightweight phenomenological ν-like extension, useful for audit/stability checks but not a replacement for Cobaya/MontePython chains.



## v9.2 patch notes
- GitHub branch fallback no longer records expected `main`/`master` misses as failed data sources when a later branch succeeds.
- T23 now uses the NASA LAMBDA BK18 direct bandpower URL first, avoiding the bicepkeck.org certificate-hostname problem and stale LAMBDA paths.
- A remaining `data_limited` status on ACT/Planck/NANOGrav/GWOSC tests usually means the script intentionally avoided large downloads; rerun with `--allow-large`.

## v9.4 patch notes

This bundle adds the requested parser/sampling fixes:

- **P40 / T23**: BK18 bandpower parser now treats column 2 as ell and columns 3--134 as BB bandpower/error-bar pairs; the output reports pair summaries and no longer averages ell as a BB amplitude.
- **P32 / T24**: GWOSC downloader now prioritizes HDF5 strain files before GWF frame files when using `h5py`; GWF now gives an explicit reader warning instead of a file-signature error.
- **P8c / T16**: NANOGrav `.par` parsing now supports `RAJ/DECJ`, `ELONG/ELAT`, and approximate `PSRJ` name fallback, and reports the number of `.par` files scanned.
- **P14/P30 / T04--T05**: ACT DR6 now downloads/extracts the real `dr6_lensing_release.tar.gz` map when `--allow-large` is used. Planck cross-check now searches actual Planck/PR4 lensing map archives; HEALPix sampling requires optional `healpy`.
- **P5 / T25**: HEPData columns are now inspected and rejected unless an explicit eta/s-like column is present. Flow columns such as `v2{2, Δη}` are no longer treated as viscosity.
- **P3/P38 / T07/T09**: Null controls were strengthened. T07 now uses within-field density-shuffle nulls. T09 now requires radius-like kurtosis to exceed both k4>4 and matched lognormal null controls; log-radius-only excess is reported separately.

Optional for Planck HEALPix sampling:

```powershell
conda install -c conda-forge healpy
```

Without `healpy`, T05 will still download/select the map but will return a structured `data_limited` result explaining the missing optional sampler.

## v9.4 result-quality patch

This patch improves scientific interpretability rather than software plumbing: DR2-only BAO screens, quality-cut high-z acceleration, redshift-residual cluster mixtures, artifact-controlled direct-detection curve extrema, physical FIRAS μ/y bounds, and generic P40 B-mode template limits.
