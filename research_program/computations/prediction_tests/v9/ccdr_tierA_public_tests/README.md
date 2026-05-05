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

## v9.6 result-quality patch

Implemented requested 10 quality upgrades while preserving the rule that test data are downloaded automatically from public sources by the scripts:

1. Added pure-NumPy HEALPix RING map sampling fallback for ACT/Planck/NANOGrav κ cross-link tests when `healpy` is unavailable.
2. Improved ACT/Planck map selection to avoid `alm`/`curl` products when map products exist.
3. Added Pantheon+ covariance download/parser and covariance-aware ν-like SN audit.
4. Added DESI DR2 public mean/cov pair covariance-aware GLS trend diagnostic.
5. Expanded Euclid Q1 TAP loading to request optional photo-z, magnitude/depth, mask/flag, and quality columns when public.
6. Added Euclid depth/mask diagnostics for patch-spread tests and downgraded depth-confounded results.
7. Added depth-residualized density splits and depth-stratified nulls for Euclid filament tests.
8. Expanded eROSITA cluster proxy aliases and numeric-column diagnostics on parser failure.
9. Added high-z galaxy/object-level bootstrap diagnostics for KMOS3D/KROSS acceleration proxies.
10. Added broader direct-detection public-source discovery, FIRAS unit audit, Planck spectrum direct fallbacks, and BK18 header/spectrum identity hints.

Important: v9.6 still refuses to convert proxy-level diagnostics into hard confirmations. Cosmology remains covariance-diagnostic unless a full RVM/BAO likelihood is added; direct detection remains readiness-only without event-level likelihoods.


## v9.6 quality/data-limited patch

This patch keeps the public-data/no-manual-input rule and adds:

- stricter ACT/Planck map selection: WCS/HEALPix pixel maps are preferred; `alm`/`klm` products are used only through optional `healpy.alm2map`;
- Euclid optional-column propagation and flux-error-to-depth conversion;
- observable-wise DESI DR2 covariance stability diagnostics;
- more robust VizieR/CDS filament table parsing;
- eROSITA redshift/selection/sky-region controls;
- NANOGrav posterior/residual column identity checks;
- safer FIRAS/Planck/BK18/KSS parsers that downgrade unverified products instead of producing false positives.


## v9.7 patch notes

- Fixed T03 crash caused by private `_pantheon_columns` use.
- `run_all_tierA.py` now writes `Txx.crash.json` files and continues after non-zero subprocess returns.
- T04/T05/T16 explicitly reject mask/coverage products and constant sampled maps as `data_limited`, not null; mask metrics are labelled `mask_proxy`.
- ACT/Planck map discovery now prefers exact non-mask κ/convergence pixel maps and only uses klm/alm products through `healpy.alm2map`.
- T08 uses a more robust VizieR/CDS parser with endpoint and sexagesimal coordinate support.
- T22 has a low-ell numeric Planck table parser for whitespace/commented products.
- T25 includes a curated eta/s registry and continues to reject flow/v2 tables unless eta/s is explicit.
- T01/T02 include simplified DESI model-vector likelihood diagnostics in addition to trend screens.
- T11 adds a robustness matrix; T06/T07 add field/depth/photo-z matched controls; T12 adds survey-split object-level summaries.

## v10 patch notes

This v10 bundle implements the 10 post-v9.9 improvements:

1. `run_all_tierA.py` prints and writes `run_meta_summary.json`, `run_meta_summary.md`, and `merged_results_with_meta.txt` so pasted logs include all-25 status counts.
2. T04/T05/T16 share the same candidate-map validation path and report candidate-attempt diagnostics.
3. κ tests report coordinate-system assumptions (`coordsys_verified`, expected frame, map family).
4. T07 keeps negative high-density-minus-low-density results as formal P3 tension unless endpoint/skeleton support reverses it.
5. T08 uses an endpoint/spine/position-angle catalogue registry and keeps kNN as secondary only.
6. T10 reports explicit redshift-bin, sky-region, and shuffle-null control gate breakdowns.
7. T12 preserves KROSS/KMOS3D source labels where possible and reports offset/trend as separate claims.
8. T22 adds an approximate low-ell cosmic-variance likelihood screen while warning it is not official Planck likelihood.
9. T11 attempts physical SPARC metadata splits before falling back to filename/order robustness splits.
10. DM/KSS/PTA tests expose conservative evidence guards: no event-level likelihood, verified η/s, or semantic posterior columns means no promotion to evidence.

Recommended full run:

```powershell
python run_all_tierA.py --outdir out_v10 --cache .cache --allow-large --prefer-healpy --strict-25
```

Use `merged_results_with_meta.txt` for analysis upload/copy-paste.
