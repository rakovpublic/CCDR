# CCDR Tier-A v9.4 hotfix: T04, T05, T16, T22

This patch targets the failures seen in the uploaded `run_all_tierA.py` output:

- **T04** ACT DR6 kappa sampling stopped at `healpix_map_requires_optional_healpy`.
- **T05** Planck PR4 kappa sampling stopped at `healpix_map_requires_optional_healpy`.
- **T16** NANOGrav positions parsed, but ACT kappa sampling stopped at the same HEALPix dependency.
- **T22** downloaded a Planck product but did not parse it as a usable spectrum table.

## What changed

### T04/T05/T16

The scripts no longer fail only because `healpy` is absent. They now:

1. download the same public ACT DR6 / Planck PR4 lensing releases;
2. select a **non-curl kappa/klm ALM** FITS product when possible;
3. parse ALM binary tables via `astropy.io.fits`;
4. evaluate a controlled **low-l spherical-harmonic projection** at Euclid or NANOGrav sky positions without `healpy`;
5. clearly report `mode = low_l_alm_projection_no_healpy` and warn that this is a proxy/sign test, not a full-resolution likelihood.

This avoids the previous false `data_limited` caused by a missing optional map package, but it does **not** claim to replace full HEALPix analysis.

### T22

The script now discovers/downloads Planck PR3 `PowerSpect` text products, parses generic numeric tables, chooses the best TT-like spectrum, and computes a no-map low-ell suppression proxy by comparing `ell=2..29` to a smooth high-ell continuation.

## How to install

Copy/overwrite these files into the root of your existing `ccdr_tierA_public_tests` folder:

```powershell
Copy-Item .\tests\_tierA_healpix_fallback.py F:\git\upd\go_on\CCDR\research_program\computations\prediction_tests\v9\ccdr_tierA_public_tests\tests\ -Force
Copy-Item .\tests\test04_p30_euclid_q1_act_dr6_density_kappa.py F:\git\upd\go_on\CCDR\research_program\computations\prediction_tests\v9\ccdr_tierA_public_tests\tests\ -Force
Copy-Item .\tests\test05_p30_planck_crosscheck_density_kappa.py F:\git\upd\go_on\CCDR\research_program\computations\prediction_tests\v9\ccdr_tierA_public_tests\tests\ -Force
Copy-Item .\tests\test16_pta_kappa_crosslink.py F:\git\upd\go_on\CCDR\research_program\computations\prediction_tests\v9\ccdr_tierA_public_tests\tests\ -Force
Copy-Item .\tests\test22_cmb_large_angle_no_map_proxy.py F:\git\upd\go_on\CCDR\research_program\computations\prediction_tests\v9\ccdr_tierA_public_tests\tests\ -Force
```

Then install the only new dependency used by the FITS fallback:

```powershell
python -m pip install astropy
```

No manual data files are required. Public data is downloaded by the scripts and cached in `--cache`.

## Run only the fixed tests

From the Tier-A root after copying files:

```powershell
python .\tests\test04_p30_euclid_q1_act_dr6_density_kappa.py --cache .cache --outdir out_hotfix --allow-large --max-rows 8000 --alm-lmax 64
python .\tests\test05_p30_planck_crosscheck_density_kappa.py --cache .cache --outdir out_hotfix --allow-large --max-rows 8000 --alm-lmax 64
python .\tests\test16_pta_kappa_crosslink.py --cache .cache --outdir out_hotfix --allow-large --alm-lmax 64
python .\tests\test22_cmb_large_angle_no_map_proxy.py --cache .cache --outdir out_hotfix
```

Or run the included mini-runner from this hotfix folder:

```powershell
python .\run_hotfix_T04_T05_T16_T22.py --cache .cache --outdir out_hotfix --allow-large --alm-lmax 64 --max-rows 8000
```

## Interpretation caution

For T04/T05/T16, `alm-lmax=64` is intentionally conservative and fast. You can try `--alm-lmax 96` or `128`, but runtime grows roughly as `N_positions × lmax²`. Treat results as a sign/proxy test. A final publication-grade result should still be rerun with `healpy` or a validated full-resolution HEALPix/ALM sampler.
