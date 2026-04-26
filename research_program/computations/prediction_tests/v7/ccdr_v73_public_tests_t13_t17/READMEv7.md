# CCDR v7.3 public-data tests T13–T17

This bundle implements the five public-data tests specified as T13–T17 in the uploaded CCDR v7.3 / Synthesis v3.3 materials.

Implemented scripts:
- `test13_p33_density_correlated_bao.py`
- `test14_cl2_p8c_pta_planck_pr4_kappa.py`
- `test15_p36_highz_a0_trend.py`
- `test16_p38_void_wall_cauchy_tail.py`
- `test17_p3_filament_orientation_density.py`
- shared helpers: `common_public_utils.py`

## Install

```bash
python -m pip install -r requirements.txt
```

Optional but helpful:
- `healpy` or `astropy-healpix` for Planck map sampling in T14
- `pint-pulsar` for a better timing-residual WRMS estimate in T14

## Run examples

```bash
python test13_p33_density_correlated_bao.py --outdir out_t13
python test14_cl2_p8c_pta_planck_pr4_kappa.py --outdir out_t14
python test15_p36_highz_a0_trend.py --outdir out_t15
python test16_p38_void_wall_cauchy_tail.py --outdir out_t16
python test17_p3_filament_orientation_density.py --outdir out_t17
```

Each script downloads its own public inputs into its cache directory and writes a `result.json` to the output directory.

## Honesty notes

These are **screening / proxy scripts**, not collaboration-grade pipelines:
- T13 uses density-split clustering subsamples and a simplified BAO peak fit.
- T14 samples PR4 kappa at pulsar positions instead of building a full sky-field estimator.
- T15 uses real SPARC curves but an integrated-quantity proxy for KMOS3D when directly parseable full resolved curves are not trivial from the public release format.
- T16 uses a radial wall-thickness proxy from public void centers + NSA galaxies.
- T17 works directly from the public cosmic-web filament catalog instead of rerunning filament finding.

That keeps them fully public-data and runnable, while staying faithful to the requested tests.
