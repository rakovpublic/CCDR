# Round-10 v56 confirm-recovery patch

v56 is a strict confirm-recovery layer on top of v55. It does not promote template or auto-generated next-run artifacts. It adds stronger artifact builders, stricter gates, and clearer dashboard accounting for the current high-value confirm paths.

## Implemented improvements

1. **P36 high-z raw-row recovery v56**
   - Broader raw catalogue scanner for KGES/KROSS/KMOS3D/SAMI-like files in `inputs/`, `measurements/`, and cache.
   - Supports common aliases for object id, redshift, Vrot, and radius.
   - Rejects generated `outputs/` and summary/reanalysis files.

2. **P36 large-radius strict gate v56**
   - Requires `R_kpc >= 0.5`, at least 30 trusted rows, at least two source groups, at least 20 large-radius rows per two sources, tiny-radius fraction <=20%, and median acceleration above local a0.

3. **P30 active protocol gate v56**
   - Requires `inputs/p30_patch_protocol_v56.json` or a valid active prior protocol filled before the run.
   - Ignores `FILL`, `TEMPLATE`, `EXAMPLE`, and `AUTO_PREDECLARED_FOR_NEXT_RUN` artifacts.

4. **P30 same-mask route proof v56**
   - Requires a non-template route artifact proving same mask, same density labels, weaker curl, positive route, and source hashes.

5. **P30 route/global split v56**
   - Route confirm can pass without independent route; global confirm requires second independent Planck/Euclid route.

6. **P33 alpha-measurement gate v56**
   - Requires numeric `alpha_high_density`, `alpha_low_density`, `delta_alpha`, positive `delta_alpha_sigma`, >=2σ delta-alpha, covariance-aware fit, DESI randoms, null p-values <=0.05, redshift jackknife stability, predeclared sign, and source hashes.

7. **PTA / CL2 weighted statistic gate v56**
   - Requires coordinate hashes, residual/TOA weights, kappa samples, weighted statistic, sky-shuffle p <=0.05, top-weight-removal stability, predeclared sign, and source hashes.

8. **P32/P40/P41 likelihood gates v56**
   - P32 requires PSD, GR fit, CCDR residual-template fit, injection null, detector split, leave-one-event-out, source hashes, and Δχ² threshold.
   - P40 requires BB bandpowers, covariance, foreground controls, Planck/BK18 cross-check, amplitude and uncertainty.
   - P41 requires q²/value/error rows, CP null, observable-bin jackknife, source hashes, and Δχ² >=9.

9. **SMD derivation gate v56**
   - Requires non-template preregistered derivation metadata with no target fit, independent derivation, source hashes, predicted value, uncertainty, and residual sigma.

10. **Dashboard v56 strict buckets**
    - Adds `dashboard_v56`, artifact index, failed-gate report, and confirm-recovery priority list.
    - Keeps claim policy strict: templates/fill/auto-next-run files never promote.

## Run

```powershell
python run_all.py --allow-large --max-mb 80000 --script-timeout 720000
```

## Most important artifacts to fill

```text
inputs/p36_highz_object_rows_raw.csv
inputs/p30_patch_protocol_v56.json
measurements/p30_same_mask_route_v56.json
measurements/p33_alpha_measurement_v56.json
measurements/pta_weighted_kappa_residual_v56.json
measurements/p32_strain_likelihood_v56.json
measurements/p40_bb_likelihood_v56.json
measurements/p41_q2_wilson_likelihood_v56.json
measurements/smd_derivation_predictions_v56.json
```
