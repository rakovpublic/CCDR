# Round-10 v63 behavior-changing confirm-recovery patch

v63 is built on v62 and avoids interface-only changes. It modifies parser/estimator behavior for the highest-yield blockers from the v62 report.

## Implemented behavior changes

1. **P36 high-z source-targeted public fetchers**
   - Adds exact public/cache fetch attempts for KROSS, KGES, KMOS3D, SAMI, MOSDEF, PHIBSS and VizieR/CDS-style mirrors.
   - Audits failures and skips oversized archives rather than requiring manual files.

2. **P36 source-specific column maps**
   - Adds source-specific aliases for object id, redshift, Vrot/Vmax/V2.2, physical radius columns, and inclination.
   - Supports FITS/CSV/TSV/DAT/JSONL through existing robust readers.

3. **P36 strict radius provenance**
   - Accepts only source-mapped physical radius columns such as R_kpc, R_e, R_d, R_turn, Rmax, R6D.
   - Rejects ambiguous generic size columns in claim mode.
   - Every accepted row gets source hash, raw file, original columns, and unit conversion metadata.

4. **P36 source-level bootstrap**
   - Adds leave-one-source-out and source bootstrap CI above local a0 once multiple sources are available.
   - Keeps strict promotion gates: >=2 sources, >=20 large-radius rows/source, tiny-radius <=20%.

5. **P30 predeclared patch/control resolver**
   - Adds a deterministic patch rejection table based on curl patch magnitude and missingness before route promotion.
   - Adds redshift-density residualization proxy from field-jackknife dispersion.

6. **P30 route recomputation diagnostics**
   - Carries same-run same-mask route output forward and adds v63 route gate with curl, variants, patch rejection, and residualization blockers.

7. **P33 DESI LSS endpoint discovery**
   - Adds concrete DESI LSS public endpoint candidates and cached LSS catalogue discovery.
   - Loads RA/DEC/Z/weight rows from FITS if astropy is installed or from table rows otherwise.

8. **P33 density-split alpha proxy**
   - Implements a real lightweight pair-histogram alpha proxy for high/low density samples.
   - Still does not promote without covariance, randoms, shuffles, jackknife, and source hashes.

9. **PTA weighted statistic attempt**
   - Parses residual-like public/cache tables when found and builds a residual RMS weighted-statistic proxy.
   - Still blocks until kappa samples, sky shuffle, and top-weight stability exist.

10. **P32/P40/P41 likelihood gate refinements**
   - P32 checks actual likelihood artifact flags for injection-null and detector split.
   - P40/P41 expose exact missing numeric likelihood/null requirements without manual artifacts.

## Validation

```text
python -m py_compile ccdr_r10_common.py run_all.py tests/*.py
```

Targeted quick checks emitted v63 fields for T13, T14, T04, T07, T17, T19, T21, T31, and dashboard.
