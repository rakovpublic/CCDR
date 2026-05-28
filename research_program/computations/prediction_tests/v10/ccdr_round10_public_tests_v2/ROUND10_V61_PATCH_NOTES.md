# Round-10 v61 no-manual real-estimator patch

v61 is intended to replace the previous wrapper/interface-only changes with concrete no-manual code paths.

## Implemented

1. P36 high-z source-targeted public fetchers for KROSS, KGES, KMOS3D, SAMI, MOSDEF, PHIBSS, plus VizieR/CDS mirrors where known.
2. P36 source-specific column maps for object id, redshift, Vrot/Vmax/V2.2, radius/Re/Rd/Rturn, inclination.
3. P36 row-level provenance: source file hash, original column names, unit conversion method, source group, raw file path.
4. P30 same-run same-mask route resolver using current baseline/curl/variant/null/jackknife stats; no manual protocol requirement for the claim path.
5. P30 control-tension resolver for curl dominance, sign flips, field-jackknife instability, patch imbalance, and mask-edge proxy state.
6. P30 route-specific sign policy for SDSS route before global aggregation.
7. P33 automated density-split BAO alpha fitter scaffold using public/cached DESI-like catalogues with RA/DEC/Z when present; it returns fit-missing if no catalogue exists.
8. PTA automatic parser for NANOGrav-like .par coordinates plus residual/TOA discovery gate.
9. P32 minimal GWOSC HDF5 ringdown likelihood builder when strain files and h5py are available.
10. P40/P41 minimal public likelihood builders for BB bandpower rows and LHCb q2/value/error rows.

## Scientific policy

No manual filling is accepted. If public/cached data cannot support a statistic, the test returns parser-blocked, fit-missing, likelihood-missing, or control-tension. v61 should not promote by template or by generated placeholder artifact.

## Expected current behavior

Without additional public/cached source files available, P36 high-z, P33, PTA, P32, P40, and P41 may remain blocked. The difference from v60 is that v61 now contains executable fetch/parse/estimate code paths rather than just artifact normalization.
