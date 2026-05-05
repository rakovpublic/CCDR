# Round-10 v14: 10 confirmation-squeeze improvements

Generated: 2026-05-05T15:51:22Z

Implemented:
1. Fixed CL2 NameError by adding local NANOGrav RA/DEC parser and full .par coordinate extraction.
2. Added manual ACT ALM reconstruction from FITS index/real/imag columns.
3. Sanitized ALM values and reports finite/nonfinite ALM/map diagnostics.
4. P30 confirmation rules: confirm_like only if real catalogue statistic has delta>0, sky p<=0.05, density-shuffle p<=0.05, and jackknife same sign.
5. P30 fallback science path: Euclid mer_catalogue first, SDSS galaxy coordinates second, sampler-validation separated from evidence.
6. P3 CDS ReadMe byte-by-byte parser; endpoint evidence requires explicit endpoint/node coordinate columns.
7. P41 CDS/PDF/attachment guard; evidence requires numeric observable values plus sign/operator-basis terms.
8. High-z a0 KROSS/KMOS Vrot^2/R parser path; SIG^2/R remains excluded from evidence.
9. Direct-detection measured mass-window coverage remains coverage-confirmation only, not detection.
10. Dashboard suite_status_v14 and confirmation-upgrade policy: confirm_like vs compatible vs ready vs guarded.
