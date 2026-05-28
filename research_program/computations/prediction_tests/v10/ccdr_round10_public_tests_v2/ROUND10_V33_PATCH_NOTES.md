# Round-10 v33 confirm-target patch

Focus: P30 curl-clean route diagnostics and measurement-consumer hardening.

Implemented improvements:

1. P30 curl-clean core science-family split: baseline/f150/tonly are tracked separately from f090/cibdeproj.
2. P30 science-vs-curl likelihood/sign diagnostic: reports how many science variants exceed |curl| and a low-N sign-tail p.
3. P30 curl-template regression/projection: with `--allow-large`, samples science and curl maps at the same SDSS coordinates and reports residual/projection deltas.
4. P30 curl-orthogonal projection diagnostics: reports whether core variants remain positive after projecting out curl.
5. P30 independent quadrant/stripe rotation controls: with `--allow-large`, rotates coordinates within RA/Dec quadrants and tests whether the curl effect weakens.
6. P30 Euclid quarantine remains explicit; global P30 stays blocked until a second repaired route passes.
7. P36 high-z source-specific parser plan: strict object-level Vrot/R/z table schema remains required.
8. P41 Wilson/SM Δχ² contract: major claim requires numerical q² likelihood, not pattern counts.
9. P33 first real alpha-split contract: requires alpha_high/alpha_low plus covariance/nulls/redshift jackknife.
10. P32 one-event strain contract: GW150914-first minimal strain run requirements are explicit.

Claim policy:

- P30-SDSS remains route-specific and cannot become global P30 without a second independent repaired route.
- Curl-contaminated P30 routes remain near-confirm only.
- Curl subtraction/projection diagnostics are not sufficient by themselves for a publication-grade confirmation.
