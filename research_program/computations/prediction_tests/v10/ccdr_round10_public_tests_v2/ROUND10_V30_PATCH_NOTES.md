# Round-10 v30 confirm-target patch

Implemented after the v29 report. Main goal: turn P30-SDSS from a near-confirm into a quantitatively auditable route-specific confirm candidate while keeping global P30 strict.

## Implemented improvements

1. P30 curl diagnostics: curl/science ratio, p-value, sign relation, and JSON artifact.
2. Quantitative curl pass rule: curl must be weaker than 0.60× median science effect and non-significant/extreme.
3. P30-SDSS near-confirm bucket: route-specific candidate separated from global P30.
4. P30 bootstrap CI/effect size artifact for SDSS random-normalized split.
5. Euclid quarantine/repair policy until photo-z/depth/quality cuts and field-level randoms pass.
6. P36 dedicated source-specific Vrot/R/z object parser aliases.
7. P36 strict object-level acceleration output v30.
8. P41 q2/value/error extraction plus Wilson/SM proxy likelihood consumer.
9. P33 measured density-split BAO-alpha consumer and publication gate.
10. P32 minimal strain-run manifest and strict strain-result gate.

## Claim policy

- `P30-SDSS_route_near_confirm_candidate` is allowed when random-normalized route, same-split science variants, and jackknife stability pass.
- `p30_sdss_route_confirm_like` requires the above plus quantitative curl pass and bootstrap CI positive or pending.
- Global P30 still requires a second independent route.
