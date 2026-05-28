# Round 10 v31 confirm-target patch

This patch implements the 10 next-step improvements from the v30 report.

## Main fixes

1. Fixes P30 curl diagnostics population: v30 read `variants`, but the v29 same-split artifact writes `variant_results`.
2. Computes `curl_abs_delta / median_abs_science_delta` from the same-split artifact.
3. Adds a strict curl p-value gate: `0.05 < p_high < 0.95`.
4. Adds `p30_sdss_route_confirm_gate_v31` and keeps global P30 separate from P30-SDSS.
5. Keeps Euclid quarantined until photo-z/depth/quality and field-random repairs pass.
6. Adds a stronger P36 object-level product consumer for source-specific Vrot/R/z tables.
7. Adds P41 JSON/CSV q² likelihood consumer and Wilson/SM proxy gate.
8. Adds P33 JSON/CSV alpha measurement consumer.
9. Adds P32 minimal strain-product consumer and explicit missing-step manifest.
10. Adds `dashboard_v31` instrumentation.

## Claim policy

P30-SDSS may only promote to route-specific confirm-like if random-normalized route, same-split variants, jackknife, bootstrap, and curl gates pass. Global P30 still needs a second independent route.
