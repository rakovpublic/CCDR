# Round-10 v36 confirm-target patch

Focus: P30-SDSS-core projection consistency.

Implemented improvements:

1. Projection consistency assertion: v35 raw pre-projection deltas must match canonical v29 same-split deltas before residual projection can count.
2. Canonical coordinate/density-label manifest with coordinate hashes, label hashes, thresholds, and sample labels.
3. Canonical ACT map sign convention manifest with low-Nside sanity stats and checksums.
4. Nside raw-delta comparison across 256/128/64 for baseline/f150/tonly/curl.
5. Reference-vs-sequential projection audit using v29/v34/v35 artifacts.
6. Pixel-level sample audit: first high/low rows include RA/Dec, density label, pixel id, and map values.
7. Residual confirm gate is blocked unless projection consistency and Nside stability pass.
8. P3 endpoint hard skip is preserved as v36 policy.
9. P36/P41/P33/P32 measurement gates are carried forward and stricter about real measurements.
10. Dashboard v36 reports P30 projection mismatch separately from physics failure.

Claim policy: if the projection raw deltas do not reproduce the canonical same-split route, label P30 as `projection_pipeline_mismatch`, not as physics evidence.
