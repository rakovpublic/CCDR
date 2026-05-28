# Round 10 v35 patch notes

Purpose: fix the v34 P30 execution blocker without weakening the claim gate.

Implemented improvements:

1. Memory-safe P30 curl-projected residual execution.
2. Sequential map sampling to avoid holding science and curl full-sky maps together.
3. Lower-Nside projection candidates, configurable with `CCDR_P30_PROJECTION_NSIDE`.
4. Mandatory residual-test JSON output on success or failure.
5. Residual delta, projected delta, beta coefficient, p-value, paired bootstrap, residual-curl null, and patch jackknife fields.
6. Route-specific P30-SDSS-core promotion gate.
7. Global P30 remains blocked until a second route passes.
8. P3 endpoint hard skip when metadata lacks endpoint/node-pair columns.
9. P36/P41/P33/P32 strict measurement contracts refreshed as v35 gates.
10. Dashboard v35 instrumentation.

Claim policy: do not promote P30 unless the v35 residual route gate passes. Even then, promote only `P30-SDSS-core`, not global P30.
