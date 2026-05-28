# Round-10 v38 confirm-target patch

Goal: fix the P30-SDSS projection sampler mismatch enough to make the next run diagnosable, while preserving strict claim discipline.

Implemented improvements:

1. P30 row-by-row sampler audit for canonical high/low objects.
2. P30 high/low inversion test comparing v35 projection raw deltas against v29/v34 canonical deltas.
3. Projection samplers are required to consume `p30_canonical_split_manifest_v37.json`; no recomputed labels are claim-usable.
4. Canonical object-order lock with row/order/label hashes.
5. Direct `healpy.ang2pix` theta/phi vs lonlat pixel-function comparison.
6. Projection-disabled guard: residual projection is disabled for claims until raw consistency passes.
7. P30-SDSS random-normalized positive route is preserved separately as a route-specific candidate, not merged with the projection path.
8. P36 high-z remains strict: only real source-specific object catalogues with `Vrot/R/z` can promote.
9. P41 remains strict: requires q²/value/error rows plus Wilson/SM Δχ² and CP-control nulls.
10. P33/P32 remain strict: first real alpha split and first GW150914 strain product are required.

Expected status after v38 if the sampler is still inconsistent:

```text
P30 = density_kappa_sdss_core_projection_pipeline_mismatch
```

This is not a physics result; it is a reproducibility guard.
