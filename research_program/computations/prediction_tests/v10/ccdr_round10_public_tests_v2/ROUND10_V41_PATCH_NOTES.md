# Round-10 v41 patch: measurement-first confirm hardening

v41 keeps the P30 route-specific policy but concentrates on the highest-value
blocked confirmations:

1. P30 route-specific gate is separated from global P30; `global_scope` no longer
   blocks the P30-SDSS-core route gate, while global P30 still requires a second
   independent route.
2. P30 residual curl-null is explicitly patch-based; object-level curl nulls are
   treated as over-counted diagnostics only.
3. P30 adds leave-one-patch-out residual stability accounting.
4. P30 family-specific quarantine remains required before any route promotion.
5. P30 writes a persistent empirical finite-support mask product and requires all
   variants/randoms to use one mask product for publication-grade claims.
6. P3 endpoint hard-skip policy remains in force.
7. P36 high-z now has a real source-specific object catalogue parser/consumer for
   KMOS3D/KGES/KROSS/MOSDEF/SAMI-style products, writing strict Vrot/R/z rows to
   `outputs/p36_highz_real_object_catalogue_rows_v41.json`.
8. P41 now has a numerical Wilson/SM likelihood consumer that scans JSON/CSV rows
   for q²/value/error/SM/Wilson predictions and writes
   `outputs/p41_wilson_sm_likelihood_rows_v41.json`.
9. P33 now consumes real density-split alpha products and writes
   `outputs/p33_density_split_alpha_measurement_v41.json`.
10. P32 now consumes one-event GW150914 strain-result products and writes
   `outputs/p32_gw150914_one_event_strain_result_v41.json`.

Claim policy stays strict: no proxy/contract-only result can be promoted. P30 can
only become `p30_sdss_core_route_confirm_like`; global P30 still needs a second
independent route.
