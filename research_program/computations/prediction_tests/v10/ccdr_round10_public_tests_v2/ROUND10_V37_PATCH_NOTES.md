# Round 10 v37 confirm hardening patch

Focus: repair the P30 v36 canonical split manifest crash and make the P30 curl-projected residual route reproducible before it can ever be promoted.

Implemented changes:

1. Fixed the canonical split manifest bug caused by NumPy array truth-value checks.
2. Made the canonical split manifest mandatory for P30 projection interpretation.
3. Stored exact per-object P30 labels: object_id, RA, Dec, density label, density value, density pixel, data count, random count, and scaled random expectation.
4. Added v29/v34/v35 raw-delta consistency checks before residual fitting is allowed to count.
5. Added canonical-label Nside raw-delta comparison at Nside 256, 128, and 64.
6. Added map-sign convention/checksum diagnostics for baseline, f150, tonly, and curl.
7. Added pixel-level sample audit using canonical labels only.
8. Kept P3 endpoint hard skip active to avoid giant non-endpoint downloads.
9. Kept P36/P41/P33/P32 strict measurement gates: no promotion without object-level rows, Wilson/SM likelihood, alpha split, or strain products.
10. Added dashboard_v37 instrumentation and blocked-gate accounting.

Claim policy:

- If `p30_canonical_split_manifest_v37.available == false`, P30 projection is pipeline-debug only.
- If `p30_projection_consistency_audit_v37.raw_projection_consistency_pass == false`, P30 projection is labelled `projection_pipeline_mismatch`, not physics evidence.
- If all P30 v37 gates pass, the maximum permitted claim is `P30-SDSS-core route confirm-like`, not global P30.
