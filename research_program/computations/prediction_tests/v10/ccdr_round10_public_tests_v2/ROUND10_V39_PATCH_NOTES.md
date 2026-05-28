# Round-10 v39 patch notes

## Purpose
v39 directly addresses the v38 diagnosis that P30 projection was likely applying high/low density labels backwards. The patch does not broaden claim gates; it makes the projection path reproduce the canonical same-split route before any residual curl projection can be interpreted.

## Implemented improvements
1. Fix/test P30 H/L mask application in projection sampler.
2. Emit dual-sign raw deltas: H−L and L−H for baseline/f150/tonly/curl.
3. Add regression assertion against canonical v29 route deltas.
4. Force projection to consume canonical manifest rows only.
5. Preserve manifest row order and use aligned boolean masks.
6. Rerun curl projection only after raw delta consistency passes.
7. Keep promotion route-specific: P30-SDSS-core, not global P30.
8. Keep P36 high-z strict source-specific object catalogue gate.
9. Keep P41 Wilson/SM likelihood and CP-control gate.
10. Keep P33/P32 first-measurement gates.

## New P30 outputs
- `outputs/p30_projection_label_fix_test_v39.json`
- `outputs/p30_fixed_curl_projection_v39.json`
- `p30_projection_raw_delta_assertion_v39` in the P30 test JSON
- `p30_sdss_core_route_confirm_gate_v39` in the P30 test JSON

## Claim policy
If raw deltas do not reproduce canonical positives after the H/L orientation fix, P30 remains `projection_pipeline_mismatch`. If the fix works and residual gates pass, the only allowed claim is `P30-SDSS-core route confirm-like`, not global P30.
