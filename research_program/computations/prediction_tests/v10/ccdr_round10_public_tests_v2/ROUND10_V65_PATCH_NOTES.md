# Round-10 v65 patch notes — deep behavior patch

This patch is intentionally not interface-only. It targets the repeated v64 blockers with concrete parser/estimator behavior.

## Implemented behavior changes

1. **P36 source-2 parser: KGES and KMOS3D.** Adds exact public parser routes for KGES via VizieR/CDS `J/MNRAS/506/323/tablea1`, KMOS3D via the MPE data-release page and table/tarball links, plus KROSS retention.
2. **P36 physical-radius claim whitelist.** Claim-mode rows accept only documented physical radii (`R_kpc`, `R_e`, `R_d`, `R_turn`, `R6D`, `R2.2`, pc→kpc conversion, etc.) and reject generic `radius`/`size` proxy columns.
3. **P36 radius audit.** Writes `outputs/p36_highz_radius_claim_audit_v65.json` with accepted/rejected row counts by source and reject reasons.
4. **P36 fallback hierarchy.** Local cached/generated public rows are tried first for speed; then exact KROSS/KGES/KMOS3D endpoints are fetched unless running in `--quick` mode.
5. **P36 source bootstrap.** Adds source-level bootstrap and leave-one-source-out checks once at least two source groups exist.
6. **P30 single empirical mask recomputation.** Builds a shared empirical mask identity from all route stats and recomputes route diagnostics under the same finite-footprint/mask note.
7. **P30 pre-sign patch rejection.** Rejects patches using only count/missingness/curl-patch metrics before route promotion, then recomputes science/curl/variant deltas.
8. **P30 redshift-density residualization proxy.** Adds route sign-disagreement and jackknife-dispersion residualization checks.
9. **P33 exact DESI LSS/random parser + alpha proxy.** Adds exact public DESI DR2 LSS clustering/random endpoint attempts and a pair-histogram redshift-split alpha proxy when RA/DEC/Z rows load.
10. **PTA/P32 focused completion.** PTA now tries residual-kappa CSV joins and sky shuffles; P32 focuses on detector split/injection-null gate completion before P40/P41.

## Expected statuses when public data still do not pass gates

- `highz_a0_public_rows_gate_failed_v65`
- `density_kappa_same_mask_route_blocked_v65`
- `p33_density_bao_alpha_measurement_required_v65`
- `dashboard_positive_current_only_v65`

These are conservative failures, not fake confirms.
