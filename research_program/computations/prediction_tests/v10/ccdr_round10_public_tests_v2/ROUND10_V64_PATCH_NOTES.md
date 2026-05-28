# Round-10 v64 deep behavior patch

This patch is intentionally not an interface/dashboard-only update. It changes test behavior, parsers, estimators, and computations.

## Deep changes

1. P36 high-z source-2 parsing: adds a v64 source-targeted parser path for KGES/KMOS3D/SAMI/MOSDEF/PHIBSS/KROSS with improved CDS/VizieR ASCII/TSV/FITS parsing.
2. P36 physical-radius claim mode: claim rows now require a physical-radius whitelist (`R_kpc`, `R_e`, `R_d`, `R_turn`, `R6D`, etc.) and reject generic `radius/size/pixel` columns.
3. P36 revalidates prior auto-public rows under v64 rules and separates discovery rows from claim rows so tiny proxy radii no longer poison claim-mode denominators.
4. P36 writes streamed CSV/JSONL artifacts and computes source-level bootstrap/leave-one-source-out medians.
5. P30 recomputes science/curl/variant summaries after pre-sign patch rejection based on shared science/curl jackknife vectors.
6. P30 adds a redshift-density/field-confounding residualization proxy and only promotes if curl is weaker and variants agree after rejection.
7. P33 adds exact DESI LSS/random endpoint discovery and a pair-histogram density-split BAO alpha proxy with density-label shuffles and redshift jackknife where possible.
8. PTA adds a v64 weighted residual×kappa statistic path if public residual/TOA and kappa-sample rows exist.
9. P40 adds a public BB-bandpower parser and minimal inverse-variance template amplitude estimator.
10. P41 adds a q²/value/error parser and crude SM-vs-shifted-Wilson likelihood proxy.

## No manual filling

No tests ask the user to fill files. If public/cached data are absent or insufficient, gates return parser/data/likelihood blockers.

## Expected conservative behavior

If public data are insufficient, the expected statuses remain conservative:

- `highz_a0_public_rows_gate_failed_v64`
- `density_kappa_same_mask_route_blocked_v64`
- `p33_density_bao_alpha_measurement_required_v64`
- `p40_bb_likelihood_required`
- `p41_q2_likelihood_gate_ready`
- `dashboard_positive_current_only_v64`
