# v63 implementation report

## Goal

Implement the 10 requested improvements as real test/parser/computation behavior, not interface-only changes.

## Summary

v63 appends a new compatibility layer to `ccdr_r10_common.py` and updates `run_all.py` to stamp v63 run IDs. It does not add fillable templates. If public/cached data are insufficient, tests return structured blockers.

## Behavior-changing areas

- **P36 high-z** now performs source-targeted public/cache fetches, source-specific column mapping, strict radius provenance, row streaming, and source-level bootstrap gates.
- **P30** now adds predeclared patch rejection and redshift-density confounding proxy to the same-mask/curl route resolver.
- **P33** now attempts DESI LSS public endpoint discovery and runs a lightweight real density-split pair-histogram alpha proxy when RA/DEC/Z rows are available.
- **PTA** now attempts to parse residual-like public/cache tables and reports a statistic proxy when possible.
- **P32/P40/P41** keep strict no-manual gates and refine claim blockers against actual artifacts.

## Expected result

The next full run may still not increase confirmations if public endpoints/cached tables are insufficient, but failures should move toward concrete parser/estimator blockers rather than generic scaffolding.

## Validation results

Compilation passed:

```text
python -m py_compile ccdr_r10_common.py run_all.py tests/*.py
```

Targeted quick statuses:

```text
T13 -> highz_a0_public_rows_gate_failed_v63
T14 -> highz_a0_public_rows_gate_failed_v63
T04 -> density_kappa_same_mask_route_blocked_v63
T07 -> p33_density_bao_alpha_measurement_required_v63
T17 -> pta_density_cross_data_availability_blocked
T19 -> ringdown_strain_analysis_required
T21 -> p40_bb_likelihood_required
T31 -> p41_q2_likelihood_gate_ready
T51 -> dashboard_positive_current_only_v63
```
