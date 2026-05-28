# v64 implementation report

## Validation

Compiled successfully with:

```bash
python -m py_compile ccdr_r10_common.py run_all.py tests/*.py
```

Targeted quick checks emitted v64 fields:

```text
T13 -> highz_a0_public_rows_gate_failed_v64
T14 -> highz_a0_public_rows_gate_failed_v64
T04 -> density_kappa_same_mask_route_blocked_v64
T07 -> p33_density_bao_alpha_measurement_required_v64
T17 -> pta_density_cross_data_availability_blocked with pta_weighted_kappa_residual_gate_v64
T19 -> ringdown_strain_analysis_required with p32_strain_likelihood_gate_v64
T21/T22 -> p40_bb_likelihood_required with p40_bb_likelihood_gate_v64
T31 -> p41_q2_likelihood_gate_ready with p41_q2_wilson_likelihood_gate_v64
T51 -> dashboard_positive_current_only_v64
```

## What changed materially

- P36 now separates claim rows from discovery rows and requires physical-radius provenance for claim rows.
- P36 has a deeper CDS/VizieR parser and source-2-targeted URL set.
- P30 no longer only reports old curl tension: it recomputes route diagnostics after pre-sign patch rejection.
- P33 now attempts exact DESI LSS/random downloads and constructs a pair-histogram alpha proxy when RA/DEC/Z rows are available.
- PTA/P40/P41 now parse public rows for actual statistic/likelihood quantities instead of only validating artifact fields.

## Scientific caveat

v64 does not fake confirmations. If a confirm gate still fails, it means the new computation could not satisfy the current strict public-data claim criteria.
