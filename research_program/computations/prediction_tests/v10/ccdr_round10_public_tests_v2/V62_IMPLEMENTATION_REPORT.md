# v62 implementation report

## Validation

Passed:

```bash
python -m py_compile ccdr_r10_common.py run_all.py tests/*.py
```

Targeted quick checks passed:

```text
T13 -> highz_a0_public_rows_gate_failed_v62
T14 -> highz_a0_public_rows_gate_failed_v62  # no MemoryError in quick validation
T04 -> density_kappa_same_mask_route_blocked_v62
T07 -> p33_density_bao_alpha_measurement_required_v62
T17 -> PTA v62 gate emitted
T19 -> P32 v62 gate emitted
T21 -> P40 v62 gate emitted
T31 -> P41 v62 gate emitted
T51 -> dashboard_positive_current_only_v62
```

## Important behavior changes

- P36 high-z no longer stores large row payloads in the test JSON. It writes `measurements/p36_highz_object_rows_v62_AUTO_PUBLIC.jsonl` and `.csv`, with a compact JSON summary.
- v62 uses prior auto-generated public/cache row products as parser inputs when available; this is still no-manual because it ignores template/fill/manual paths.
- P30 resolver now computes additional control-tension metrics rather than only reporting missing gates.
- P33 now attempts a simple automated density-split BAO alpha proxy from public/cache RA/DEC/Z catalogues.

## Remaining blockers expected after v62

- P36 high-z still needs two independent source groups with >=20 large-radius rows/source and a tiny-radius fraction <=20%.
- P30 remains blocked if curl dominates or variants flip sign.
- P33 remains blocked unless public DESI-like RA/DEC/Z catalogues are parsed and covariance/random/shuffle products are available.
- PTA/P32/P40/P41 remain blocked unless public residual/strain/BB/q² likelihood products are parseable.
