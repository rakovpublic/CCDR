# v61 implementation report

## Validation performed

- `python -m py_compile ccdr_r10_common.py run_all.py tests/*.py` passed.
- Targeted quick checks passed:
  - T13 P36 high-z returns `highz_a0_public_rows_gate_failed_v61` with `p36_highz_public_parser_v61` and row-provenance gate fields.
  - T04 P30 returns `density_kappa_same_mask_route_blocked_v61` with `p30_same_mask_recompute_gate_v61` and control-tension classes.
  - T07 P33 returns `p33_density_bao_alpha_measurement_required_v61` with `p33_alpha_autofit_v61`.
  - T19 P32 emits `p32_strain_likelihood_gate_v61`.
  - T21 P40 emits `p40_bb_likelihood_gate_v61`.
  - T31 P41 emits `p41_q2_wilson_likelihood_gate_v61`.
  - T51 dashboard emits `dashboard_v61`.

## Important limitations

The v61 code paths are real, but they still depend on public endpoints/cached files being reachable and schema-compatible. No result is promoted unless the strict gate passes. In quick mode, network fetching is skipped to keep validation fast.

## No-manual guarantee

The v61 active claim paths do not require the user to fill CSV/JSON files. Manual, template, fill, example, summary, rejected, and reanalysis paths are ignored for claim-building.
