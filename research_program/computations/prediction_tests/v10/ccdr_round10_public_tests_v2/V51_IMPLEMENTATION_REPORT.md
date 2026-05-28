# CCDR Round-10 v51 implementation report

Implemented as a source-level patch to `ccdr_r10_common.py` and `run_all.py` on top of v50.

## Implemented files

- `ccdr_r10_common.py`: appended v51 runner overrides and measurement-template generators.
- `run_all.py`: replaced with v51 run-completeness aware runner.
- `ROUND10_V51_PATCH_NOTES.md`: this patch note.
- `V51_IMPLEMENTATION_REPORT.md`: implementation summary.

## New/expected input artifacts

These are generated automatically on first run if missing:

- `inputs/p36_highz_object_rows_raw_template_v51.csv`
- `inputs/p30_patch_protocol_v51_template.json`
- `measurements/p30_independent_route_measurement_template_v51.json`
- `measurements/p33_alpha_measurement_template_v51.json`
- `measurements/p38_void_measurement_template_v51.json`
- `measurements/pta_weighted_kappa_residual_template_v51.json`
- `measurements/p32_strain_likelihood_template_v51.json`
- `measurements/p40_bb_likelihood_template_v51.json`
- `measurements/p41_q2_wilson_likelihood_template_v51.json`
- `measurements/smd_derivation_predictions_template_v51.json`

## Validation

The patched bundle was checked with:

```bash
python -m py_compile ccdr_r10_common.py run_all.py tests/*.py
```

Targeted quick checks were run for P38, P36 high-z, P30, P33, and the dashboard. They returned structured JSON and emitted v51 gate fields/templates.

## Expected next-run behavior

- Partial runs are no longer ambiguous: `run_complete_v51=false` and `missing_tests_v51` are written.
- P38 should no longer silently appear as a lost result; it writes recovery requirements and can ingest a measurement file.
- P36 high-z remains blocked until raw object rows are provided or parsed, with large-radius quality passing.
- P30 remains blocked until same-mask/curl/patch protocol and independent route pass.
- P33 remains blocked until a real alpha high/low measurement artifact exists.
