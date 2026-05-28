# CCDR Round-10 v51 patch notes

v51 is a confirm-recovery patch based on the partial v50 run where T10/P38, T13/T14/P36 high-z, T04/P30, and T07/P33 were blocked by stricter gates.

## What changed

1. `run_all.py` now writes `round10_partial_summary_v51.json` and `current_run_progress_v51.json` after every test, so interrupted runs are clearly marked as incomplete.
2. `run_all.py` adds `run_complete_v51`, `expected_tests`, and `missing_tests_v51` to `round10_summary.json`.
3. P38 now has a v51 recovery manifest, Zenodo API candidates, cached-file audit, and an optional `p38_void_measurement_v51.json` ingest gate.
4. P36 high-z now prefers raw external object-row inputs in `inputs/p36_highz_object_rows_raw*.csv/json` and refuses output/self-ingested rows for confirm.
5. P36 high-z writes a large-radius reanalysis artifact and requires radius >= 0.5 kpc, >=30 rows, >=2 source groups, >=20 rows/source, and tiny-radius fraction <=20% for global confirm.
6. P30 now requires a predeclared `p30_patch_protocol_v51.json` to pass route confirmation; post-hoc patch quarantine is explicitly blocked.
7. P30 global confirmation now separately requires an independent Planck/Euclid route artifact: `p30_independent_route_measurement_v51.json`.
8. P33 now ingests `p33_alpha_measurement_v51.json/csv` with alpha_high/alpha_low, covariance, DESI randoms, shuffles, jackknife, and predeclared sign.
9. PTA, P32, P40, P41, and SMD gates now have v51 measurement/derivation templates and clearer next-required artifacts.
10. The dashboard now reports v51 buckets with strict semantics: non-SM confirm-like, SM constant consistency, coverage confirmed, ready/compatible, blocked/gate-failed, and failed gates.

## Claim policy

v51 does not loosen confirmation thresholds. It is designed to recover real confirms only when the required raw measurements are supplied or produced by the tests.
