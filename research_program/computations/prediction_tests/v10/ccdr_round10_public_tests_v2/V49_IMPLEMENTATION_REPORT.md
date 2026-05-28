# V49 implementation report

Implemented the requested 10 improvements as a new v49 patch layer on top of v48.

## Implemented changes

1. `run_all.py` now cleans stale `outputs/test*.json` and `outputs/round10_summary.json` at run start by default and stamps each row with `current_run_id_v49`.
2. `test51_round10_joint_dashboard.py` now uses current per-test JSON outputs only by default. External/reference summary comparisons are opt-in via `CCDR_DASHBOARD_COMPARE_REFERENCES=1`.
3. P36 high-z T13/T14 now add `p36_highz_raw_catalogue_contract_v49`, blocking promotion if rows come from outputs/summaries/guard files or are dominated by tiny-radius/extreme-ratio rows.
4. P30 T04 now adds `p30_global_confirm_gate_v49`, requiring v48 mask/curl/patch pass plus two positive science variants and an independent second route beyond SDSS.
5. P33 T07 now adds `p33_alpha_effect_confirm_gate_v49`, requiring numeric alpha_high/alpha_low, DESI randoms, covariance-aware fit, sky shuffle, density-label shuffle, redshift jackknife, and predeclared sign.
6. P8/P8c T16/T17 now add stricter weighted-statistic gates requiring source-hashed pulsar coordinates, residual/TOA weights, kappa samples, weighted statistic, sky-shuffle null, and sign contract.
7. P32 T19 now adds `p32_strain_likelihood_confirm_gate_v49`, requiring strain-level likelihood evidence before confirmation.
8. P40 T21/T22 now add `p40_bb_likelihood_confirm_gate_v49`, requiring BB bandpowers, covariance, foreground controls, template amplitude+uncertainty, and Planck/BK18 cross-check.
9. P41 T31/T32 now add `p41_major_likelihood_confirm_gate_v49`, requiring q² rows, sign basis, source hashes, Δχ² >= 9, CP null, and observable-bin jackknife.
10. Direct detection T25-T29 and SMD T34-T38+ now separate coverage/consistency from detection/derivation confirmation.

## Validation performed in sandbox

- `python -m py_compile ccdr_r10_common.py run_all.py tests/*.py` passed.
- Quick targeted checks were run for P30, P33, P8/P8c, P32, P40, P41, direct detection, SMD, and dashboard.
- Dashboard no longer reports source conflicts by default; reference conflict mode is opt-in.

## Important note

This zip intentionally does not include fresh scientific result outputs. Run locally with:

```powershell
python run_all.py --allow-large --max-mb 80000 --script-timeout 720000
```

Then upload `outputs/round10_summary.json` and `outputs/test51_round10_joint_dashboard.json` for analysis.
