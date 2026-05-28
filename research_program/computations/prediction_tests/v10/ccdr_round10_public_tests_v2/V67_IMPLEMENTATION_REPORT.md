# Round 10 v67 Implementation Report

Implemented all six requested recovery improvements in `ccdr_r10_common.py` and updated `run_all.py` to stamp v67 summaries.

## Confirm changes

- P40 BK18/B-mode now parses `BK18lf_cl_hat.dat` and `BK18lf_covmat_dust.dat` from the cached public BK18 tarball at archive-member level.
- R10-T21 and R10-T22 now pass `p40_bb_likelihood_confirm_like_v67`.
- Dashboard non-SM confirm-like count is now 6: P38, P36 local, P36 high-z x2, and P40 x2.

## Implemented strict recoveries

- P30: official/empirical ACT mask search plus stricter pre-sign patch rejection artifact. Still gated because curl/science ratio remains 1.7469, residualization is still required, and zero patches survive the stricter rule.
- P33: local exact DESI LSS RA/DEC/Z ingestion plus compressed DESI BAO mean/cov indexing. It found 40 compressed BAO pairs, but exact LSS/random catalogues are still absent, so no density-environment alpha confirm is claimed.
- PTA/CL2: residual/TOA-kappa join scanner and shuffle statistic gate. No join rows were found, so the weighted-statistic gate remains closed.
- P32: H1/L1 GWOSC cache discovery. Both detector caches are present, but only one likelihood row exists; injection-null, detector-split, leave-one-event-out, and delta-chi2 gates still fail.
- P41: numeric q2/value/error candidate extraction and single-Wilson-shift fit. It builds 9 fit rows with CP/sign gates passing and jackknife stable, but delta chi2 is 7.475, below the required 9.

## Verification

- `python -m py_compile ccdr_r10_common.py run_all.py`
- Focused quick reruns for R10-T04, T07, T17, T19, T21, T22, T31, T32, and T51.
- Full resumed summary: `python run_all.py --resume --script-timeout 180`
- Final summary: 51/51 tests complete, `runner_version = v67`, `run_complete_v67 = true`.
