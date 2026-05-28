# Round-10 v49 confirm-path hardening

Implemented 10 improvements from the v48 result report:

1. Dashboard now counts current per-test JSON outputs only by default; external/reference summary conflict checks are opt-in via `CCDR_DASHBOARD_COMPARE_REFERENCES=1`.
2. `run_all.py` cleans stale `outputs/test*.json` and `outputs/round10_summary.json` at start unless `CCDR_KEEP_PREVIOUS_OUTPUTS=1`, and stamps `current_run_id_v49`.
3. P36 high-z adds a raw-catalogue/source-quality contract: source files must not come from outputs/summary/guard files; rows need hashes, units, source stability, and tiny-radius/extreme-ratio domination checks.
4. P30 adds a global confirm gate requiring v48 mask/curl/patch pass plus two positive variants and an independent second route beyond SDSS.
5. P33 adds an explicit alpha_high/alpha_low effect contract with DESI randoms, covariance, sky/label shuffles, redshift jackknife, and predeclared sign.
6. P8/P8c require source-hashed pulsar coordinates, residual/TOA weights, kappa samples, weighted statistic, and sky-shuffle null.
7. P32 requires strain-level likelihood evidence: PSD, GR fit, CCDR residual fit, Δχ²/AIC, injection nulls, detector split, leave-one-event-out.
8. P40 requires BB bandpowers, covariance, foreground controls, template amplitude+uncertainty, and Planck/BK18 cross-check.
9. P41 requires q²/value/error rows, sign basis, source hashes, SM-vs-Wilson Δχ² >= 9, CP null, and observable-bin jackknife.
10. Direct detection and SMD claims are separated: exclusion-curve overlap can confirm coverage only; detection needs event-level likelihood. SMD constants remain consistency checks until preregistered derivation gates pass.

Run example:

```powershell
python run_all.py --allow-large --max-mb 80000 --script-timeout 720000
```

For dashboard reference comparison only:

```powershell
$env:CCDR_DASHBOARD_COMPARE_REFERENCES=1
python tests/test51_round10_joint_dashboard.py
```
