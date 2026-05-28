# Round-10 v60 patch notes — no-manual public-parser confirm recovery

v60 implements the 10 requested improvements from the v59 report while preserving the rule that tests must not require manual file filling.

## Main changes

1. **P36 high-z source-targeted public fetchers**
   - Added best-effort public/CDS/VizieR-style candidate downloads for KROSS, KGES, KMOS3D, SAMI, MOSDEF, PHIBSS, and related high-z kinematic catalogues.
   - Downloads are cached under `.ccdr_round10_cache/v60_public_fetches/`.
   - Network failure is non-fatal and becomes parser/public-data-limited.

2. **P36 high-z source-specific column maps**
   - Added alias maps for velocity, radius, redshift, object ID, and inclination.
   - Handles CSV/TSV/TXT/DAT/JSON/HTML/FITS when dependencies permit.

3. **P36 unit provenance and strict claim gate**
   - Added row-level radius unit policy: explicit kpc, explicit pc-to-kpc, large-value pc-to-kpc, safe assumed-kpc range, or ambiguous rejection.
   - Strict claim gate still requires large-radius rows, two source groups, per-source counts, low tiny-radius fraction, and median acceleration above local a0.

4. **P30 same-mask recomputation diagnostics**
   - Removed reliance on manual active protocol files for the current claim path.
   - Adds same-run current-output route gate using science/curl/variant metrics.

5. **P30 curl/variant tension audit**
   - Adds explicit `control_tension` classification for curl dominance, variant sign flip, and field-jackknife instability.

6. **P30 route-specific sign policy support**
   - Keeps route confirmation separate from global confirmation and records route-specific diagnostics.

7. **P33 automated alpha artifact finder**
   - Searches public/cache outputs for real density-split alpha artifacts and normalizes fields.
   - Does not accept fill/template/example/manual artifacts.
   - Missing alpha fit remains `p33_density_bao_alpha_measurement_required_v60`.

8. **PTA/P32/P40/P41 no-manual gates**
   - Adds stricter v60 gates for weighted PTA statistic, strain likelihood, BB likelihood, and q²/Wilson likelihood.

9. **SMD no-target-fit derivation gate**
   - Keeps SMD constants as consistency confirms only unless a real no-target-fit derivation artifact exists.

10. **Dashboard why-not-confirm classes**
   - Adds `dashboard_v60` with `why_not_confirm_class_counts` and a confirm-recovery priority list.
   - Legacy manual/template dashboard sections are removed from the current dashboard output.

## Expected behavior

v60 may still produce the same confirm count as v59 unless public/cached data are sufficient. That is intentional. The patch improves automatic public parsing, diagnostics, and no-manual policy enforcement; it does not create confirms from incomplete data.

## Run

```powershell
python run_all.py --allow-large --max-mb 80000 --script-timeout 720000
```

Send after run:

```text
outputs/round10_summary.json
outputs/test51_round10_joint_dashboard.json
```
