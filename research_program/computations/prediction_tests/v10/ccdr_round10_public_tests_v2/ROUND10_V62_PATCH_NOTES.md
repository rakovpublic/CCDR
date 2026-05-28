# Round-10 v62 patch notes — behavior-changing confirm recovery

v62 continues the no-manual-fill rule and changes actual test behavior, not only dashboard/interface labels.

## Implemented behavior changes

1. **P36 T14 MemoryError fix**
   - P36 high-z rows are now written as external JSONL/CSV streams.
   - Test JSON stores only summary statistics, file paths, and hashes.
   - This avoids `json.dumps()` over huge row payloads.

2. **P36 second-source harvest**
   - v62 scans prior auto-generated public/cache P36 CSV/JSON/JSONL rows and source-targeted downloads.
   - It preserves `source_group` when present and infers KROSS/KGES/KMOS3D/SAMI/MOSDEF/PHIBSS from row metadata or path.

3. **P36 stricter radius/source gate**
   - Claim gate still requires >=30 trusted rows, >=30 large-radius rows, >=2 source groups, >=20 large-radius rows/source for two sources, tiny-radius fraction <=20%, and median large-radius acceleration above local a0.

4. **P36 row provenance**
   - Accepted rows carry source file hash, raw source path, original column mapping, and unit-conversion method.

5. **P30 deeper control-tension resolver**
   - Adds patch imbalance, science/curl correlation, mask-edge proxy, galactic-cut proxy, and field/redshift confounding proxy.
   - Keeps route confirmation blocked when curl/variant/patch controls fail.

6. **P30 route-specific policy**
   - Makes SDSS positive high-minus-low sign policy explicit before route aggregation.

7. **P33 automated alpha proxy**
   - Searches public/cache RA/DEC/Z catalogues and runs a density-split pair-distance BAO alpha proxy when enough rows exist.
   - Keeps claim gate strict: covariance, DESI randoms, shuffles, and jackknife are still required for confirmation.

8. **PTA gate cleanup**
   - Emits a v62 no-manual weighted-statistic gate with exact missing products: residual/TOA weights, kappa samples, weighted statistic, sky-shuffle p.

9. **P32/P40/P41 likelihood gates**
   - Carries forward real v61 minimal builders but adds explicit v62 behavior-change gates and publication blockers.

10. **Dashboard v62**
   - Adds `dashboard_v62`, behavior-change claim policy, v62 artifact index, and confirm-recovery priorities.

## Expected outcome

v62 should not fake confirmations. It should:
- fix T14 MemoryError,
- improve P36 row handling and source harvest,
- improve P30 control diagnosis,
- attempt P33 alpha proxy when public RA/DEC/Z rows exist,
- keep blocked tests blocked when controls/likelihoods are insufficient.
