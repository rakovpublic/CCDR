# Round-10 v32 confirm-target hardening patch

v32 is a targeted follow-up to the v31 report.  It does not relax claim discipline; it adds explicit remediation diagnostics for the remaining near-confirm blockers.

Implemented improvements:

1. P30 curl-remediation report: writes `outputs/p30_curl_remediation_v32.json`.
2. P30 curl permutation/rotation controls: with `--allow-large`, recomputes curl label-shuffle and sky-rotation controls on the SDSS random-normalized split.
3. P30 science/curl ratio uncertainty: consumes `p30_sdss_bootstrap_ci_v30.json` and reports conservative ratio CI when bootstrap data are available.
4. P30 curl-subtracted diagnostics: reports `science_delta - curl_delta` and `science_delta - abs(curl_delta)` per science variant; diagnostic only.
5. P30 frequency/systematic family split: groups f090/f150, baseline/tonly, and cibdeproj to identify whether the signal is broad or map-family-specific.
6. P30 route-specific promotion policy: `P30-SDSS_route_confirm_like` may only appear if random-normalized route, same-split variants, bootstrap CI, and curl remediation pass. Global P30 still requires a second route.
7. P36 high-z object parser manifest: writes `outputs/p36_highz_source_specific_parser_manifest_v32.json` and keeps strict Vrot/R/z object-table gates.
8. P41 Wilson/SM contract: writes `outputs/p41_major_claim_contract_v32.json` and keeps major claim blocked without q2 numeric rows, CP controls, and Wilson/SM likelihood.
9. P33 measured BAO-alpha contract: writes `outputs/p33_alpha_measurement_contract_v32.json` and blocks publication confirm without measured alpha_high/alpha_low and nulls.
10. P32 minimal strain-run contract: writes `outputs/p32_minimal_strain_run_contract_v32.json` and keeps ringdown confirm blocked without actual strain/PSD/GR/CCDR/injection/split products.

Recommended P30 command:

```powershell
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
```

Inspect after run:

```text
outputs/p30_curl_remediation_v32.json
outputs/p30_curl_diagnostics_v31.json
outputs/p30_sdss_same_split_variant_rerun_v29.json
outputs/p30_sdss_bootstrap_ci_v30.json
outputs/test04_p30_act_dr6_lensing_inventory.json
```
