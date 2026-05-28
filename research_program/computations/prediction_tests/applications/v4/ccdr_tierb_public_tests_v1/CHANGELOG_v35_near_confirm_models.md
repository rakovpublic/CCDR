# v35 near-confirm models and endpoint fixes

Implements the nine requested improvements after v34:

1. T44 NAND normalization + first layer-vs-year model (`t44_nand_normalized_rows_v35.csv`).
2. T48b NREL/NLR XLSX parser repair and descriptor-model row writer (`t48b_pv_descriptor_auto_rows_v35.csv`).
3. T53 ProteinGym residual-model attempt using auto-join rows (`t53_residual_model_input_rows_v35.csv`).
4. T31/T32/T33 decisive microstructure scoring (`*_decisive_microstructure_rows_v35.csv`).
5. Fusion structured-file-only funnel that rejects OSF/Zenodo metadata frames as physical tables.
6. T57/T59 official HEPData record/submission/table endpoint fallback.
7. All-tests fallback JSON restoration so T34 and any timed-out/missing-output test is represented in dashboards.
8. T50-T52 hard bound-only rule preserved.
9. v35 `near_confirm_score` dashboard and ranked blockers.

Run all tests:

```powershell
python .\run_all_tier_b.py --cache tierb_cache_v35 --outdir tierb_out_v35 --max-papers 40 --max-tables 120 --script-timeout 900
```

Check:

```powershell
notepad .\tierb_out_v35\positive_dashboard.json
```

Use only `v35_confirm_status.strict_confirm_allowed_now` for confirmation language.
