# v63 behavioral/runtime exact-source patch

Implemented behavioral changes (not dashboard-only):

1. T31/T32 scans now exclude `data/generated`, `tierb_out*`, dashboards, confirm targets, and rejection diagnostics by default.
2. Added dtype-safe/chunked table reader (`dtype=str`, chunking, numeric conversion only in normalizers) to reduce mixed-type CSV warnings and timeout pressure.
3. T31/T32 now use exact-source directories/manifests only unless `CCDR_V63_ALLOW_LEGACY_SOURCE_SCAN=1` is set.
4. T31/T32 estimator changed to source/material-family residualized fitting plus source-balanced bootstrap and temperature-bin model-win tests.
5. T44 broad NAND crawling is disabled; exact NAND directories/manifests are required for Tier-A rows.
6. T44 density model runs only after true Tier-A rows exist and includes company fixed effects.
7. T53 now supports a real two-stage join: ProteinGym assay rows joined to separate UniProt/PDB/AlphaFold mapping rows.
8. T34 uses exact thermoelectric source tables and fits `ZT ~ cos(6θ) + temperature` when enough rows exist.
9. T57/T59/T45/T47 exact-row parsers use strict source directories and compute residual/benchmark summaries.
10. T26-T30 remain exact-row diagnostic only; T50-T52 remain bound-only; T60 remains anchor-only.

Main command:

```powershell
.\.venv\Scripts\python.exe run_all_and_confirm_v63.py `
  --cache tierb_cache_v63_all `
  --outdir tierb_out_v63_all `
  --timeout 240 `
  --max-tables 300 `
  --force
```

Trust only `confirm_only_dashboard_v63.json -> confirmed_public_now`.
