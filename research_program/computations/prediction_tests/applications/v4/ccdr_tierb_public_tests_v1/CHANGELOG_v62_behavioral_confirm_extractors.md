# v62 behavioral confirm extractors

This patch implements behavior changes, not interface-only changes.

1. Adds manifest-driven public supplement downloader for exact CSV/XLSX/YAML sources.
2. Expands T31/T32 material table scanning with unit conversion for temperature and grain-size columns.
3. Adds source/family fixed-effect and source-balanced bootstrap estimators for T31/T32.
4. Adds row-level provenance hashes and rejection-summary CSVs for T31/T32.
5. Tightens T44 exact NAND parsing with capacity-unit conversion and company fixed effects.
6. Improves T53 ProteinGym->UniProt/PDB/AlphaFold join normalization and rejection summaries.
7. Improves T34 Bi2Te3/Sb2Te3 angular ZT parsing and cos(6θ) model gate.
8. Improves T57/T59 HEPData exact YAML/CSV residual computation.
9. Improves T45/T47 exact benchmark parsers and rejection summaries.
10. Keeps fusion/bounds/anchor safety unchanged: T26-T30 require exact certified physical rows, T50-T52 are bound-only, T60 is anchor-only.

Use one command:

```powershell
.\.venv\Scripts\python.exe run_all_and_confirm_v62.py `
  --cache tierb_cache_v62_all `
  --outdir tierb_out_v62_all `
  --timeout 240 `
  --max-tables 300 `
  --force
```

Trust only `confirm_only_dashboard_v62.json -> confirmed_public_now` for public claims.
