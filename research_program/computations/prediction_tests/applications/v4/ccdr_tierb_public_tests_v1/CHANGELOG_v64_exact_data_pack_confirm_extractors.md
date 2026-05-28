# v64 exact-data-pack confirm extractors

Behavioral/runtime changes, not dashboard-only:

1. Adds exact source pack directories and schema templates for T31/T32, T44, T53, T34, T57/T59, T45/T47, and fusion exact rows.
2. T31/T32 now ingest only filled exact materials source packs and per-family packs; generated dashboards/rejection files/templates are excluded.
3. T31/T32 add small curated set mode via `data/exact_sources/materials/*.csv` and `data/exact_sources/materials/families/*.csv`.
4. T31/T32 add stricter source/family/temperature diversity gates and a source+family residualized OLS estimator with family-source balanced bootstrap.
5. T44 uses exact Tier-A NAND rows only and rejects inferred/derived die-area rows.
6. T44 runs a density-vs-layers model only after complete rows and >=3 companies exist.
7. T53 implements a two-stage ProteinGym assay to UniProt to PDB/AlphaFold structure-feature join cache.
8. T34 parses exact Bi2Te3/Sb2Te3 ZT+angle rows and fits a cos(6θ)+temperature model when enough rows exist.
9. T57/T59 parse exact HEPData manifests/local tables into observed-model-uncertainty residual rows.
10. T45/T47 parse exact benchmark source packs only; metadata remains non-evidence.

Templates are not evidence. A public confirm appears only when filled physical rows pass the v64 gates.
