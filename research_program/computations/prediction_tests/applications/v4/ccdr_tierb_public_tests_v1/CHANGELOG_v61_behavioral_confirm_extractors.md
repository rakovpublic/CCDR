# v61 behavioral confirm extractors

This patch avoids interface-only improvements. It changes actual test behavior and computations:

1. Adds local/cache table scanners for exact public supplement rows.
2. Adds T31/T32 measured κ(T)+microstructure normalization, source/sample deduplication, rejection diagnostics.
3. Adds T31/T32 source/family-demeaned OLS estimator, source-balanced bootstrap, and temperature-bin model-win checks.
4. Adds strict T44 NAND Tier-A row parser and density-vs-layers model, refusing rows missing die area or bits-per-cell.
5. Adds T53 ProteinGym/UniProt/PDB/AlphaFold raw join parser and model/FDR gate inputs.
6. Adds T34 Bi2Te3/Sb2Te3 exact thermoelectric angle parser with cos(6θ) model fit.
7. Adds T57/T59 exact HEPData YAML/CSV residual parser and standardized-residual computation.
8. Adds T45/T47 exact benchmark table parsers.
9. Adds T26-T30 fusion exact-row scanners, but keeps fusion diagnostic-only unless raw rows are explicitly certified.
10. Adds one-command v61 runner that runs full Tier-B tests and then the behavioral confirm extractor dashboard.

Public claims remain conservative: use `confirm_only_dashboard_v61.json -> confirmed_public_now`.
