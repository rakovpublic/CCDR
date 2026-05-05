# v9 script-quality upgrades

Implemented script-quality improvements without relaxing evidence standards.

## Added manual-curated public table routes

New files:

- `data/manual_curated_fusion_tables.csv`
- `data/manual_curated_electronics_specs.csv`

These are empty templates by default. If rows are added with public `source_url` and an `evidence_tier`, T26/T27/T28/T29/T30 and T44/T45/T47 can run actual model checks instead of staying permanently PDF/data-limited. Results are explicitly marked as `manual_curated_model_fit_done` and remain separate from primary machine-readable-supplement evidence.

## T46 quality upgrade

T46 now uses a GF(2) rank erasure/burst correctability benchmark. A burst is correctable only if the restricted parity-check submatrix has full column rank. Baselines include local LDPC, surface-like, protograph/QC, spatially coupled, interleaved, random regular, and CDT-like irregular/nonlocal constructions.

## MAT1/MAT3 decisive microstructure gate

T31/T32 now include a v9 decisive-quality gate. Confirm/falsify language is disallowed unless measured/explicit microstructure rows reach the preregistered threshold.

## T48 family-level multiple-testing gate

T48 now applies BH-FDR diagnostics across global and family-level proxy tests. Support requires positive global direction plus at least one family-level result with q<0.10.
