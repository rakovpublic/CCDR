# v22 Positive paths + EL/fusion implementation

This patch implements all six requested positive-focused improvements and the additional positive paths from the v21 report, without relaxing confirmation/falsification evidence gates.

## Implemented

1. **T44 / EL1+EL3 NAND exact parser first**
   - Adds v22 NAND structured parser hooks for WikiChip/TechInsights-style text/HTML/PDF sources.
   - Extracts/normalizes manufacturer, product/generation text, layer count, die capacity, die area, bits/cell, and density hints.
   - Adds `t44_nand_exact_parser_v22` with the model formula and success rule.

2. **T53 enrichment/model path**
   - Adds `t53_uniprot_pdb_enrichment_v22` with explicit join keys and proxy features.
   - Keeps readiness-positive language unless outcome/proxy columns are actually present.

3. **T31/T32 materials flagship path**
   - Adds `materials_positive_score_v22`, `grain_size_expansion_v22`, and MAT3b-to-T31 linkage.
   - Rewards grain/nano CCDR-vs-powerlaw AIC support, decisive microstructure rows, and usable fits.

4. **T45 / EL8 optical interconnect parser**
   - Adds v22 pJ/bit/fJ/bit, Gb/s/Tb/s, and reach-unit extraction hooks.
   - Adds `t45_optical_interconnect_parser_v22` diagnostics and success rule.

5. **T46b optimization mode**
   - Adds a deterministic 50-seed proxy search across CDT-hybrid and matched baseline families.
   - Emits `t46b_optimization_run_v22`. This remains synthetic engineering search, not CCDR physics evidence.

6. **Fusion secondary diagnostics**
   - Adds a stronger multi-backend PDF/text unit-line extractor for exact Loarte/JET/ITER/ITPA/W7-X-style sources.
   - Emits `fusion_secondary_diagnostic_v22`. Secondary rows cannot confirm/falsify.

7. **Dashboard v22**
   - Expands `positive_dashboard.json` with ranked positive priorities and recommended next actions.

## Evidence policy

- Primary confirmation still requires public machine-readable physical tables passing all contract groups.
- Secondary PDF/text extracted rows are diagnostic only.
- Broad HTML/search metadata remains discovery-only.
