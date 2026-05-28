# CCDR Tier-B v30 implementation report

This bundle implements the 10 priority improvements requested from the v29 analysis.

## Implemented code paths

1. **T53 ProteinGym confirm model**
   - Added `_v30_t53` in `tierb/tierb_runner.py`.
   - Best-effort auto-downloads ProteinGym metadata/substitution rows.
   - Confirmation requires enriched rows with real `OrganismalFitness`/`DMS_score` and a PDB/UniProt symmetry/contact proxy.
   - Gate: proxy BH-FDR q < 0.10, bootstrap CI excludes zero, and family/assay jackknife keeps the sign.

2. **T31/T32 measured grain/nano material gate**
   - Added `_v30_materials`.
   - Reads `data/generated/grain_size_known_manifest_v30.csv`.
   - Broad MAT3 is demoted to null/pressure control; MAT3b is confirmable only on measured grain/nano rows.

3. **T44 exact 3D NAND parser/model**
   - Added `_v30_t44`.
   - Reads `data/manual_curated_electronics_specs.csv` and `data/generated/t44_nand_exact_rows_v30.csv`.
   - Fits layer model vs year/bits/manufacturer baseline.

4. **T48b PV descriptor model only**
   - Added `_v30_t48`.
   - T48a is retired as null control.
   - Confirms only if family-level BH-FDR q < 0.10 after tandem/concentrator exclusions.

5. **T57/T59 exact HEPData manifest gate**
   - Added `_v30_hepdata_exact`.
   - Prefers `data/exact_hepdata_manifest_v30.csv` and scores direct CSV/YAML downloads only.

6. **T50-T52 hard bound-only**
   - Added `_v30_bounds`.
   - These tests cannot be promoted to confirmation.

7. **Demote exact-table-missing tests**
   - Added `_v30_demote_exact_missing` and generic wrappers for T33-T43, T49, T55, T56, T58.

8. **Fusion priority split**
   - Added `_v30_fusion`.
   - T28/T30 are prioritized; T26/T27/T29 remain secondary-only without primary event/profile tables.

9. **T60 blocking gates**
   - Added `_v30_t60`.
   - T60a remains a positive consistency anchor.
   - Full T60 confirmation is blocked until T60b/T60c/T60d pass.

10. **Global v30 dashboard**
   - `run_all_tier_b.py` now writes `ccdr-tierb-positive-dashboard-v30`.
   - `v30_confirm_status.strict_confirm_allowed_now` is the only automated confirmation-language list.

## Added templates

- `data/generated/t53_proteingym_enriched_rows_v30.csv`
- `data/generated/grain_size_known_manifest_v30.csv`
- `data/generated/t44_nand_exact_rows_v30.csv`
- `data/generated/t48b_pv_descriptor_rows_v30.csv`
- `data/generated/t60_quark_lattice_masses_v30.csv`
- `data/exact_hepdata_manifest_v30.csv`

These templates are intentionally empty except for headers. No synthetic confirm rows are inserted.

## Smoke tests run

- `python -m compileall -q .` passed.
- `python run_all_tier_b.py --only T46 ...` passed and produced a v30 dashboard.
- `python run_all_tier_b.py --only T33 --timeout 1 --max-papers 1 --max-tables 1 ...` passed and showed v30 demotion behavior.
