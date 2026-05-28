# v32 confirm + primary-table-hunt patch

Implemented requested improvements 1-6 and 8-10, explicitly excluding the previous item 7 demotion policy.

## Included
1. T53 ProteinGym/DMS source locator and enriched-row confirmation gate.
2. T31/T32 measured grain/nanocrystalline source hunt and v32 template.
3. T44 exact 3D-NAND source hunt and v32 template.
4. T48b-only PV descriptor route; T48a remains a null control.
5. T57/T59 HEPData record/table CSV/YAML download gates.
6. Fusion priority preserved: T28/T30 H-mode DB first.
8. T50-T52 hard bound-only role preserved.
9. T34 restored in v32 dashboard and primary-table-hunt route.
10. Added `--confirm-candidates` and `--primary-table-hunt` run modes.

## New manifest/templates
- `data/primary_table_candidate_manifest_v32.csv`
- `data/generated/t53_proteingym_enriched_rows_v32.csv`
- `data/generated/grain_size_known_manifest_v32.csv`
- `data/generated/t44_nand_exact_rows_v32.csv`
- `data/generated/t48b_pv_descriptor_rows_v32.csv`
- T60b/T60c/T60d v32 templates

No synthetic confirm rows were added.
