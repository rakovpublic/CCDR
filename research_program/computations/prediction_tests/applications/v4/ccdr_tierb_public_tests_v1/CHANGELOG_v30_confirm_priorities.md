# v30 confirm-priority implementation

Implements the 10 priority improvements requested after the v29 report:

1. T53 now has an executable ProteinGym/DMS residual gate. It writes best-effort ProteinGym metadata/substitution rows to `data/generated/t53_proteingym_auto_rows_v30.csv` and requires an enriched `t53_proteingym_enriched_rows_v30.csv` with a real PDB/UniProt symmetry/contact proxy before confirmation.
2. T31/T32 now use a measured grain/nanocrystalline-only material gate from `grain_size_known_manifest_v30.csv`; broad MAT3 remains a null/pressure control.
3. T44 now reads exact/manual 3D NAND rows and fits `density_Gb_per_mm2 ~ layers + year + bits/cell + manufacturer` versus a year/bits baseline.
4. T48 now treats T48a as a retired null control and promotes only T48b descriptor-family FDR modeling.
5. T57/T59 now prefer `exact_hepdata_manifest_v30.csv` and score only direct HEPData CSV/YAML-like downloads.
6. T50/T51/T52 are hard bound-only tests and cannot be promoted to confirmation.
7. Exact-table-missing tests are demoted until real structured tables exist.
8. Fusion priority is split: T28/T30 H-mode exact tables first; T26/T27/T29 remain secondary diagnostics without primary tables.
9. T60 keeps T60a as a consistency anchor and adds blocking T60b/T60c/T60d gates.
10. `positive_dashboard.json` now has v30 confirm status, with `strict_confirm_allowed_now` as the only automated confirm-language list.

No synthetic rows are inserted into confirm manifests. Empty v30 CSV templates are included so users can add exact rows without changing code.
