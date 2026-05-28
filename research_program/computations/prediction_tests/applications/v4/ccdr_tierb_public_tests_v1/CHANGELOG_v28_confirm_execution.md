# v28 confirm execution

Implemented requested confirm-focused improvements from the v27 report.

- T44: consumes/generated `data/generated/t44_nand_exact_rows_v28.csv`, runs layer-vs-year/manufacturer OLS gate when N>=20.
- T31/T32: consumes `grain_size_known_manifest_v28.csv`, keeps broad MAT3 as null control and MAT3b as measured-nanostructure-only confirm path.
- T48b: primary descriptor confirm route with family BH-FDR readiness.
- T60: stronger random-triplet null plus sector/look-elsewhere blocking gates.
- T53: OrganismalFitness/PDB/UniProt mapper target and confirm gate.
- T45/T47: exact row templates and parser hooks.
- Fusion: secondary numeric-context rows template; no confirmation from secondary rows.
- Bounds/T54/HEPData: exact-table manifests and confirm-forbidden/bound-only gates.
