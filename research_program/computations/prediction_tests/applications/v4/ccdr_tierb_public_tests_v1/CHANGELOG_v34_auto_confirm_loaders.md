# v34 automatic confirm-loader patch

Implemented the nine requested improvements after v33:

1. T53 ProteinGym/DMS + UniProt/PDB/RCSB auto-join audit rows.
2. T31/T32 CMB-S4 reference/microstructure miner for grain/nano branch.
3. T44 NAND fallback parser for Wikipedia/vendor/press-style public tables.
4. T48b NREL/NLR loader fix; T48a stays a null control.
5. Fusion OSF/Zenodo recursive structured-file funnel with DB5.2.3 aliases.
6. T57/T59 HEPData JSON/YAML/original fallback before browser CSV URLs.
7. T50-T52 hard bound-only rule preserved.
8. Split branch policies for T32a/T32b and T40a/T40b.
9. Per-test blocker dashboard fields: why_not_confirmed, single_next_blocker, best_auto_data_source_next.

Generated CSVs in data/generated are audit/cache outputs written by scripts. They are not manual user inputs.
