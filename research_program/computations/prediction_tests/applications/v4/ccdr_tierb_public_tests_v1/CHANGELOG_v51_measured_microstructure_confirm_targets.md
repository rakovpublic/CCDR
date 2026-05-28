# v51 measured microstructure confirm-target hardening

Implements the next 10 improvements after v50 analysis:

1. T31 measured-only κ(T)+microstructure rows; proxy-only rows cannot confirm.
2. T32 measured-only κ(T)+microstructure rows; broad T^0.5 remains control/pressure.
3. Grouped/hierarchical bootstrap and material/source/temperature-bin jackknife for T31/T32.
4. T44 true Tier-A NAND audit only: company/year/layers/capacity/die_area/bits_per_cell/source_url.
5. T48b frozen compatible-positive robustness manifest only; no gate changes.
6. T53 real ProteinGym DMS -> UniProt/PDB/AlphaFold join manifest.
7. T34 exact Bi2Te3/Sb2Te3 thermoelectric export manifest.
8. T57/T59 exact HEPData record/table/column manifest only.
9. T45/T47 exact benchmark manifests only; generic pages do not score.
10. Fusion exact-attachment-only policy, T50-T52 bound-only, and T60 anchor-only null-suite status.

The dashboard now emits `ccdr-tierb-positive-dashboard-v51` and `v51_confirm_status`.
