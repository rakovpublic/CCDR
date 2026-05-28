# v59 confirm-extractor patch

Implements the 10 requested confirm-focused improvements over v58:

1. T31/T32 strict measured microstructure parser with source/sample/material/temperature de-duplication.
2. T31/T32 row-by-row rejection diagnostics.
3. T31/T32 grouped bootstrap plus material/source/temperature-bin jackknife outputs.
4. T44 curated exact NAND source manifest.
5. T44 strict Tier-A parser refuses confirmation if `die_area_mm2` or `bits_per_cell` is missing.
6. T53 ProteinGym -> UniProt/PDB/AlphaFold join gate with family/assay/sequence-cluster requirements.
7. T34 exact Bi2Te3/Sb2Te3 thermoelectric ZT + temperature + orientation/grain-angle contract.
8. T57/T59 exact HEPData record/table/column manifest loader.
9. T45/T47 exact benchmark row contracts.
10. Fusion T26-T30 stay diagnostic unless exact public physical row tables appear.

Public confirm claims must use only:

```text
positive_dashboard.json -> v59_confirm_only_dashboard.confirmed_public_now
```

Expected current claimable result remains `T48` only unless new exact rows are added.
