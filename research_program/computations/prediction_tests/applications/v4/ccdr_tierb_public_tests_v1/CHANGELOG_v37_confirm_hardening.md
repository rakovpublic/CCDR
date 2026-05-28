# v37 confirm-hardening patch

Implements the 9 requested improvements after the v36 report:

1. T48b PV descriptor-model hardening with material/family descriptors, family-level FDR, and tandem/concentrator exclusion control.
2. T44 3D NAND normalized-row expansion and layer-vs-year/bits-per-cell model with manufacturer jackknife diagnostics.
3. T53 ProteinGym symmetry/contact proxy residual model with family/assay-style jackknife diagnostics.
4. T31/T32 microstructure metadata miner for grain/nano, SEM/TEM, isotope, defect and porosity terms.
5. T34 thermoelectric export parser hardening with orientation/grain-angle row diagnostics.
6. T57/T59 exact HEPData registry artifact and official YAML/JSON endpoint fallback.
7. Fusion T26-T30 metadata hard-filtering; OSF/Zenodo metadata wrappers are rejected before physical scoring.
8. T45/T47 exact benchmark narrowing for optical interconnect and neuromorphic-energy tests.
9. T60b/T60c/T60d null-suite scaffolding while preserving T60a as a consistency anchor only.

The v37 dashboard continues to require strict gates before any test can be called confirmed. Near-confirm scores are prioritization only.
