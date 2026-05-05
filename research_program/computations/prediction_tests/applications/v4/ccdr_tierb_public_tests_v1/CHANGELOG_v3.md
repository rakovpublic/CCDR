# Tier-B v3 result-quality patch

Implemented all requested result-quality fixes:

- Fusion T26-T30: replaced broad literature scraping with structured-source manifests. Required named physical columns are enforced per prediction.
- T31/T32: promoted to serious negative/positive tests with material classification and primary boundary/grain subset inference.
- T32: added fixed-exponent model comparison for κ ∝ T^0.5, T^1, T^2, T^3 plus free exponent.
- T48: upgraded NREL PV analysis to baseline residual model: year + material class + cell bucket + area; primary proxy is within-class crystallinity/texture rather than circular material-class proxy.
- T57/T59: removed HEPData search API dependence; added direct table/INSPIRE manifests and exact table qualification gates.
- T60: split charged-lepton and quark/lattice sectors internally.
- T44: added 3D NAND structured spec-table gate.
- T50-T52: converted to upper-limit/constraint tests only.
- T53: replaced weak PDB metadata proxy with stability/Tm/ΔG dataset gate.

No generic term-number extraction is evidence anywhere in v3.
