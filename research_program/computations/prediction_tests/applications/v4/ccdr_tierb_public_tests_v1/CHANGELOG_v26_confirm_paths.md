# v26 confirm-focused patch

Implements all requested confirm/improve sections from the v25 analysis:

- T60a: always surfaces random-triplet null and full-confirmation gates for sector reshuffling/look-elsewhere/T60b.
- T31/T32: confirm-aware grain/nano scorer and MAT3b gate tied to the T31 grain-size manifest.
- T44: EL1/EL3 NAND exact-row confirm route with layer-vs-year/manufacturer model threshold.
- T45: EL8 optical pJ/bit + bandwidth + reach extraction route.
- T46: 500-seed optimizer-style engineering gate; strong baselines still required.
- T47: exact neuromorphic benchmark row gate.
- T48b: descriptor PV confirm-model route and FDR/jackknife requirements.
- T53: OrganismalFitness/PDB/UniProt confirm gate; MSA_bitscore is not confirmatory.
- Fusion: stronger secondary-only numeric context extraction route and CSV artifact.
- T50-T52: bound-only confirm-forbidden outputs.
- T54/T57/T59: exact supplement/HEPData table gates.

No confirmation gate is relaxed: secondary PDF/prose/figure extraction remains non-decisive.
