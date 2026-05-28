# v43 confirm robustness and next-target hardening

Implemented the next 9 improvements requested after the v42 all-tests report:

1. Preserve T48b and T44 as frozen test-level positives and add robustness-only artifacts.
2. Add v43 T53 DMS/PDB/AlphaFold structure-contact proxy rows and readiness accounting.
3. Add stricter T31/T32 narrow grain/nano κ(T)+microstructure normalization rows.
4. Add v43 T34 exact thermoelectric Bi2Te3/Sb2Te3 orientation/ZT mapping rows.
5. Add v43 exact HEPData registry artifacts for T57/T59.
6. Add v43 exact optical benchmark parsing rows for T45.
7. Add v43 exact neuromorphic benchmark parsing rows for T47.
8. Preserve fusion exact-attachment-only policy and reject metadata/schema/PDF wrappers as evidence.
9. Preserve T60a as anchor-only and add v43 full-null-suite readiness artifacts.

T48b and T44 remain confirmed only if their frozen prior strict gates are still present in the run artifacts. T50-T52 remain bound-only.
