# v41 confirm robustness and next-target hardening

Implemented the next nine improvements requested after the v40 report:

1. Preserve T48b and T44 confirmations; add robustness-only dashboards and artifacts.
2. Add final T53 DMS + PDB/AlphaFold/RCSB structure-contact proxy model scaffold.
3. Harden T31/T32 temperature/Kappa/microstructure normalization for the narrow grain/nano branch.
4. Improve T34 Bi2Te3/Sb2Te3 thermoelectric orientation/ZT row mapping.
5. Add v41 exact HEPData registry artifacts for T57/T59.
6. Tighten T45 optical exact benchmark row extraction.
7. Tighten T47 neuromorphic exact benchmark row extraction.
8. Keep fusion as exact structured-attachment-only evidence.
9. Add T60 full null-suite input gate audit while preserving T60a as anchor-only.

The v41 dashboard separates confirmed_now, near_confirm_next, positive_anchor_only, bound_only, and open/data-limited tests.
