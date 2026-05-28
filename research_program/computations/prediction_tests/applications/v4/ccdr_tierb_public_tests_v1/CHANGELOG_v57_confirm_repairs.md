# v57 confirm repairs

Implements the 10 requested improvements after the v56 analysis:

1. Hard repaired T31 result emission via `materials_confirm_v57()`.
2. Hard repaired T32 result emission via the same strict schema.
3. Added standalone `run_materials_confirm_v57.py` for T31/T32 only.
4. Added row-by-row T31/T32 rejection diagnostics.
5. Added exact T44 NAND Tier-A parser/audit with derived rows marked audit-only.
6. Added T48 robustness-only family/source/year/permutation outputs without moving the frozen confirm gate.
7. Added real T53 ProteinGym -> UniProt/PDB/AlphaFold joined-row gate parser.
8. Added T29 raw PDF text-block extraction diagnostics for Stroth/W7-X/AUG parser debugging.
9. Forced T28/T29/T30 diagnostic non-confirm JSON/CSV outputs even when PDF/parser rows are missing.
10. Added `public_claim_check_v57.json`; public claims must use only `positive_dashboard.json -> v57_confirm_status.confirmed_now`.
