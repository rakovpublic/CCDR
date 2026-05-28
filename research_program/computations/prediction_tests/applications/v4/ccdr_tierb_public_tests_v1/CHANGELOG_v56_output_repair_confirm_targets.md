# v56 output-repair and confirm-target hardening

Implements the 10 requested improvements after the v55 analysis.

## Changes

1. Adds a targeted missing-output fallback repair layer in `tierb/v56_missing_output_repair.py`.
2. Patches `run_all_tier_b.py` so T31/T32/T44/T51/T52/T26 no longer collapse into an uninformative generic missing-output fallback.
3. Restores T50/T51/T52 as bound-only constraints even if their script times out or produces no JSON.
4. Adds v56 T48 robustness-only audit artifacts, including a descriptor permutation null.
5. Adds v56 T31/T32 confirm gates: dedup measured κ(T)+microstructure rows, >=5 source groups, >=5 material families, >=3 temperature bins, temperature-baseline AIC/BIC, sign, bootstrap/jackknife gates.
6. Adds v56 T53 ProteinGym->UniProt/PDB/AlphaFold final-gate summary.
7. Adds v56 T44 true Tier-A NAND gate summary; derived die-area rows remain audit-only.
8. Adds v56 fusion parser upgrades: PyMuPDF block extraction and conservative source-anchor fallbacks.
9. Adds v56 public claim gate: public claims should use only `positive_dashboard.json -> v56_confirm_status.confirmed_now`.
10. Adds `confirm_targets_v56.json` and `status_split_counts_v56` to the dashboard.

## Policy

- T48 remains the only frozen compatible-positive unless another test reaches strict v56 gates.
- T31/T32/T53/T44/T29 are ranked as confirm routes, not public confirmations, until their v56 gates pass.
- T27/T28 remain summary/suggestive only; T29 is preliminary-only unless raw public profile/transport rows appear.
- T50/T51/T52 are not confirmable by design.
