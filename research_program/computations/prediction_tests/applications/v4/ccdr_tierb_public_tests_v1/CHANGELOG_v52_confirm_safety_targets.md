# v52 confirm-safety and target-dashboard patch

Implements the requested 10 improvements after v51 analysis:

1. Confirmation-conflict detector/demoter, including T44 strict+zero true Tier-A-row protection.
2. Explicit split of `execution_status_v52`, `data_status_v52`, `evidence_status_v52`, and `confirmation_status_v52`.
3. T48b publication-grade robustness scaffold: descriptor model, absorber-family/source/year jackknife, and permutation null artifacts.
4. T31/T32 measured-only microstructure registry shared across materials tests.
5. T34 exact Bi2Te3/Sb2Te3 thermoelectric source contract.
6. T53 ProteinGym-to-structure source contract.
7. T57/T59 exact HEPData record/table/column source contract.
8. T26-T30 fusion exact measurement attachment source contracts.
9. T45/T47 exact benchmark source contracts.
10. `confirm_targets_v52.json` dashboard ranking blockers, next sources, effort, and confirmation eligibility.

Safety rule: only `v52_confirm_status.confirmed_now` should be used for confirm claims.
