# v58 Tier-B confirm-focus patch

Implements the 10 requested confirm-focused improvements over v57:

1. Strict T31/T32 measured κ(T)+SEM/TEM/XRD material loader contracts.
2. T31/T32 row rejection and gate artifacts are preserved and surfaced as v58 contracts.
3. Strict T44 true Tier-A NAND row gate; derived die-area rows remain audit-only.
4. T53 ProteinGym → UniProt/PDB/AlphaFold joined-row and model/FDR gate.
5. T48 remains the only frozen public confirm; robustness-only audit is written.
6. T34 exact Bi2Te3/Sb2Te3 ZT+temperature+angle row contract.
7. T57/T59 exact HEPData record/table/column manifest gates.
8. T45/T47 exact benchmark row gates.
9. T26–T30 fusion strict row-table contracts; PDFs/summaries stay diagnostic only.
10. `confirm_only_dashboard_v58.json` and `public_claim_check_v58.json` enforce public claims.

Public confirm claims must use only:

```text
positive_dashboard.json -> v58_confirm_only_dashboard.confirmed_public_now
```

Expected current output remains `T48` only unless later strict row/model gates pass.
