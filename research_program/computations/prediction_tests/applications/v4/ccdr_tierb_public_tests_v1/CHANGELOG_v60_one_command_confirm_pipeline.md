# v60 one-command confirm pipeline

Implemented requested improvements over v59:

1. Added `run_all_and_confirm_v60.py`, a single wrapper that runs full Tier-B tests and then v60 confirm-only/public-claim dashboards.
2. Added T31/T32 measured microstructure rejection summaries by reason.
3. Added T31/T32 source/material-family/temperature-bin balance gates.
4. Added T44 exact NAND source manifest and empty fixture with required Tier-A columns.
5. Added T44 hard public gate: no confirmation without explicit die area and bits-per-cell plus manufacturer/year jackknife.
6. Added T53 ProteinGym→UniProt/PDB/AlphaFold v60 join normalization and rejection summaries.
7. Added T34 exact ZT/temperature/orientation/grain-angle row parser gate.
8. Added T57/T59 exact HEPData manifest parser gate.
9. Added T45/T47 exact benchmark table gates.
10. Preserved fusion T26–T30 diagnostic-only policy and bound/anchor safety; v60 public claims use only `confirmed_public_now`.

Main one-command usage:

```powershell
.\.venv\Scripts\python.exe run_all_and_confirm_v60.py `
  --cache tierb_cache_v60_all `
  --outdir tierb_out_v60_all `
  --timeout 240 `
  --max-tables 300 `
  --force
```

Trust only:

```text
tierb_out_v60_all\confirm_only_dashboard_v60.json -> confirmed_public_now
```
