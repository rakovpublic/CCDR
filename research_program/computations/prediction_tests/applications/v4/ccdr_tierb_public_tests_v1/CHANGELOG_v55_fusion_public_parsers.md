# v55 Fusion public-source parsers

Implements actual public-source parser modules for the expected fusion sources added in v54.

## Added

- `tierb/fusion_public_parsers_v55.py`
  - T26 parser for public ELM-loss / pedestal summary and figure-context rows from the IAEA/ITPA PDF.
  - T27 parser for Paz-Soldan 2024 RMP-ELM compilation summary/text-table rows.
  - T28 parser for Verdoolaege 2021 / DB5.2.3 public summary and regression rows, plus OSF structured-attachment probe.
  - T29 parser for Stroth 2021 W7-X/AUG/W7-AS public comparison text/table rows.
  - T30 derived parser that reports reusable T28/T29 anchors but does not claim an independent result.
- `run_fusion_public_parsers_v55.py`, a parser-only runner that bypasses older broad discovery layers.
- v55 result fields in `tierb/tierb_runner.py` for T26-T30:
  - `auto_data_improvements_v55.fusion_public_source_parser_v55`
  - `status_split_v55`
  - `confirm_target_v55`
  - `near_confirm_score_v55`
  - `public_claim_gate_v55`
- v55 dashboard overlay in `run_all_tier_b.py`:
  - `v55_confirm_status`
  - `confirm_targets_v55.json`
  - `recommended_next_v55`

## Confirmation policy

v55 intentionally keeps all fusion parser outputs non-confirm unless an exact public raw table appears.

- T26/T27: parsed PDF rows are partial/suggestive only.
- T28: parsed DB5 summary/regression rows are public ingredient support only; full DB5.2.3 per-timeslice rows are still required.
- T29: parsed Stroth rows can create a preliminary structured-public test, not strict confirmation.
- T30: secondary diagnostic only.

## Parser-only command

```powershell
.\.venv\Scripts\python.exe run_fusion_public_parsers_v55.py `
  --cache tierb_cache_v55_fusion `
  --outdir tierb_out_v55_fusion_parsers `
  --only T29 T28 T27 T26 T30 `
  --timeout 90 `
  --max-tables 120
```
