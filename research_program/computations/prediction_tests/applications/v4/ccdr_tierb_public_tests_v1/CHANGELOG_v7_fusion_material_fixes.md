# v7 fusion/material fixes

Implemented after v6 run-time report:

- Fixed missing `ensure_dir`, `to_jsonable`, and `find_col` imports in `tierb/tierb_runner.py`.
- Replaced TODO fusion manifest rows with curated source entries for:
  - T26 ELM energy / pedestal supplements
  - T27 RMP / ELM-frequency supplements
  - T29 W7-X/W7-AS profile-only stellarator-vs-tokamak proxies
- Added v7 override runners:
  - T26/T27 curated source-only structured-table probes
  - T28 exact OSF ITPA DB5.2.3 recursive parser
  - T30 same OSF ITPA DB parser plus density+shaping residual model
  - T29 profile-only proxy readiness runner
- Expanded `data/microstructure_manifest.csv` for MAT1/MAT3 with explicit evidence classes:
  - explicit nanostructure / grain-size keyword rows
  - composite/fiber boundary proxies
  - amorphous polymer controls
  - porous/powder/sinter proxies
  - bulk metal/crystal controls
- Added v7 T31/T32 wrappers using a fresh cache namespace so old v5/v6 cached summaries do not mask the expanded microstructure manifest.

Notes:
- PDF/article prose is still not evidence. Curated PDF/HTML sources are only used to locate/parse machine-readable tables if present.
- T28/T30 classify OSF variable dictionaries as metadata-only and return data_limited if no structured DB table is exposed.
- T29 is intentionally a profile-only proxy unless comparable structured stellarator and tokamak rows are parsed.
