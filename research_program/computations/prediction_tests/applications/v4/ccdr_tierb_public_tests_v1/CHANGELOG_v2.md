# CCDR Tier-B Public Tests v2 changes

## Evidence-gating changes

- Disabled generic term-window number extraction from article/PDF/HTML text as evidence.
- A literature parser now returns `partial`/`ok` only if it finds direct structured public data or discovered supplements with named physical columns required by the test.
- If named physical columns are not identified, status is `data_limited`, not `partial`.
- Added test-specific strict column rules for fusion, materials, quantum, electronics, sensors, bio, and aerospace literature tests.

## Fusion T26-T30

- Stopped treating article HTML tables or citation metadata as data.
- Require headers/columns such as `E_ELM`, `W_ELM`, pedestal pressure, pedestal volume, `ΔP/P`, ELM frequency, RMP current/phasing, confinement time, shaping/curvature, density.
- If only papers are found and no structured supplement has those columns, returns `data_limited`.

## Materials T31-T32

- Added heuristic material classification from public CMB-S4 repository path/columns:
  - crystalline_or_metal
  - amorphous
  - composite_or_polymer
  - unknown
  - grain_size_known true/false
  - nanocrystalline_yes_no true/false
  - boundary_dominated_candidate true/false
- T31/T32 now report the main inference on the boundary-dominated subset only.
- Added `falsification_pressure` fields for serious negative readings.

## T48 NREL PV

- Added a public-data-only baseline model: `efficiency_pct ~ year + material_class fixed effects`.
- Tests residual efficiency against a deterministic acoustic-optical / mass-contrast proxy inferred from public NREL material/cell text.
- Still marked as proxy-level, not equivalent to a Materials Project phonon/symmetry calculation.

## T57/T59 HEP/CR

- Replaced HEPData search endpoint reliance with known HEPData record URLs plus INSPIRE API endpoints.
- Added JSON table parsing in the common table reader so INSPIRE/HEPData JSON can be summarized.

## Other fixes

- T53 support logic now requires a meaningful positive correlation and p-value instead of any rho > 0.
- T60 now tries to parse tau from public PDG MC mass-width data instead of requiring an opt-in hard-coded seed.
