# v54 fusion expected-source patch

Adds targeted expected-source manifests and source-aware triage for T26-T30 based on supplied fusion-source assessment.

## Policy

Papers, figures, metadata pages, and summary regressions may create partial/suggestive/preliminary anchors, but strict confirmation still requires exact machine-readable physical rows.

## Main changes

- T26 remains blocked; Loarte/ITPA/JET/AUG/MAST figure-level sources are partial trend diagnostics only.
- T27 gains Paz-Soldan 2024 as suggestive public RMP-ELM compilation; no raw per-discharge confirm.
- T28 gains Verdoolaege 2021 DB5.2.3-STD5 as strongest public summary anchor; row access still required for confirmation.
- T29 becomes strongest fusion preliminary path using Stroth 2021 W7-X/AUG/W7-AS comparison tables.
- T30 remains dependent on exact T28/T29-like rows.

New outputs include `data/generated/fusion_expected_sources_v54.csv`, per-test `t26...t30_fusion_expected_sources_v54.csv`, and `confirm_targets_v54.json`.
