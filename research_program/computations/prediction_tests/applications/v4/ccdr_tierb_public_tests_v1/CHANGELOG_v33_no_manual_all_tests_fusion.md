# v33 no-manual all-tests fusion-primary patch

Implemented requested changes:

1. Removes the workflow where the user must fill/commit generated CSV input files.
2. Generated CSVs are now treated only as auto-generated public-data cache/audit artifacts.
3. `--confirm-candidates` and `--primary-table-hunt` are backward-compatible no-op selectors; default run executes all tests T26-T60.
4. Adds `primary_table_candidate_manifest_v33.csv` and `fusion_primary_candidate_manifest_v33.csv`.
5. Adds OSF ITPA DB5.2.3 API/direct-schema routes for T28/T30 and stronger fusion source leads for T26/T27/T29.
6. Adds v33 dashboard policy and recommended all-test command.
7. Keeps T50-T52 bound-only and T60a consistency-anchor-only.
8. Preserves strict confirmation gate: no synthetic rows, no manual-curated row promotion.
9. Writes v33 source-hunt/audit metadata for all tests in one scientific run.
