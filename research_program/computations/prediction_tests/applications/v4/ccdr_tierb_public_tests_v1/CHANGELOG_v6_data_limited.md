# v6 data-limited-group fixes

Implemented requested fixes for the data-limited groups from the latest report.

## Fusion T26-T30
- Added compound fusion gates for ELM/RMP to reject false ELM contexts such as elm trees, Earth Land Model, white dwarfs and squirrels.
- Removed dangerous plain `roll` negative keyword.
- T28 and T30 now share exact recursive OSF/ITPA DB5.2.3 traversal and follow curated discovered links even in manifest-only mode.
- Added finer readiness states: `source_found_variables_dictionary_only`, `candidate_zip_found_header_failed`, `xlsx_found_header_scan_failed`, `structured_table_found_missing_required_columns`, `model_fit_done`.

## PV / T48
- Added robust NREL page/workbook parser: direct roots + script/href asset discovery + Excel sheet/header-row scan.
- Fixed date/year extraction from date-like cells.
- Requires candidate_rows_count >= 100 for full model-fit status.

## Electronics T44/T45/T47
- Added curated electronics source manifest and dedicated structured gates for 3D NAND, optical interconnect and neuromorphic benchmarks.
- Broad article prose remains non-evidence.

## Metrology T50-T52
- Converted to explicit upper-limit-only tests. Confirmation claims are disabled.
- Results now report upper-limit protocol and bound-oriented interpretation.

## HEP / cosmic T57/T59
- Added exact HEPData manifest with URL mirrors: www/non-www and CSV/YAML/JSON variants.
- T59 is split into category subtests: MET, Drell-Yan, di-Higgs.
