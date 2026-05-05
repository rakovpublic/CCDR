# Tier-B v4 result-quality patch

Implemented requested 5 improvements plus stronger data-limited recovery attempts.

## 1. Discovery APIs no longer count as tables
Zenodo/OSF/INSPIRE/HEPData search responses are now treated as discovery metadata only. 
The runner follows attached file/download links, then parses only real structured files.

## 2. More public data-source crawling
Added direct OSF file API crawling for the International Global H-mode confinement database (`drwcq`), 
recursive structured-file crawling from Zenodo records, Figshare article files, and ZIP supplement parsing.

## 3. Preregistered manifests
Added `data/microstructure_manifest.csv` for T31/T32 classification and `data/pv_proxy_manifest.csv` for T48 proxies.

## 4. Evidence vs readiness split
Every result now includes:
- `readiness_status`: no_source_found / source_found_no_usable_table / structured_table_found_no_required_physical_columns / physical_columns_found / model_fit_done
- `evidence_status`: data_limited / confirm_like / confirm_like_synthetic_only / null / null_with_falsification_pressure / plausible_but_incomplete / diagnostic_ok_no_directional_claim

## 5. Stronger result-quality tests
T31/T32 use the microstructure manifest for boundary/grain/nano subsets.
T48 uses the public NREL/NLR full-data spreadsheet URL plus the preregistered PV proxy manifest.
Fusion tests crawl OSF/Zenodo attached files and only count named physics columns.
HEP tests keep exact table-link parsing only; metadata is never evidence.
