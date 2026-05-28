# Implemented Automated Public-Source Confirm Improvements v67

Constraint: no manual input rows. Countable rows must be discovered, downloaded or read from cache, parsed, normalized, provenance-linked, and validated by code from public structured sources.

## What Changed

1. Added `tierb/v67_public_source_harvesters.py`, an automated public-source harvester for all v64 exact-source packs.
2. Added `harvest_v67_public_sources.py`, a CLI entry point for discovery, parsing, row normalization, rejection logging, and validation.
3. Wired `validate_v64_source_packs.py --harvest-public-sources` so validation can run discovery before source-pack checks.
4. Wired `run_confirm_only_v64.py --harvest-public-sources` so confirm-only runs can harvest before applying gates.
5. Wired `run_all_and_confirm_v64.py --harvest-public-sources` so the one-command wrapper can run harvest -> validation -> confirm artifacts.
6. Added opt-in network control with `--allow-network`; without it, the harvester uses cached files and writes manifests only.
7. Added `--dry-run-harvest`/`--dry-run` for source planning without countable row writes.
8. Implemented T31/T32 materials public-source seeds and structured table parsing.
9. Implemented T31/T32 microstructure normalization for material family, grain size, temperature, kappa, method, and boundary-density proxy.
10. Implemented T31/T32 coverage summaries for source domains, material families, and temperature bins.
11. Implemented T44 NAND public spec-table seeds, row normalization, bits-per-cell mapping, and dedup identity support.
12. Implemented T53 ProteinGym assay ingestion and protein-structure input harvesting paths.
13. Implemented T34 thermoelectric, T57/T59 HEPData, T45 optical, T47 neuromorphic, T26-T30 fusion, and T46 LDPC/burst-channel pack harvest paths.
14. Added harvest manifests and candidate rejection CSVs under `data/generated` in each output directory.
15. Bounded legacy generated-table reads with size/file/row caps and dtype-safe parsing to reduce T32/T44 memory spikes and DtypeWarnings.

## New Commands

Dry-run discovery plan:

```powershell
python .\harvest_v67_public_sources.py --outdir .\tierb_out_v67_public_harvest_dryrun --cache .\tierb_cache --dry-run
```

Pre-confirm validation with harvest planning:

```powershell
python .\validate_v64_source_packs.py --outdir .\tierb_out_v67_public_harvest_validate --cache .\tierb_cache --harvest-public-sources --dry-run-harvest
```

Network-enabled automated harvest:

```powershell
python .\run_all_and_confirm_v64.py --skip-full-run --outdir .\tierb_out_v67_network_harvest --cache .\tierb_cache --harvest-public-sources --allow-network
```

## Verification Performed

`python -m py_compile` passed for:

- `tierb/v67_public_source_harvesters.py`
- `harvest_v67_public_sources.py`
- `validate_v64_source_packs.py`
- `run_confirm_only_v64.py`
- `run_all_and_confirm_v64.py`
- `tierb/v64_exact_data_packs.py`
- `tierb/tierb_runner.py`
- `tierb/v61_behavioral_confirm_extractors.py`

Dry-run harvester result:

- packs attempted: 11
- sources downloaded: 0
- structured sources parsed: 0
- rows written: 0
- expected, because network was not enabled

Dry-run validation result:

- all existing rows valid: true
- problem packs: 0
- empty required packs: 11
- expected, because no public downloads were allowed

Dry-run confirm wrapper result:

- confirmed_public_now: `["T48"]`
- no new confirms claimed without validated public rows

## Confirmation Guard

The harvester can push tests toward confirmation only when public rows are actually parsed and the existing strict v64 gates pass. Search pages, metadata wrappers, templates, generated dashboards, and manual placeholder rows do not count.

