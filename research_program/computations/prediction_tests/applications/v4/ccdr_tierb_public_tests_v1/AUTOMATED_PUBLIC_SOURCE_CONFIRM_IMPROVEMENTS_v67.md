# Automated Public-Source Confirm Improvements v67

No manual input rows are allowed. All data must be discovered, downloaded, parsed, normalized, validated, and provenance-linked from public sources by code.

## Improvement Plan

1. Replace the T32 legacy generated-CSV scan with bounded exact-source parsing so T32 no longer times out before the confirm gate can run.
2. Implement an automated T31/T32 materials harvester for public supplements, public data repositories, and open structured tables.
3. Implement automatic T31/T32 microstructure extraction for measured grain size and SEM/TEM/XRD/EBSD-style method fields.
4. Implement automatic T31/T32 family balancing so the harvester keeps searching until family/source/temperature jackknife coverage is plausible.
5. Implement an automated T44 NAND spec-table harvester for public product/spec tables with true die area and bits/cell.
6. Implement T44 product deduplication and source-domain audit so mirrors and repeated specs do not inflate evidence.
7. Implement automated ProteinGym assay ingestion for T53.
8. Implement automated UniProt/PDB/AlphaFold structure-feature joins for T53.
9. Implement automated public thermoelectric table parsing for T34.
10. Implement automated HEPData API/table download and residual-column mapping for T57/T59.
11. Implement automated optical-interconnect benchmark table parsing for T45.
12. Implement automated neuromorphic benchmark table parsing for T47.
13. Implement automated fusion public-row connectors for T26-T30.
14. Implement automated external-public LDPC/burst-channel benchmark harvesting for T46.
15. Turn `validate_v64_source_packs.py` into the pre-confirm pipeline that runs discovery, parsing, validation, row rejection, and confirm-gate checks end to end.

## Confirmation Rule

Rows parsed from public sources can push tests toward confirmation only after they pass strict pack validation and the relevant confirm gate. `ok`, synthetic/engineering results, generated dashboards, templates, and manually prepared rows are not public confirms.

## Implementation Status

Implemented in `tierb/v67_public_source_harvesters.py` and exposed through:

- `harvest_v67_public_sources.py`
- `validate_v64_source_packs.py --harvest-public-sources`
- `run_confirm_only_v64.py --harvest-public-sources`
- `run_all_and_confirm_v64.py --harvest-public-sources`

Network downloads are opt-in with `--allow-network`. Without that flag the pipeline writes an auditable plan and parses only cached public payloads, so it does not fabricate rows or request manual input.
