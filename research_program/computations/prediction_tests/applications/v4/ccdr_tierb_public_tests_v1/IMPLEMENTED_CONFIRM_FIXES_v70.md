# Implemented Confirm Fixes v70

Scope requested: implement the four T53 confirm-path fixes from the latest report and add source-specific adapters for T31/T32 and T44.

## Implemented

1. Fixed the v63/v64 source-pack CSV reader.
   - Removed the invalid `low_memory=False` argument when using pandas `engine="python"`.
   - This fixes the validator error: `ValueError: The 'low_memory' option is not supported with the 'python' engine`.

2. Added T53 pack-specific validation diagnostics.
   - `proteingym` validation now reports assay, UniProt, family, and sequence-cluster counts.
   - `protein_structures` validation now reports UniProt, AlphaFold, and PDB row counts.
   - `protein_structures` now accepts `pdb_id OR alphafold_id`, matching the confirm gate.

3. Added a T53 ProteinGym raw-DMS adapter.
   - The harvester no longer treats ProteinGym metadata fields such as MSA bitscore as DMS evidence.
   - It now tries public raw DMS file layouts from the ProteinGym repository and writes variant-level rows only when mutation and score columns are present.
   - Existing metadata-only rows are rejected as `proteingym_reference_manifest_is_metadata_not_variant_scores`.

4. Added a T53 AlphaFold structure adapter.
   - It uses harvested ProteinGym UniProt IDs.
   - It fetches public AlphaFold prediction metadata and CIF files.
   - It writes `protein_structures` rows with AlphaFold id, single-chain model state, symmetry proxy, contact-network proxy from CA contacts, fold/family label, and source URL.

5. Added a source-specific T31/T32 materials adapter.
   - It targets the public CMB-S4 `Cryogenic_Material_Properties` repository.
   - It parses temperature/kappa tables and records candidate rows.
   - It keeps these rows non-confirming unless public grain-size and microstructure fields are present or joinable.

6. Added a source-specific T44 NAND adapter.
   - It targets WikiChip 3D NAND and flash-memory cell pages.
   - It parses HTML tables for company, year, layers, capacity, die area, bits per cell, product, and source URL.
   - It writes only rows that satisfy the exact NAND pack schema.

## Verification

Commands run:

```powershell
python -m py_compile .\tierb\v63_behavioral_confirm_extractors.py .\tierb\v64_exact_data_packs.py .\tierb\v67_public_source_harvesters.py .\validate_v64_source_packs.py .\run_confirm_only_v64.py .\run_all_and_confirm_v64.py .\harvest_v67_public_sources.py
python .\validate_v64_source_packs.py --outdir .\tierb_out_v70_revalidate_after_pack_guard --cache .\tierb_cache_v70_all --only T53 T31 T32 T44
python .\harvest_v67_public_sources.py --outdir .\tierb_out_v70_adapter_dryrun2 --cache .\tierb_cache_v70_all --only T31 T32 T44 T53 --dry-run --max-sources-per-pack 2 --max-rows-per-source 20
python .\run_confirm_only_v64.py --outdir .\tierb_out_v70_confirm_after_adapter_fixes --cache .\tierb_cache_v70_all --only T31 T32 T44 T53 T48
```

Results:

- Compile passed.
- Dry-run adapter path passed.
- Confirm-only still reports `confirmed_public_now: ["T48"]`.
- The old ProteinGym metadata rows are now correctly rejected rather than counted as usable DMS evidence.

## Next Network Run

To actually populate the new raw DMS, AlphaFold, CMB-S4, and WikiChip adapters:

```powershell
python .\run_all_and_confirm_v64.py --skip-full-run --cache tierb_cache_v71_all --outdir tierb_out_v71_adapters --harvest-public-sources --allow-network --confirm-only T31 T32 T44 T53 T48 --max-sources-per-pack 25 --max-rows-per-source 5000
```

After that, inspect:

- `tierb_out_v71_adapters\public_source_harvest_v67.json`
- `tierb_out_v71_adapters\v64_source_pack_validation.json`
- `tierb_out_v71_adapters\confirm_only_dashboard_v64.json`

