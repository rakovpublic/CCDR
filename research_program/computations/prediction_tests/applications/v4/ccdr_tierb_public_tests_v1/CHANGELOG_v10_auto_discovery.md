# v10 automated data-discovery quality patch

This patch implements a no-manual-steps quality layer for data-limited tests.

## Core changes

- Adds `tierb/tierb_autodiscovery.py`, a reusable automated discovery layer.
- Adds data-contract JSON generation under `data/contracts/` at runtime.
- Adds automated source expansion:
  - HTML supplementary/data/source links
  - arXiv `e-print` source packages
  - DOI -> Crossref/DataCite/OpenAlex metadata expansion
  - HEPData search links for DOI/arXiv IDs
  - OSF related/parent/sibling links
  - Zenodo/Figshare/OSF file links when metadata JSON is found
- Adds optional PDF table extraction through `pdfplumber`.
- Adds optional vector-PDF/figure diagnostics through `PyMuPDF`.
- Adds automated schema extraction from dictionary/variables PDFs, especially for ITPA DB5.2.3.
- Adds unit-normalization hints for parsed columns.
- Adds nearest-miss reports and sensitivity classification for candidate tables.
- Adds automatic microstructure-manifest triage from public source paths/metadata for MAT1/MAT3.

## Evidence policy

Evidence tiers are explicit:

- E0: no source found
- E1: source found but no usable table
- E2: secondary auto-extracted table/model possible; not decisive
- E3: primary machine-readable public table/model possible
- E4: primary table with uncertainties/controls and adequate sensitivity

Secondary PDF/figure/source-package extraction is never allowed to confirm/falsify.
Only primary machine-readable public tables can be decisive.

## Affected tests

Automated discovery wrappers were added for:

- Fusion: T26, T27, T28, T29, T30
- Materials: T31, T32
- Electronics: T44, T45, T47
- ECC: T46 reporting retained from v9 GF(2) benchmark
- PV: T48 contract/descriptor diagnostics
- Metrology: T50, T51, T52
- Bio/coherence: T54
- HEP/cosmic: T57, T59

## Dependencies

`pdfplumber` and `PyMuPDF` were added as optional automated extraction helpers.
If unavailable, the scripts still run and simply report those extractors as unavailable.
