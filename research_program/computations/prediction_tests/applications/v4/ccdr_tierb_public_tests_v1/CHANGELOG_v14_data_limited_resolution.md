# v14 data-limited resolution patch

Implements the 10 requested fixes for the Tier-B data-limited pipeline:

1. Hard README/plain-text metadata filter.
2. Strict candidate counting: numeric columns plus domain/contract match required.
3. Source-quality ladder: exact/source-data links before broad repository search.
4. Secondary fusion/metrology/coherence PDF text+unit extractor.
5. Repository record relevance gate before following files.
6. T28/T30 schema extraction and alternative H-mode search seeds.
7. Electronics exact/spec-source discovery seeds.
8. Metrology bound-only connectors and bound summary.
9. Biology/coherence direct-source discovery seeds.
10. HEP/cosmic exact-table manifest and HEPData search seeds.

Scientific policy unchanged: only primary machine-readable physical tables can confirm/falsify. Secondary PDF/source-package extraction is diagnostic only.
