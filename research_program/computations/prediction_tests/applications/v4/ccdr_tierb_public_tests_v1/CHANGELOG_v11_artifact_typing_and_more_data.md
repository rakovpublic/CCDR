# v11 artifact typing and automated data expansion

Implemented all seven requested quality improvements without manual data-entry steps.

1. Metadata APIs (Crossref/OpenAlex/DataCite/HEPData search) are now link-discovery records only, never physical table candidates.
2. PDF extraction strengthened with pdfplumber over more pages, optional Camelot, and conservative text-line table fallback.
3. arXiv source parser now handles tabular, tabular*, longtable, and deluxetable style LaTeX tables.
4. Domain-specific additional repository seeds were added for fusion, electronics, metrology, HEP, and coherence tests.
5. Every candidate table now includes rejection reasons and nearest-miss strategy.
6. MAT1/MAT3 auto microstructure triage is stricter and keeps decisive language gated.
7. Strict E3/E4 primary physical-table rules are recorded in outputs; secondary extraction remains diagnostic only.
