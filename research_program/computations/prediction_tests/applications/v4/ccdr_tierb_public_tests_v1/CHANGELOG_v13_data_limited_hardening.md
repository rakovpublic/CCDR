# v13 data-limited hardening patch

Tier-B only. Implements the five requested improvements for data-limited tests:

1. **Metadata evidence-tier fix**: repository/search/bibliographic JSON from Zenodo, Figshare, OSF, Crossref, OpenAlex, DataCite, Semantic Scholar, and HEPData is link-source metadata only. It is never parsed as a primary physical table.
2. **HTML/SVG/icon noise blocking**: generic web boilerplate, SVG/icon/css/js/analytics fragments, and zero-numeric/no-domain-hint HTML tables are rejected before unit normalization or sensitivity scoring.
3. **File-first repository connectors**: Zenodo/Figshare/OSF/HEPData connectors now prefer record files, content/download URLs, and table download endpoints over metadata frames.
4. **Recursive archive traversal**: zip/tar/tar.gz/tgz/gz artifacts are recursively inspected; table/source-data members are parsed, LaTeX tables are extracted, nested archives are followed, and common false positives are rejected.
5. **Domain-specific physical gates**: every candidate must contain contract/domain physical hints before candidate_table_count can increase. Candidate tables now mean physical candidates, not metadata or page fragments.

Scientific policy unchanged: only primary machine-readable physical tables may confirm/falsify. PDF/figure/arXiv-derived tables remain secondary diagnostics.
