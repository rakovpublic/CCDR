# v12 strict physical-artifact autodiscovery and data expansion

This patch implements the seven requested result-quality updates for data-limited tests without adding manual steps.

## 1. Repository metadata is no longer counted as candidate physical data

Zenodo search/record JSON, Figshare article/search JSON, OSF API JSON, Crossref/OpenAlex/DataCite/Semantic Scholar JSON, and HEPData search/record metadata are now `metadata_record` artifacts. They can discover links, but they do not increment `candidate_table_count` and cannot be evidence.

## 2. File-level repository connectors

The autodiscovery layer now follows repository metadata to actual downloadable files:

- Zenodo `hits.hits[].files[]`, record `files[]`, files endpoint, and archive links.
- Figshare POST search via `figshare_search://...`, article metadata, and `download_url` file links.
- OSF API file tree download links and parent/related folders.
- HEPData search/record links to actual `/download/table/.../csv` artifacts.

Only downloaded physical files can become primary candidates.

## 3. Domain-specific physical table gates and relevance scoring

A parsed frame is now a candidate only if it contains physical-observable hints for its test. Reports distinguish:

- `metadata_records_seen_count`
- `nonphysical_tables_parsed_count`
- `physical_candidate_table_count`
- `candidate_table_count` = physical candidate table count only

Tables include `physical_hint_score`, `table_relevance_score`, and specific rejection reasons.

## 4. Better PDF extraction but safe by default

PDF extraction now uses bounded `pdfplumber` and text-line physical table fallback. Camelot is available only when explicitly enabled with:

```powershell
$env:CCDR_ENABLE_CAMELOT=1
```

The default PDF page budget is bounded by:

```powershell
$env:CCDR_PDF_TABLE_PAGES=4
```

PDF-derived tables remain secondary and cannot confirm/falsify.

## 5. More exact automated seeds

The patch adds additional query seeds and exact manifest-driven seeds for:

- fusion T26/T27/T29,
- electronics T44/T45/T47,
- HEP/cosmic T57/T59 using bundled HEPData exact URLs,
- metrology T50/T51/T52,
- photosynthetic coherence T54.

## 6. Auto microstructure enrichment

MAT1/MAT3 now harvest CMB-S4 sibling `references.txt`, `README.md`, and fit metadata URLs when available. It labels automatic rows as measured/explicit nanocrystalline, grain-boundary candidate, composite/fiber proxy, amorphous control, or bulk control. This remains triage unless enough high-confidence measured/explicit rows are found.

## 7. Runtime bounding

Autodiscovery now has a wall-clock budget per test based on `--timeout`, and `run_all_tier_b.py` has a `--script-timeout` argument to prevent runaway discovery processes.

Example:

```powershell
python run_all_tier_b.py --only T26 T27 T29 T57 T59 --outdir tierb_out_v13 --cache tierb_cache_v13 --force --timeout 15 --script-timeout 900
```

