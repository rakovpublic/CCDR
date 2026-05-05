# v15 exact-source resolution and data-limited hardening

Implemented four requested improvements:

1. Exact source manifests for fusion, electronics, metrology, biology/coherence and HEP/cosmic tests. These contain public URLs only; no manual numeric rows.
2. Source-specific extractors/connectors through stricter repository record relevance, HEPData direct CSV seeds, fusion PDF/schema routes, and electronics/metrology/bio exact discovery routes.
3. Archive-size and relevance guards to avoid multi-GB broad false-positive downloads while allowing exact curated artifacts.
4. Explicit public-unavailability statuses such as schema_found_data_file_not_public_or_not_discovered and no_public_primary_physical_table_found_after_exact_source_search.

Also added more automated source seeds for T26-T30, T44/T45/T47, T50-T52, T54, T57 and T59.
