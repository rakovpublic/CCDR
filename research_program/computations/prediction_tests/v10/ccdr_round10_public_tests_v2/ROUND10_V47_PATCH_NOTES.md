# Round-10 v47 patch notes

Focus: fix P36 high-z T13/T14 executable failures.

Implemented:

1. Replaced the heavy inherited high-z runner path for `highz_unit_field_table_v22` with `run_highz_unit_field_table_v47`.
2. T13/T14 now consume persisted local high-z object rows/audit products instead of running an unbounded catalogue crawler.
3. If no rows exist, T13/T14 return `highz_object_catalogue_data_limited_v47`, not `broken`.
4. If rows exist but hard gates fail, T13/T14 return `highz_a0_objectlevel_second_source_needed` or `highz_a0_objectlevel_gate_failed_v47`, not `broken`.
5. If rows pass all gates, T13/T14 return `highz_a0_objectlevel_publication_confirm_like`.
6. Writes `outputs/p36_t13_executable_guard_v47.json` and `outputs/p36_t14_executable_guard_v47.json`.
7. Writes row-level CSV audit outputs for T13/T14.
8. Adds bounded landing-page probes only when no local rows exist; no heavy downloads are attempted.
9. Adds dashboard v47 accounting so broken/not-run high-z tests cannot promote P36 high-z.
10. Keeps P36 local a0 unaffected.

Claim policy: P36 high-z promotion is only allowed when executable T13/T14 v47 guard outputs exist and pass. Missing/catalogue-unavailable data is data-limited, not broken.
