# Round-10 v58 patch notes — stronger no-manual confirm recovery

v58 keeps the v57 rule: the user does not need to hand-fill artifacts. The runner tries to auto-build strict artifacts from public/cached data, but does not promote tests unless the evidence passes predeclared gates.

Implemented improvements:

1. Stronger P36 high-z public/cache parser for KROSS/KGES/KMOS3D/SAMI/MOSDEF/PHIBSS-like tables.
2. P36 high-z unit heuristics for radius columns, including pc→kpc conversion when explicit or strongly implied.
3. P36 high-z diagnostics now separate `public_rows_absent` from `rows_parsed_but_gate_failed`.
4. P30 route-specific same-mask diagnostics with science/curl ratio and same-sign variant count.
5. P30 protocol artifact now documents active route assumptions while still blocking post-hoc/global promotion.
6. P33 alpha measurement normalization is stricter and reports why no density-split BAO fit exists.
7. PTA/CL2 gate now distinguishes coordinate readiness from missing weighted κ×residual statistic.
8. P32/P40/P41 likelihood gates now write v58 artifacts with clearer missing-field diagnostics.
9. SMD derivation gate remains strict and refuses consistency→derivation promotion without no-target-fit metadata.
10. Dashboard v58 reports diagnostic classes: no public rows, curl/variant tension, missing likelihood artifacts, etc.

Expected behavior: no extra confirms unless public/cached data contain real rows/statistics. P38 and P36-local should remain the main confirm core until P36-high-z/P30/P33 evidence is genuinely available.
