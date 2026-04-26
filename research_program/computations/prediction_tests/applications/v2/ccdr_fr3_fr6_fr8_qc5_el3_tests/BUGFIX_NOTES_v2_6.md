# v2.6 improvements

## EL3

* Added explicit volume-to-area transition scoring. A proxy can no longer pass only because the advanced/small-node side has an area-like exponent; the larger-node side must be steeper by a minimum alpha drop.
* Added `alpha_drop_large_minus_small`, `crossover_like_alpha_drop`, `strong_volume_to_area_alpha_drop`, and `large_segment_volume_like` flags to global and constrained-window fits.
* Kept the v2.5 split: raw historical full-chip density, node-median historical full-chip density, and strict foundry standard-cell density.
* Added `density_kind_counts` so mixed realized-chip / foundry-standard-cell inputs are auditable.
* Overall status now distinguishes weak-foundry support as `mixed_to_supportive_proxy_weak_foundry` instead of overclaiming a robust pass.

## FR3 / FR8

* Improved DIII-D Knolker Table-2 extraction. The parser now tries a local window, the full PDF text, and a downloaded-text-gated fingerprint fallback for range-level rows.
* The fallback is explicitly range-level only and never promotes FR3 to a decisive shot-cycle result.
* FR8 now includes the Knolker extraction metadata and an additional W7-X public PDF source for stellarator context.
* FR8 still refuses to compute eta/s or M_KSS from f_ELM/collisionality unless a public source gives a valid conversion or machine-readable rows.
