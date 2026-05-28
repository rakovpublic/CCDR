# Round-10 v46 confirm-target hardening

Implemented the next 10 improvements from the v45 report:

1. P30 mask-equivalence acceptance gate: writes a candidate appendix but requires an official ACT mask or a user-supplied accepted appendix before promotion.
2. P30 frozen patch protocol: defines min counts, curl-amplitude threshold, positive science-minus-|curl| rule, leave-one-patch stability, and no post-hoc patch rescue.
3. P30 fresh-rerun guard: promotion requires a fresh rerun manifest matching the frozen protocol hash.
4. P36 publication audit appendix: writes high-z source/object/z/Vrot/radius/acceleration/unit provenance JSON and CSV.
5. P36 blinded unit-conversion stress: recomputes acceleration from Vrot and radius and blocks if unit consistency fails.
6. P41 measured q² Wilson likelihood consumer: requires measured q²/value/error rows, SM and Wilson predictions, Δχ²>=9, and CP controls null/weak.
7. P33 measured BAO α-split consumer: requires alpha_high, alpha_low, delta_alpha, covariance, density/sky/null and redshift jackknife.
8. P32 GWOSC endpoint resolver v46: API-first endpoint discovery plus H1/L1 product tracking; strain download alone still cannot promote.
9. SMD prediction metadata v46: supports external pre-registered prediction JSON and residual-sigma reporting; constants stay constant-level until derivation/cross-prediction passes.
10. Dashboard v46: keeps Bucket A non-SM publication confirms, Bucket B SM constant-level checks, Bucket C near-confirm/readiness targets.

Promotion policy remains strict: readiness rows, templates, diagnostic masks, and same-run patch quarantine cannot become confirm-like.
