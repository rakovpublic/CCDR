# Round-10 v19: all 10 confirmation-from-positive-ready improvements

Generated: 2026-05-07T08:02:56Z

Implemented:
1. P30 Euclid quality cuts: clean non-spurious/non-point, extended-only, low-spurious-prob, high-confidence galaxy subsamples.
2. P30 alternate density estimators: HEALPix nside 64, HEALPix nside 128, and KNN-k16 density proxy.
3. P30 ACT mask-aware diagnostics: finite-footprint remains explicit; no confirm-like without explicit mask-aware robustness.
4. P30 variant matrix retained/formalized: baseline, frequency/cleaned science variants, curl control, science/curl summary.
5. P30 tension-preserving promotion: quality-cut near-confirm only when multiple quality subsamples plus variants agree.
6. P3 endpoint orientation parser: exact limited endpoint rows, endpoint-shuffle statistic, redshift/density null plan.
7. P41 structured supplementary extraction: structured rows, sign basis, CP-asymmetry null summary.
8. High-z Vrot confirm-candidate guard: FITS unit hits, quality gates, leave-one-table-out readiness.
9. CL2 residual-weighted path and direct-detection unit-verified coverage guard.
10. Dashboard tension bucket above positive-ready: confirm_like, near_confirm, tension, guarded, coverage.
