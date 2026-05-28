# Round-10 v16: blocker fix and confirmation squeeze

Generated: 2026-05-06T06:38:11Z

Implemented 10 improvements:
1. Canonical _ra_dec_to_theta_phi helper and backward alias to fix P30 T04 NameError.
2. ACT map science gate: n_finite_pixels > 100000, finite_fraction > 0.1, std > 0.
3. ACT ALM probes: row-order, sparse zero/one-based index, ell/m, healpy direct.
4. P30 degrades to act_map_reconstruction_failed_positive_ready instead of broken when map fails.
5. CL2 uses fixed coordinate helper and ACT science gate.
6. P38 family-split hardening for robust-confirm label.
7. High-z a0 Vrot unit audit; no confirm-like without unit verification.
8. P41 structured row extraction; remains guarded unless rows+sign basis exist.
9. Direct detection coverage schema separated from detection claim.
10. P3 limited VizieR metadata mode with -out.max=50 to avoid huge ASU downloads.
