# CCDR v7.2 / Synthesis v3.2 public-data nine-test battery

This bundle implements nine empirical tests derived from the current CCDR v7.2 / Synthesis v3.2 scorecard and follow-up priorities.

Included tests:
1. test01_p30_non_euclid_lensing_replication.py — P30 replication with Euclid Q1 galaxy density against ACT/Planck public lensing products (proxy-level if no readable map is available).
2. test02_p30_euclid_systematics_audit.py — Euclid-Q1 internal nuisance/split audit of the density–kappa signal.
3. test03_highz_a0_sparc_kmos3d.py — local SPARC anchor plus public high-z KMOS3D-like proxy for a0(z).
4. test04_nu_public_redrive.py — Pantheon+ + DESI DR2 + Planck public re-drive of the nu claim with tomography and nuisance splits.
5. test05_p3_euclid_filament_ordering.py — Euclid-Q1 transparent-finder filament ordering with density-stratified null controls.
6. test06_p29_multi_redshift_growth.py — public DESI full-shape/RSD proxy test for ongoing DM abundance growth.
7. test07_dm_window_readiness_tracker.py — hard-core readiness tracker for the 0.5–3 TeV second-peak window.
8. test08_p31_dm_peak_drift_audit.py — direct-detection drift audit on public HEPData tables.
9. test09_p8c_nanograv_cosmic_web_sign.py — NANOGrav 15-year × cosmic-web sign cross-check proxy.

All scripts download from public sources at runtime. Several are explicitly proxy/screening/readiness implementations rather than collaboration-grade analyses, because fully machine-readable public products are uneven across the relevant surveys.
