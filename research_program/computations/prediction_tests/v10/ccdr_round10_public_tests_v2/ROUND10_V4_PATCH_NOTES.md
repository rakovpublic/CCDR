# Round-10 v4 blocker-resolution patch

Generated: 2026-05-03T21:05:25Z

Implemented blocker fixes:
- DESI DR2 BAO switched from 401 NERSC/file-server paths to public CobayaSampler/bao_data GitHub contents.
- ACT DR6 endpoint corrected to ACT AdvACT DR6 lensing maps LAMBDA pages.
- Planck lensing/y-map endpoints updated to PR3 ancillary index and PLA product-action candidates.
- VAST Zenodo record fixed from unrelated 6944382 to VAST SDSS DR7 record 7406035.
- GW170817 path changed from brittle per-event URL to GWOSC GWTC-1 catalogue/event candidates.
- Direct-detection pages changed from blocked HEPData search to exact LZ HEPData record 155182, official XENON results/arXiv, and PandaX public CSV endpoints.
- P41 changed from blocked HEPData search to CDS/arXiv parser.

Implemented high-priority upgrades:
1. SPARC full RAR/a0 grid fit: `run_sparc_rar_a0`.
2. Pantheon+ plus DESI DR2 BAO diagnostic hook: `run_pantheon_bao_joint`.
3. BK18 tarball unpacker and BB/bandpower candidate finder: `run_bk18_unpack_summary`.
4. FIRAS blackbody + toy μ/y least-squares bound: `run_firas_mu_y_fit`.
5. P41 CDS/arXiv observable-hook parser: `run_p41_cds_parser`.

Also added SM-D5 Koide constant-level checker to turn the previous readiness inventory into a clean consistency-positive.
