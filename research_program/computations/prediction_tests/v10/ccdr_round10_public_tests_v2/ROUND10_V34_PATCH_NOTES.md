# Round-10 v34 confirm-target patch

Focus: convert the P30-SDSS curl-clean near-confirm into a quantitative route-specific residual test, while keeping global P30 blocked until an independent route passes.

Implemented improvements:

1. P30-SDSS-core route: baseline/f150/tonly are treated as the core family; f090/cibdeproj remain systematics-sensitive controls.
2. Curl-template regression is quantitative: beta, residual delta, projected delta, and residual-shuffle p-values are written per core variant.
3. Residual-curl null after projection: curl is regressed against core science and the remaining high-low curl residual is tested.
4. Paired bootstrap science-vs-curl: object-level high/low resampling tests whether core science remains above absolute curl.
5. Patch-level jackknife: RA/Dec tiles test whether core residual signal is positive across sky patches and stronger than curl.
6. Route-specific claim separation: `P30-SDSS-core` can be promoted only if v34 residual tests pass; global P30 still requires a second independent route.
7. P3 endpoint prefilter: avoids giant non-endpoint downloads; full table parsing only proceeds after explicit endpoint/node-pair metadata is detected.
8. P36 high-z object parser contract: source-specific KMOS3D/KGES/KROSS/MOSDEF/SAMI object catalog requirements are written to outputs.
9. P41 Wilson/SM likelihood contract: major P41 claim remains blocked until q²/value/error rows and a Wilson-vs-SM Δχ² exist.
10. P33/P32 measurement contracts: P33 requires the first measured alpha split; P32 requires a minimal GW150914-style strain run.

Key output files after the next run:

- outputs/p30_sdss_core_curl_projected_residual_v34.json
- outputs/p36_highz_object_parser_contract_v34.json
- outputs/p41_wilson_sm_likelihood_contract_v34.json
- outputs/p33_first_alpha_split_contract_v34.json
- outputs/p32_one_event_strain_contract_v34.json
