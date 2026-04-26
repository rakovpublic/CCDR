# v2.4 EL3 improvement notes

## Main scientific/code fix

`piecewise_log_slope()` sorts node labels from small to large. In earlier bundles the
first segment was labelled as `pre_alpha_large_nodes` even though it was actually the
advanced/small-node segment. This could invert EL3 interpretation.

v2.4 adds explicit corrected fields:

- `small_node_alpha`
- `large_node_alpha`

The legacy aliases are retained but corrected:

- `post_alpha_small_nodes` = `small_node_alpha`
- `pre_alpha_large_nodes` = `large_node_alpha`

## EL3 analysis improvement

EL3 no longer uses only one global best breakpoint. Each separated observable now reports:

- `global_fit`
- `hypothesis_fit_10_100nm`
- `hypothesis_penalty_vs_global`

A result can be marked `window_compatible_proxy` when the constrained 10-100 nm fit is
area-like and nearly as good as the global fit. This avoids rejecting the EL3 window just
because a broad historical trend has a slightly lower global SSE outside the target window.

## Provenance improvement

Foundry rows now include additional proxy/provenance fields when available:

- `advertised_node_nm`
- `density_kind`
- `source_url`
- `source_sha256`
- `evidence_tokens`
- `evidence_snippet`
- `confidence`
- `data_role`

These fields make clear that modern foundry rows are public secondary standard-cell / reported
density proxies, not direct physical length measurements.
