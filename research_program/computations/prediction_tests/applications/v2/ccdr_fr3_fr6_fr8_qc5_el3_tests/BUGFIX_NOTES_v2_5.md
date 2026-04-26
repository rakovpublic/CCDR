# v2.5 EL3 robustness improvement

This release makes EL3 more conservative and more informative:

- Separates three EL3 views:
  - `EL3a_historical_chip_density_raw`
  - `EL3a_historical_chip_density_node_median`
  - `EL3b_foundry_standard_cell_density`
- Filters EL3b rows to foundry-standard-cell / reported-density rows only; realized full-chip densities are kept out of the foundry decision.
- Adds deterministic subsample robustness diagnostics:
  - `subsample_supportive_fraction`
  - `pass_proxy_fraction`
  - `window_compatible_fraction`
  - break-node and alpha quantiles
- Adds rolling local-slope diagnostics by node window, so EL3 does not depend only on one best breakpoint.
- The final EL3 decision remains non-decisive unless like-for-like observables show stable support.

The combined historical+foundry fit is still retained only as a diagnostic and is not used for pass/fail.
