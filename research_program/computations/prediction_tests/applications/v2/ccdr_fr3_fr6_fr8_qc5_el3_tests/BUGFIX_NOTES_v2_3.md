# v2.3 EL3 fix

This patch fixes the EL3 interpretation problem found in the v2.2 run.

## Problem

v2.2 merged two heterogeneous observables into one piecewise fit:

1. historical full-chip transistor density from processor rows, and
2. modern foundry standard-cell density from public process-node pages.

That combined fit produced a misleading overall `fail_or_inconclusive_proxy` and an artifact-like breakpoint because the two datasets measure different objects.

## Fix

EL3 is now split into two explicit subtests:

- `EL3a_historical_chip_density`: full-chip transistor density vs process/node proxy.
- `EL3b_foundry_standard_cell_density`: modern foundry standard-cell density vs process-node proxy.

The top-level decision is now an evidence synthesis:

- `robust_proxy_pass`: both separated proxies pass.
- `mixed_inconclusive`: one separated proxy passes and the other does not.
- `not_supported_by_current_public_proxy`: both separated proxies fail/inconclusive.

The combined historical+foundry fit is retained as `combined_diagnostic_fit_not_used_for_decision` only.

## Numerical stability

`piecewise_log_slope()` now skips candidate splits with too few unique node values or poorly conditioned polynomial fits. This suppresses the repeated `RankWarning: Polyfit may be poorly conditioned` messages that appeared for modern foundry tables with many repeated node labels such as 3, 5, 7, and 10 nm.

## Foundry-node handling

Foundry rows are fit using node-median aggregation by default, so multiple process variants at the same nominal node do not dominate the breakpoint search.
