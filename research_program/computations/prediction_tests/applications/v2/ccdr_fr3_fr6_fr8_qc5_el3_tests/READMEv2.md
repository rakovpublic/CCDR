# CCDR engineering public-data proxy tests: FR3, FR6, FR8, QC5, EL3

These scripts implement public-data-only tests for the engineering predictions in the uploaded CDT–CCDR–TN engineering document.

## Philosophy

Every source artifact is downloaded by the script from a public URL. No manual files are accepted. When the public literature exposes only plots, ranges, or prose rather than machine-readable shot/device rows, the relevant test returns `data_insufficient_public_proxy` instead of manufacturing a positive result.

## Install / update dependencies

The PDF tests need `pdfminer.six`. If your previous run printed `No module named 'pdfminer'`, reinstall requirements with `--upgrade` inside the same environment used to run the tests.

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install --upgrade -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt
```

## Run everything

```bash
python run_all.py
```

Outputs are JSON files:

* `out_fr3_fr6.json`
* `out_fr8.json`
* `out_qc5.json`
* `out_el3.json`
* `out_summary.json`

`run_all.py` v2.2 flattens nested decisions, so the summary now reports separate FR3/FR6 statuses instead of `decision: null` for the combined script.

## Individual tests

### FR3 / FR6

```bash
python test_fr3_fr6_literature_proxy.py --out out_fr3_fr6.json
```

FR3 strict mode can accept a public CSV URL if a real open supplement is found later:

```bash
python test_fr3_fr6_literature_proxy.py --fr3-csv-url https://example.org/public_elm_rows.csv
```

Required FR3 columns, with flexible aliases: `E_ELM`, `P_ped`, `V_ped`, `dP_frac`.

The FR6 proxy uses downloaded MAST RMP-current/frequency values from the public paper text. It tests the sign/linearity of applied-field response, not the full helicity integral `H_mag = ∫A·B d³x`.

### FR8

```bash
python test_fr8_eta_s_collisionality_proxy.py --out out_fr8.json
```

This currently reports whether a decisive public eta/s-vs-collisionality table exists. It downloads QGP/KSS public literature and DIII-D collisionality ranges. It does not convert ELM frequency or pedestal pressure into eta/s unless a public paper provides that conversion.

### QC5

```bash
python test_qc5_phononic_crystal_t1_public.py --out out_qc5.json
```

QC5 is now explicitly split:

* `QC5a_TLS_mechanism`: complete phononic/acoustic bandgap suppresses embedded TLS/defect relaxation and reaches ms scale with >2–5x improvement.
* `QC5b_transmon_T1`: standalone transmon-qubit `T1 > 1 ms` inside a complete bandgap and >2–5x over controls.

TLS-only >1 ms is mechanism support, not full transmon confirmation.

### EL3

```bash
python test_el3_si_area_crossover_public.py --include-wikipedia --include-foundry-nodes --out out_el3.json
```

The default historical proxy uses transistor count, area, and process node data to fit `density ∝ L^-alpha` and detect a two-slope breakpoint. `--include-wikipedia` adds best-effort parsing of process-node density tables from downloaded public HTML. `--include-foundry-nodes` verifies modern foundry-node density rows against downloaded public HTML and writes an inspectable CSV:

```text
el3_cache/foundry_node_density_public_extracted.csv
```

This strengthens the previous run where `n_wikipedia_rows = 0` by adding public-source-gated rows for Intel/Samsung/TSMC/SMIC 22/16/14/10/7/5/4/3 nm-class nodes when the page text contains the quoted density values.

## Decisiveness levels

* `pass` / `pass_proxy`: metric satisfied on downloaded public data.
* `partial_support_*`: mechanism supported, but not the exact prediction.
* `data_insufficient_public_proxy`: public data are inadequate for a falsification-safe metric.
* `fail_or_inconclusive_proxy`: metric did not pass or was unstable at proxy level.


## v2.3 EL3 interpretation fix

EL3 is now split into two like-for-like public proxies:

- `EL3a_historical_chip_density`: historical full-chip transistor density.
- `EL3b_foundry_standard_cell_density`: modern foundry standard-cell density.

The script no longer uses the heterogeneous combined fit for the pass/fail decision. The combined fit is still written as `combined_diagnostic_fit_not_used_for_decision` so you can inspect it, but the top-level EL3 status is now an evidence synthesis (`robust_proxy_pass`, `mixed_inconclusive`, or `not_supported_by_current_public_proxy`).

Foundry rows are also aggregated by node median before fitting to avoid repeated 3/5/7/10 nm variants dominating the breakpoint search. Poorly conditioned candidate splits are skipped to avoid `RankWarning` noise.

## v2.4 EL3 note

EL3 was improved in v2.4. The script now corrects small-node vs large-node slope labels,
adds a constrained 10-100 nm hypothesis fit, and records stronger row-level provenance for
foundry-node proxy rows. Treat `advertised_node_nm` as a marketing/process-node proxy, not a
literal physical gate length.

## v2.5 EL3 robustness notes

EL3 now reports three separated views instead of one combined result:

- `EL3a_historical_chip_density_raw`
- `EL3a_historical_chip_density_node_median`
- `EL3b_foundry_standard_cell_density`

For each view, inspect:

- `global_fit`
- `hypothesis_fit_10_100nm`
- `subsample_robustness`
- `local_slope_profile`
- `decision.subsample_supportive_fraction`

A `window_compatible_proxy` result means the constrained 10--100 nm fit is competitive with the global fit, not that EL3 is confirmed. A `weak_window_compatible_proxy` result means compatibility exists in the full fit but is fragile under subsampling.

## v2.7 additions

v2.7 adds a uniform evidence ladder and new sub-branches:

* FR3 public supplement crawler and reduced pedestal proxy.
* FR6 multi-paper RMP/frequency text extraction plus helicity-quality tiers.
* FR8 split into QGP context, fusion transport proxy, and true edge eta/s/M_KSS.
* EL3 confidence-filtered decisions plus `EL3c_physical_pitch_density_proxy`.
* QC5 split remains permanent.

The evidence ladder is:

```text
0 = data_missing
1 = qualitative_context
2 = partial_proxy
3 = window_compatible_proxy
4 = robust_proxy
5 = decisive_public_test
```

To disable FR3 supplement crawling:

```bash
python test_fr3_fr6_literature_proxy.py --no-fr3-supplement-crawl --out out_fr3_fr6.json
```


## v2.8 notes

v2.8 fixes four audit issues from v2.7:

1. FR3 now exposes separate evidence ladders for full FR3 and reduced FR3b.
2. FR6 multi-paper rows are deduplicated and validated; rejected parser rows are reported.
3. FR8 reports separate decisive and context evidence levels.
4. EL3 separates primary physical-pitch CSV rows from secondary placeholder rows.

Optional EL3 primary physical-pitch input:

```powershell
python test_el3_si_area_crossover_public.py --include-wikipedia --include-foundry-nodes --el3-physical-csv-url https://example.org/public_cpp_mmp_sram.csv --out out_el3.json
```

Expected CSV columns are flexible, for example:

```text
company,process_name,advertised_node_nm,cpp_nm,mmp_nm,density_MTr_per_mm2
```

or

```text
company,node_name,advertised_node_nm,sram_bitcell_um2,standard_cell_density_MTr_per_mm2
```

If no public physical-pitch CSV URL is supplied, EL3 primary-only physical-pitch evidence remains `data_insufficient_public_proxy`.
