# CCDR public-data application tests — patched v3

This patch changes the philosophy of the bundle:

- scripts must either compute a numeric statistic from public machine-readable data, or
- explicitly return `not_executable_*` with the exact missing columns/table needed.

The previous `seed_corpus_only` Crossref/article-harvesting outputs were removed for FR3, FR8, MAT2, EL3, and EL8 because article lists are not prediction tests.

## Main changes

- `test_mat1_ccdr_mu_vs_casimir_public.py`: now scans public CMB-S4 cryogenic conductivity tables and computes low-T exponents as a proxy audit.
- `test_mat3_lowT_sqrtT_public.py`: now scans many public conductivity files and tests whether grain/nano candidates have alpha near 0.5.
- `test_fr4_transport_vs_collisionality_public.py`: broader HDB column recognition; computes a non-monotonicity proxy only if required columns are found.
- `test_fr7_mkss_vs_tauE_public.py`: broader HDB column recognition; computes Spearman correlation only if required columns are found.
- `test_mat7_hbn_si_public.py`: adds direct Bristol static-file fallback for Figure CSVs if CKAN resets the connection.
- FR3, FR8, MAT2, EL3, EL8: no longer harvest articles; they now return data-gate reports explaining why a public machine-readable numeric test is not executable yet.

## Running

```powershell
python .\run_all_public_data_tests.py
```

Outputs are written under `outputs/<prediction>/` as JSON and plots where applicable.

## Patch 5 notes
- FR7 sign convention corrected: if `M_KSS` is a transport margin, the expected relation with `tau_E` is negative, not positive. The script now labels the old `a^2/tau_E` proxy as mechanical/weak because it contains `tau_E` by construction, and it searches for H-factor or independent transport columns.
- MAT1 now reports a stricter confirmation ladder: full confirmation requires grain-size metadata; near-half alpha is stronger proxy support; broad deviation from alpha≈3 controls is weak support only. It also probes the public ACS Figshare nanocrystalline-silicon collection for machine-readable files.

### Patch 10: EL3 / EL8 batch handling

`test_el3_si_area_volume_crossover_public.py` and `test_el8_optical_vs_via_public.py` are no longer run by default because without harmonized numeric benchmark tables they only produce data-gate/no-verdict reports.

Run them explicitly with:

```bash
python run_all_public_data_tests.py --include-data-gates
# or
python run_all_public_data_tests.py --only test_el3_si_area_volume_crossover_public.py
```

To turn them into numeric tests, provide public CSV/XLSX URLs:

```powershell
$env:EL3_NUMERIC_URLS="https://example.org/interconnect_scaling.csv"
$env:EL8_NUMERIC_URLS="https://example.org/optical_vs_electrical_links.csv"
python .\run_all_public_data_tests.py
```

EL3 expected columns: `node_nm`/`linewidth_nm` plus resistance/delay/energy metric.  
EL8 expected columns: technology type, node/feature size, energy per bit, and optionally bandwidth/reach.


## Patch 12 notes

- `MAT6` is no longer run by default because the bundled public Figshare source is PDF-only and produced no machine-readable verdict. Enable it with `MAT6_NUMERIC_URLS=<public CSV/XLSX URL>` or `--include-data-gates`.
- `run_all_public_data_tests.py --compact` prints one compact JSON verdict per test instead of huge full reports.
- `MAT7` now reuses cached Bristol CSVs before retrying slow direct downloads.

## Patch 14 notes

- `_common_public_data.py`: `CCDR_HTTP_TIMEOUT` now controls HTTP timeout globally (default 30 seconds), so blocked networks do not hang for long default request windows.
- `run_all_public_data_tests.py`: added `--script-timeout` and `--http-timeout`; compact JSON extraction now scans for the final valid JSON object instead of using a greedy regex.
- `MAT1`, `MAT3`, `MAT6`, and `MAT7`: source-unavailable/network failures now produce structured `source_unavailable` / `no_physics_verdict` reports instead of crashing or being misread as null physics evidence.
- `MAT7`: avoids repeated static URL probes when CKAN metadata is unavailable unless `MAT7_TRY_STATIC_ON_CKAN_FAIL=1` is set; cached Bristol CSVs are still reused.
- `EL1`: aborts the live scrape after a global DNS/network failure and still reports the curated benchmark separately with `live_scrape_complete=false`.
- `FR3`, `EL3`, and `EL8`: direct `requests.get(... timeout=60)` calls now use the shared configurable timeout.

Recommended quick smoke run on unreliable networks:

```powershell
python .\run_all_public_data_tests.py --compact --http-timeout 8 --script-timeout 180
```

Use the default run when network access is stable:

```powershell
python .\run_all_public_data_tests.py
```

## Patch 15: HDB parser + FR7 upgrade

- Fixed FR4/FR7 failure on public HDB OSF exports whose first bytes look like `\ufeff,1,2,3,...`.
  These files are transposed variable-by-row CSV matrices; `_common_public_data.repair_header_if_needed()` now converts them to one-shot-per-row tables before column matching.
- `read_public_table()` now has a ragged-CSV fallback and UTF-8-BOM handling, so public CSV exports with uneven rows do not fail before header repair.
- FR7 now tests all recognized H-factor columns, adds global and device-stratified permutation checks, reports device-stratified residual summaries when a device column exists, and keeps the final verdict conservative: the mechanical `a^2/tau_E` proxy alone is not confirmation.
- FR7 still requires a true independent transport/viscosity/diffusivity column, or published edge `eta/s`/`chi_edge`, for a decisive physics verdict.

## Patch 16 notes
- Fixed FR4/FR7 crash in `_maybe_transpose_public_matrix`: pandas on the Windows run passed a float into the BOM-stripping lambda despite `astype(str)`, causing `AttributeError: 'float' object has no attribute 'replace'`.
- HDB transposed-matrix detection now uses plain Python scalar conversion and can handle more public HDB variants.
- FR7 p15 residual/H-factor improvements are preserved; p16 only hardens the shared HDB parser so the test reaches physics analysis.
