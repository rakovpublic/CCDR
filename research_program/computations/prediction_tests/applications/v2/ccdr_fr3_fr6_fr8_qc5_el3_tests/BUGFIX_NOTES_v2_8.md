# v2.8 bugfix notes

This release fixes the four audit issues identified after the v2.7 run.

## 1. FR3 ladder clarity

`FR3.evidence_ladder` still refers only to the full parameter-free FR3 scaling
test. v2.8 adds explicit split fields:

- `FR3_full_evidence_ladder`
- `FR3b_reduced_proxy_evidence_ladder`
- `evidence_ladder_note`

This avoids hiding the weak FR3b reduced proxy under the full-FR3 level-0 verdict.

## 2. FR6 text-row validation and tiering

`parse_rmp_frequency_text_proxies()` now:

- de-duplicates rows by source/current/frequency/mode rather than full context;
- rejects obvious 0 kAt / RMP-applied frequency mispairings;
- reports `n_rejected_rows` and `rejected_rows_sample`;
- promotes rows with current plus `n = ...` or q-profile hints to Tier 2;
- reports `max_tier_observed`.

Tier 3 remains reserved for true public equilibrium/coil-geometry based helicity or
separatrix perturbation calculations.

## 3. FR8 split-level evidence summary

FR8 now reports:

- `FR8_decisive_evidence_level`
- `FR8_context_evidence_level`
- `FR8_split_evidence_summary`

The decisive FR8c eta/s or M_KSS branch can remain level 0 while the QGP/fusion
context branches remain level 1 without being confused for confirmation.

## 4. EL3 primary vs secondary physical-pitch branches

EL3 now separates:

- `EL3c_primary_physical_pitch_density_proxy`
- `EL3c_secondary_physical_pitch_density_proxy`
- backward-compatible alias: `EL3c_physical_pitch_density_proxy`, which points to the primary branch.

The new optional argument:

```bash
--el3-physical-csv-url <public_csv_url>
```

loads public primary/semi-primary CPP/MMP/SRAM physical-feature rows. Expected
columns are flexible: `advertised_node_nm`, `physical_feature_nm` or `cpp_nm` /
`mmp_nm` / `metal_pitch_nm`, and optionally `density_MTr_per_mm2`. If density is
missing, the script may pair the physical feature with foundry density at the
same advertised node and marks this explicitly in `density_source_kind`.

The default run remains conservative: without such a public CSV, primary-only
physical-pitch evidence stays data-insufficient.
