# v2.7 improvements

Implements the ten requested evidence-quality improvements.

## 1. FR3 public supplement crawler

`test_fr3_fr6_literature_proxy.py` now crawls public FR3 source pages/PDF endpoints for supplement links (`csv`, `tsv`, `txt`, `dat`, `xlsx`, `xls`, `zip`, `json`, `yaml`). Parsed supplement rows are passed through the strict FR3 column-alias filter. The result is included under `FR3.literature_evidence.supplement_crawler`.

This crawler is conservative: it does not digitize figures and does not turn range-level PDF text into decisive rows.

## 2. FR3 reduced proxy

Adds `fr3b_reduced_pedestal_pressure_proxy`, which only uses range-level pedestal pressure / ELM-frequency information. This can never confirm FR3, but it reports whether a public reduced trend is worth pursuing.

## 3. FR6 multi-paper text extractor

Adds multi-source RMP/frequency text extraction across the downloaded fusion PDFs. Extracted snippets are reported under `FR6.multi_paper_text_proxy`.

## 4. FR6 helicity-quality tiers

FR6 rows are now classified by evidence quality:

- Tier 0: RMP/frequency text only.
- Tier 1: coil current + frequency.
- Tier 2: coil current + mode number/q-profile proxy.
- Tier 3: public equilibrium/coil geometry sufficient for true separatrix perturbation/helicity proxy.

The existing MAST numeric regression remains Tier-1 proxy evidence, not decisive FR6.

## 5. FR8 split

`test_fr8_eta_s_collisionality_proxy.py` now reports:

- `FR8a_QGP_eta_over_s_context`
- `FR8b_fusion_transport_proxy`
- `FR8c_true_edge_eta_or_MKSS`

This prevents QGP context or fusion f_ELM/collisionality proxies from being mistaken for a true edge eta/s or M_KSS test.

## 6. FR8 computable transport proxy scaffold

Adds a named `M_transport_proxy` scaffold for future public profile rows. The current script still refuses to compute eta/s or M_KSS from f_ELM unless a public source provides a defensible conversion.

## 7. EL3 source-quality fields

EL3 rows now keep stronger provenance fields and confidence-filtered decision blocks, including `EL3_primary_only`, `EL3_all_public_sources`, and `EL3_wikipedia_proxy_only`.

## 8. EL3 physical-pitch/SRAM branch

Adds `EL3c_physical_pitch_density_proxy`, a conservative low-confidence physical-feature branch. It currently uses verified public process-page tokens and paired foundry densities, so it is marked as proxy-level until real CPP/MMP/SRAM rows are added.

## 9. QC5 permanent split

QC5 remains permanently split into:

- `QC5a_TLS_mechanism`
- `QC5b_transmon_T1`

Each now has an evidence ladder entry.

## 10. Uniform evidence ladder

All scripts now expose a common evidence ladder where possible:

0. `data_missing`
1. `qualitative_context`
2. `partial_proxy`
3. `window_compatible_proxy`
4. `robust_proxy`
5. `decisive_public_test`

`run_all.py` now includes evidence levels in `out_summary.json` when available.
