# CCDR v7.5 Tier-B Public-Data Test Bundle

This bundle implements Tier-B engineering/application tests **T26–T60** from the uploaded Tier-B table.

## Core rule

No script requires manual local data files. Every external input is downloaded automatically into `--cache` from public URLs or public APIs. If a prediction currently has no reliably machine-readable public table, its script returns `status: "data_limited"` or `"partial"` and lists the attempted public sources. This is intentional: data availability is kept separate from CCDR confirmation/falsification.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run all tests

```bash
python run_all_tier_b.py --cache tierb_cache --outdir tierb_out --max-papers 30 --timeout 60
```

## Run a specific test

```bash
python tests/test31_cryogenic_kappa_ccdr_vs_casimir.py --cache tierb_cache --outdir tierb_out
python tests/test46_ldpc_burst_channel_benchmark.py --outdir tierb_out
python tests/test58_exoplanet_stellar_chronometry_null.py --cache tierb_cache --outdir tierb_out
```

## Result status meaning

- `ok`: enough public structured data were parsed to compute the intended metric.
- `partial`: some useful public data/metadata were parsed, but not enough for a strong claim.
- `data_limited`: public sources were attempted, but machine-readable tables were not available or not parseable enough.
- `error`: unexpected code/runtime failure; see traceback in the JSON.

## Implemented tests

- T26 FR3 fusion ELM energy scaling
- T27 FR6 ELM helicity proxy
- T28 FR7/FR10 global H-mode confinement/KSS margin
- T29 FR8 stellarator vs tokamak edge transport
- T30 FR10 fusion residual curvature coupling
- T31 MAT1/MAT3 cryogenic κ CCDR-vs-Casimir using CMB-S4 GitHub data
- T32 MAT3 low-T κ exponent using CMB-S4 GitHub data
- T33 MAT6/MAT7 diamond/hBN thermal ceiling audit
- T34 MAT4 Bi₂Te₃ ZT angle meta-analysis
- T35 MAT5/MAT10 Kibble–Zurek grain-size exponent
- T36 MAT11/CL4 density-stratified grain scattering
- T37 MAT12 auxetic phonon transport
- T38 MAT8 skyrmion lifetime literature audit
- T39 MAT9/QC10 moiré phononic bandgap twist scaling
- T40 QC5 transmon phononic-substrate T1 audit
- T41 QC11 qubit T2 plateau meta-analysis
- T42 QC8 spin-qubit T2 in isotopically pure Si
- T43 QC7 DTC qubit error-per-cycle trend
- T44 EL1/EL3 3D NAND area vs volume scaling
- T45 EL8 optical interconnect trend
- T46 EL6 synthetic LDPC/CDT-like burst-channel benchmark
- T47 EL7 neuromorphic graph energy audit
- T48 NREL PV efficiency public-data readiness audit
- T49 EN?/MAT4/MAT7 materials symmetry audit
- T50 SE2/SE6 Casimir residual public-table audit
- T51 SE3/AE3 optical-clock drift bound
- T52 SE6/SE7 atom-interferometer noise-floor audit
- T53 RCSB PDB biological symmetry proxy
- T54 photosynthetic coherence meta-analysis
- T55 AE5 spacecraft anomaly audit
- T56 AE4 solar-sail residual proxy
- T57 HEPData cosmic-ray flux ledger
- T58 NASA Exoplanet Archive chronometry readiness/null audit
- T59 HEPData public HEP anomaly ledger
- T60 Koide sector-distance public constants audit

## Notes on the intentionally conservative design

Several Tier-B items depend on literature supplements that may be on publisher pages, PDFs, OSF records, Zenodo, GitHub, or HEPData. The generic literature tests therefore use OpenAlex and Crossref public APIs, discover candidate open-access/supplement links, download public pages/tables, and parse HTML/CSV/XLSX when available. They do **not** invent numbers when public tables cannot be parsed.

For T60, the script downloads public NIST/PDG/FLAG pages. It refuses to hard-code tau/quark values by default; set `CCDR_T60_USE_PUBLIC_PDG_TAU_SEED=1` only if you explicitly accept the public PDG tau mass as a seed for a charged-lepton-only smoke test.


## v2 evidence policy

Tier-B v2 disables generic term-number evidence extraction. A test result is `partial` or `ok` only when the script finds a direct structured public data file or discovered supplement with named physical columns required by that test. Otherwise it returns `data_limited`. This prevents DOI fragments, publication years, page numbers, HTML/CSS ids, and citation metadata from being treated as evidence.

T31/T32 now classify CMB-S4 thermal-conductivity files heuristically and evaluate the main claim only on boundary-dominated candidates. T48 now fits an NREL PV efficiency baseline by material class and year before testing residuals. T57/T59 avoid the HEPData search endpoint and use known HEPData/INSPIRE endpoints.

## v3 result-quality patch

This bundle implements the eight requested result-quality fixes:

1. Fusion T26-T30 now use structured-source manifests only. Article/PDF text numerals are never evidence.
2. T31/T32 now classify materials into crystalline/metal, amorphous, composite/polymer, boundary candidate, and grain/nano-known subsets. The primary inference uses grain/nano-known when available, otherwise boundary-candidate.
3. T32 now compares fixed low-T exponents T^0.5, T^1, T^2, T^3 against a free exponent.
4. T48 uses a NREL PV residual model with year + material class + cell bucket + log(area) controls, then tests a predefined within-class crystallinity/texture proxy.
5. T57/T59 use exact/direct table manifests and INSPIRE metadata discovery, not the HEPData search API.
6. T60 is split internally into T60a charged leptons and T60b quark/lattice sector; charged-lepton success no longer implies full sector confirmation.
7. T44 uses a curated 3D-NAND spec-table gate requiring layer count, capacity and die area before any result is counted.
8. T50-T52 are explicitly upper-limit/constraint tests. They cannot claim confirmation from generic residual/noise literature.
9. T53 no longer uses PDB resolution/assembly-count as a stability proxy; it requires public stability/Tm/ΔG-style tables.

Expected behavior: more `data_limited` statuses are correct when public structured tables are absent. That improves result quality by removing false partial positives.


## v4 strict result-quality notes

This bundle adds `evidence_status` and `readiness_status` to every result. Use
`evidence_status` for scientific reporting and `readiness_status` for debugging
public-data access.

New public-data recovery features:
- OSF file API crawling for the International Global H-mode confinement database.
- Recursive discovery of attached files from Zenodo/OSF/Figshare-style APIs.
- ZIP supplement parsing for CSV/XLS/JSON/DAT files.
- Preregistered manifests in `data/` for material microstructure classification and PV acoustic-optical proxy scoring.

Recommended rerun:

```powershell
python run_all_tier_b.py --cache tierb_cache_v4 --outdir tierb_out_v4 --max-papers 50 --max-tables 120 --timeout 90 --force
```

For faster checks:

```powershell
python run_all_tier_b.py --only T28 T31 T32 T48 T46 T60 --cache tierb_cache_v4 --outdir tierb_out_v4 --timeout 90 --force
```

## v5 source-quality mode

Tier-B v5 defaults to a manifest-only scientific pipeline. It is designed to download fewer, better-targeted public tables:

- source metadata is inspected first;
- only relevant records pass the metadata gate;
- only structured attached files pass to header-only parsing;
- full parsing happens only after required physical columns are found.

Run v5:

```bash
python run_all_tier_b.py --cache tierb_cache_v5 --outdir tierb_out_v5 --mode scientific --manifest-only --timeout 90 --max-bytes 50000000 --header-rows 50 --force
```

Use broad discovery only for source scouting, not evidence:

```bash
python run_all_tier_b.py --only T28 --mode discovery --allow-broad-discovery --cache tierb_cache_discovery --outdir discovery_out
```

New manifests live under `data/source_manifests/`. Add exact URLs there before treating a source as evidence-grade.
