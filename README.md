# CCDR — Crystallographic Cosmology from Dimensional Reduction

A theoretical physics research programme in which the observed large-scale structure of the universe, the existence and abundance of dark matter, and the small value of the cosmological constant are all consequences of a single mechanism: **local, asynchronous dimensional reduction triggered by saturation of the holographic bound**.

The post-inflationary universe is treated as a supersaturated medium that undergoes symmetry-breaking nucleation at sites determined by topological defects in a higher-dimensional bulk, producing a lattice-like condensate whose preferred spacing is the baryon acoustic oscillation (BAO) scale r\* ≈ 147 Mpc. The cosmic web's filaments, walls, nodes, and voids are, in this picture, the grain boundaries, facets, triple junctions, and grains of a cosmological crystal. Dark matter is the collection of residue species left behind at each successive dimensional reduction. Dark energy is the zero-point energy of a cosmological-scale time crystal.

The current form of the theory is **CCDR v7.3** (core paper) and **Extended Crystallographic Synthesis v3.3** (architecture), both tracked in five empirical rounds against public data from Planck, Pantheon+, DESI DR2, NANOGrav, SPARC, KMOS3D, Euclid Q1, LHCb, CMS, ALICE, XENONnT, LZ, and PandaX-4T.

---

## Current Status (through round 5)

- **One robust confirmation.** Local a₀ → Milgrom constant (SPARC rotation curves; re-fit round 5 on n = 8 galaxies, 80 points, 0.047 dex RMS).
- **Two suggestive positive signals.** P30 density-correlated κ in Euclid Q1 (cross-access-path stable on ACT + Planck proxies, six-way subset stable); CL2 PTA × density-κ cross-correlation r = +0.524, same sign as P30.
- **One suggestive high-z trend.** a₀ → cH₀ at z > 0.3 (Spearman r = 0.886, p = 0.019, six points).
- **One empirically anchored triangle.** The CL1/CL2/CL3 three-way ν extractor disagrees by 30× between the Pantheon+ SN fit, the three-point MOND sequence, and the time-crystal Q-factor proxy — interpreted as a quantitative measurement of the low-z SN systematic rather than a contradiction.
- **Hard core window reached.** Six HEPData limit curves cross the 0.5–3 TeV target window for direct-detection peak resolution; peak resolution not yet achieved.
- **Two proxy-level nulls.** P37 phase-space drift (z ≈ −0.43), P38 void-wall kurtosis (k₄ = 2.81 < 4). Not falsifying; limited by proxy scale.
- **38 numbered predictions.** Full scorecard in Synthesis v3.3 §8.

---

## Repository Contents

### Core manuscripts

The primary paper has gone through v5 → v6 (v6, v6.1, v6.2, v6.3) → v7 (v7.1, v7.2, v7.3). Each version exists in English and Ukrainian `.docx` form. The **current version is v7.3**:

- `CCDR_v7_3_EN.docx` — current canonical English CCDR paper
- `CCDR_v7_3_UK.docx` — current canonical Ukrainian translation

Older versions are retained for provenance and peer-review trail. If you are reading for the first time, start with v7.3.

### Synthesis architecture

The Synthesis document describes the 12 primary frameworks (CDT, TN, GFT, EPRL, TGFT-RG, SSM, RVM, AS, AQN, DA/O, HaPPY, LGT), 3 alternative swappable branches (CST, EG, CFS), 9 resolution paths for the remaining open problems, and the 5-layer falsifiability hierarchy around which the CCDR programme is organised. It has gone through v2 → v3 (v3.1, v3.2, v3.3) in English and Ukrainian:

- `Synthesis_v3_3_EN.docx` — current canonical English Synthesis
- `Synthesis_v3_3_UK.docx` — current canonical Ukrainian translation

### Companion papers

Each companion paper extends one specific mechanism of CCDR or applies it to an adjacent problem. English and Ukrainian versions are kept in parallel.

- `Filament_Alignment_Paper_{EN,UK}.docx` — the P3 filament orientational correlation test, with independent transparent-finder reproductions of Tempel-style alignment catalogues on SDSS DR8 and Euclid Q1.
- `mond_ccdr_{en,uk}.docx`, `mond_cdt_ccdr_{en,uk}.docx` — derivation of the MOND a₀ constant from grain-boundary phonon scattering, and its embedding in the CDT simplicial framework.
- `rvm_aqn_cdt_tn_ccdr_{en,uk}.docx`, `aqn_cdt_tn_ccdr_{en,uk}.docx`, `cdt_ccdr_tn_{en,uk}.docx` — the combined Running Vacuum Model + axion-quark-nugget + causal dynamical triangulation + tensor network treatment of dark matter and baryogenesis.
- `baryogenesis_rvm_aqn_cdt_tn_ccdr_{en,uk}.docx` — the baryogenesis-specific layer (Layer 5 of the falsifiability hierarchy).
- `sm_beyond_{en,uk}.docx` — Standard Model embedding of CCDR via the spectral action on CDT geometry.
- `aurora_{en,uk}.docx`, `mag_{en,uk}.docx`, `pole_{en,uk}.docx` — focused analyses on auroral / magnetospheric / polar anomalies as cascade residue detection channels.
- `cdt_ccdr_{en,uk}.docx`, `cdt_ccdr_tn_revised_{en,uk}.docx` — the original and revised CDT + tensor network foundations.
- `ccdr_applications_{english,ukrainian}.docx` — broader applications.
- `ccdr_article*.docx`, `ccdr_v5_*.docx`, `ccdr_v6_*.docx`, `ccdr_revised_*.docx` — historical drafts; superseded by `CCDR_v7_3_*.docx` but retained.
- `economic_consequences_ccdr_jneopallium{,_ukrainian}.docx` — adjacent essay on broader implications.

### Figures

- `P8a_hexapolar.png` — hexapolar (ℓ = 6) Hellings-Downs correction figure for PTA data (P8a prediction).
- `P8c_correlation.png` — PTA × cosmic-web correlation figure (P8c / CL2).
- `SMD5_koide_v2.png` — fifth Standard Model Koide-closure attempt.
- `cdt_ccdr_t1_results.png` — round-1 CDT test output visualisation.

### Computational tests

- `cdt_ccdr_t1_test.py` — round-1 CDT consistency test (Python).
- `cdt_ccdr_t1_report.json` — machine-readable output of the round-1 test.

### Directories

- `data/` — supporting numerical inputs and intermediate outputs.
- `research_program/` — the twelve-test public-data battery (round 5) plus earlier batteries. Each test is a standalone Python script that downloads or generates proxy data, runs the analysis, and emits JSON for diff-friendly tracking across rounds.

---

## Which Document to Read First

1. **First-time reader — physics.** Start with `CCDR_v7_3_EN.docx`. It gives the complete framework: single rule (χ_k → 1 triggers local dimensional reduction), mechanism (running vacuum provides supersaturation, higher-dimensional defects seed nucleation, BAO scale is the lattice constant), derived phenomenology (MOND, dark matter spectrum, black hole thermodynamics, cosmological constant), full prediction list (P1–P38), round-5 empirical status, and the §18 round-5 addendum with the twelve-test screening battery.

2. **First-time reader — architecture.** Start with `Synthesis_v3_3_EN.docx`. It shows how the 12 frameworks fit together, which components can be swapped out if falsified, where the 9 resolution paths close remaining gaps, and how the 5-layer falsifiability hierarchy locates every claim. §8 is the complete tabulated prediction scorecard.

3. **Ukrainian reader.** Every file has a `_UK` twin; `CCDR_v7_3_UK.docx` and `Synthesis_v3_3_UK.docx` are the current canonical Ukrainian versions.

4. **Reproducing the empirical tests.** Start in `research_program/` with the round-5 `test01_*_v73.py` through `test12_*_v73.py` scripts and their `resultsv6.txt` output. The `READMEv6.md` in that directory describes the twelve-test battery and its differences from the prior nine-test round-4 battery.

5. **Specific prediction status.** Synthesis v3.3 §8 gives a 38-row table with mechanism, observable, round-5 status, decisive test, and falsifiability layer for each numbered prediction.

---

## How CCDR Is Organised (One-Paragraph Summary)

A single mechanism — dimensional reduction triggered when the information content of a local region saturates the holographic bound for its current dimensionality — is applied consistently across all scales and epochs. The Big Bang is the first such reduction in our patch; black holes are ongoing local reductions; the cosmological constant is the end-state of a de Sitter asymptote that the cascade approaches. Each reduction stage k → k−1 leaves behind a residue species that cannot compactify, and these residues are what we observe as dark matter. Because the cascade is local and asynchronous, different patches of our universe can have different cascade histories, giving density-correlated variation in the dark-sector composition (P30, CL2). Because the reducing volume does not sample the full higher-dimensional bulk uniformly, the probability of a species originating from stage k being present in our patch is P_presence(k) ~ (r_k/L_k)^{k−4} — a "dimension-origin prior" that predicts an asymmetric peak distribution in direct detection, skewed toward lighter masses. The framework's number of bulk dimensions N is not a theoretical input; it is an observable, determined by peak counting under this prior. The algebraic upper bound from the division algebra C ⊗ H ⊗ O → SU(3) × SU(2) × U(1) gives N ≤ 11.

---

## Falsifiability

The programme is organised around five explicit falsification layers, with the cascade core at Layer 0 and the peripheral predictions at Layer 4. No layer is currently falsified. The most immediate falsifiers are:

- Tonne-scale direct detection fires with mass ratios that do not match the RVM-cooling-predicted geometric pattern (kills Layer 0).
- DESI DR3 returns ν inconsistent with the MOND-sequence extraction (forces a choice between CL1 and the cascade).
- Euclid DR1 cross-correlation κ collapses the P30 signal (kills Layer 4).
- Observation of genuine superluminal information transfer or closed timelike curves (kills the framework entirely — CCDR explicitly forbids both as consequences of the holographic bound and RVM monotonic cooling).

See `Synthesis_v3_3_EN.docx` §5 for the full layer-by-layer falsification table and exit conditions.

---

## Languages

All substantive documents are maintained in parallel English (`_EN.docx` or `_english.docx`) and Ukrainian (`_UK.docx` or `_ukrainian.docx`) versions. Where the two have diverged in older drafts, the newer version in each language is canonical.

---

## Empirical Data Sources Used

Planck PR3/PR4 · Pantheon+ SNe Ia · DESI DR2 BAO · NANOGrav 15-year PTA · SPARC rotation curves · KMOS3D high-z rotation curves · Euclid Q1 galaxy density · ACT DR6 lensing · SDSS DR8 · HEPData (XENONnT 2023 SI/SDn/SDp, LZ, PandaX-4T) · LHCb Run 3 · CMS/ATLAS Run 3 · ALICE QGP viscosity · LIGO/Virgo compact object mergers.

Every test script in `research_program/` is written to auto-download from the public endpoints when available and fall back to documented deterministic proxy data when machine-readable collaboration-grade products are unavailable. Distinction between "public overlap" and "collaboration-grade replication" is preserved in every result.

---

## Citing

Manuscripts in this repository are prepared for double-blind peer review. Author names are withheld in the distributed files. If citing an internal mechanism or prediction, cite the specific manuscript (e.g. "CCDR v7.3 §4.5 — probability-weighted reducing volume") and the version label.

---

## License

Not yet specified. Until a LICENSE file is added, all rights are reserved by the author(s). Please contact the repository owner before redistribution or derivative work.

---

## Contact

Issues and pull requests welcome via this repository's GitHub issue tracker.
