# Test 06 — LHCb 1–2 GeV Hadronic Spectral Excess

**Predictions tested:** CCDR v6 P9a (acoustic-optical phonon avoided crossing produces a narrow ~10–50 MeV excess at m_inv ≈ 1.0–1.8 GeV in pp → μ⁺μ⁻ + X and pp → γγ + X).

## Statement
At the electroweak–QCD crystallisation boundary, the SM ("acoustic") phonon branch and the dark ("optical") phonon branch undergo an avoided crossing. The repulsion produces a narrow resonance-like enhancement in the dimuon and diphoton invariant mass spectrum at 1.0–1.8 GeV, distinguishable from known hadronic resonances by its isotropic angular distribution.

## Data
- **HEPData LHCb low-mass dimuon:** https://www.hepdata.net/ — search "LHCb low mass dimuon prompt"
- **HEPData ALICE prompt dimuon in pp:** ALICE pp 13 TeV measurements of low-mass dimuon continuum.
- **CMS scouting dimuon:** https://www.hepdata.net/record/ins1837084 (Run 2 scouting search).

## Method
1. Pull the dimuon invariant mass spectrum in the 0.5–3 GeV range, properly background-subtracted (combinatorial + known resonances ρ, ω, φ, J/ψ).
2. Fit the residual continuum with a smooth template (third-order polynomial or QCD shape).
3. Search for narrow excesses with width 10–50 MeV in the 1.0–1.8 GeV window.
4. For any excess found: (a) compute local significance, (b) check angular distribution from the published moments, (c) compare against PDG hadronic resonances at the same mass.
5. Look-elsewhere correction over the 1.0–1.8 GeV mass window.

## Pass / fail
| Criterion | Pass | Fail |
|---|---|---|
| Narrow excess in 1.0–1.8 GeV at local > 3σ | supports v6 P9a | — |
| Angular distribution isotropic (not resonance-like) | strong support | — |
| Excess matches known hadronic resonance | refutes interpretation | — |
| No excess above 2σ | consistent with no signal at current sensitivity | — |

## Output
`result.json`: `excess_mass_gev`, `excess_width_mev`, `local_significance_sigma`, `look_elsewhere_significance`, `angular_distribution_isotropic`, `pass_v6_p9a`. Plot: spectrum, fit residuals, angular moments.

## Notes
LHCb has published low-mass dimuon spectra with sufficient statistics. The Run 2 scouting search by CMS is the best dataset for narrow resonances in this range. Known issue: the 1.0–1.8 GeV region contains the φ(1020) tail and several narrow charmonium-like states; subtraction is non-trivial.
