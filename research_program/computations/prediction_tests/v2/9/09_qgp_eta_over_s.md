# Test 09 — QGP η/s Saturation and Oscillation Near T_c

**Predictions tested:** CCDR v6 P5 (KSS bound saturated by QGP at the holographic boundary); P9c (η/s exhibits dip-peak structure near T_c ≈ 155 MeV).

## Statement
The quark-gluon plasma produced in Pb–Pb collisions at the LHC sits near the QCD crystallisation boundary. CCDR v6 predicts that η/s(T) (a) saturates the KSS bound η/s = ℏ/4πk_B = 1/4π in the high-T limit, and (b) shows a non-monotonic dip-peak structure across T_c ≈ 155 MeV: a dip at the dimensional phase boundary, a peak in the hadronic phase.

## Data
- **ALICE elliptic and triangular flow:** https://www.hepdata.net/ — Pb-Pb 5.02 TeV v_2{2}, v_3{2} measurements (multiple HEPData records).
- **ATLAS / CMS** equivalent flow harmonics in Pb–Pb at 5.02 and 2.76 TeV.
- **JETSCAPE Bayesian extraction of η/s(T):** https://jetscape.org/papers/ — published posterior on η/s(T) from global fits to flow + spectra (Phys. Rev. C 103, 054904, 2021).

## Method
1. Use the JETSCAPE published η/s(T) posterior as the primary input — it already does the inversion from observables. Alternatively re-derive from raw v_n with a hydro emulator.
2. Compute η/s at T = 200, 250, 300 MeV (high-T regime). Compare to KSS = 1/(4π) ≈ 0.0796.
3. Compute η/s at T = 140, 150, 160, 170 MeV (transition region). Search for non-monotonicity.
4. Fit a smooth model (constant, linear, dip-peak) and use BIC to compare.

## Pass / fail
| Criterion | Pass | Fail |
|---|---|---|
| η/s(T → ∞) within 50% of KSS | supports v6 P5 | — |
| η/s(T) shows dip-peak with BIC preferred over monotonic | supports v6 P9c | — |
| η/s(T) monotonically rising or constant in T = 140–200 MeV | refutes P9c (P5 may survive) | — |
| η/s ≫ KSS at all T | refutes P5 universal saturation | — |

## Output
`result.json`: `eta_over_s_high_T`, `kss_bound`, `ratio_to_kss`, `dip_peak_bic_preferred`, `min_T_mev`, `max_T_mev`, `pass_p5`, `pass_p9c`. Plot: η/s(T) posterior with KSS line and dip-peak fit.

## Notes
JETSCAPE Bayesian results give η/s ≈ 0.08–0.20 at T = 150–300 MeV — within a factor of 2.5 of KSS, which is the closest any system has come. The dip-peak structure has been hinted at but not robustly established; this test is **discovery-grade**, not consistency-grade.
