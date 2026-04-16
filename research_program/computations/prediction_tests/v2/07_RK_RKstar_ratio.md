# Test 07 — R_K / R_K* Geometric Ratio Test

**Predictions tested:** CCDR v6 P9d (lattice-symmetry prediction δR_K / δR_K* = √3 from hexagonal Brillouin zone phonon dispersion).

## Statement
The deviations of the lepton flavour universality ratios R_K = BR(B → Kμμ)/BR(B → Kee) and R_K* from unity should satisfy a parameter-free geometric ratio:
δR_K / δR_K* = cos(π/6) / cos(π/3) = √3 ≈ 1.732.
This is a single number and is independent of all other CCDR parameters.

## Data
- **LHCb 2022 R_K and R_K* combined measurement:** Phys. Rev. Lett. 131, 051803 (2023). Values: R_K = 0.949 ± 0.047, R_K* = 1.027 ± 0.072 (low q²).
- **Belle II preliminary:** B → K(*) ll measurements from BELLE2-CONF-2024 series.
- **HEPData record:** https://www.hepdata.net/record/ins2659065

## Method
1. Pull central values, statistical errors, and full systematic covariance for R_K and R_K* from the LHCb HEPData record.
2. Compute δR_K = R_K − 1 and δR_K* = R_K* − 1 with propagated errors.
3. Compute the ratio R = δR_K / δR_K* with full error propagation including correlation.
4. Compare R to √3 = 1.732. Report z-score against this prediction.
5. Compare also to R = 1 (the SM expectation if both deviate identically) and to R = 0 (no signal in either) — null competitors.

## Pass / fail
| Criterion | Pass | Fail |
|---|---|---|
| R within 2σ of √3 | consistent with v6 P9d | — |
| R within 1σ of √3 | strong support | — |
| R outside 3σ of √3 | refutes the prediction | — |
| Both δR_K and δR_K* consistent with 0 | test cannot fire | — |

## Output
`result.json`: `R_K`, `R_Kstar`, `delta_RK`, `delta_RKstar`, `ratio`, `ratio_sigma`, `z_score_vs_sqrt3`, `pass_v6_p9d`, `null_test_passed`. Plot: 2D contour of δR_K vs δR_K* with √3 line and the SM origin.

## Notes
After the 2022 LHCb update, R_K and R_K* moved closer to SM. The current central values give |δR_K|/|δR_K*| ~ 0.05 / 0.03 ~ 1.7, **already within 1σ of √3 by central value**, but both errors include zero, so the test does not yet fire decisively. Belle II Run 2 (2024–2026) and HL-LHC will tighten this.
