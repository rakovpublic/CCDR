# Test 04 — CMB μ/y Distortion Limits vs Cascade Releases

**Predictions tested:** CCDR v6 P11; Synthesis v2 P28 (each cascade stage releases its own thermal flash, imprinting one distortion feature; N − 4 features total).

## Statement
A cascade with N − 4 stages produces N − 4 separate energy injections into the CMB photon bath at characteristic redshifts z_k set by the cascade timing. Each injection produces a μ-type distortion (10⁴ < z < 2×10⁶) or y-type (z < 10⁴) depending on its epoch. Current FIRAS limits constrain the *integrated* distortion at |μ| < 9×10⁻⁵, |y| < 1.5×10⁻⁵ (95%). For ν ~ 10⁻³ the expected per-stage injection is ΔE/E ~ ν² ~ 10⁻⁶, below FIRAS but above PIXIE projected sensitivity ~10⁻⁸.

## Data
- **COBE/FIRAS monopole spectrum:** https://lambda.gsfc.nasa.gov/product/cobe/firas_products.html (`firas_monopole_spec_v1.txt`)
- **Fixsen et al. (1996) calibrated residuals:** Table II from the paper.

## Method
1. Load the FIRAS residual spectrum I_ν − B_ν(T_CMB) with errors.
2. Build the joint μ + y distortion template using the standard Chluba–Sunyaev kernels.
3. Fit (μ, y, ΔT_CMB) jointly with their full covariance.
4. Compute the upper limit on |μ| and |y|.
5. **For v6 / v2:** under the cascade model with two stages (N = 6) and ν = 10⁻³, predict (μ_pred, y_pred) ~ (10⁻⁶, 10⁻⁶). Compare to limits.
6. Report whether current data can already exclude a fraction of the (ν, N) plane.

## Pass / fail
| Criterion | Pass | Fail |
|---|---|---|
| Predicted (μ, y) < FIRAS limits | consistent (current data cannot test) | model already ruled out by FIRAS |
| (μ, y) within future PIXIE reach | future test definable | — |

## Output
`result.json`: `mu_limit_95`, `y_limit_95`, `mu_predicted`, `y_predicted`, `consistent_with_firas`, `pixie_detectable`. Plot: FIRAS residuals with best-fit distortion overlay.

## Notes
This test is currently **a consistency check, not a discovery test**. Its value is to confirm that the cascade picture is not already ruled out by FIRAS, and to set the target sensitivity for PIXIE/PRISM. The decisive measurement is ~5 years away.
