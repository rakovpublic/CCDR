# Test 01 — RVM ν Coefficient from Joint Cosmological Fit

**Predictions tested:** CCDR v6 P1 (BAO scale shift δr*/r* ≈ ν/2); Synthesis v2 P21 (ν as TGFT-RG IR fixed-point output, predicted 10⁻³ ± 10⁻⁴).

## Statement
The Running Vacuum Model coefficient ν, fit jointly to Planck TT+TE+EE, Pantheon+ SNe Ia, and DESI DR2 BAO, should equal 10⁻³ within errors. ν = 0 (ΛCDM) should be disfavoured at ≥ 2σ. CCDR v6 §6 holds that this is the *final-stage* (k = 4) value of ν; earlier-stage values are not constrained by this test.

## Data
- **Planck 2018 likelihoods:** https://pla.esac.esa.int/ (Plik, lite TT+TE+EE)
- **Pantheon+SH0ES:** https://github.com/PantheonPlusSH0ES/DataRelease (`Pantheon+SH0ES.dat`, `Pantheon+SH0ES_STAT+SYS.cov`)
- **DESI DR2 BAO:** https://data.desi.lbl.gov/public/dr2/ (DV/rd, DM/rd, DH/rd at the seven effective z bins)

## Method
1. Implement RVM background expansion: ρ_vac(H) = C₀ + ν m² H²/M²_pl, with continuity dρ_vac/dt = −dρ_matter/dt.
2. Build a 3-parameter likelihood (Ω_m, h, ν) with Planck prior covariance, full Pantheon+ Cholesky, and DESI Gaussian BAO.
3. Run emcee: 64 walkers × 30000 steps. Discard 5τ_autocorr burn-in. Compute marginal posteriors.
4. Compute Bayes factor between ν = 0 (ΛCDM) and free-ν via Savage–Dickey ratio at ν = 0.

## Pass / fail
| Criterion | Pass | Fail |
|---|---|---|
| ν central value within 2σ of 10⁻³ | yes | no |
| ν > 0 at ≥ 2σ | yes | no |
| Bayes factor BF(ν=0 vs free) | < 3 | ≥ 10 (decisive against) |

## Output
`result.json` keys: `nu_mean`, `nu_sigma`, `nu_95ci`, `bayes_factor`, `pass_pred`, `pass_lcdm_disfavored`. Plot: 1D posterior on ν with prediction band and ν = 0 line.

## Notes
The existing `P01_rvm_fit_v2_fixed.py` already implements most of this and returns ν = 0.00116 ± 0.00098 with BF = 20.3 against the model. Re-run only needed if data have updated since. CCDR v6 reads the current result as **consistent but not confirmatory** — the BF disfavours the model and that needs explicit acknowledgment in the output.
