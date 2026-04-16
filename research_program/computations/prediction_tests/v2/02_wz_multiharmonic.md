# Test 02 — Multi-Harmonic w(z) Search

**Predictions tested:** CCDR v6 P7-extended (sum of cascade-stage harmonics in dark-energy EOS).

## Statement
The dark-energy equation-of-state w(z) is not a constant or a smooth drift but a sum of N − 4 harmonics, one per active cascade stage, with frequencies set by the residual oscillation period of each stage attractor:
δw(z) = Σₖ Aₖ cos(ωₖ z + φₖ), Aₖ ~ ν ~ 10⁻³.

For N = 6 the prediction is **two harmonics** detectable in current data; for N = 5, **one**.

## Data
- **DESI DR2 BAO:** as in Test 01.
- **Pantheon+SH0ES:** as in Test 01.
- **SDSS BOSS+eBOSS BAO:** https://www.sdss4.org/science/final-bao-and-rsd-measurements/ (legacy points outside DESI z-range).

## Method
1. Fit w(z) with a flexible spline (8 knots in z = 0 → 2.5) jointly with Ω_m, h.
2. Compute residuals Δw(z) = w_fit(z) − w_ΛCDM(z).
3. Run a Lomb–Scargle periodogram on Δw(z) with frequency range corresponding to oscillation periods 1/H_∞ (= 10¹¹ yr → ω ~ 10⁻¹⁸ Hz, mapped onto the redshift baseline) down to 10× shorter.
4. Identify peaks above false-alarm probability 1% (single-trial) and 10% (look-elsewhere corrected).
5. Count significant peaks → infer number of active cascade stages → infer N − 4.

## Pass / fail
| Criterion | Pass | Fail |
|---|---|---|
| ≥ 1 peak at FAP < 1% | consistent with v6 (N ≥ 5) | refutes cascade |
| ≥ 2 peaks with frequency ratio matching RVM | strong support v6, N ≥ 6 | — |
| 0 peaks (flat residuals) | consistent with v5 single-step | refutes v6 |

## Output
`result.json`: `n_peaks_significant`, `peak_frequencies`, `peak_amplitudes`, `inferred_N_minus_4`, `pass_v6`, `pass_v5_only`. Plot: spline-fit w(z), residuals, periodogram with FAP lines.

## Notes
Roman Space Telescope dark-energy survey will improve sensitivity by ~10× on the relevant frequencies. Current data are at the edge of the predicted amplitude (10⁻³); a null result is informative but not decisive.
