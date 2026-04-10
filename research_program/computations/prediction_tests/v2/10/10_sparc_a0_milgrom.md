# Test 10 — SPARC a₀ Local vs cH₀ Asymptotic

**Predictions tested:** CCDR v6 §6.1 (cH₀ is the *asymptotic* a₀ in late de Sitter; the local present-day value is the Milgrom 1.2 × 10⁻¹⁰ m s⁻², reproduced by cascade-residue weighting).

## Statement
The radial-acceleration relation a₀ extracted from rotation-curve fits at z ≈ 0 should equal the Milgrom value, *not* cH₀ ≈ 6.5 × 10⁻¹⁰ m s⁻². CCDR v6 retreats from the v5 identification a₀ = cH₀ (which P-MOND-2 already excluded at the 82% level) to the prediction that a₀ → cH₀ only as z → ∞. This test confirms the local part of the v6 retreat and **specifies the redshift trajectory** for future high-z tests.

## Data
- **SPARC RAR:** http://astroweb.cwru.edu/SPARC/ (`RAR.mrt`, 2693 points).
- **SPARC parent catalog:** Lelli et al. (2016), 175 galaxies with rotation curves.

## Method
1. Load SPARC RAR (g_obs, g_bar with errors).
2. Fit the simple-μ Milgrom interpolating function: g_obs = g_bar × (1 + (a₀/g_bar)^(1/2))^(1/2)... or use the published parameterisation.
3. Free parameter: a₀. Minimise reduced χ² in log space.
4. Report best-fit a₀ with statistical and systematic errors.
5. **Two reference comparisons:** (a) Milgrom a₀ = 1.20 × 10⁻¹⁰ m s⁻²; (b) cH₀ = 6.55 × 10⁻¹⁰ m s⁻² (using H₀ = 67.4).
6. Compute the fractional deviation from each.

## Pass / fail
| Criterion | Pass v6 | Fail v6 |
|---|---|---|
| Best-fit a₀ within 5% of Milgrom | confirms v6 local retreat | — |
| Best-fit a₀ within 5% of cH₀ | refutes v6 retreat (and v5) | — |
| a₀ between Milgrom and cH₀ | partial support; needs cascade-weighting calc | — |

## Output
`result.json`: `a0_fit`, `a0_fit_sigma`, `chi2_reduced`, `dev_from_milgrom_pct`, `dev_from_cH0_pct`, `pass_v6_local`. Plot: g_obs vs g_bar with both reference fits, residuals.

## Notes
The existing `P_MOND_2_sparc_test_v2.py` already implements this and returns a₀ = 1.16 × 10⁻¹⁰ m s⁻² (3.3% from Milgrom, 82% from cH₀). The result is **already in hand** and supports v6. The remaining piece is the *high-z* test: collect lensing-derived a₀ values from cluster studies at z > 0.5 and check whether a₀(z) trends toward cH₀. The PMOND1 template CSV is the input format.
