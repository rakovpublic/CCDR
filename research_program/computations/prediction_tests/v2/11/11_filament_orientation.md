# Test 11 — Cosmic Filament Orientation Correlation

**Predictions tested:** CCDR v6 P3 (filament orientational correlation C_fila ~ A exp(−r/r_texture) with r_texture > r* ≈ 147 Mpc, A ~ 10⁻²–10⁻³).

## Statement
If the cosmic web is the grain structure of a 3+1D crystal nucleated from an N-dim bulk, filaments should not be randomly oriented — they should exhibit a long-range orientational correlation with characteristic length above the BAO scale. The texture length r_texture > r* is the "domain size" of the cosmological crystal grains.

## Data
- **Tempel et al. (2014) filament catalog:** http://cosmodb.to.ee/ (SDSS DR8/DR12 filament spine catalog).
- **DisPerSE filament catalog from SDSS:** https://www2.iap.fr/users/sousbie/web/html/index.html
- **NEXUS+ catalog (if public):** Cautun et al. (2013).

## Method
1. Load filament spine catalog. For each filament, extract the unit tangent vector at the midpoint and the 3D position.
2. Bin filament pairs by 3D separation r in [10, 30, 50, 75, 100, 150, 200, 300] Mpc/h.
3. In each bin, compute the mean of |cos θ_ij|² − 1/3 where θ_ij is the angle between filament tangents (the standard orientational correlation, vanishing for isotropic).
4. Fit C(r) = A exp(−r/r_texture) over r > 30 Mpc (above peculiar-velocity smearing scale).
5. Report A, r_texture, and the χ² of the fit.

## Pass / fail
| Criterion | Pass | Fail |
|---|---|---|
| A > 0 at 3σ | non-trivial alignment exists | random orientations |
| r_texture > 147 Mpc (BAO scale) | supports v6 P3 | — |
| A in [10⁻³, 10⁻²] | within predicted band | outside band |

## Output
`result.json`: `correlation_amplitude`, `correlation_amplitude_sigma`, `r_texture_mpc`, `r_texture_sigma`, `chi2_reduced`, `pass_amplitude`, `pass_length_above_bao`. Plot: C(r) data with exponential fit and BAO scale marked.

## Notes
Existing literature (Tempel et al. 2014, Hirv et al. 2017) finds significant filament alignment at scales below ~30 Mpc/h. The CCDR v6 prediction is at *larger* scales (texture length > 147 Mpc), which is qualitatively new and not yet measured robustly. SDSS coverage is sufficient; the analysis just needs to be extended to the larger separation bins.
