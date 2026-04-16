# Test 03 — CMB Large-Angle Anomaly Common-Origin Test

**Predictions tested:** CCDR v6 P4 (quadrupole deficit, l=2/l=3 alignment, hemispherical asymmetry share a geometric origin from N-dim nucleation).

## Statement
The four canonical large-angle CMB anomalies — low quadrupole power, quadrupole-octupole alignment ("axis of evil"), hemispherical power asymmetry, and the cold spot — should not be statistically independent. CCDR v6 predicts a common preferred direction inherited from the higher-dimensional pre-image. The test: the directions associated with each anomaly should cluster on the sphere with significance > 95% against isotropy.

## Data
- **Planck PR4 / NPIPE maps:** https://pla.esac.esa.int/ (SMICA component-separated, 2048 nside).
- **Existing anomaly direction catalogs:** Schwarz et al. (2016) compilation; Planck 2018 Isotropy & Statistics paper Table 18.

## Method
1. Load SMICA map. Mask using Planck PR4 common mask.
2. Compute multipole vectors for l = 2, 3 (Copi–Huterer–Schwarz definition). Extract the area vectors w_2, w_3.
3. Compute the dipole modulation direction via local-variance hemispherical asymmetry estimator.
4. Take the cold spot direction from the catalog.
5. Compute pairwise great-circle distances between the four directions. Compare against 10⁵ Monte Carlo isotropic samples.
6. Report the probability that the four directions are mutually within 30° of a common axis.

## Pass / fail
| Criterion | Pass | Fail |
|---|---|---|
| Common-axis p-value < 0.05 | supports v6 P4 | inconclusive |
| Common-axis p-value < 0.01 | strong support | — |
| p-value > 0.5 | refutes shared-origin claim | — |

## Output
`result.json`: `directions` (list of unit vectors), `pairwise_angles_deg`, `common_axis_pvalue`, `pass_p05`, `pass_p01`. Plot: Mollweide projection with the four directions, common-axis fit, and isotropic null distribution.

## Notes
This test does not by itself confirm CCDR v6 — alignment can arise from other mechanisms (e.g., local foreground residuals). It is a *necessary* condition only. Multiple groups have run versions of this; the consensus current p-value for the l = 2 / l = 3 alignment alone is ~0.01–0.05.
