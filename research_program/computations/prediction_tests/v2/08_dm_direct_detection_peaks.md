# Test 08 — Dark-Matter Mass-Peak Counting from Direct Detection

**Predictions tested:** CCDR v6 P10/P12 + Synthesis v2 P25/P27 (the number of resolvable DM mass peaks equals N − 4; consecutive ratios are geometric, set by RVM cooling). **This is the single most diagnostic test for v6.**

## Statement
N is determined by counting distinct mass peaks in direct-detection spectra. For N = 5, one peak (the v5 ~100 GeV optical phonon). For N = 6, two peaks with geometric ratio 10–100. For N = 7, three peaks. The lightest peak should anchor near 100 GeV; the heaviest at the cascade-onset scale.

## Data
- **XENONnT:** https://www.xenonexperiment.org/data-releases (latest WIMP search, 2024–2025 release).
- **LZ:** https://lz.lbl.gov/results/ (2024 first science result and updates).
- **PandaX-4T:** http://pandax.sjtu.edu.cn/ (independent cross-check).

## Method
1. Pull the unbinned event lists or the published differential rate dR/dE_recoil from each experiment.
2. For each: build a likelihood for DM signal as a sum of K Gaussian-shaped excesses over the background model. Fit K = 1, 2, 3, 4 and compare via AIC/BIC.
3. For each significant peak (TS > 9): extract central mass, width, amplitude.
4. **Combine experiments** via a stacked likelihood — independent detector systematics must be propagated.
5. Compute consecutive mass ratios m_{k+1}/m_k. Compare to the RVM cooling prediction (10–100 per stage for ν = 10⁻³).
6. Infer N = 4 + (number of significant peaks).
7. **Crucial:** report upper-limit shape if 0 or 1 peaks are found. The shape constrains m_5 from below.

## Pass / fail
| Criterion | Pass v6 | Fail v6 |
|---|---|---|
| ≥ 2 peaks with geometric ratio 10–100 | supports cascade, fixes N | — |
| ≥ 2 peaks with ratio of order 1 or non-monotonic | — | refutes cascade |
| 1 peak found, no upper-limit constraint on m_5 < 10 TeV | inconclusive (N = 5 or m_5 > 10 TeV) | — |
| 1 peak found, m_5 < 10 TeV excluded by limit shape | N = 5, supports v5 | refutes v6 in narrow form |
| 0 peaks above background | inconclusive (sensitivity issue) | — |

## Output
`result.json`: `n_peaks`, `peak_masses_gev`, `peak_significances`, `consecutive_ratios`, `ratios_geometric`, `inferred_N`, `pass_v6_cascade`, `pass_v5_single_step`, `m5_lower_limit_gev`. Plot: combined dR/dE with fitted peaks and background; secondary panel: ratio test vs prediction band.

## Notes
This test is the v6 hard core. Current XENONnT/LZ data show a mild excess near ~30 GeV that has been variously interpreted as background fluctuation, a sub-100 GeV WIMP, or a systematic artifact. **The v6 framework predicts the lightest peak at the optical-phonon scale ~100 GeV**, so a confirmed sub-30 GeV peak would already be tension. Fitting with K > 1 has not been done in the public releases — this is the most important new analysis to run.
