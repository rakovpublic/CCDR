# Test 05 — Pulsar Timing Dark-Matter Spectral Comb

**Predictions tested:** CCDR v6 P8 (each cascade residue contributes a beat-frequency line in pulsar timing residuals; the dark-sector signature is a *comb*, not a single line).

## Statement
Galactic dark matter, in the cascade picture, comprises N − 4 species with distinct masses m_k. Each species couples gravitationally to pulsars and produces a periodic timing residual at beat frequency ω_k,beat ~ ω_k × (v/c)² where v ~ 10⁻³ c is the local virial velocity. For m_k spanning 10–10⁵ GeV, the beat periods span 10⁻²⁰ – 10⁻²⁴ s, well above NANOGrav's nHz band — but the *envelope* of the comb appears at the lowest beat frequency (lightest residue) and can in principle imprint a slow modulation on the timing residual variance.

## Data
- **NANOGrav 15-year data set:** https://nanograv.org/science/data (DR3, 67 pulsars, 16 years of TOAs).
- **EPTA DR2:** https://www.epta.eu.org/dr2.html (independent cross-check).

## Method
1. Load TOA residuals for the 30 best-timed NANOGrav pulsars after subtracting timing model.
2. Compute the autocorrelation of residual variance over the full baseline.
3. Search for periodic modulation at frequencies corresponding to virial dark-matter beats: in the cascade picture, *N − 4 distinct lines* with frequency ratios matching the predicted mass ratios from RVM cooling.
4. Compare line count and ratios against (a) v5 single-species expectation (one line) and (b) v6 cascade expectation (multiple lines with geometric ratios).
5. Use the Hellings–Downs cross-correlation as a sanity check that any signal is dark-matter-like, not GW-like.

## Pass / fail
| Criterion | Pass | Fail |
|---|---|---|
| ≥ 2 significant comb lines (FAP < 1%) | supports v6 cascade | — |
| Line frequency ratio matches RVM cooling prediction within 30% | strong support | — |
| 1 line only | consistent with v5, ambiguous for v6 | — |
| 0 lines | inconclusive (cosmological DM may not modulate) | — |

## Output
`result.json`: `n_lines_detected`, `line_frequencies_hz`, `line_amplitudes`, `frequency_ratios`, `ratios_match_rvm`, `pass_v6`. Plot: residual variance autocorrelation, with detected lines marked and predicted positions overlaid.

## Notes
This is the most speculative of the tests in this batch — the coupling between cosmological dark matter and pulsar timing is weak and the predicted amplitude is at the boundary of NANOGrav sensitivity. A null result is not informative; a positive result would be diagnostic.
