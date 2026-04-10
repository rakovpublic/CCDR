# Test 07 — R_K / R_K* Geometric Ratio Test

## Source
- Public paper PDF: `https://arxiv.org/pdf/2212.09152.pdf`
- q² region used for the main test: `central`

## Extracted measurements
- R_K = 0.949 +0.047/-0.047 (total)
- R_K* = 1.027 +0.077/-0.073 (total)

## Derived quantities
- delta_RK = -0.051000
- delta_RKstar = +0.027000
- magnitude ratio |delta_RK|/|delta_RKstar| = 1.888889 ± 1.661514
- signed ratio delta_RK/delta_RKstar = -1.888889 ± 1.668429
- sqrt(3) = 1.732051
- z-score vs sqrt(3) = 0.094395

## Verdict
- status = `cannot_fire`
- pass_v6_p9d = `False`
- null_test_passed = `True`

## Important ambiguity handled explicitly
The uploaded Test 07 note compares **|delta_RK|/|delta_RKstar|** to ~1.7.
However, the published central values in the central-q² bin give delta_RK < 0 and
delta_RKstar > 0, so the **signed** ratio is negative. To keep the implementation aligned
with the note, the primary `ratio` reported in `result.json` is the **magnitude ratio**,
while `signed_ratio` is also reported for transparency.

## Covariance treatment
No public machine-readable RK/RK* covariance matrix was located in the source used here; the Monte Carlo propagation therefore assumes rho = 0 between delta_RK and delta_RKstar.
