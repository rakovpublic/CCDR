# Round-10 v24 confirm-first hardening

Implemented 10 confirm-oriented improvements on top of v23:

1. P36 high-z: strict object-level V^2/R scanner with FITS/CSV/TXT support.
2. P36 high-z: source/table leave-one-out publication gate and explicit unit-hint gate.
3. P36 high-z: rejection audit for unsafe velocity/radius columns such as VEL_PA, angle, PA, sigma, inclination and error columns.
4. P30: ACT mask/weight/window candidate search and official-mask/empirical-mask distinction.
5. P30: random-catalogue density normalization gate and estimator harmonization plan for HEALPix32/64/128/256 + KNN16/32/64.
6. P33: measured-density BAO-alpha plan with required randoms, covariance and null outputs.
7. P41: q^2-binned numeric row audit and stricter likelihood/promotion gate.
8. P32: GWOSC strain download and residual-fit manifest with injection, detector split and leave-one-event-out gates.
9. P5/P9/DM/HEPData: endpoint-funnel schema policy and coverage-only direct-detection claim policy.
10. SM-D1–D5: derivation gates separating constant consistency from actual CCDR derivation confirmation.

v24 intentionally avoids inflating confirmation counts: proxy/coverage/readiness results keep their old status or receive a v24 claim-scope field until the stricter publication/object/table gates pass.
