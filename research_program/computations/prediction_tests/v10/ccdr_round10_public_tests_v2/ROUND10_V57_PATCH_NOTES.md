# Round-10 v57 autonomous confirm-artifact builders

v57 removes the requirement that the user manually fill strict measurement artifacts.
The runner now attempts to auto-build the confirm-recovery artifacts from public/cached
inputs during the normal run, then applies the same strict gates.

Key behavior:
- No hand-edited CSV/JSON files are required.
- Auto-built artifacts are diagnostic unless they contain the real required fields.
- FILL/TEMPLATE/example files are ignored for promotion.
- If public/cached data are unavailable, tests remain data_limited/required rather than fabricating confirms.

Implemented:
1. P36 high-z raw-row auto-builder from cache/public raw-looking tables.
2. P36 strict large-radius gate reused after auto-build.
3. P30 active protocol auto-creation and same-mask route proof from current ACT/SDSS stats.
4. P33 alpha-measurement auto-normalizer from cached/public alpha artifacts.
5. PTA weighted-statistic auto-normalizer from cached/public CL2 artifacts.
6. P32 strain-likelihood auto-normalizer.
7. P40 BB-likelihood auto-normalizer.
8. P41 q2/Wilson-likelihood auto-normalizer.
9. SMD derivation prediction auto-normalizer.
10. v57 dashboard with no-manual-fill claim policy and artifact index.

Important: v57 can remove manual file filling, but cannot guarantee new confirms if the public data required for a strict measurement are not present or not significant.
