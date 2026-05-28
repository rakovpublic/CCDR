# Round-10 v28 confirm-execution patch

Implemented all 10 requested improvements from the v27 report.

Highlights:
- P30-SDSS random-normalized route rerun machinery and output JSON.
- P30 same-split ACT variant/curl gate consuming the v28 route split.
- P30 Euclid repair contract with photo-z/depth/quality/random requirements.
- P36 high-z source-specific object-table scanner and strict object acceleration output.
- P33 measured density-split BAO alpha handoff/consumer.
- P41 q²/CP/Wilson-SM likelihood handoff/consumer.
- P32 strain execution manifest and fit-product consumer.
- SM-D derivation JSON consumer gate.

Claim policy remains strict: v28 can promote only if measured artifacts actually pass the gate; otherwise it records exact missing items.
