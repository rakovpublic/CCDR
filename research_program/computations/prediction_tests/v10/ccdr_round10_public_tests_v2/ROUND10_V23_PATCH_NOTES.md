# Round-10 v23 confirm-hardening patch

This patch adds 10 confirm-oriented improvements on top of v22 while keeping the old test files and manifest stable. The v22 runner names are overridden in `ccdr_r10_common.py`, so existing `run_all.py` commands automatically use v23 behavior.

## Implemented improvements

1. **P36 high-z a0 object-level V²/R audit**
   - Adds a strict table scanner for FITS/CSV/TXT products.
   - Computes object-level `(Vrot km/s)^2 / (R kpc)` in SI units when strict columns exist.
   - Requires >=30 object rows, >=2 tables, median above SPARC a0, and leave-one-table medians above SPARC for object-level promotion.

2. **P36 velocity-column false-positive guard**
   - Rejects broad velocity labels such as `VEL_PA`, position-angle, angle, inclination, sigma/dispersion, and error columns.
   - Keeps previous v22 proxy confirmation scoped as proxy-only until the object-level gate passes.

3. **P38 radius-preserving angular/sector null**
   - Downloads/parses VAST void catalogues when not in quick mode.
   - Uses coordinate-like columns to test whether radius morphology is sky-sector driven.
   - Adds a publication-grade gate distinct from the existing robust confirm-like label.

4. **P30 estimator sign-flip diagnosis**
   - Adds per-subsample HEALPix64/HEALPix128/KNN sign-conflict summaries.
   - Explicitly labels the failure mode as mask/edge/density-normalization until random-catalogue normalization is added.

5. **P8/P8c NANOGrav metadata fallback**
   - If full tar extraction remains data-limited, v23 still records Zenodo tar/par-like file metadata and exposes the member-streaming retry path.

6. **P33 density–BAO measured-split design**
   - Recovers DESI BAO baseline observables if v22 saw an empty vector.
   - Adds the required measured high/low-density BAO schema, covariance gate, and random-catalogue/null checklist.

7. **P39 model-penalty gate**
   - Adds approximate ΔAIC/ΔBIC from Δχ² and number of observables.
   - Prevents Δχ²≈1 compatibility from becoming confirmation.

8. **P32 ringdown strain execution plan**
   - Converts metadata ranking into an explicit strain-level execution plan with H1/L1, PSD, injection null, detector split, and leave-one-event-out requirements.

9. **P41 CP-asymmetry null scaffold**
   - Classifies parsed rows into CP-control and CP-averaged candidates.
   - Only allows future P41 promotion if structured rows, sign basis, and CP-control null all pass.

10. **Direct-detection + HEPData schema hardening**
   - Direct detection can promote only to coverage-confirmed-no-detection when mass and limit units are explicit.
   - HEPData now follows a source → record → table → CSV/JSON/YAML endpoint funnel; HTML/API previews remain inventory only.

## Claim policy

v23 is intentionally stricter. It should increase trustworthy confirms, not inflate labels. In particular:

- P36 high-z may remain proxy-confirm-like unless object-level V²/R passes.
- P30 remains frozen until estimator signs agree after explicit mask/random-density normalization.
- P33/P39/P32/P41 remain non-confirm until their measured/null gates pass.
