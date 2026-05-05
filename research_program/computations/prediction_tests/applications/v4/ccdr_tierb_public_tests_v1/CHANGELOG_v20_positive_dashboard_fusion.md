# v20 positive dashboard + fusion parser/data improvements

Implemented all six positive-focused improvements from the v19 analysis and added extra fusion support.

Highlights:
- Materials positive score for T31/T32 to reward microstructure-specific support and penalize broad overfit.
- T53 residual-model contract with bootstrap/jackknife requirements.
- T48a/T48b split: coarse proxy frozen as null control; descriptor-enriched PV model promoted as positive path.
- Electronics exact-parser plans for T44/T45/T47.
- Fusion is not abandoned: added v20 unit-line PDF extractor, more exact Zenodo queries, exact/curated secondary diagnostic mode, and candidate-manifest export metadata.
- Batch runner writes `positive_dashboard.json`.

Evidence gates remain strict: secondary PDF/figure extraction cannot confirm/falsify; primary physical tables must still pass contracts.
