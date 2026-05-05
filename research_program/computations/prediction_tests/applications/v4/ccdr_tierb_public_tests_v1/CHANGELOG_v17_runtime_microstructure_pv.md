# v17 Runtime, Microstructure, and PV Artifact Fixes

Implemented targeted fixes requested after the v16 run:

1. **Runtime fix:** `tierb_autodiscovery.py` now defines `ROOT_DIR` and `DATA_DIR`, fixing the `NameError: DATA_DIR is not defined` crash in T26–T30, T47, T54, T57, and T59.
2. **MAT1/MAT3 data-quality improvement:** the automatic microstructure manifest now honors existing public manifest fields (`grain_size_known`, `nanocrystalline_yes_no`, `decisive_primary`, `nominal_grain_size_um`) and mines public reference text for grain-size / nanocrystalline phrases. It emits `measured_microstructure_manifest_v17` while keeping the strict no-manual policy.
3. **T48 artifact cleanup:** HTML/JS/CSS boilerplate table summaries from the NREL PV page are filtered from report summaries; support remains governed by physical PV rows and the global + family FDR gate.

No evidence rules were relaxed: only primary public physical tables can confirm/falsify; secondary or metadata artifacts remain diagnostic only.
