# Tier-A v9.6 quality patch

Implemented improvements:

1. Fixed T03 `_pantheon_columns` NameError by importing a robust Pantheon+/SH0ES column selector.
2. Added robust kappa map product classifier/sampler helpers: WCS maps, HEALPix maps, and ALM detection with optional healpy alm2map.
3. Added Euclid Q1 depth/quality proxy helper for T06/T07 residualization.
4. Added VizieR/CDS parser fallback for T08 filament catalogues.
5. Added supplemental BAO/SN likelihood-support helpers and posterior/chain readers for PTA/GW/ringdown tests.
6. Added automated source seeds for FIRAS/Planck/BK/HEPData/NANOGrav/GWOSC/eta-s posterior discovery.
7. Added strict artifact typing so metadata records are link sources, not physical evidence tables.
8. Added run_tiera_v96_quality.py supplemental diagnostics.

Run:

```powershell
python run_tiera_v96_quality.py --cache .cache --outdir out_v9_6_quality --allow-large
```
