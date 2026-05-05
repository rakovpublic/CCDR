# Tier-B v5 manifest-only source-quality patch

Implemented requested result-quality changes:

1. Three-stage funnel: source metadata -> relevant record -> structured file. Discovery APIs are not evidence.
2. Exact public table/API routes where possible:
   - T28 uses the exact OSF ITPA Global H-mode DB node (`drwcq`) file API.
   - T48 uses NREL/NLR/PVDPC interactive page parsing and fixed PV proxy manifest.
   - T53/T54 use direct FireProtDB/ProteinGym-style stability data gates instead of PDB metadata.
   - T57/T59 use exact HEPData table download manifests, not broad HEPData search.
3. Cache levels: metadata, files, fit_result_cache. T31/T32 cache fit summaries separately.
4. Header-only parsing first: full parsing happens only after required physical columns are visible.
5. Negative keyword filters stop irrelevant Zenodo/OSF records (COVID, survey, yaw/roll, GPS, SIGFOX, etc.).
6. Streaming/size limits: large non-manifest files are skipped before full decode/parse.
7. Manifest-only scientific mode is the default. `--allow-broad-discovery` is opt-in and diagnostic only.

Recommended run:

```powershell
python run_all_tier_b.py --cache tierb_cache_v5 --outdir tierb_out_v5 --mode scientific --manifest-only --max-bytes 50000000 --header-rows 50 --timeout 90 --force
```

Discovery/source scouting should be treated as manifest update generation, not evidence.
