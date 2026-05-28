# Round-10 v40 confirm-target patch

Focus: P30-SDSS-core curl-residual separation after the v39 H/L orientation fix.

Implemented improvements:
1. Object-level science = alpha + beta*curl + residual model.
2. Patch-level paired bootstrap for science-minus-abs-curl.
3. Residual curl-null gate after projection/regression.
4. Curl-family specificity diagnostics for baseline/f150/tonly.
5. Route-specific P30-SDSS-core confirm gate only; global P30 remains blocked.
6. Cached low-Nside sampled ACT values to reduce repeated ALM reconstruction and memory failures.
7. P3 endpoint hard skip preserved.
8. P36 high-z strict object-level catalogue gate preserved.
9. P41 Wilson/SM likelihood gate preserved.
10. P33/P32 measured-output gates preserved.

Primary output:
- outputs/p30_patch_level_curl_residual_separation_v40.json
- outputs/p30_low_nside_sample_cache_manifest_v40.json

Run:
python run_all.py --only P30 --allow-large --max-mb 12000 --script-timeout 9000
