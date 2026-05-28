# Round-10 v17.1 hotfix

Fixes uploaded T04 failure:

```text
NameError: name '_v16_act_alm_probe_modes' is not defined
```

Change:
- Adds `_v16_act_alm_probe_modes()` compatibility wrapper around the existing v15 ACT ALM reader.
- Keeps the same return shape expected by v17 P30/CL2 runners: `_best_map`, `best_method`, `best_sanity`, `probes`, `reader_info`.
- No science thresholds changed.
