# Round-10 v3 patch notes

Generated: 2026-05-03T20:45:11Z

Fixes:
1. Fixed Windows runner import failure:
   `ModuleNotFoundError: No module named 'ccdr_r10_common'`.
2. Every `tests/test*.py` now prepends the parent bundle directory to `sys.path`.
3. `run_all.py` now passes `PYTHONPATH=<bundle-root>` to subprocesses.
4. Added this patch notes file.

The uploaded `round10_summary(1).json` showed 51/51 `runner_parse_error` statuses, all caused by the same import failure.
