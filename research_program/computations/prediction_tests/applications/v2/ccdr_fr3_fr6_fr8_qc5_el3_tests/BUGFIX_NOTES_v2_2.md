# v2.2 implementation notes

Implemented the four requested improvements from the prediction-output analysis.

1. **PDF extraction dependency**
   * `pdfminer.six` remains in `requirements.txt`; `html5lib` was added for more robust `pandas.read_html` parsing.
   * README now tells users to reinstall requirements with `--upgrade` if the run prints `No module named 'pdfminer'`.

2. **`run_all.py` nested summary fix**
   * Combined FR3/FR6 output is flattened in `out_summary.json`.
   * Summary now reports `FR3: data_insufficient_public_proxy` and `FR6: proxy_pass` instead of `decision: null`.

3. **QC5 split**
   * QC5 is now reported as `QC5a_TLS_mechanism` and `QC5b_transmon_T1`.
   * TLS bandgap support is no longer conflated with full standalone-transmon confirmation.

4. **EL3 foundry-node strengthening**
   * Added `--include-foundry-nodes` to verify modern foundry density rows against downloaded public process-node pages.
   * The script writes `el3_cache/foundry_node_density_public_extracted.csv` for auditability.
   * `run_all.py` enables this flag by default.
