# v8 all five quality fixes

Implemented the five requested result-quality fixes:

1. Manifest loading diagnostics for fusion/electronics curated-source runners.
   Outputs now include manifest_exists, manifest_rows_total, manifest_columns,
   manifest_test_ids_seen, rows_selected_for_test, and selected_labels.

2. OSF ITPA DB5.2.3 item-level diagnostics for T28/T30.
   The OSF walker now reports item names, kinds, paths, download URLs, related
   folder links, exact DB5/STD5 match flags, structured-extension flags, and
   candidate rejection reasons.

3. Honest MAT1/MAT3 microstructure manifest diagnostics.
   Results now report decisive/grain-size-known manifest rows, matched evidence
   classes, matched table samples, and decisive_microstructure_status. Proxy rows
   are explicitly treated as controls, not decisive measured-grain evidence.

4. T48 PV family/proxy split analysis.
   The NREL PV parser now reports global and technology-family residual tests
   for silicon, III-V/tandem, thin-film CdTe/CIGS, perovskite, organic/dye, and
   other families when enough rows exist.

5. Stronger T46 ECC baselines.
   T46 now compares the CDT-like irregular/nonlocal proxy against local LDPC,
   surface-like, protograph/QC-like, spatially-coupled LDPC-like, and interleaved
   RS-like parity proxies at matched n/checks/weight.
