# v2.1 bugfix notes

Fixed after `run_all.py` crash report:

1. `test_fr3_fr6_literature_proxy.py`
   - Fixed `IndexError: no such group` in `parse_diiid_knolker_table2()`.
   - Cause: fallback regex alternatives for exact table sequences had no capture group, but the parser always called `m.group(1)`.
   - Fix: parse `m.group(1)` only when a capture group exists; otherwise parse `m.group(0)`.
   - Added tolerant float matching for table fallback checks.

2. `test_fr8_eta_s_collisionality_proxy.py`
   - This failed only because it imports and reuses the FR3/FR6 Knolker parser. The shared parser fix resolves the FR8 crash.

3. `test_qc5_phononic_crystal_t1_public.py`
   - Improved Chen/Painter arXiv abstract parsing so `$5$ milliseconds`, `5 ms`, and `two-orders` variants are recognized.
   - Corrected the Odeh/Sipahigil 34 us result classification: it is now treated as an engineered TLS / qubit-probe result, not as standalone transmon-qubit T1.

Expected status after patch:

- FR3: usually `data_insufficient_public_proxy` unless a public CSV URL with full shot/cycle rows is supplied.
- FR6: should run and report a sign-level RMP-current proxy result if the MAST text extraction succeeds.
- FR8: should run; likely `data_insufficient_public_proxy` because public sources do not give matched edge `eta/s` / `M_KSS` rows.
- QC5: should become `partial_support_TLS_not_qubit` when the Chen/Painter abstract text is extracted normally.
- EL3: unchanged.
