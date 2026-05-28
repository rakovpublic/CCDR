# Round-10 v43 patch notes

v43 is a confirm-target patch built from the v42 report.

## Implemented improvements

1. P36 high-z second-source discovery and audit.
2. P36 source-level bootstrap and leave-one-source stability gate.
3. P36 compact audit rows with source, object_id, z, Vrot, radius, computed acceleration, and unit provenance.
4. P30 bad-patch isolation, including worst-patch quarantine diagnostic.
5. P30 leave-one-patch-out stability gate.
6. P30 empirical/official mask consistency gate.
7. P41 explicit Wilson/SM coefficient-fit layer and CP-control split.
8. P33 measured alpha_high/alpha_low density-split consumer hardening.
9. P32 optional GW150914 GWOSC strain download/product gate and strict measured-product requirements.
10. Dashboard v43 strict claim separation.

## Claim policy

- P36 high-z cannot promote unless there are at least two independent real-object sources and source bootstrap CI16 > 1.
- P30 cannot promote if it depends on a post-hoc bad-patch quarantine or lacks official/equivalence mask support.
- P41 cannot promote without numerical Wilson/SM Δχ² and CP controls.
- P33 cannot promote from design contracts; it needs measured alpha split rows.
- P32 cannot promote from metadata or strain download alone; it needs PSD, GR fit, CCDR residual fit, injection nulls, and detector split.
