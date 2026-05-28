# Round-10 v20: 10 priorities implemented with P3 concentration

Generated: 2026-05-07T09:46:55Z

Implemented:
1. P3 metadata-first VizieR/CDS discovery: CDS ReadMe parsing, table/file candidates, endpoint-column lines.
2. P3 exact endpoint parser: named RA1/DEC1/RA2/DEC2 or X1/Y1/Z1/X2/Y2/Z2 rows extracted from limited VizieR/CDS tables.
3. P3 orientation statistic: axis vectors, endpoint-shuffle p-value, length-split proxy, redshift-bin shuffle if z exists.
4. P3 guarded promotion: endpoint columns -> ready; endpoint p<=0.05 -> positive-compatible; endpoint+redshift null -> confirm-like candidate.
5. P30 moved to diagnostic/tension workflow with quality/density/mask/variant matrix retained; no confirm squeeze while tension persists.
6. High-z a0 Vrot confirm guard: unit verification + leave-one-table/field readiness required before confirm-like.
7. P41 structured table/sign/CP-null guard retained and made explicit.
8. CL2 residual-weighted kappa path retained as ready scaffold.
9. Direct-detection unit-verified coverage guard retained; no detection claims from limit curves.
10. Dashboard candidate/tension buckets: confirm_like, confirm_like_candidate, near_confirm, tension, guarded_ready, coverage_confirmed.

Updated test files:
{
  "R10-T09": "test09_p03_filament_catalogue_inventory.py",
  "R10-T04": "test04_p30_act_dr6_lensing_inventory.py",
  "R10-T13": "test13_p36_kmos3d_inventory.py",
  "R10-T14": "test14_p36_highz_a0_cross_catalogue_inventory.py",
  "R10-T31": "test31_p41_lhcb_bsll_inventory.py",
  "R10-T32": "test32_p41_hepdata_api_inventory.py",
  "R10-T17": "test17_p08c_nanograv_density_cross_scaffold.py",
  "R10-T25": "test25_p10_xenonnt_2025_inventory.py",
  "R10-T26": "test26_p10_lz_inventory.py",
  "R10-T05": "test05_p30_planck_lensing_inventory.py",
  "R10-T07": "test07_p33_desi_density_bao_inventory.py",
  "R10-T19": "test19_p32_ringdown_high_snr_scaffold.py",
  "R10-DASH": "test51_round10_joint_dashboard.py"
}
