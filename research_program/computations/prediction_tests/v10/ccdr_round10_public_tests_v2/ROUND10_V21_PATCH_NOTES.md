# Round-10 v21: 10 confirmation-oriented improvements, P3-first

Generated: 2026-05-07T11:52:10Z

Implemented:
1. P3 exact metadata-first table discovery: metadata/table candidates before row-heavy downloads.
2. P3 fallback filament catalogue metadata routes.
3. P3 strict endpoint parser: named endpoint columns only.
4. P3 orientation/nulls: endpoint axes, endpoint shuffle, length split, redshift-bin shuffle if available.
5. High-z a0 Vrot: unit + field/table robustness gate.
6. P41: deeper supplementary ZIP/archive member parsing for structured q2/value/sign rows.
7. CL2: residual/TOA-weighted parser path scaffold.
8. Direct detection: explicit unit-label coverage guard.
9. P33 and ringdown: confirmation scaffolds; P30 frozen as diagnostic tension until mask/random catalogue exists.
10. Dashboard: P3 status, confirm-like candidates, tension, and guarded buckets.

Updated test files:
{
  "R10-T09": "test09_p03_filament_catalogue_inventory.py",
  "R10-T13": "test13_p36_kmos3d_inventory.py",
  "R10-T14": "test14_p36_highz_a0_cross_catalogue_inventory.py",
  "R10-T31": "test31_p41_lhcb_bsll_inventory.py",
  "R10-T32": "test32_p41_hepdata_api_inventory.py",
  "R10-T17": "test17_p08c_nanograv_density_cross_scaffold.py",
  "R10-T25": "test25_p10_xenonnt_2025_inventory.py",
  "R10-T26": "test26_p10_lz_inventory.py",
  "R10-T07": "test07_p33_desi_density_bao_inventory.py",
  "R10-T19": "test19_p32_ringdown_high_snr_scaffold.py",
  "R10-T04": "test04_p30_act_dr6_lensing_inventory.py",
  "R10-DASH": "test51_round10_joint_dashboard.py"
}
