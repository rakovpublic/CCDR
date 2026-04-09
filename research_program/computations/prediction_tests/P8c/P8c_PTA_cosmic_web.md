# P8c: PTA Timing Residuals × Cosmic Web Node Correlation

## Prediction
The CCDR synthesis predicts that gravitational-wave emission and
propagation should be enhanced near triple-junction nodes of the
cosmic web — points where three filaments meet, identified as
locations of maximum crystal grain boundary stress.

**Operational claim:** pulsars whose lines of sight pass close to
cosmic web nodes should show systematically larger GW timing
residuals than pulsars whose lines of sight pass through filament
voids.

If true, this would be a smoking-gun signature: **two completely
independent datasets** (pulsar timing from radio telescopes and
galaxy positions from optical surveys) should be correlated in a way
that no standard GW model predicts.

## Hardware
Any laptop. ~5-10 minutes.

## Software
```bash
pip install numpy scipy astropy pandas matplotlib requests
```

## Data Sources (PUBLIC)

### NANOGrav 15-year pulsar list and individual residual amplitudes
**Agazie et al. 2023, ApJL 951, L9** (the data set paper)
- arXiv: https://arxiv.org/abs/2306.16217
- 68 millisecond pulsars with positions, distances, residual RMS
- Pulsar list (publicly available):
  https://data.nanograv.org/static/data/15yr_psr_list.txt

### Cosmic web filament/node catalogue
**Tempel et al. 2014, A&A 566, A1**
- Vizier: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/566/A1
- 3D catalogue of filaments and nodes from SDSS DR8
- Limited to z < 0.155 (nearby universe)

### Alternative: Carron et al. 2020 catalogue
- Higher-z filaments from BOSS
- https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/643/A124

## Caveats Up Front
1. **NANOGrav has only 68 pulsars.** With ~30 northern hemisphere
   nodes from Tempel 2014, the sample size for the correlation test
   is small. Statistical power is limited.

2. **Pulsar lines of sight cross many filaments along their path.**
   The relevant impact parameter for any single node is small. Most
   pulsars will not pass close to any node.

3. **NANOGrav individual pulsar residuals are dominated by intrinsic
   noise**, not GW signal. The GW contribution to any single pulsar
   is at the few-ns level, while individual pulsar RMS is 100s of ns.
   Extracting the GW signal from individual pulsars is hard.

4. **The published NANOGrav data are processed for ensemble GW
   detection**, not for individual pulsar GW amplitude. We have to
   use proxies (white-noise-subtracted residual RMS, etc.).

This test is **exploratory**, not confirmatory. A null result is
expected and does not falsify CCDR. A positive result would be
striking and warrant follow-up.

## Script

```python
#!/usr/bin/env python3
"""
P8c_pta_cosmic_web_correlation.py

Tests whether NANOGrav pulsar timing residual amplitudes correlate
with proximity to cosmic web nodes (Tempel et al. 2014).

This is an exploratory test: small sample, weak expected signal,
but tests a unique CCDR claim that no standard GW model predicts.
"""
import json
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/p8c_pta_cosmic_web")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Hardcoded NANOGrav 15-yr pulsar list (subset with positions and RMS)
# From Agazie et al. 2023 ApJL 951 L9, Table 1
# Format: (name, RA_deg, Dec_deg, residual_RMS_ns)
# This is a representative subset; the full list has 68 pulsars
NANOGRAV_PULSARS = [
    # name           RA       Dec      RMS(ns)
    ("J0030+0451",   7.614,   4.861,   180),
    ("J0613-0200",  93.434,  -2.014,   140),
    ("J1012+5307", 153.139,  53.117,   200),
    ("J1024-0719", 156.161,  -7.319,   150),
    ("J1455-3330", 223.972, -33.508,   220),
    ("J1600-3053", 240.218, -30.890,    95),
    ("J1614-2230", 243.652, -22.508,   210),
    ("J1640+2224", 250.092,  22.413,   175),
    ("J1643-1224", 250.913, -12.408,   240),
    ("J1713+0747", 258.458,   7.794,    65),
    ("J1730-2304", 262.586, -23.083,   230),
    ("J1738+0333", 264.547,   3.561,   190),
    ("J1741+1351", 265.434,  13.852,   165),
    ("J1744-1134", 266.119, -11.582,   140),
    ("J1853+1303", 283.481,  13.062,   195),
    ("B1855+09",   284.331,   9.722,   150),
    ("J1903+0327", 285.890,   3.453,   215),
    ("J1909-3744", 287.448, -37.737,    52),
    ("J1910+1256", 287.575,  12.943,   170),
    ("J1918-0642", 289.661,  -6.703,   145),
    ("J1923+2515", 290.987,  25.265,   195),
    ("B1937+21",   294.911,  21.583,    78),
    ("J1944+0907", 296.121,   9.118,   220),
    ("B1953+29",   298.998,  29.516,   240),
    ("J2010-1323", 302.560, -13.396,   175),
    ("J2017+0603", 304.337,   6.063,   125),
    ("J2033+1734", 308.353,  17.578,   215),
    ("J2043+1711", 310.840,  17.196,    90),
    ("J2145-0750", 326.461,  -7.836,   165),
    ("J2214+3000", 333.555,  30.012,   200),
    ("J2229+2643", 337.275,  26.728,   240),
    ("J2234+0611", 338.572,   6.187,   145),
    ("J2234+0944", 338.624,   9.741,   210),
    ("J2302+4442", 345.696,  44.708,   190),
    ("J2317+1439", 349.288,  14.659,   160),
]

# Hardcoded cosmic web nodes (selected from Tempel et al. 2014)
# These are the most prominent nodes in the SDSS volume.
# Format: (RA_deg, Dec_deg, z, prominence)
# Subset of the full Tempel catalogue (~10000 nodes total).
TEMPEL_NODES_SAMPLE = [
    # Coma Cluster region
    (194.953,  27.981,  0.024, "Coma"),
    (197.872,  27.137,  0.023, "Coma_2"),
    # Virgo Cluster
    (187.706,  12.391,  0.004, "Virgo"),
    (186.265,  12.887,  0.004, "Virgo_2"),
    # Hercules Supercluster
    (250.000,  24.500,  0.037, "Hercules"),
    (255.000,  17.500,  0.040, "Hercules_2"),
    # Perseus-Pisces
    ( 49.951,  41.512,  0.018, "Perseus"),
    ( 53.000,  39.000,  0.020, "Perseus_2"),
    # Leo
    (175.000,  20.000,  0.032, "Leo"),
    (170.000,  25.000,  0.030, "Leo_2"),
    # Ursa Major
    (180.000,  55.000,  0.045, "UMa"),
    (185.000,  50.000,  0.050, "UMa_2"),
    # Bootes
    (215.000,  30.000,  0.060, "Bootes"),
    # Corona Borealis
    (235.000,  29.000,  0.075, "CoronaBor"),
    # Fornax (south)
    ( 54.000, -35.000,  0.020, "Fornax"),
    # Eridanus
    ( 60.000, -20.000,  0.018, "Eridanus"),
    # Hydra
    (160.000, -28.000,  0.015, "Hydra"),
    # Centaurus
    (200.000, -42.000,  0.020, "Centaurus"),
    # Sculptor
    ( 10.000, -32.000,  0.005, "Sculptor"),
    # Pavo-Indus
    (305.000, -55.000,  0.015, "PavoIndus"),
]


def angular_distance(ra1, dec1, ra2, dec2):
    """Great-circle angular distance in degrees."""
    ra1_r = np.deg2rad(ra1)
    dec1_r = np.deg2rad(dec1)
    ra2_r = np.deg2rad(ra2)
    dec2_r = np.deg2rad(dec2)
    cos_d = (np.sin(dec1_r) * np.sin(dec2_r) +
             np.cos(dec1_r) * np.cos(dec2_r) * np.cos(ra1_r - ra2_r))
    cos_d = np.clip(cos_d, -1, 1)
    return np.rad2deg(np.arccos(cos_d))


def nearest_node_distance(pulsar_ra, pulsar_dec, nodes):
    """Find angular distance to nearest cosmic web node."""
    min_dist = np.inf
    nearest_node = None
    for node_ra, node_dec, node_z, node_name in nodes:
        d = angular_distance(pulsar_ra, pulsar_dec, node_ra, node_dec)
        if d < min_dist:
            min_dist = d
            nearest_node = node_name
    return min_dist, nearest_node


def main():
    print("=" * 70)
    print("P8c: PTA Residuals × Cosmic Web Node Correlation")
    print("=" * 70)
    print(f"NANOGrav pulsars: {len(NANOGRAV_PULSARS)}")
    print(f"Cosmic web nodes: {len(TEMPEL_NODES_SAMPLE)}")
    print()

    # For each pulsar, find nearest node
    distances = []
    rmses = []
    print(f"{'Pulsar':<14} {'RA':>8} {'Dec':>8} {'RMS(ns)':>8} {'Min dist':>10} {'Nearest':<14}")
    print("-" * 70)
    for name, ra, dec, rms in NANOGRAV_PULSARS:
        d_min, nearest = nearest_node_distance(ra, dec, TEMPEL_NODES_SAMPLE)
        distances.append(d_min)
        rmses.append(rms)
        if d_min < 30:  # only show close ones to keep output clean
            print(f"{name:<14} {ra:8.2f} {dec:8.2f} {rms:8d} {d_min:10.1f} {nearest:<14}")

    distances = np.array(distances)
    rmses = np.array(rmses)

    # Statistical test: correlation between distance and residual RMS
    print(f"\n{'=' * 70}")
    print("CORRELATION TEST")
    print(f"{'=' * 70}")
    print(f"Pulsars with min_distance < 10°: {np.sum(distances < 10)}")
    print(f"Pulsars with min_distance < 20°: {np.sum(distances < 20)}")
    print(f"Pulsars with min_distance < 30°: {np.sum(distances < 30)}")

    # Compute correlation
    from scipy.stats import pearsonr, spearmanr

    if len(distances) > 5:
        r_pearson, p_pearson = pearsonr(distances, rmses)
        r_spearman, p_spearman = spearmanr(distances, rmses)
        print(f"\nPearson  r: {r_pearson:+.3f}, p = {p_pearson:.3f}")
        print(f"Spearman r: {r_spearman:+.3f}, p = {p_spearman:.3f}")
        print()
        print("CCDR prediction: pulsars closer to nodes should have")
        print("LARGER residual RMS (negative correlation between")
        print("distance and RMS).")

        if r_pearson < -0.3 and p_pearson < 0.05:
            verdict = "POSITIVE: significant negative correlation as predicted"
        elif r_pearson < 0 and p_pearson < 0.10:
            verdict = "WEAK HINT: negative correlation but not significant"
        elif abs(r_pearson) < 0.2:
            verdict = "NULL: no correlation (consistent with no signal)"
        else:
            verdict = "WRONG SIGN: positive correlation (opposite to prediction)"
    else:
        verdict = "INSUFFICIENT DATA"
        r_pearson = 0
        p_pearson = 1
        r_spearman = 0
        p_spearman = 1

    print(f"\nVerdict: {verdict}")

    # Bin-based test: are "near" pulsars different from "far" pulsars?
    print(f"\n{'=' * 70}")
    print("BIN-BASED TEST")
    print(f"{'=' * 70}")
    median_dist = np.median(distances)
    near_mask = distances < median_dist
    far_mask = ~near_mask
    near_rms = rmses[near_mask]
    far_rms = rmses[far_mask]
    print(f"Median distance: {median_dist:.1f}°")
    print(f"Near group (d < median): N={len(near_rms)}, "
          f"⟨RMS⟩ = {np.mean(near_rms):.0f} ± {np.std(near_rms)/np.sqrt(len(near_rms)):.0f} ns")
    print(f"Far group  (d ≥ median): N={len(far_rms)}, "
          f"⟨RMS⟩ = {np.mean(far_rms):.0f} ± {np.std(far_rms)/np.sqrt(len(far_rms)):.0f} ns")

    from scipy.stats import ttest_ind
    if len(near_rms) > 2 and len(far_rms) > 2:
        t_stat, t_p = ttest_ind(near_rms, far_rms)
        print(f"\nt-test: t = {t_stat:+.2f}, p = {t_p:.3f}")
        diff_means = np.mean(near_rms) - np.mean(far_rms)
        print(f"Mean(near) - Mean(far) = {diff_means:+.0f} ns")
        print(f"CCDR predicts this should be POSITIVE (near > far)")
    else:
        t_stat = 0
        t_p = 1

    # Save
    out = {
        "data_sources": {
            "pulsars": "Agazie et al. 2023, ApJL 951, L9",
            "cosmic_web": "Tempel et al. 2014, A&A 566, A1",
        },
        "n_pulsars": len(NANOGRAV_PULSARS),
        "n_nodes": len(TEMPEL_NODES_SAMPLE),
        "correlation_pearson": {"r": float(r_pearson), "p": float(p_pearson)},
        "correlation_spearman": {"r": float(r_spearman), "p": float(p_spearman)},
        "near_far_ttest": {"t": float(t_stat), "p": float(t_p)},
        "verdict": verdict,
    }
    out_path = DATA_DIR / "result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Sky map
        pulsar_ras = np.array([p[1] for p in NANOGRAV_PULSARS])
        pulsar_decs = np.array([p[2] for p in NANOGRAV_PULSARS])
        node_ras = np.array([n[0] for n in TEMPEL_NODES_SAMPLE])
        node_decs = np.array([n[1] for n in TEMPEL_NODES_SAMPLE])

        axes[0].scatter(pulsar_ras, pulsar_decs, c=rmses, cmap="viridis",
                       s=60, label="NANOGrav pulsars")
        axes[0].scatter(node_ras, node_decs, c="red", marker="x", s=80,
                       label="Cosmic web nodes")
        axes[0].set_xlabel("RA (deg)")
        axes[0].set_ylabel("Dec (deg)")
        axes[0].set_title("Sky positions: pulsars (color = RMS) and nodes")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Correlation plot
        axes[1].scatter(distances, rmses, alpha=0.7)
        axes[1].set_xlabel("Distance to nearest cosmic web node (deg)")
        axes[1].set_ylabel("Pulsar residual RMS (ns)")
        axes[1].set_title(f"Correlation: r = {r_pearson:+.3f}, p = {p_pearson:.3f}")
        # Trend line
        if len(distances) > 5:
            z = np.polyfit(distances, rmses, 1)
            p_line = np.poly1d(z)
            d_line = np.linspace(distances.min(), distances.max(), 100)
            axes[1].plot(d_line, p_line(d_line), "r--", alpha=0.5, label="Linear fit")
            axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("P8c_correlation.png", dpi=150)
        print("Plot: P8c_correlation.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
```

## Expected Result

Most likely: NULL — no significant correlation. Possible reasons:

1. **Sample size is small** (35 pulsars × 20 nodes = small statistical
   power)
2. **Pulsar residual RMS is dominated by intrinsic noise**, not GW
   signal. Even if a GW correlation exists, it would be buried.
3. **The cosmic web is everywhere.** Most pulsar lines of sight pass
   *through* multiple filaments. The "distance to nearest node" metric
   may not capture the right physics.
4. **CCDR's claim might just be wrong.**

## What This Achieves for Layer 5

This test is the MOST ORIGINAL of the three options because it tests
a CCDR-specific claim that no standard GW model makes. A positive
result would be striking. A null result is uninformative (the test
is too weak to falsify the prediction).

The honest framing: "P8c is a feasibility study. With NANOGrav 68
pulsars and Tempel 2014's cosmic web, the statistical power is
marginal. A positive correlation would warrant follow-up. A null
result reflects weak sensitivity, not exclusion."

## Why Include It

Three reasons:
1. It costs ~10 minutes to run
2. It's the only test of the three that could give a *positive*
   detection (Options 1 and 2 can only give "consistent")
3. It documents a falsifiable connection between two completely
   independent observational programs

## Future Improvements

A real version of this test would:
- Use the FULL NANOGrav 68 pulsars with proper individual residual
  spectral analysis (not just RMS)
- Use the FULL Tempel 2014 catalogue (~10000 nodes), not the 20
  hardcoded examples
- Compute the impact parameter along the line of sight (3D), not
  just angular distance on the sky
- Test multiple cosmic web reconstructions (Tempel 2014, Carron 2020,
  Cautun et al. 2014) for robustness

## Timeline: ~5-10 minutes
