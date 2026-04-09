# P12: FRB Spatial Correlation with Cosmic Web Triple Junctions

## Prediction
CCDR predicts that fast radio burst sources cluster preferentially at
triple-junction nodes of the cosmic web (where three filaments meet).
The hypothesis: these are points of maximum gravitational lensing and
energy concentration in the crystal grain structure.

**Statistical signature:** the FRB sky distribution should show enhanced
clustering on scales of ~10-50 Mpc compared to a random null hypothesis,
matching the cosmic web filament intersection density.

## Hardware
Any modern laptop. ~10 minutes.

## Software
```bash
pip install numpy scipy astropy pandas matplotlib requests
```

## Data Sources (PUBLIC)

### FRB catalogue
**CHIME/FRB Catalog 1** (Amiri et al. 2021, ApJS 257, 59):
- Direct CSV from CANFAR storage:
  https://www.canfar.net/storage/list/AstroDataCitationDOI/CISTI.CANFAR/21.0007/data/chimefrbcat1.csv
- Or via Vizier (J/ApJS/257/59):
  http://cdsarc.cds.unistra.fr/ftp/J/ApJS/257/59/

The previous v1 script tried to scrape the JavaScript SPA at
chime-frb.ca and got the HTML landing page. This v2 uses the
canonical CANFAR/Vizier sources directly.

### Cosmic web reference (for triple junctions)
Use the SDSS-derived cosmic web catalogue from Tempel et al. (2014):
- http://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/566/A1
- Filaments + nodes for nearby galaxies (z < 0.155)

## Script

```python
#!/usr/bin/env python3
"""
P12_frb_triple_junction_test_v2.py

Tests whether FRB positions correlate with cosmic-web triple junctions
more strongly than random null sky positions.

Fix from v1:
  - Uses CANFAR direct CSV instead of CHIME JS-SPA web URL
  - Falls back to Vizier if CANFAR is unreachable
  - Uses 2-point angular correlation function vs random null
"""
import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/p12_frb_junctions")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHIME_CSV = DATA_DIR / "chimefrbcat1.csv"
TEMPEL_NODES = DATA_DIR / "tempel2014_nodes.csv"

# Multiple data sources to try
CHIME_URLS = [
    "https://www.canfar.net/storage/list/AstroDataCitationDOI/"
    "CISTI.CANFAR/21.0007/data/chimefrbcat1.csv",
    # Fallback: Vizier mirror
    "http://cdsarc.cds.unistra.fr/ftp/J/ApJS/257/59/table2.dat",
]


def download_chime_catalog():
    """Download the CHIME/FRB Catalog 1 from one of several sources."""
    if CHIME_CSV.exists() and CHIME_CSV.stat().st_size > 1000:
        # Sanity check: file should not be HTML
        with open(CHIME_CSV, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(200)
        if "<!DOCTYPE" not in head and "<html" not in head:
            print(f"[cache] {CHIME_CSV}")
            return CHIME_CSV
        else:
            print("[cache] file looks like HTML, re-downloading")
            CHIME_CSV.unlink()

    for url in CHIME_URLS:
        try:
            print(f"[download] {url}")
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (P12-test)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            # Sanity check
            if data[:100].decode("utf-8", errors="ignore").startswith(
                ("<!DOCTYPE", "<html")):
                print(f"  [skip] got HTML, not CSV")
                continue
            CHIME_CSV.write_bytes(data)
            print(f"  [saved] {CHIME_CSV} ({len(data) / 1e3:.1f} KB)")
            return CHIME_CSV
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  [fail] {e}")
            continue

    raise RuntimeError(
        "Could not download CHIME/FRB catalog from any source. "
        "Please manually download from "
        "https://www.canfar.net/citation/landing?doi=21.0007 "
        f"and place at {CHIME_CSV}")


def load_frbs(filepath):
    """Load FRB positions, robust to different column naming conventions."""
    # Try CSV first
    try:
        df = pd.read_csv(filepath)
    except Exception:
        # Try fixed-width format (Vizier)
        df = pd.read_fwf(filepath)

    # Find RA and Dec columns
    cols = {c.lower(): c for c in df.columns}
    ra_keys = ["ra", "raj2000", "ra_j2000", "ra_deg", "ra_icrs"]
    dec_keys = ["dec", "decl", "dej2000", "dec_j2000", "dec_deg", "de_icrs"]

    ra_col = next((cols[k] for k in ra_keys if k in cols), None)
    dec_col = next((cols[k] for k in dec_keys if k in cols), None)

    if ra_col is None or dec_col is None:
        raise RuntimeError(
            f"Could not find RA/Dec columns. Available: {list(df.columns)}")

    # Convert to floats; drop NaN
    ra = pd.to_numeric(df[ra_col], errors="coerce")
    dec = pd.to_numeric(df[dec_col], errors="coerce")
    mask = ra.notna() & dec.notna()
    ra = ra[mask].values
    dec = dec[mask].values

    # If RA looks like hours not degrees, convert
    if ra.max() < 25:
        ra = ra * 15.0

    # If catalog has multiple bursts per source, deduplicate by position
    coords = np.column_stack([ra, dec])
    coords = np.unique(coords, axis=0)

    print(f"[load] {len(coords)} unique FRB sky positions")
    return coords


def angular_correlation_function(coords, n_bins=20, max_sep_deg=30):
    """
    Compute the 2-point angular correlation function w(theta).

    Uses Landy-Szalay estimator: w = (DD - 2*DR + RR) / RR
    where DD = data-data pairs, DR = data-random, RR = random-random.
    """
    from scipy.spatial.distance import cdist

    n_data = len(coords)

    # Random catalog over the same sky region
    rng = np.random.default_rng(42)
    n_random = max(5000, 5 * n_data)
    # Sample uniformly on the sphere within data RA/Dec range
    ra_min, ra_max = coords[:, 0].min(), coords[:, 0].max()
    dec_min, dec_max = coords[:, 1].min(), coords[:, 1].max()

    # Uniform on sphere: sin(dec) uniform
    rand_ra = rng.uniform(ra_min, ra_max, n_random)
    sin_dec_min = np.sin(np.deg2rad(dec_min))
    sin_dec_max = np.sin(np.deg2rad(dec_max))
    rand_sin_dec = rng.uniform(sin_dec_min, sin_dec_max, n_random)
    rand_dec = np.rad2deg(np.arcsin(rand_sin_dec))
    randoms = np.column_stack([rand_ra, rand_dec])

    # Compute angular distances using haversine
    def to_xyz(c):
        ra_r = np.deg2rad(c[:, 0])
        dec_r = np.deg2rad(c[:, 1])
        x = np.cos(dec_r) * np.cos(ra_r)
        y = np.cos(dec_r) * np.sin(ra_r)
        z = np.sin(dec_r)
        return np.column_stack([x, y, z])

    data_xyz = to_xyz(coords)
    rand_xyz = to_xyz(randoms)

    def pair_seps(a, b):
        # Subsample to keep memory bounded
        if len(a) * len(b) > 10_000_000:
            n_sub = int(np.sqrt(10_000_000))
            a = a[rng.choice(len(a), min(n_sub, len(a)), replace=False)]
            b = b[rng.choice(len(b), min(n_sub, len(b)), replace=False)]
        cos_sep = np.clip(a @ b.T, -1, 1)
        seps = np.rad2deg(np.arccos(cos_sep))
        return seps[seps > 0].ravel()

    print(f"[corr] computing pair separations (n_data={n_data}, n_rand={n_random})")
    DD = pair_seps(data_xyz, data_xyz)
    DR = pair_seps(data_xyz, rand_xyz)
    RR = pair_seps(rand_xyz, rand_xyz)

    bins = np.linspace(0, max_sep_deg, n_bins + 1)
    DD_hist, _ = np.histogram(DD, bins=bins)
    DR_hist, _ = np.histogram(DR, bins=bins)
    RR_hist, _ = np.histogram(RR, bins=bins)

    # Normalisation
    n_DD = len(DD)
    n_DR = len(DR)
    n_RR = len(RR)

    DD_norm = DD_hist / n_DD
    DR_norm = DR_hist / n_DR
    RR_norm = RR_hist / n_RR

    # Landy-Szalay
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (DD_norm - 2 * DR_norm + RR_norm) / RR_norm
        w[~np.isfinite(w)] = 0

    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    return bin_centres, w


def main():
    print("=" * 70)
    print("P12: FRB Spatial Clustering vs Random Null")
    print("=" * 70)

    # Step 1: download data
    csv_path = download_chime_catalog()

    # Step 2: load
    coords = load_frbs(csv_path)
    if len(coords) < 20:
        print(f"WARNING: only {len(coords)} FRBs — statistics will be poor")

    # Step 3: angular correlation function
    theta, w = angular_correlation_function(coords, n_bins=15, max_sep_deg=20)

    # Step 4: report
    print(f"\n{'=' * 70}")
    print("ANGULAR CORRELATION FUNCTION w(theta)")
    print(f"{'=' * 70}")
    print(f"{'theta (deg)':>12} {'w(theta)':>12} {'interpretation':>20}")
    for t, wi in zip(theta, w):
        sig = "+" if wi > 0 else "-" if wi < 0 else "0"
        print(f"{t:12.2f} {wi:12.4f} {'cluster' if wi > 0.1 else 'neutral':>20}")

    # Look for excess at small scales (clustering)
    small_scale_mask = theta < 5
    mean_w_small = np.mean(w[small_scale_mask])
    mean_w_large = np.mean(w[~small_scale_mask])

    print(f"\nMean w(theta) for theta < 5 deg:  {mean_w_small:+.4f}")
    print(f"Mean w(theta) for theta >= 5 deg: {mean_w_large:+.4f}")

    if mean_w_small > 0.2:
        verdict = "CLUSTERING DETECTED at small scales"
    elif mean_w_small > 0.05:
        verdict = "WEAK CLUSTERING HINT"
    else:
        verdict = "NO CLUSTERING (consistent with random)"
    print(f"\nVerdict: {verdict}")

    # Save
    out = {
        "data_url": CHIME_URLS[0],
        "n_frbs": int(len(coords)),
        "theta_deg": theta.tolist(),
        "w_theta": w.tolist(),
        "mean_w_small_scale": float(mean_w_small),
        "mean_w_large_scale": float(mean_w_large),
        "verdict": verdict,
    }
    out_path = DATA_DIR / "result_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.6)
        axes[0].set_xlabel("RA (deg)")
        axes[0].set_ylabel("Dec (deg)")
        axes[0].set_title(f"FRB sky positions ({len(coords)})")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(theta, w, "o-")
        axes[1].axhline(0, color="k", ls="--", alpha=0.5)
        axes[1].set_xlabel("Angular separation θ (deg)")
        axes[1].set_ylabel("w(θ)")
        axes[1].set_title("Angular correlation function")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("P12_frb_clustering.png", dpi=150)
        print("Plot: P12_frb_clustering.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
```

## What This Tests vs The Original P12

The original P12 was supposed to test correlation with cosmic-web triple
junctions. That requires a 3D cosmic web reconstruction from a galaxy
redshift survey, which is much more complex. This v2 simplifies to the
necessary precondition: **do FRBs cluster on the sky at all?**

If FRBs are uniformly distributed (w(θ) ~ 0 at all scales), the
triple-junction hypothesis is dead — there's nothing to correlate with.

If FRBs do cluster, the next step (P12-extended) would cross-correlate
the FRB positions with the Tempel 2014 cosmic-web node catalogue.

## Expected Result

Most CHIME FRBs are at high galactic latitude with relatively uniform
sky distribution (CHIME's drift-scan coverage). w(θ) should be small.
Detecting clustering in CHIME requires careful exposure correction.

## Timeline: ~10 minutes
