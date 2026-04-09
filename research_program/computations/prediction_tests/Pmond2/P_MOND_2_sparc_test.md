# P-MOND-2: SPARC Radial Acceleration Relation — a₀(z) Test

## Prediction
**Standard MOND:** a₀ = 1.20 × 10⁻¹⁰ m/s² (constant) [Milgrom 1983,
McGaugh, Lelli & Schombert 2016]

**CCDR alternative:** a₀(z) = c × H(z), giving a₀(z=0) = c × H₀ ≈
6.8 × 10⁻¹⁰ m/s² (varying with redshift)

The standard MOND value is well-established from SPARC. The CCDR
prediction is 5.7× larger at z=0 — easily distinguishable.

**The v1 script reported a₀ = 3.0 × 10⁻¹⁰** which matches NEITHER
prediction and was an optimisation boundary artefact, not a real fit.

## Hardware
Any laptop. ~2 minutes.

## Software
```bash
pip install numpy scipy pandas matplotlib requests
```

## Data
**SPARC Radial Acceleration Relation table** (Lelli et al. 2017,
ApJ 836, 152, and McGaugh et al. 2016, PRL 117, 201101):
- Direct: http://astroweb.case.edu/SPARC/RAR.mrt
- Alternative: http://astroweb.cwru.edu/SPARC/

## Script

```python
#!/usr/bin/env python3
"""
P_MOND_2_sparc_test_v2.py

Fits the MOND a₀ parameter to the SPARC RAR data, with:
  1. Sanity check: must recover Milgrom's 1.2e-10 with the standard
     interpolation function
  2. Search bounds wide enough to detect both Milgrom (1.2e-10) and
     CCDR (6.8e-10) values
  3. Real chi-squared minimum (not boundary pinning)
  4. Comparison with both predictions

Fix from v1: v1 hit a parameter boundary at exactly 3.0e-10 (with
floating-point artefact 3.000000000000001), which is between the two
predictions but matches neither. The optimiser was hitting an upper
bound, not finding a real minimum.
"""
import json
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

DATA_DIR = Path("data/p_mond_2_sparc")
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL_PRIMARY = "http://astroweb.case.edu/SPARC/RAR.mrt"
URL_FALLBACK = "http://astroweb.cwru.edu/SPARC/RAR.mrt"
RAR_FILE = DATA_DIR / "RAR.mrt"

# Predictions
A0_MILGROM = 1.20e-10  # m/s² (standard MOND)
H0_KMSMPC = 67.4       # km/s/Mpc (Planck 2018)
H0_SI = H0_KMSMPC * 1e3 / 3.086e22  # 1/s
C_LIGHT = 2.998e8      # m/s
A0_CCDR = C_LIGHT * H0_SI  # ~6.78e-10 m/s²


def download_rar():
    """Download SPARC RAR table from one of two mirrors."""
    if RAR_FILE.exists() and RAR_FILE.stat().st_size > 1000:
        print(f"[cache] {RAR_FILE}")
        return RAR_FILE

    for url in (URL_PRIMARY, URL_FALLBACK):
        try:
            print(f"[download] {url}")
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (P-MOND-2)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            RAR_FILE.write_bytes(data)
            print(f"  [saved] {RAR_FILE} ({len(data) / 1e3:.1f} KB)")
            return RAR_FILE
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  [fail] {e}")
            continue

    raise RuntimeError(
        "Could not download SPARC RAR table. Please manually download "
        f"from {URL_PRIMARY} or {URL_FALLBACK} and place at {RAR_FILE}")


def load_rar(filepath):
    """
    Load the SPARC RAR table.

    The .mrt file is in CDS Machine-Readable Table format. The relevant
    columns are g_obs (observed acceleration) and g_bar (baryonic
    acceleration), both in m/s².
    """
    # Skip header lines until we find the data
    with open(filepath, "r") as f:
        lines = f.readlines()

    # CDS .mrt header ends with a line of dashes
    data_start = 0
    dash_count = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("-" * 5):
            dash_count += 1
            if dash_count == 2:  # second dash line is end of column descriptions
                data_start = i + 1
                break

    # Try to detect g_obs and g_bar columns from header
    # The standard SPARC RAR file has columns:
    # ID, R, g_bar, e_g_bar, g_obs, e_g_obs (or similar)
    g_bar_list = []
    g_obs_list = []
    err_obs_list = []

    for line in lines[data_start:]:
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            # Different SPARC versions have slightly different layouts
            # Try the most common: ID, R(kpc), g_obs, e_g_obs, g_bar, e_g_bar
            # or:                    ID, R(kpc), g_bar, e_g_bar, g_obs, e_g_obs
            nums = [float(p) for p in parts[1:6]]
            # Heuristic: g_obs > g_bar usually (in MOND regime)
            # We'll just take both and let user verify
            r_kpc = nums[0]
            g_a, e_a, g_b, e_b = nums[1], nums[2], nums[3], nums[4]
            # g_obs > g_bar in low-acc regime, so:
            if g_a > g_b:
                g_obs, e_obs, g_bar = g_a, e_a, g_b
            else:
                g_obs, e_obs, g_bar = g_b, e_b, g_a
            g_obs_list.append(g_obs)
            err_obs_list.append(e_obs)
            g_bar_list.append(g_bar)
        except (ValueError, IndexError):
            continue

    g_obs = np.array(g_obs_list)
    g_bar = np.array(g_bar_list)
    e_obs = np.array(err_obs_list)
    print(f"[load] {len(g_obs)} (g_bar, g_obs) data points")
    if len(g_obs) > 0:
        print(f"       g_obs range: {g_obs.min():.2e} to {g_obs.max():.2e} m/s²")
        print(f"       g_bar range: {g_bar.min():.2e} to {g_bar.max():.2e} m/s²")
    return g_bar, g_obs, e_obs


def mond_simple_interp(g_bar, a0):
    """
    Simple MOND interpolation function (Milgrom 1983):
        g_obs = g_bar / (1 - exp(-sqrt(g_bar / a0)))
    This is the function McGaugh, Lelli & Schombert (2016) fitted SPARC with.
    """
    x = np.sqrt(g_bar / a0)
    return g_bar / (1.0 - np.exp(-x))


def chi2_log(a0, g_bar, g_obs):
    """Chi-squared in log space (more robust to outliers)."""
    g_pred = mond_simple_interp(g_bar, a0)
    log_obs = np.log10(g_obs)
    log_pred = np.log10(g_pred)
    return np.mean((log_obs - log_pred) ** 2)


def fit_a0(g_bar, g_obs, bounds=(1e-12, 1e-8)):
    """
    Fit a0 with WIDE bounds (1e-12 to 1e-8 m/s²) so neither Milgrom's
    1.2e-10 nor CCDR's 6.8e-10 sits at a boundary.

    Returns (best_a0, mse, hit_boundary_flag)
    """
    result = minimize_scalar(
        lambda a0: chi2_log(a0, g_bar, g_obs),
        bounds=bounds, method="bounded",
        options={"xatol": 1e-15})

    best_a0 = result.x
    best_mse = result.fun

    # Check if we hit a boundary (within 1% of either edge)
    hit_lower = abs(best_a0 - bounds[0]) / bounds[0] < 0.01
    hit_upper = abs(best_a0 - bounds[1]) / bounds[1] < 0.01
    hit_boundary = hit_lower or hit_upper

    return best_a0, best_mse, hit_boundary


def main():
    print("=" * 70)
    print("P-MOND-2: SPARC RAR Test (Milgrom vs CCDR)")
    print("=" * 70)
    print(f"Milgrom prediction: a0 = {A0_MILGROM:.2e} m/s²")
    print(f"CCDR prediction:    a0 = c·H0 = {A0_CCDR:.2e} m/s²")
    print(f"Ratio: a0(CCDR) / a0(Milgrom) = {A0_CCDR / A0_MILGROM:.2f}×")
    print()

    # Step 1: download
    rar_path = download_rar()

    # Step 2: load
    g_bar, g_obs, e_obs = load_rar(rar_path)
    if len(g_bar) < 100:
        print(f"WARNING: only {len(g_bar)} data points loaded — check file format")
        print("Inspect the first 20 lines of the file:")
        with open(rar_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 30:
                    break
                print(f"  {line.rstrip()}")
        return

    # Step 3: SANITY CHECK — recover Milgrom's value
    print(f"\n[sanity] Fitting with WIDE bounds (1e-12 to 1e-8 m/s²)")
    best_a0, best_mse, hit_boundary = fit_a0(g_bar, g_obs)
    print(f"  Best a0:          {best_a0:.4e} m/s²")
    print(f"  MSE (log space):  {best_mse:.4f}")
    print(f"  Hit boundary:     {hit_boundary}")

    if hit_boundary:
        print("  ✗ FIT FAILED: hit boundary, result unreliable")
        print("  This was the v1 problem.")
        return

    # Compare with predictions
    sigma_milgrom = abs(best_a0 - A0_MILGROM) / A0_MILGROM
    sigma_ccdr = abs(best_a0 - A0_CCDR) / A0_CCDR

    print(f"\n  Distance from Milgrom (1.2e-10): {sigma_milgrom * 100:.1f}%")
    print(f"  Distance from CCDR    (6.8e-10): {sigma_ccdr * 100:.1f}%")

    # Step 4: explicit chi-squared at the two predictions
    chi2_at_milgrom = chi2_log(A0_MILGROM, g_bar, g_obs)
    chi2_at_ccdr = chi2_log(A0_CCDR, g_bar, g_obs)
    chi2_at_best = chi2_log(best_a0, g_bar, g_obs)

    print(f"\n{'=' * 70}")
    print("CHI-SQUARED AT EACH PREDICTION")
    print(f"{'=' * 70}")
    print(f"  At Milgrom (1.2e-10): MSE = {chi2_at_milgrom:.4f}")
    print(f"  At CCDR    (6.8e-10): MSE = {chi2_at_ccdr:.4f}")
    print(f"  At best    ({best_a0:.2e}): MSE = {chi2_at_best:.4f}")

    delta_milgrom = chi2_at_milgrom - chi2_at_best
    delta_ccdr = chi2_at_ccdr - chi2_at_best
    print(f"\n  Δ(MSE) Milgrom - best: {delta_milgrom:.4f}")
    print(f"  Δ(MSE) CCDR - best:    {delta_ccdr:.4f}")

    # Step 5: verdict
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    if sigma_milgrom < 0.10:
        print(f"✓ Best fit a0 = {best_a0:.2e} is within 10% of Milgrom's value.")
        print(f"  The script can recover the published result.")
        print(f"  CCDR's prediction (6.8e-10) is FAR from the data.")
        verdict = "MILGROM CONFIRMED, CCDR EXCLUDED at z=0"
    elif sigma_ccdr < 0.10:
        print(f"✓ Best fit a0 = {best_a0:.2e} is within 10% of the CCDR value.")
        verdict = "CCDR CONFIRMED at z=0"
    else:
        print(f"✗ Best fit a0 = {best_a0:.2e} matches neither.")
        print(f"  Possible causes: column extraction error, wrong interpolation")
        print(f"  function, or systematic in the data.")
        verdict = "INCONCLUSIVE — investigate data extraction"
    print(f"\nVerdict: {verdict}")

    # Note about z dependence
    print(f"\nNOTE: This test only constrains a0(z=0). The CCDR prediction")
    print(f"a0(z) = c·H(z) requires high-z rotation curves to test.")
    print(f"At z=0, CCDR predicts 6.8e-10 which is grossly inconsistent")
    print(f"with SPARC if Milgrom's value is correct.")

    out = {
        "data_url": URL_PRIMARY,
        "n_points": int(len(g_bar)),
        "best_fit_a0": float(best_a0),
        "best_mse_log10": float(best_mse),
        "hit_boundary": bool(hit_boundary),
        "predictions": {
            "milgrom_a0": A0_MILGROM,
            "ccdr_a0_z0": A0_CCDR,
            "ratio_ccdr_to_milgrom": A0_CCDR / A0_MILGROM,
        },
        "comparison": {
            "frac_dev_from_milgrom": float(sigma_milgrom),
            "frac_dev_from_ccdr": float(sigma_ccdr),
            "mse_at_milgrom": float(chi2_at_milgrom),
            "mse_at_ccdr": float(chi2_at_ccdr),
            "mse_at_best": float(chi2_at_best),
        },
        "verdict": verdict,
    }
    out_path = DATA_DIR / "result_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # RAR data + fit
        sort_idx = np.argsort(g_bar)
        axes[0].loglog(g_bar, g_obs, ".", markersize=2, alpha=0.5,
                      label="SPARC data")
        g_smooth = np.logspace(np.log10(g_bar.min()),
                               np.log10(g_bar.max()), 200)
        axes[0].loglog(g_smooth, mond_simple_interp(g_smooth, best_a0),
                      "g-", lw=2, label=f"Fit: a0={best_a0:.2e}")
        axes[0].loglog(g_smooth, mond_simple_interp(g_smooth, A0_MILGROM),
                      "b--", lw=1.5, label=f"Milgrom: a0={A0_MILGROM:.2e}")
        axes[0].loglog(g_smooth, mond_simple_interp(g_smooth, A0_CCDR),
                      "r:", lw=1.5, label=f"CCDR: a0={A0_CCDR:.2e}")
        axes[0].set_xlabel("g_bar (m/s²)")
        axes[0].set_ylabel("g_obs (m/s²)")
        axes[0].set_title("SPARC Radial Acceleration Relation")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, which="both")

        # Chi² scan
        a0_scan = np.logspace(-12, -8, 200)
        mse_scan = [chi2_log(a, g_bar, g_obs) for a in a0_scan]
        axes[1].loglog(a0_scan, mse_scan)
        axes[1].axvline(A0_MILGROM, color="blue", ls="--", label="Milgrom")
        axes[1].axvline(A0_CCDR, color="red", ls=":", label="CCDR")
        axes[1].axvline(best_a0, color="green", ls="-",
                       label=f"Best fit")
        axes[1].set_xlabel("a0 (m/s²)")
        axes[1].set_ylabel("MSE (log space)")
        axes[1].set_title("Chi-squared scan")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig("P_MOND_2_v2.png", dpi=150)
        print("Plot: P_MOND_2_v2.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
```

## Why v1 Failed

The v1 result was a₀ = 3.000000000000001 × 10⁻¹⁰ — that 1 at the end of
14 zeros is the floating-point signature of an optimisation hitting a
parameter boundary. The bounds were probably (1e-11, 3e-10) or similar,
forcing the fit to stop at the upper edge instead of finding a real minimum.

This v2:

1. Uses **wide bounds (1e-12 to 1e-8)** so neither prediction sits near
   an edge.
2. **Checks for boundary hits** explicitly and aborts if the fit pegged.
3. **Reports the chi-squared at both predicted values** so you can see
   which one the data prefers.
4. **Sanity-checks against Milgrom's published 1.2e-10** — if it can
   recover that, the column extraction and interpolation are correct.

## Expected Result

The literature is clear: SPARC fits Milgrom's a₀ = 1.20 × 10⁻¹⁰ at high
significance. CCDR's a₀ = 6.8 × 10⁻¹⁰ is nearly 6× too large. **At z=0,
the CCDR a₀(z) prediction is excluded by SPARC.**

The interesting CCDR test would need high-z rotation curves where
H(z) > H₀ to test whether a₀ scales with H. SPARC alone cannot test
this because all SPARC galaxies are local (z ≈ 0).

**Honest interpretation:** if this script confirms a₀ ≈ 1.2 × 10⁻¹⁰
at z=0, the CCDR prediction a₀(0) = c·H₀ is falsified at z=0. The
synthesis must either drop this prediction or argue that the relevant
a₀ at the galactic scale differs from c·H₀ by some calculable factor.

## Timeline: ~2 minutes
