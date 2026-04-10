#!/usr/bin/env python3
"""
P12_frb_triple_junction_test_v2.py

Tests whether FRB sky positions show angular clustering relative to a
random null hypothesis. This is the implementable v2 version of the P12
idea from the markdown spec: before attempting a full FRB × cosmic-web
triple-junction cross-correlation, first check whether the FRB sky
positions cluster on the sky at all.

Practical fixes relative to the markdown draft:
  - Prefers the live Vizier table2.dat endpoint because the published
    CANFAR direct CSV path in the note currently returns 404.
  - Parses the CHIME catalog with explicit fixed-width column handling
    using the Vizier byte layout.
  - Uses unique sky positions by default, so repeat bursts from the same
    source do not inflate the clustering signal.
  - Computes the Landy-Szalay estimator with chunked pair counting to
    avoid large in-memory distance matrices.

Outputs:
  data/p12_frb_junctions/result_v2.json
  data/p12_frb_junctions/P12_frb_clustering.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path("data/p12_frb_junctions")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHIME_TABLE = DATA_DIR / "chimefrbcat1_table2.dat"

# The markdown spec points to a CANFAR CSV URL, but that path currently 404s.
# The Vizier table2.dat endpoint is live and documented in the catalog ReadMe.
CHIME_URLS = [
    "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/257/59/table2.dat",
    "http://cdsarc.cds.unistra.fr/ftp/J/ApJS/257/59/table2.dat",
]

# Explicit fixed-width colspecs derived from the Vizier ReadMe byte layout.
# Pandas uses 0-based, half-open intervals.
COLSPECS = [
    (0, 12),    # Name
    (13, 28),   # OName
    (29, 41),   # RpName
    (42, 53),   # RAdeg
    (54, 60),   # e_RAdeg
    (61, 62),   # n_RAdeg
    (63, 73),   # DEdeg
    (74, 79),   # e_DEdeg
    (80, 81),   # n_DEdeg
    (82, 88),   # GLON
    (89, 95),   # GLAT
]
COLUMN_NAMES = [
    "Name",
    "OName",
    "RpName",
    "RAdeg",
    "e_RAdeg",
    "n_RAdeg",
    "DEdeg",
    "e_DEdeg",
    "n_DEdeg",
    "GLON",
    "GLAT",
]


def download_file(urls: Iterable[str], dest: Path) -> Path:
    """Download the first working catalog source."""
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[cache] {dest}")
        return dest

    last_error: Exception | None = None
    for url in urls:
        try:
            print(f"[download] {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (P12-FRB-test)"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if not data or len(data) < 1000:
                raise RuntimeError("downloaded file is unexpectedly small")
            head = data[:200].decode("utf-8", errors="ignore").lower()
            if "<html" in head or "<!doctype" in head:
                raise RuntimeError("downloaded HTML instead of catalog data")
            dest.write_bytes(data)
            print(f"  [saved] {dest} ({len(data) / 1e3:.1f} KB)")
            return dest
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  [fail] {exc}")
            continue

    raise RuntimeError(f"Could not download CHIME/FRB catalog: {last_error}")


def load_frb_catalog(path: Path, deduplicate_positions: bool = True) -> pd.DataFrame:
    """Load the Vizier fixed-width CHIME/FRB table and return clean coordinates."""
    df = pd.read_fwf(
        path,
        colspecs=COLSPECS,
        names=COLUMN_NAMES,
        comment="#",
        dtype=str,
    )

    # Convert only the coordinate fields we need for the angular test.
    for col in ("RAdeg", "DEdeg", "e_RAdeg", "e_DEdeg", "GLON", "GLAT"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = (
        df["RAdeg"].notna()
        & df["DEdeg"].notna()
        & (df["RAdeg"] >= 0.0)
        & (df["RAdeg"] <= 360.0)
        & (df["DEdeg"] >= -90.0)
        & (df["DEdeg"] <= 90.0)
    )
    df = df.loc[mask].copy()

    # Remove placeholder entries such as -9999 and repeated bursts at the same sky position.
    df = df[(df["RAdeg"] > -1000) & (df["DEdeg"] > -1000)].copy()

    if deduplicate_positions:
        before = len(df)
        df = df.drop_duplicates(subset=["RAdeg", "DEdeg"]).copy()
        print(f"[load] {len(df)} unique FRB sky positions (from {before} catalog rows)")
    else:
        print(f"[load] {len(df)} FRB catalog rows")

    return df.reset_index(drop=True)


def spherical_to_xyz(coords_deg: np.ndarray) -> np.ndarray:
    """Convert RA/Dec in degrees to 3D unit vectors."""
    ra = np.deg2rad(coords_deg[:, 0])
    dec = np.deg2rad(coords_deg[:, 1])
    cos_dec = np.cos(dec)
    x = cos_dec * np.cos(ra)
    y = cos_dec * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack((x, y, z))


def generate_random_catalog(
    n_random: int,
    dec_min_deg: float,
    dec_max_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a simple random catalog in the CHIME-accessible sky region.

    This is intentionally simple because P12-v2 is only a precondition test.
    It does not model CHIME exposure in detail.
    """
    rand_ra = rng.uniform(0.0, 360.0, size=n_random)
    sin_min = math.sin(math.radians(dec_min_deg))
    sin_max = math.sin(math.radians(dec_max_deg))
    rand_sin_dec = rng.uniform(sin_min, sin_max, size=n_random)
    rand_dec = np.degrees(np.arcsin(rand_sin_dec))
    return np.column_stack((rand_ra, rand_dec))


def accumulate_hist_same(
    xyz: np.ndarray,
    bins_rad: np.ndarray,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, int]:
    """Histogram unique pair separations within one catalog."""
    n = len(xyz)
    hist = np.zeros(len(bins_rad) - 1, dtype=np.int64)
    n_pairs = 0

    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        a = xyz[i0:i1]
        dots = np.clip(a @ xyz.T, -1.0, 1.0)
        seps = np.arccos(dots)

        row_idx = np.arange(i0, i1)[:, None]
        col_idx = np.arange(n)[None, :]
        keep = col_idx > row_idx
        vals = seps[keep]
        if vals.size:
            hist += np.histogram(vals, bins=bins_rad)[0]
            n_pairs += int(vals.size)

    return hist, n_pairs


def accumulate_hist_cross(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    bins_rad: np.ndarray,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, int]:
    """Histogram pair separations between two catalogs."""
    hist = np.zeros(len(bins_rad) - 1, dtype=np.int64)
    n_pairs = 0

    for i0 in range(0, len(xyz_a), chunk_size):
        i1 = min(i0 + chunk_size, len(xyz_a))
        a = xyz_a[i0:i1]
        dots = np.clip(a @ xyz_b.T, -1.0, 1.0)
        vals = np.arccos(dots).ravel()
        if vals.size:
            hist += np.histogram(vals, bins=bins_rad)[0]
            n_pairs += int(vals.size)

    return hist, n_pairs


def angular_correlation_landy_szalay(
    coords_deg: np.ndarray,
    n_bins: int = 15,
    max_sep_deg: float = 20.0,
    n_random: int | None = None,
    dec_min_deg: float = -11.0,
    dec_max_deg: float = 90.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute w(theta) with the Landy-Szalay estimator."""
    if len(coords_deg) < 4:
        raise RuntimeError("Need at least 4 FRB positions for an angular correlation estimate.")

    rng = np.random.default_rng(seed)
    if n_random is None:
        n_random = max(5000, 10 * len(coords_deg))

    random_coords = generate_random_catalog(
        n_random=n_random,
        dec_min_deg=dec_min_deg,
        dec_max_deg=dec_max_deg,
        rng=rng,
    )

    data_xyz = spherical_to_xyz(coords_deg)
    rand_xyz = spherical_to_xyz(random_coords)

    bins_deg = np.linspace(0.0, max_sep_deg, n_bins + 1)
    bins_rad = np.deg2rad(bins_deg)

    print(f"[corr] DD with {len(coords_deg)} data points")
    dd_hist, n_dd = accumulate_hist_same(data_xyz, bins_rad)
    print(f"[corr] DR with {len(coords_deg)} × {len(random_coords)} pairs")
    dr_hist, n_dr = accumulate_hist_cross(data_xyz, rand_xyz, bins_rad)
    print(f"[corr] RR with {len(random_coords)} random points")
    rr_hist, n_rr = accumulate_hist_same(rand_xyz, bins_rad)

    dd = dd_hist / max(n_dd, 1)
    dr = dr_hist / max(n_dr, 1)
    rr = rr_hist / max(n_rr, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        w_theta = (dd - 2.0 * dr + rr) / rr
        w_theta[~np.isfinite(w_theta)] = 0.0

    theta_deg = 0.5 * (bins_deg[:-1] + bins_deg[1:])
    meta = {
        "n_data": int(len(coords_deg)),
        "n_random": int(len(random_coords)),
        "n_DD_pairs": int(n_dd),
        "n_DR_pairs": int(n_dr),
        "n_RR_pairs": int(n_rr),
        "bins_deg": bins_deg.tolist(),
        "dec_range_deg": [float(dec_min_deg), float(dec_max_deg)],
    }
    return theta_deg, w_theta, meta


def make_plot(coords: np.ndarray, theta: np.ndarray, w_theta: np.ndarray, out_path: Path) -> None:
    """Write a simple diagnostic figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.7)
    axes[0].set_xlabel("RA (deg)")
    axes[0].set_ylabel("Dec (deg)")
    axes[0].set_title(f"FRB sky positions ({len(coords)})")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(theta, w_theta, marker="o")
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Angular separation θ (deg)")
    axes[1].set_ylabel("w(θ)")
    axes[1].set_title("Angular correlation function")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_verdict(theta: np.ndarray, w_theta: np.ndarray, small_scale_deg: float) -> tuple[str, float, float]:
    """Return a simple textual verdict consistent with the markdown spec."""
    small_mask = theta < small_scale_deg
    if not np.any(small_mask):
        raise RuntimeError("No bins fall below the chosen small-scale threshold.")

    mean_small = float(np.mean(w_theta[small_mask]))
    mean_large = float(np.mean(w_theta[~small_mask])) if np.any(~small_mask) else 0.0

    if mean_small > 0.2:
        verdict = "CLUSTERING DETECTED at small scales"
    elif mean_small > 0.05:
        verdict = "WEAK CLUSTERING HINT"
    else:
        verdict = "NO CLUSTERING (consistent with random)"
    return verdict, mean_small, mean_large


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P12 v2: FRB angular clustering test against a random sky null."
    )
    parser.add_argument("--n-bins", type=int, default=15, help="Number of angular bins.")
    parser.add_argument(
        "--max-sep-deg",
        type=float,
        default=20.0,
        help="Maximum angular separation included in w(theta).",
    )
    parser.add_argument(
        "--small-scale-deg",
        type=float,
        default=5.0,
        help="Threshold defining the small-scale regime for the simple verdict.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=None,
        help="Size of the random catalog. Default: max(5000, 10 * N_FRB).",
    )
    parser.add_argument(
        "--dec-min",
        type=float,
        default=-11.0,
        help="Lower declination bound for the random null catalog.",
    )
    parser.add_argument(
        "--dec-max",
        type=float,
        default=90.0,
        help="Upper declination bound for the random null catalog.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the null catalog.",
    )
    parser.add_argument(
        "--keep-duplicate-bursts",
        action="store_true",
        help="Use all catalog rows instead of unique FRB positions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    print("=" * 70)
    print("P12: FRB Spatial Clustering vs Random Null")
    print("=" * 70)

    table_path = download_file(CHIME_URLS, CHIME_TABLE)
    frb_df = load_frb_catalog(table_path, deduplicate_positions=not args.keep_duplicate_bursts)

    coords = frb_df[["RAdeg", "DEdeg"]].to_numpy(dtype=float)
    if len(coords) < 20:
        print(f"[warning] only {len(coords)} FRB positions available; statistics will be weak")

    theta, w_theta, meta = angular_correlation_landy_szalay(
        coords_deg=coords,
        n_bins=args.n_bins,
        max_sep_deg=args.max_sep_deg,
        n_random=args.n_random,
        dec_min_deg=args.dec_min,
        dec_max_deg=args.dec_max,
        seed=args.seed,
    )

    verdict, mean_small, mean_large = build_verdict(theta, w_theta, args.small_scale_deg)

    print(f"\n{'=' * 70}")
    print("ANGULAR CORRELATION FUNCTION w(theta)")
    print(f"{'=' * 70}")
    print(f"{'theta (deg)':>12} {'w(theta)':>12} {'interpretation':>20}")
    for t, wi in zip(theta, w_theta):
        label = "cluster" if wi > 0.1 else "neutral" if wi > -0.1 else "anti-cluster"
        print(f"{t:12.2f} {wi:12.4f} {label:>20}")

    print(f"\nMean w(theta) for theta < {args.small_scale_deg:g} deg:  {mean_small:+.4f}")
    print(f"Mean w(theta) for theta >= {args.small_scale_deg:g} deg: {mean_large:+.4f}")
    print(f"\nVerdict: {verdict}")

    result = {
        "catalog_source_urls": CHIME_URLS,
        "catalog_path": str(table_path),
        "n_frbs": int(len(coords)),
        "theta_deg": theta.tolist(),
        "w_theta": [float(x) for x in w_theta],
        "mean_w_small_scale": mean_small,
        "mean_w_large_scale": mean_large,
        "small_scale_threshold_deg": float(args.small_scale_deg),
        "verdict": verdict,
        "meta": meta,
    }

    out_json = DATA_DIR / "result_v2.json"
    out_png = DATA_DIR / "P12_frb_clustering.png"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_json}")

    try:
        make_plot(coords, theta, w_theta, out_png)
        print(f"Plot: {out_png}")
    except ImportError:
        print("[warning] matplotlib not installed; skipping plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
