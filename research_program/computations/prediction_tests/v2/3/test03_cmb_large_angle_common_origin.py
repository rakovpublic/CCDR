#!/usr/bin/env python3
"""
Self-contained implementation of:
    Test 03 — CMB Large-Angle Anomaly Common-Origin Test

Design goal:
- No input FITS files are passed on the command line.
- Public Planck products are downloaded by the script and cached locally.
- The script writes result.json and a diagnostic plot.

What this script does
---------------------
1. Downloads public Planck CMB products.
2. Builds a working large-angle temperature map.
3. Extracts four direction proxies:
   - quadrupole preferred axis (l=2)
   - octupole preferred axis (l=3)
   - hemispherical asymmetry axis
   - cold spot direction
4. Fits a common axis and evaluates its Monte Carlo p-value.
5. Saves JSON output and a diagnostic figure.

Important methodological note
-----------------------------
The original test note asks for Copi–Huterer–Schwarz multipole vectors.
To keep this implementation self-contained and dependency-light, this script
uses the angular-momentum-dispersion / power-tensor preferred-axis proxy for
l=2 and l=3. That is a closely related and commonly used low-l alignment
estimator, but it is not an exact multipole-vector reproduction.

Modes
-----
--source pr3_smica
    Fastest practical default. Downloads the public PR3 SMICA CMB map and the
    public common intensity mask.

--source pr4_143
    Downloads the public PR4 / NPIPE 143 GHz full-mission map and uses it as a
    large-angle temperature tracer, together with the common intensity mask.

--source pr4_ilc
    Downloads public PR4 / NPIPE full-mission frequency maps (70, 100, 143,
    217 GHz), downgrades them to the requested analysis NSIDE, and constructs
    a simple global pixel-space ILC CMB estimate. This is the closest option
    here to a component-separated PR4 workflow, but it is much heavier because
    the public frequency files are multi-GB.

Dependencies
------------
    pip install numpy matplotlib healpy

Example
-------
    python test03_cmb_large_angle_common_origin.py
    python test03_cmb_large_angle_common_origin.py --source pr4_ilc --nside 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import healpy as hp
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "healpy is required. Install it with: pip install healpy"
    ) from exc


# -----------------------------
# Public data URLs
# -----------------------------
PR3_SMICA_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/"
    "maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits"
)
PR3_COMMON_MASK_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/masks/"
    "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
)

PR4_FREQ_URLS = {
    "070": (
        "https://portal.nersc.gov/cfs/cmb/planck2020/frequency_maps/"
        "Single-frequency/LFI_SkyMap_070_1024_R4.00_full.fits"
    ),
    "100": (
        "https://portal.nersc.gov/cfs/cmb/planck2020/frequency_maps/"
        "Single-frequency/HFI_SkyMap_100_2048_R4.00_full.fits"
    ),
    "143": (
        "https://portal.nersc.gov/cfs/cmb/planck2020/frequency_maps/"
        "Single-frequency/HFI_SkyMap_143_2048_R4.00_full.fits"
    ),
    "217": (
        "https://portal.nersc.gov/cfs/cmb/planck2020/frequency_maps/"
        "Single-frequency/HFI_SkyMap_217_2048_R4.00_full.fits"
    ),
}

# Frequently quoted cold-spot center in Galactic coordinates.
COLD_SPOT_LON_DEG = 209.0
COLD_SPOT_LAT_DEG = -57.0


@dataclass
class DirectionResult:
    name: str
    vector: np.ndarray

    @property
    def lon_lat_deg(self) -> Tuple[float, float]:
        return vec_to_lonlat_deg(self.vector)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["pr3_smica", "pr4_143", "pr4_ilc"],
        default="pr3_smica",
        help="Public data source / map construction mode.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache_test03"),
        help="Directory used to cache downloaded public data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./test03_output"),
        help="Directory where result.json and plots are written.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=64,
        help="Analysis NSIDE after downgrading maps. Use 16, 32, or 64.",
    )
    parser.add_argument(
        "--lmax-clean",
        type=int,
        default=16,
        help="Maximum multipole retained when smoothing large-angle map.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.9,
        help="Threshold applied after downgrading the common mask.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=100_000,
        help="Number of isotropic Monte Carlo realizations.",
    )
    parser.add_argument(
        "--common-angle-threshold-deg",
        type=float,
        default=30.0,
        help="Angular threshold used for the 30-degree common-axis check.",
    )
    parser.add_argument(
        "--hemisphere-scan-nside",
        type=int,
        default=16,
        help="Healpix NSIDE for hemisphere-axis scan candidates.",
    )
    parser.add_argument(
        "--pr4-ilc-channels",
        default="070,100,143,217",
        help="Comma-separated PR4 channels used in pr4_ilc mode.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_if_needed(url: str, dst: Path) -> Path:
    """Download a public file once and reuse it from cache."""
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[download] {url}")
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dst


def unit_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return vec / norm


def lonlat_deg_to_vec(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z], dtype=float)


def vec_to_lonlat_deg(vec: np.ndarray) -> Tuple[float, float]:
    x, y, z = unit_vector(vec)
    lon = np.rad2deg(np.arctan2(y, x)) % 360.0
    lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    return float(lon), float(lat)


def downgrade_mask(mask_path: Path, nside_out: int, threshold: float) -> np.ndarray:
    mask = hp.read_map(str(mask_path), field=0, dtype=np.float64, memmap=True, verbose=False)
    downgraded = hp.ud_grade(mask.astype(np.float64), nside_out=nside_out)
    good = np.isfinite(downgraded) & (downgraded >= threshold)
    return good


def sanitize_map(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    bad = ~np.isfinite(m)
    bad |= np.abs(m) > 1e29
    out = m.copy()
    out[bad] = 0.0
    return out


def smooth_large_angle_map(m: np.ndarray, mask: np.ndarray, lmax_clean: int) -> np.ndarray:
    temp = np.where(mask, m, 0.0)
    temp = hp.remove_dipole(temp, fitval=False, verbose=False)
    alm = hp.map2alm(temp, lmax=max(lmax_clean, 3), iter=3)
    cleaned = hp.alm2map(alm, nside=hp.get_nside(temp), verbose=False)
    cleaned = hp.remove_dipole(cleaned, fitval=False, verbose=False)
    cleaned[~mask] = 0.0
    return cleaned


def load_pr3_smica(cache_dir: Path, nside_out: int, lmax_clean: int, mask_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    smica_path = download_if_needed(PR3_SMICA_URL, cache_dir / Path(PR3_SMICA_URL).name)
    mask_path = download_if_needed(PR3_COMMON_MASK_URL, cache_dir / Path(PR3_COMMON_MASK_URL).name)

    m = hp.read_map(str(smica_path), field=0, dtype=np.float64, memmap=True, verbose=False)
    m = sanitize_map(m)
    m = hp.ud_grade(m, nside_out=nside_out)
    mask = downgrade_mask(mask_path, nside_out=nside_out, threshold=mask_threshold)
    return smooth_large_angle_map(m, mask, lmax_clean=lmax_clean), mask


def load_pr4_single_frequency(
    cache_dir: Path,
    channel: str,
    nside_out: int,
    lmax_clean: int,
    mask_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if channel not in PR4_FREQ_URLS:
        raise ValueError(f"Unsupported PR4 channel: {channel}")

    map_path = download_if_needed(PR4_FREQ_URLS[channel], cache_dir / Path(PR4_FREQ_URLS[channel]).name)
    mask_path = download_if_needed(PR3_COMMON_MASK_URL, cache_dir / Path(PR3_COMMON_MASK_URL).name)

    m = hp.read_map(str(map_path), field=0, dtype=np.float64, memmap=True, verbose=False)
    m = sanitize_map(m)
    m = hp.ud_grade(m, nside_out=nside_out)
    mask = downgrade_mask(mask_path, nside_out=nside_out, threshold=mask_threshold)
    return smooth_large_angle_map(m, mask, lmax_clean=lmax_clean), mask


def global_ilc(stack: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple pixel-space ILC weights: w = C^-1 1 / (1^T C^-1 1)."""
    if stack.ndim != 2:
        raise ValueError("stack must have shape (nchan, npix)")

    data = stack[:, valid]
    data = data - data.mean(axis=1, keepdims=True)
    cov = np.cov(data)
    ones = np.ones(cov.shape[0], dtype=np.float64)
    inv_cov = np.linalg.pinv(cov)
    weights = inv_cov @ ones
    denom = float(ones @ weights)
    if abs(denom) < 1e-20:
        raise RuntimeError("ILC denominator is numerically zero.")
    weights = weights / denom
    cmb = weights @ stack
    return cmb, weights


def load_pr4_ilc(
    cache_dir: Path,
    channels: Iterable[str],
    nside_out: int,
    lmax_clean: int,
    mask_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    mask_path = download_if_needed(PR3_COMMON_MASK_URL, cache_dir / Path(PR3_COMMON_MASK_URL).name)
    mask = downgrade_mask(mask_path, nside_out=nside_out, threshold=mask_threshold)

    maps: List[np.ndarray] = []
    channel_list = list(channels)
    for ch in channel_list:
        if ch not in PR4_FREQ_URLS:
            raise ValueError(f"Unsupported PR4 channel in ILC: {ch}")
        path = download_if_needed(PR4_FREQ_URLS[ch], cache_dir / Path(PR4_FREQ_URLS[ch]).name)
        m = hp.read_map(str(path), field=0, dtype=np.float64, memmap=True, verbose=False)
        m = sanitize_map(m)
        m = hp.ud_grade(m, nside_out=nside_out)
        maps.append(m)

    stack = np.vstack(maps)
    valid = mask & np.all(np.isfinite(stack), axis=0)
    cmb, weights = global_ilc(stack, valid)
    cmb = smooth_large_angle_map(cmb, mask, lmax_clean=lmax_clean)
    weight_dict = {ch: float(w) for ch, w in zip(channel_list, weights)}
    return cmb, mask, weight_dict


def alm_vector_for_l(alm: np.ndarray, ell: int) -> np.ndarray:
    """Return a_{ell m} for m=-ell..ell in the standard complex Y_lm basis."""
    coeffs: List[complex] = []
    for m in range(-ell, ell + 1):
        if m < 0:
            ap = alm[hp.Alm.getidx(ell, ell, -m)]
            coeffs.append(((-1) ** (-m)) * np.conj(ap))
        else:
            coeffs.append(alm[hp.Alm.getidx(ell, ell, m)])
    return np.asarray(coeffs, dtype=np.complex128)


def angular_momentum_matrices(ell: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mvals = np.arange(-ell, ell + 1, dtype=np.float64)
    dim = 2 * ell + 1
    jp = np.zeros((dim, dim), dtype=np.complex128)
    jm = np.zeros((dim, dim), dtype=np.complex128)

    for i, m in enumerate(mvals[:-1]):
        coeff = math.sqrt(ell * (ell + 1) - m * (m + 1))
        jp[i + 1, i] = coeff
    for i, m in enumerate(mvals[1:], start=1):
        coeff = math.sqrt(ell * (ell + 1) - m * (m - 1))
        jm[i - 1, i] = coeff

    jx = 0.5 * (jp + jm)
    jy = -0.5j * (jp - jm)
    jz = np.diag(mvals.astype(np.complex128))
    return jx, jy, jz


def preferred_axis_power_tensor(m: np.ndarray, ell: int) -> np.ndarray:
    lmax = max(ell, 3)
    alm = hp.map2alm(m, lmax=lmax, iter=3)
    a = alm_vector_for_l(alm, ell)
    jx, jy, jz = angular_momentum_matrices(ell)
    js = [jx, jy, jz]

    tensor = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            val = np.vdot(a, js[i] @ (js[j] @ a))
            tensor[i, j] = np.real(val)

    tensor = 0.5 * (tensor + tensor.T)
    evals, evecs = np.linalg.eigh(tensor)
    axis = evecs[:, np.argmax(evals)]
    return unit_vector(axis)


def hemispherical_asymmetry_axis(
    m: np.ndarray,
    mask: np.ndarray,
    scan_nside: int,
) -> Tuple[np.ndarray, float]:
    pixvec = np.vstack(hp.pix2vec(hp.get_nside(m), np.arange(m.size))).T
    axes = np.vstack(hp.pix2vec(scan_nside, np.arange(hp.nside2npix(scan_nside)))).T
    valid = mask & np.isfinite(m)
    vals = m[valid]
    vecs = pixvec[valid]

    best_score = -np.inf
    best_axis = None

    for axis in axes:
        dots = vecs @ axis
        north = vals[dots >= 0.0]
        south = vals[dots < 0.0]
        if north.size < 16 or south.size < 16:
            continue
        var_n = float(np.var(north))
        var_s = float(np.var(south))
        denom = var_n + var_s
        if denom <= 0:
            continue
        score = abs(var_n - var_s) / denom
        signed_axis = axis if var_n >= var_s else -axis
        if score > best_score:
            best_score = score
            best_axis = signed_axis

    if best_axis is None:
        raise RuntimeError("Failed to determine hemispherical asymmetry axis.")
    return unit_vector(best_axis), float(best_score)


def acute_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.clip(abs(float(np.dot(unit_vector(a), unit_vector(b)))), 0.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def pairwise_acute_angles(directions: List[DirectionResult]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            name = f"{directions[i].name}__{directions[j].name}"
            out[name] = acute_angle_deg(directions[i].vector, directions[j].vector)
    return out


def fit_common_axis(vectors: np.ndarray) -> np.ndarray:
    scatter = np.einsum("ni,nj->ij", vectors, vectors)
    evals, evecs = np.linalg.eigh(scatter)
    axis = evecs[:, np.argmax(evals)]
    return unit_vector(axis)


def common_axis_statistic_deg(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis = fit_common_axis(vectors)
    dots = np.clip(np.abs(vectors @ axis), 0.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    return axis, angles


def monte_carlo_common_axis(
    n_samples: int,
    n_dirs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dirs = rng.normal(size=(n_samples, n_dirs, 3))
    norms = np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs = dirs / norms

    mats = np.einsum("nki,nkj->nij", dirs, dirs)
    _, evecs = np.linalg.eigh(mats)
    axes = evecs[:, :, -1]
    dots = np.clip(np.abs(np.einsum("nki,ni->nk", dirs, axes)), 0.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    return angles.max(axis=1)


def wrap_mollweide_lon(lon_deg: float) -> float:
    lon = ((lon_deg + 180.0) % 360.0) - 180.0
    return -np.deg2rad(lon)


def plot_results(
    directions: List[DirectionResult],
    common_axis: np.ndarray,
    observed_angles_deg: np.ndarray,
    null_stats_deg: np.ndarray,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    fig = plt.figure(figsize=(13, 5))

    ax0 = fig.add_subplot(1, 2, 1, projection="mollweide")
    for d in directions:
        lon, lat = d.lon_lat_deg
        ax0.scatter(wrap_mollweide_lon(lon), np.deg2rad(lat), s=50, label=d.name)
        ax0.text(wrap_mollweide_lon(lon), np.deg2rad(lat), f" {d.name}", fontsize=8)

    common_lon, common_lat = vec_to_lonlat_deg(common_axis)
    anti_lon, anti_lat = vec_to_lonlat_deg(-common_axis)
    ax0.scatter(wrap_mollweide_lon(common_lon), np.deg2rad(common_lat), marker="x", s=100, label="common_axis")
    ax0.scatter(wrap_mollweide_lon(anti_lon), np.deg2rad(anti_lat), marker="x", s=100)
    ax0.grid(True, alpha=0.4)
    ax0.set_title("Anomaly directions and fitted common axis")
    ax0.legend(loc="lower left", fontsize=8)

    ax1 = fig.add_subplot(1, 2, 2)
    observed_stat = float(np.max(observed_angles_deg))
    ax1.hist(null_stats_deg, bins=60)
    ax1.axvline(observed_stat)
    ax1.set_xlabel("Max acute angle to best-fit axis [deg]")
    ax1.set_ylabel("Monte Carlo count")
    ax1.set_title("Isotropic null distribution")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.cache_dir)
    ensure_dir(args.output_dir)

    rng = np.random.default_rng(12345)

    ilc_weights: Dict[str, float] | None = None
    if args.source == "pr3_smica":
        cmb_map, mask = load_pr3_smica(
            cache_dir=args.cache_dir,
            nside_out=args.nside,
            lmax_clean=args.lmax_clean,
            mask_threshold=args.mask_threshold,
        )
    elif args.source == "pr4_143":
        cmb_map, mask = load_pr4_single_frequency(
            cache_dir=args.cache_dir,
            channel="143",
            nside_out=args.nside,
            lmax_clean=args.lmax_clean,
            mask_threshold=args.mask_threshold,
        )
    elif args.source == "pr4_ilc":
        channels = [part.strip() for part in args.pr4_ilc_channels.split(",") if part.strip()]
        cmb_map, mask, ilc_weights = load_pr4_ilc(
            cache_dir=args.cache_dir,
            channels=channels,
            nside_out=args.nside,
            lmax_clean=args.lmax_clean,
            mask_threshold=args.mask_threshold,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported source: {args.source}")

    quadrupole_axis = preferred_axis_power_tensor(cmb_map, ell=2)
    octupole_axis = preferred_axis_power_tensor(cmb_map, ell=3)
    hemi_axis, hemi_score = hemispherical_asymmetry_axis(
        cmb_map,
        mask,
        scan_nside=args.hemisphere_scan_nside,
    )
    cold_spot_axis = lonlat_deg_to_vec(COLD_SPOT_LON_DEG, COLD_SPOT_LAT_DEG)

    directions = [
        DirectionResult("quadrupole", quadrupole_axis),
        DirectionResult("octupole", octupole_axis),
        DirectionResult("hemis_asym", hemi_axis),
        DirectionResult("cold_spot", cold_spot_axis),
    ]

    vectors = np.vstack([d.vector for d in directions])
    common_axis, observed_angles_deg = common_axis_statistic_deg(vectors)
    observed_stat = float(np.max(observed_angles_deg))

    null_stats_deg = monte_carlo_common_axis(
        n_samples=args.mc_samples,
        n_dirs=len(directions),
        rng=rng,
    )
    common_axis_pvalue = float(np.mean(null_stats_deg <= observed_stat))
    threshold_pvalue = float(np.mean(null_stats_deg <= args.common_angle_threshold_deg))
    within_threshold = bool(observed_stat <= args.common_angle_threshold_deg)

    common_lon, common_lat = vec_to_lonlat_deg(common_axis)
    pairwise = pairwise_acute_angles(directions)

    result = {
        "source": args.source,
        "analysis_nside": args.nside,
        "lmax_clean": args.lmax_clean,
        "mask_threshold": args.mask_threshold,
        "mc_samples": args.mc_samples,
        "directions": {
            d.name: {
                "vector": [float(x) for x in d.vector],
                "galactic_lon_lat_deg": [float(d.lon_lat_deg[0]), float(d.lon_lat_deg[1])],
            }
            for d in directions
        },
        "pairwise_angles_deg": pairwise,
        "common_axis": {
            "vector": [float(x) for x in common_axis],
            "galactic_lon_lat_deg": [float(common_lon), float(common_lat)],
            "angles_to_common_axis_deg": {
                d.name: float(a) for d, a in zip(directions, observed_angles_deg)
            },
            "max_angle_deg": observed_stat,
            "threshold_deg": float(args.common_angle_threshold_deg),
            "within_threshold": within_threshold,
        },
        "common_axis_pvalue": common_axis_pvalue,
        "pvalue_for_30deg_event_under_isotropy": threshold_pvalue,
        "pass_p05": bool(common_axis_pvalue < 0.05 and within_threshold),
        "pass_p01": bool(common_axis_pvalue < 0.01 and within_threshold),
        "hemispherical_asymmetry_score": hemi_score,
    }
    if ilc_weights is not None:
        result["pr4_ilc_weights"] = ilc_weights

    result_path = args.output_dir / "result.json"
    plot_path = args.output_dir / "anomaly_common_axis.png"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    plot_results(
        directions=directions,
        common_axis=common_axis,
        observed_angles_deg=observed_angles_deg,
        null_stats_deg=null_stats_deg,
        output_path=plot_path,
    )

    print(f"Saved: {result_path}")
    print(f"Saved: {plot_path}")
    print(f"common_axis_pvalue = {common_axis_pvalue:.6g}")
    print(f"max_angle_deg = {observed_stat:.3f}")
    if ilc_weights is not None:
        print("ILC weights:")
        for ch, w in ilc_weights.items():
            print(f"  {ch}: {w:+.6f}")


if __name__ == "__main__":
    main()
