#!/usr/bin/env python3
"""
Self-contained implementation of:
    Test 03 — CMB Large-Angle Anomaly Common-Origin Test

Design goals
------------
- No input FITS files are passed on the command line.
- Public Planck products are downloaded by the script and cached locally.
- No ``healpy`` import is used.
- The script writes ``result.json`` and a diagnostic plot.

Methodological notes
--------------------
1. HEALPix I/O and geometry are handled with ``astropy.io.fits`` and
   ``astropy-healpix`` rather than ``healpy``.
2. Instead of a full spherical-harmonic transform on the complete sky, the
   script resamples the public map to a low-resolution analysis grid and fits
   spherical harmonics up to ``lmax_clean`` by masked least squares. That is
   enough for the large-angle / low-l test here.
3. The l=2 and l=3 preferred directions are estimated with the angular-
   momentum-dispersion / power-tensor proxy, not exact Copi–Huterer–Schwarz
   multipole vectors.

Dependencies
------------
    pip install numpy scipy matplotlib astropy astropy-healpix

Examples
--------
    python test03_cmb_large_angle_common_origin_no_healpy.py
    python test03_cmb_large_angle_common_origin_no_healpy.py --source pr4_143
    python test03_cmb_large_angle_common_origin_no_healpy.py --source pr4_ilc --nside 64
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm

try:
    import astropy.units as u
    from astropy.io import fits
    from astropy_healpix import HEALPix
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This version does not use healpy, but it does require astropy and astropy-healpix. "
        "Install them with: pip install astropy astropy-healpix"
    ) from exc


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
        default=32,
        help="Analysis NSIDE after resampling. Use 16, 32, or 64.",
    )
    parser.add_argument(
        "--lmax-clean",
        type=int,
        default=16,
        help="Maximum multipole retained when smoothing / cleaning the large-angle map.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold for the resampled common mask.",
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
        default=8,
        help="HEALPix NSIDE for hemisphere-axis scan candidates.",
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


def sanitize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    bad = ~np.isfinite(values)
    bad |= np.abs(values) > 1e29
    out = values.copy()
    out[bad] = 0.0
    return out


def unit_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return vec / norm


def lonlat_deg_to_vec(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    return np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def vec_to_lonlat_deg(vec: np.ndarray) -> Tuple[float, float]:
    x, y, z = unit_vector(vec)
    lon = np.rad2deg(np.arctan2(y, x)) % 360.0
    lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    return float(lon), float(lat)


def infer_nside_from_length(npix: int) -> int:
    nside = int(round((npix / 12.0) ** 0.5))
    if 12 * nside * nside != npix:
        raise ValueError(f"Cannot infer HEALPix NSIDE from npix={npix}")
    return nside


def extract_map_column(
    path: Path,
    preferred_names: Sequence[str] | None = None,
    field_index: int | None = None,
) -> np.ndarray:
    """Read a 1D HEALPix map column from a FITS image or table."""
    preferred = [name.upper() for name in (preferred_names or [])]

    with fits.open(path, memmap=True) as hdul:
        # First search table HDUs with named columns.
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            cols = getattr(hdu, "columns", None)
            if data is None or cols is None:
                continue

            names = [name.upper() for name in cols.names]
            for wanted in preferred:
                if wanted in names:
                    arr = np.asarray(data[cols.names[names.index(wanted)]], dtype=np.float64)
                    return arr.reshape(-1)

            if field_index is not None and 0 <= field_index < len(cols.names):
                arr = np.asarray(data[cols.names[field_index]], dtype=np.float64)
                return arr.reshape(-1)

        # Fallback: first 1D image-like array.
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            arr = np.asarray(data, dtype=np.float64)
            if arr.ndim == 1:
                return arr.reshape(-1)
            if arr.ndim == 2 and 1 in arr.shape:
                return arr.reshape(-1)

    raise RuntimeError(f"Could not extract a HEALPix map column from {path}")


def healpix_geometry(nside: int) -> Tuple[HEALPix, np.ndarray, np.ndarray, np.ndarray]:
    hp = HEALPix(nside=nside, order="ring")
    idx = np.arange(hp.npix)
    lon_q, lat_q = hp.healpix_to_lonlat(idx)
    lon = lon_q.to_value(u.rad)
    lat = lat_q.to_value(u.rad)
    theta = 0.5 * np.pi - lat
    xyz = np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])
    return hp, lon, lat, xyz


def resample_ring_map(
    hi_values: np.ndarray,
    nside_out: int,
    interpolate: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hi_values = sanitize_map(hi_values)
    hi_nside = infer_nside_from_length(hi_values.size)
    hi_hp = HEALPix(nside=hi_nside, order="ring")
    _, lon, lat, xyz = healpix_geometry(nside_out)

    lon_q = lon * u.rad
    lat_q = lat * u.rad

    if interpolate:
        lo_values = np.asarray(
            hi_hp.interpolate_bilinear_lonlat(lon_q, lat_q, hi_values),
            dtype=np.float64,
        )
    else:
        idx = hi_hp.lonlat_to_healpix(lon_q, lat_q)
        lo_values = np.asarray(hi_values[idx], dtype=np.float64)

    lo_values = sanitize_map(lo_values)
    return lo_values, lon, lat, xyz


def prepare_mask(mask_values: np.ndarray, nside_out: int, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lo_mask_values, lon, lat, xyz = resample_ring_map(mask_values, nside_out=nside_out, interpolate=True)
    good = np.isfinite(lo_mask_values) & (lo_mask_values >= threshold)
    return good, lon, lat, xyz


def alm_index_list(lmax: int) -> List[Tuple[int, int]]:
    return [(ell, m) for ell in range(lmax + 1) for m in range(-ell, ell + 1)]


def spherical_harmonic_design_matrix(theta: np.ndarray, lon: np.ndarray, lmax: int) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    keys = alm_index_list(lmax)
    cols = [sph_harm(m, ell, lon, theta) for ell, m in keys]
    design = np.column_stack(cols).astype(np.complex128, copy=False)
    return keys, design


def fit_alm_masked_least_squares(
    values: np.ndarray,
    mask: np.ndarray,
    theta: np.ndarray,
    lon: np.ndarray,
    lmax: int,
) -> Dict[Tuple[int, int], complex]:
    valid = mask & np.isfinite(values)
    if valid.sum() <= (lmax + 1) ** 2:
        raise RuntimeError("Too few unmasked pixels for harmonic fit.")

    keys, design = spherical_harmonic_design_matrix(theta[valid], lon[valid], lmax)
    coeffs, *_ = np.linalg.lstsq(design, values[valid], rcond=None)
    return {key: coeff for key, coeff in zip(keys, coeffs)}


def reconstruct_from_alm(
    theta: np.ndarray,
    lon: np.ndarray,
    alm: Dict[Tuple[int, int], complex],
    ell_min: int = 0,
    ell_max: int | None = None,
) -> np.ndarray:
    if ell_max is None:
        ell_max = max(ell for ell, _ in alm)

    out = np.zeros(theta.shape, dtype=np.complex128)
    for (ell, m), coeff in alm.items():
        if ell < ell_min or ell > ell_max:
            continue
        out += coeff * sph_harm(m, ell, lon, theta)
    return np.real(out)


def smooth_large_angle_map(
    values: np.ndarray,
    mask: np.ndarray,
    theta: np.ndarray,
    lon: np.ndarray,
    lmax_clean: int,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], complex]]:
    alm = fit_alm_masked_least_squares(values, mask, theta, lon, lmax=lmax_clean)
    cleaned = reconstruct_from_alm(theta, lon, alm, ell_min=2, ell_max=lmax_clean)
    cleaned[~mask] = 0.0
    return cleaned, alm


def global_ilc(stack: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def preferred_axis_power_tensor_from_alm(alm: Dict[Tuple[int, int], complex], ell: int) -> np.ndarray:
    a = np.array([alm.get((ell, m), 0.0 + 0.0j) for m in range(-ell, ell + 1)], dtype=np.complex128)
    jx, jy, jz = angular_momentum_matrices(ell)
    js = [jx, jy, jz]

    tensor = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            tensor[i, j] = float(np.real(np.vdot(a, js[i] @ (js[j] @ a))))

    tensor = 0.5 * (tensor + tensor.T)
    evals, evecs = np.linalg.eigh(tensor)
    axis = evecs[:, np.argmax(evals)]
    return unit_vector(axis)


def hemispherical_asymmetry_axis(
    values: np.ndarray,
    mask: np.ndarray,
    pixel_xyz: np.ndarray,
    scan_nside: int,
) -> Tuple[np.ndarray, float]:
    _, _, _, axes = healpix_geometry(scan_nside)
    valid = mask & np.isfinite(values)
    vals = values[valid]
    vecs = pixel_xyz[valid]

    best_score = -np.inf
    best_axis: np.ndarray | None = None

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
            out[f"{directions[i].name}__{directions[j].name}"] = acute_angle_deg(
                directions[i].vector,
                directions[j].vector,
            )
    return out


def fit_common_axis(vectors: np.ndarray) -> np.ndarray:
    scatter = np.einsum("ni,nj->ij", vectors, vectors)
    _, evecs = np.linalg.eigh(scatter)
    return unit_vector(evecs[:, -1])


def common_axis_statistic_deg(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis = fit_common_axis(vectors)
    dots = np.clip(np.abs(vectors @ axis), 0.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    return axis, angles


def monte_carlo_common_axis(n_samples: int, n_dirs: int, rng: np.random.Generator) -> np.ndarray:
    dirs = rng.normal(size=(n_samples, n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
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


def load_pr3_smica(
    cache_dir: Path,
    nside_out: int,
    lmax_clean: int,
    mask_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int], complex]]:
    smica_path = download_if_needed(PR3_SMICA_URL, cache_dir / Path(PR3_SMICA_URL).name)
    mask_path = download_if_needed(PR3_COMMON_MASK_URL, cache_dir / Path(PR3_COMMON_MASK_URL).name)

    hi_map = extract_map_column(smica_path, preferred_names=["I_STOKES", "TEMPERATURE", "I"], field_index=0)
    hi_mask = extract_map_column(mask_path, field_index=0)

    lo_map, lon, lat, xyz = resample_ring_map(hi_map, nside_out=nside_out, interpolate=True)
    mask, _, _, _ = prepare_mask(hi_mask, nside_out=nside_out, threshold=mask_threshold)
    cleaned, alm = smooth_large_angle_map(lo_map, mask, theta=0.5 * np.pi - lat, lon=lon, lmax_clean=lmax_clean)
    return cleaned, mask, lon, lat, xyz, alm


def load_pr4_single_frequency(
    cache_dir: Path,
    channel: str,
    nside_out: int,
    lmax_clean: int,
    mask_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int], complex]]:
    if channel not in PR4_FREQ_URLS:
        raise ValueError(f"Unsupported PR4 channel: {channel}")

    map_path = download_if_needed(PR4_FREQ_URLS[channel], cache_dir / Path(PR4_FREQ_URLS[channel]).name)
    mask_path = download_if_needed(PR3_COMMON_MASK_URL, cache_dir / Path(PR3_COMMON_MASK_URL).name)

    hi_map = extract_map_column(map_path, preferred_names=["I_STOKES", "TEMPERATURE", "I"], field_index=0)
    hi_mask = extract_map_column(mask_path, field_index=0)

    lo_map, lon, lat, xyz = resample_ring_map(hi_map, nside_out=nside_out, interpolate=True)
    mask, _, _, _ = prepare_mask(hi_mask, nside_out=nside_out, threshold=mask_threshold)
    cleaned, alm = smooth_large_angle_map(lo_map, mask, theta=0.5 * np.pi - lat, lon=lon, lmax_clean=lmax_clean)
    return cleaned, mask, lon, lat, xyz, alm


def load_pr4_ilc(
    cache_dir: Path,
    channels: Iterable[str],
    nside_out: int,
    lmax_clean: int,
    mask_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int], complex], Dict[str, float]]:
    mask_path = download_if_needed(PR3_COMMON_MASK_URL, cache_dir / Path(PR3_COMMON_MASK_URL).name)
    hi_mask = extract_map_column(mask_path, field_index=0)
    mask, lon, lat, xyz = prepare_mask(hi_mask, nside_out=nside_out, threshold=mask_threshold)

    maps: List[np.ndarray] = []
    channel_list = list(channels)
    for ch in channel_list:
        if ch not in PR4_FREQ_URLS:
            raise ValueError(f"Unsupported PR4 channel in ILC: {ch}")
        path = download_if_needed(PR4_FREQ_URLS[ch], cache_dir / Path(PR4_FREQ_URLS[ch]).name)
        hi_map = extract_map_column(path, preferred_names=["I_STOKES", "TEMPERATURE", "I"], field_index=0)
        lo_map, _, _, _ = resample_ring_map(hi_map, nside_out=nside_out, interpolate=True)
        maps.append(lo_map)

    stack = np.vstack(maps)
    valid = mask & np.all(np.isfinite(stack), axis=0)
    cmb, weights = global_ilc(stack, valid)
    cleaned, alm = smooth_large_angle_map(cmb, mask, theta=0.5 * np.pi - lat, lon=lon, lmax_clean=lmax_clean)
    weight_dict = {ch: float(w) for ch, w in zip(channel_list, weights)}
    return cleaned, mask, lon, lat, xyz, alm, weight_dict


def main() -> None:
    args = parse_args()
    ensure_dir(args.cache_dir)
    ensure_dir(args.output_dir)

    rng = np.random.default_rng(12345)
    ilc_weights: Dict[str, float] | None = None

    if args.source == "pr3_smica":
        cmb_map, mask, lon, lat, xyz, alm = load_pr3_smica(
            cache_dir=args.cache_dir,
            nside_out=args.nside,
            lmax_clean=args.lmax_clean,
            mask_threshold=args.mask_threshold,
        )
    elif args.source == "pr4_143":
        cmb_map, mask, lon, lat, xyz, alm = load_pr4_single_frequency(
            cache_dir=args.cache_dir,
            channel="143",
            nside_out=args.nside,
            lmax_clean=args.lmax_clean,
            mask_threshold=args.mask_threshold,
        )
    elif args.source == "pr4_ilc":
        channels = [part.strip() for part in args.pr4_ilc_channels.split(",") if part.strip()]
        cmb_map, mask, lon, lat, xyz, alm, ilc_weights = load_pr4_ilc(
            cache_dir=args.cache_dir,
            channels=channels,
            nside_out=args.nside,
            lmax_clean=args.lmax_clean,
            mask_threshold=args.mask_threshold,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported source: {args.source}")

    quadrupole_axis = preferred_axis_power_tensor_from_alm(alm, ell=2)
    octupole_axis = preferred_axis_power_tensor_from_alm(alm, ell=3)
    hemi_axis, hemi_score = hemispherical_asymmetry_axis(
        cmb_map,
        mask,
        pixel_xyz=xyz,
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
        "notes": {
            "implementation": "No healpy import. Uses astropy.io.fits + astropy-healpix + masked least-squares low-l fit.",
            "multipole_axis_estimator": "angular_momentum_dispersion_power_tensor_proxy",
        },
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
