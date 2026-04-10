#!/usr/bin/env python3
"""
Test 11 — Cosmic Filament Orientation Correlation

Standalone script that downloads the public Tempel et al. (2014) SDSS filament
catalog from CDS/VizieR and measures the large-scale filament orientational
correlation

    C(r) = < |cos(theta_ij)|^2 - 1/3 >

in user-specified separation bins.

The script:
  1. Downloads the public filament-points catalogue.
  2. Extracts one representative midpoint position and tangent per filament.
  3. Computes the pairwise orientational correlation in 3D separation bins.
  4. Fits C(r) = A exp(-r / r_texture) above 30 Mpc.
  5. Writes result.json and a plot.

Notes / assumptions
-------------------
- The Tempel CDS table stores distances and coordinates in Mpc/h. The script
  converts fitted texture lengths and bin centres to physical Mpc using the
  user-supplied Hubble parameter h (default 0.7).
- The midpoint representative of a filament is taken as the middle row of the
  filament-point sequence in table2.
- The implementation uses the public point-level catalogue because it provides
  both 3D positions and local filament orientations.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


TABLE2_URLS = [
    "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/438/3465/table2.dat.gz",
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/438/3465/table2&-out=ID,IDpt,Npts,Len,x,y,z,D,dx,dy,dz,vmap,fden,fori&-out.max=unlimited",
    "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=J/MNRAS/438/3465/table2&-out=ID,IDpt,Npts,Len,x,y,z,D,dx,dy,dz,vmap,fden,fori&-out.max=unlimited",
]

EXPECTED_COLS = 14
DEFAULT_BINS_MPC_H = [10.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0]


@dataclass
class FilamentMidpoint:
    filament_id: int
    npts: int
    length_mpc_h: float
    x_mpc_h: float
    y_mpc_h: float
    z_mpc_h: float
    dx: float
    dy: float
    dz: float
    orientation_strength: float


@dataclass
class FitResult:
    amplitude: float
    amplitude_sigma: float
    r_texture_mpc: float
    r_texture_sigma_mpc: float
    chi2: float
    chi2_reduced: float
    n_fit_bins: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="test11_output", help="Output directory")
    p.add_argument("--cache-dir", default="test11_cache", help="Download cache directory")
    p.add_argument("--force-download", action="store_true", help="Redownload public inputs")
    p.add_argument(
        "--bin-edges-mpc-h",
        default=",".join(str(x) for x in DEFAULT_BINS_MPC_H),
        help="Comma-separated separation-bin edges in Mpc/h",
    )
    p.add_argument("--h", type=float, default=0.7, help="Dimensionless Hubble parameter for Mpc/h -> Mpc conversion")
    p.add_argument("--bao-scale-mpc", type=float, default=147.0, help="BAO scale in physical Mpc")
    p.add_argument("--fit-min-mpc", type=float, default=30.0, help="Minimum separation in physical Mpc used in exponential fit")
    p.add_argument("--min-filament-points", type=int, default=5, help="Minimum number of points required per filament")
    p.add_argument("--max-filaments", type=int, default=None, help="Optional random subsample size for faster exploratory runs")
    p.add_argument("--block-size", type=int, default=512, help="Block size for pairwise accumulation")
    p.add_argument("--seed", type=int, default=12345, help="RNG seed used only if --max-filaments is set")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_with_retries(urls: Sequence[str], dest: Path, force: bool = False, timeout: int = 60) -> Path:
    if dest.exists() and not force:
        logging.info("Using cached file %s", dest)
        return dest

    ensure_dir(dest.parent)
    errors: List[str] = []

    for url in urls:
        logging.info("Downloading %s", url)
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read()
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(dest)
                logging.info("Saved %s (%.2f MB)", dest, dest.stat().st_size / 1024**2)
                return dest
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url} [attempt {attempt}]: {exc}")
                logging.warning("Download failed for %s (attempt %d): %s", url, attempt, exc)
                time.sleep(1.0 * attempt)

    raise RuntimeError("Failed to download required public data. Errors:\n" + "\n".join(errors))


def _iter_noncomment_lines_from_bytes(raw: bytes) -> Iterable[str]:
    text = raw.decode("utf-8", "replace")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("---"):
            continue
        yield s


def _looks_like_int(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def _looks_like_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_tempel_table2(path: Path) -> Dict[int, List[Tuple[int, float, float, float, float, float, float, float, float]]]:
    """
    Parse the public Tempel et al. table2 file.

    Returns
    -------
    dict mapping filament ID -> list of tuples
        (Npts, Len, x, y, z, dx, dy, dz, fori)
    """
    logging.info("Parsing %s", path)
    rows_by_id: Dict[int, List[Tuple[int, float, float, float, float, float, float, float, float]]] = {}

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            lines = f
            for raw in lines:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < EXPECTED_COLS:
                    continue
                if not (_looks_like_int(parts[0]) and _looks_like_int(parts[1])):
                    continue
                if not all(_looks_like_float(tok) for tok in parts[2:14]):
                    continue
                fid = int(parts[0])
                npts = int(parts[2])
                flen = float(parts[3])
                x, y, z = map(float, parts[4:7])
                dx, dy, dz = map(float, parts[8:11])
                fori = float(parts[13])
                rows_by_id.setdefault(fid, []).append((npts, flen, x, y, z, dx, dy, dz, fori))
    else:
        raw = path.read_bytes()
        for line in _iter_noncomment_lines_from_bytes(raw):
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < EXPECTED_COLS:
                continue
            parts = [p.strip() for p in parts[:EXPECTED_COLS]]
            if not (_looks_like_int(parts[0]) and _looks_like_int(parts[1])):
                continue
            if not all(_looks_like_float(tok) for tok in parts[2:14]):
                continue
            fid = int(parts[0])
            npts = int(parts[2])
            flen = float(parts[3])
            x, y, z = map(float, parts[4:7])
            dx, dy, dz = map(float, parts[8:11])
            fori = float(parts[13])
            rows_by_id.setdefault(fid, []).append((npts, flen, x, y, z, dx, dy, dz, fori))

    if not rows_by_id:
        raise RuntimeError(f"No valid filament rows found in {path}")

    logging.info("Parsed %d filament groups", len(rows_by_id))
    return rows_by_id


def build_midpoint_catalog(
    rows_by_id: Dict[int, List[Tuple[int, float, float, float, float, float, float, float, float]]],
    min_points: int = 5,
    max_filaments: int | None = None,
    seed: int = 12345,
) -> List[FilamentMidpoint]:
    mids: List[FilamentMidpoint] = []
    for fid, rows in rows_by_id.items():
        if not rows:
            continue
        mid = rows[len(rows) // 2]
        npts, flen, x, y, z, dx, dy, dz, fori = mid
        if npts < min_points:
            continue
        vec = np.array([dx, dy, dz], dtype=float)
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm <= 0:
            continue
        vec /= norm
        mids.append(
            FilamentMidpoint(
                filament_id=int(fid),
                npts=int(npts),
                length_mpc_h=float(flen),
                x_mpc_h=float(x),
                y_mpc_h=float(y),
                z_mpc_h=float(z),
                dx=float(vec[0]),
                dy=float(vec[1]),
                dz=float(vec[2]),
                orientation_strength=float(fori),
            )
        )

    if max_filaments is not None and len(mids) > max_filaments:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(mids), size=max_filaments, replace=False)
        mids = [mids[i] for i in sorted(idx)]
        logging.info("Randomly subsampled to %d filaments", len(mids))

    if len(mids) < 10:
        raise RuntimeError(f"Too few usable filaments after midpoint extraction: {len(mids)}")

    logging.info("Using %d filament midpoints", len(mids))
    return mids


def _accumulate_block(
    Xi: np.ndarray,
    Xj: np.ndarray,
    Ui: np.ndarray,
    Uj: np.ndarray,
    pos2_i: np.ndarray,
    pos2_j: np.ndarray,
    edges: np.ndarray,
    same_block: bool,
    sums: np.ndarray,
    sums2: np.ndarray,
    counts: np.ndarray,
) -> None:
    dist2 = pos2_i[:, None] + pos2_j[None, :] - 2.0 * (Xi @ Xj.T)
    np.maximum(dist2, 0.0, out=dist2)
    dist = np.sqrt(dist2, out=dist2)
    corr = (Ui @ Uj.T) ** 2 - (1.0 / 3.0)

    if same_block:
        tri = np.triu_indices(dist.shape[0], k=1)
        d = dist[tri]
        c = corr[tri]
    else:
        d = dist.ravel()
        c = corr.ravel()

    mask = (d >= edges[0]) & (d < edges[-1]) & np.isfinite(c)
    if not np.any(mask):
        return

    d = d[mask]
    c = c[mask]
    bin_index = np.digitize(d, edges, right=False) - 1
    valid = (bin_index >= 0) & (bin_index < len(edges) - 1)
    if not np.any(valid):
        return

    bin_index = bin_index[valid]
    c = c[valid]

    np.add.at(counts, bin_index, 1)
    np.add.at(sums, bin_index, c)
    np.add.at(sums2, bin_index, c * c)


def compute_orientational_correlation(
    positions_mpc_h: np.ndarray,
    tangents: np.ndarray,
    edges_mpc_h: np.ndarray,
    block_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = positions_mpc_h.shape[0]
    nbins = len(edges_mpc_h) - 1
    sums = np.zeros(nbins, dtype=np.float64)
    sums2 = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    pos2 = np.einsum("ij,ij->i", positions_mpc_h, positions_mpc_h)

    n_blocks = math.ceil(n / block_size)
    logging.info("Accumulating pair statistics over %d filaments in %d blocks", n, n_blocks)

    for bi, i0 in enumerate(range(0, n, block_size), start=1):
        i1 = min(i0 + block_size, n)
        Xi = positions_mpc_h[i0:i1]
        Ui = tangents[i0:i1]
        pos2_i = pos2[i0:i1]

        for j0 in range(i0, n, block_size):
            j1 = min(j0 + block_size, n)
            Xj = positions_mpc_h[j0:j1]
            Uj = tangents[j0:j1]
            pos2_j = pos2[j0:j1]
            _accumulate_block(
                Xi,
                Xj,
                Ui,
                Uj,
                pos2_i,
                pos2_j,
                edges_mpc_h,
                same_block=(i0 == j0),
                sums=sums,
                sums2=sums2,
                counts=counts,
            )

        logging.info("Processed block %d/%d", bi, n_blocks)

    means = np.full(nbins, np.nan, dtype=float)
    stderr = np.full(nbins, np.nan, dtype=float)
    for k in range(nbins):
        if counts[k] == 0:
            continue
        means[k] = sums[k] / counts[k]
        if counts[k] > 1:
            var = max((sums2[k] / counts[k]) - means[k] ** 2, 0.0)
            stderr[k] = math.sqrt(var / counts[k])
        else:
            stderr[k] = np.nan

    centers = 0.5 * (edges_mpc_h[:-1] + edges_mpc_h[1:])
    return centers, means, stderr, counts


def fit_exponential_grid(x_mpc: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> FitResult:
    mask = np.isfinite(x_mpc) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    x = x_mpc[mask]
    yv = y[mask]
    sv = sigma[mask]

    if len(x) < 3:
        raise RuntimeError("Too few valid bins for exponential fit")

    w = 1.0 / (sv * sv)
    r_grid = np.geomspace(max(1.0, float(np.min(x)) / 2.0), max(2000.0, float(np.max(x)) * 8.0), 4000)

    best = None
    best_A = np.nan
    best_r = np.nan
    best_chi2 = np.inf

    for r in r_grid:
        e = np.exp(-x / r)
        denom = np.sum(w * e * e)
        if denom <= 0:
            continue
        A = np.sum(w * e * yv) / denom
        model = A * e
        chi2 = float(np.sum(w * (yv - model) ** 2))
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_A = float(A)
            best_r = float(r)
            best = e

    if best is None or not np.isfinite(best_chi2):
        raise RuntimeError("Exponential fit failed")

    # Local covariance from Jacobian at best fit.
    e = np.exp(-x / best_r)
    jac = np.column_stack(
        [
            e,
            best_A * e * (x / (best_r * best_r)),
        ]
    )
    jt_w = jac.T * w
    fisher = jt_w @ jac
    try:
        cov = np.linalg.inv(fisher)
        sigma_A = float(math.sqrt(max(cov[0, 0], 0.0)))
        sigma_r = float(math.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        sigma_A = float("nan")
        sigma_r = float("nan")

    dof = max(len(x) - 2, 0)
    chi2_red = best_chi2 / dof if dof > 0 else float("nan")

    return FitResult(
        amplitude=best_A,
        amplitude_sigma=sigma_A,
        r_texture_mpc=best_r,
        r_texture_sigma_mpc=sigma_r,
        chi2=best_chi2,
        chi2_reduced=chi2_red,
        n_fit_bins=len(x),
    )


def make_plot(
    outpath: Path,
    centers_mpc: np.ndarray,
    corr: np.ndarray,
    stderr: np.ndarray,
    fit_mask: np.ndarray,
    fit: FitResult,
    bao_scale_mpc: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(centers_mpc, corr, yerr=stderr, fmt="o", capsize=3, label="Data")

    xfit = np.linspace(max(centers_mpc[fit_mask].min(), 1.0), centers_mpc.max() * 1.05, 400)
    yfit = fit.amplitude * np.exp(-xfit / fit.r_texture_mpc)
    ax.plot(xfit, yfit, label="Exponential fit")
    ax.axvline(bao_scale_mpc, linestyle="--", label=f"BAO scale = {bao_scale_mpc:.0f} Mpc")
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Filament separation [Mpc]")
    ax.set_ylabel(r"$\langle |\cos\theta|^2 - 1/3 \rangle$")
    ax.set_title("Cosmic filament orientation correlation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    outdir = Path(args.outdir)
    cache_dir = Path(args.cache_dir)
    ensure_dir(outdir)
    ensure_dir(cache_dir)

    edges_mpc_h = np.array([float(x) for x in args.bin_edges_mpc_h.split(",") if x.strip()], dtype=float)
    if len(edges_mpc_h) < 2 or not np.all(np.diff(edges_mpc_h) > 0):
        raise ValueError("--bin-edges-mpc-h must contain at least two strictly increasing values")
    if args.h <= 0:
        raise ValueError("--h must be positive")

    download_path = cache_dir / ("table2.dat.gz")
    table2_path = download_with_retries(TABLE2_URLS, download_path, force=args.force_download)

    rows_by_id = parse_tempel_table2(table2_path)
    mids = build_midpoint_catalog(
        rows_by_id,
        min_points=args.min_filament_points,
        max_filaments=args.max_filaments,
        seed=args.seed,
    )

    positions = np.array([[m.x_mpc_h, m.y_mpc_h, m.z_mpc_h] for m in mids], dtype=np.float64)
    tangents = np.array([[m.dx, m.dy, m.dz] for m in mids], dtype=np.float64)

    centers_mpc_h, corr, stderr, counts = compute_orientational_correlation(
        positions_mpc_h=positions,
        tangents=tangents,
        edges_mpc_h=edges_mpc_h,
        block_size=args.block_size,
    )

    centers_mpc = centers_mpc_h / args.h
    stderr_safe = np.where(np.isfinite(stderr) & (stderr > 0), stderr, np.nan)
    fit_mask = np.isfinite(corr) & np.isfinite(stderr_safe) & (stderr_safe > 0) & (centers_mpc >= args.fit_min_mpc)
    fit = fit_exponential_grid(centers_mpc[fit_mask], corr[fit_mask], stderr_safe[fit_mask])

    amp_sig = fit.amplitude / fit.amplitude_sigma if np.isfinite(fit.amplitude_sigma) and fit.amplitude_sigma > 0 else float("nan")
    pass_amplitude = bool(np.isfinite(amp_sig) and fit.amplitude > 0 and amp_sig >= 3.0)
    pass_length_above_bao = bool(np.isfinite(fit.r_texture_mpc) and fit.r_texture_mpc > args.bao_scale_mpc)

    result = {
        "source_catalog": "Tempel et al. 2014 filament points catalog (CDS/VizieR table2)",
        "n_filaments_used": int(len(mids)),
        "bin_edges_mpc_h": edges_mpc_h.tolist(),
        "bin_centers_mpc": centers_mpc.tolist(),
        "pair_counts": counts.astype(int).tolist(),
        "correlation_values": [None if not np.isfinite(x) else float(x) for x in corr],
        "correlation_errors": [None if not np.isfinite(x) else float(x) for x in stderr],
        "correlation_amplitude": float(fit.amplitude),
        "correlation_amplitude_sigma": None if not np.isfinite(fit.amplitude_sigma) else float(fit.amplitude_sigma),
        "r_texture_mpc": float(fit.r_texture_mpc),
        "r_texture_sigma": None if not np.isfinite(fit.r_texture_sigma_mpc) else float(fit.r_texture_sigma_mpc),
        "chi2_reduced": None if not np.isfinite(fit.chi2_reduced) else float(fit.chi2_reduced),
        "pass_amplitude": pass_amplitude,
        "pass_length_above_bao": pass_length_above_bao,
        "bao_scale_mpc": float(args.bao_scale_mpc),
        "fit_min_mpc": float(args.fit_min_mpc),
        "significance_amplitude_sigma_units": None if not np.isfinite(amp_sig) else float(amp_sig),
    }

    json_path = outdir / "result.json"
    plot_path = outdir / "filament_orientation_correlation.png"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_plot(plot_path, centers_mpc, corr, stderr_safe, fit_mask, fit, args.bao_scale_mpc)

    logging.info("Wrote %s", json_path)
    logging.info("Wrote %s", plot_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
