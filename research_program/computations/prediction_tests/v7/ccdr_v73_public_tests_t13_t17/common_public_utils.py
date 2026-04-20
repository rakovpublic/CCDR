
#!/usr/bin/env python3
"""
Shared helpers for CCDR v7.3 public-data screening tests T13–T17.

These helpers favor:
- public downloads only
- cached archives
- minimal but common dependencies
- proxy-friendly fallbacks when collaboration-grade products are hard to reproduce

Dependencies:
    numpy, scipy, pandas, requests, astropy
Optional:
    healpy or astropy-healpix
    pint-pulsar
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, Planck18
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.stats import kurtosis, pearsonr, spearmanr

USER_AGENT = "CCDR-v73-public-tests/1.0 (+OpenAI ChatGPT)"
DEFAULT_TIMEOUT = 60
DEFAULT_COSMO = FlatLambdaCDM(H0=Planck18.H0, Om0=Planck18.Om0, Tcmb0=Planck18.Tcmb0)

MPS2_PER_KMS2_PER_KPC = 1e6 / (3.085677581491367e19)  # (km/s)^2 per kpc -> m/s^2


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(*parts: object) -> None:
    print(*parts, file=sys.stderr, flush=True)


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    r = session().get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def fetch_json(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    r = session().get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def download_file(url: str, dest: os.PathLike | str, overwrite: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return dest
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session().get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0") or 0)
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=2**20):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
        if total and written < total * 0.95:
            raise IOError(f"Incomplete download for {url}: {written} < {total}")
    tmp.replace(dest)
    return dest


def download_first_available(urls: Sequence[str], dest: os.PathLike | str, overwrite: bool = False) -> Path:
    last_err = None
    for url in urls:
        try:
            return download_file(url, dest, overwrite=overwrite)
        except Exception as e:
            last_err = e
            log(f"[download] failed {url}: {e}")
    raise RuntimeError(f"All download candidates failed for {dest}: {last_err}")


def extract_archive(archive: os.PathLike | str, outdir: os.PathLike | str, overwrite: bool = False) -> Path:
    archive = Path(archive)
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        return outdir
    ensure_dir(outdir)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(outdir)
    elif archive.suffix in {".gz", ".tgz"} or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(outdir)
    elif archive.suffix == ".tar":
        with tarfile.open(archive, "r") as tf:
            tf.extractall(outdir)
    else:
        raise ValueError(f"Unsupported archive type: {archive}")
    return outdir


def parse_zenodo_record_id(record_id_or_doi: str | int) -> int:
    if isinstance(record_id_or_doi, int):
        return record_id_or_doi
    s = str(record_id_or_doi)
    m = re.search(r"zenodo[./](\d+)", s)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    raise ValueError(f"Cannot parse Zenodo record id from {record_id_or_doi!r}")


def zenodo_record(record_id_or_doi: str | int) -> dict:
    rid = parse_zenodo_record_id(record_id_or_doi)
    return fetch_json(f"https://zenodo.org/api/records/{rid}")


def find_zenodo_file(record_id_or_doi: str | int, patterns: Sequence[str]) -> dict:
    rec = zenodo_record(record_id_or_doi)
    files = rec.get("files", [])
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for f in files:
            name = f.get("key") or f.get("filename") or ""
            if rx.search(name):
                return f
    available = [f.get("key") or f.get("filename") for f in files]
    raise FileNotFoundError(f"No Zenodo file matched {patterns}; available={available}")


def download_zenodo_file(record_ids: Sequence[str | int], patterns: Sequence[str], dest: os.PathLike | str) -> Path:
    last_err = None
    for rid in record_ids:
        try:
            f = find_zenodo_file(rid, patterns)
            url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
            if not url:
                raise KeyError(f"No download link in Zenodo metadata for {f}")
            return download_file(url, dest)
        except Exception as e:
            last_err = e
            log(f"[zenodo] record {rid} failed for {patterns}: {e}")
    raise RuntimeError(f"Zenodo candidates failed for {dest}: {last_err}")


def github_releases(owner: str, repo: str) -> list[dict]:
    return fetch_json(f"https://api.github.com/repos/{owner}/{repo}/releases")


def download_github_release_asset(
    owner: str,
    repo: str,
    asset_name_regex: str,
    dest: os.PathLike | str,
    release_name_regex: Optional[str] = None,
    tag_name_regex: Optional[str] = None,
) -> Path:
    releases = github_releases(owner, repo)
    name_rx = re.compile(release_name_regex or ".*", re.I)
    tag_rx = re.compile(tag_name_regex or ".*", re.I)
    asset_rx = re.compile(asset_name_regex, re.I)
    for rel in releases:
        rel_name = rel.get("name") or ""
        tag_name = rel.get("tag_name") or ""
        if not name_rx.search(rel_name):
            continue
        if not tag_rx.search(tag_name):
            continue
        for asset in rel.get("assets", []):
            asset_name = asset.get("name") or ""
            if asset_rx.search(asset_name):
                url = asset.get("browser_download_url")
                if url:
                    return download_file(url, dest)
    raise FileNotFoundError(
        f"No GitHub asset found: repo={owner}/{repo}, asset={asset_name_regex}, release={release_name_regex}, tag={tag_name_regex}"
    )


def discover_links(url: str) -> list[str]:
    html = fetch_text(url)
    links = re.findall(r'href=["\\\']([^"\\\']+)["\\\']', html, flags=re.I)
    abs_links = []
    for link in links:
        if link.startswith("//"):
            abs_links.append("https:" + link)
        elif link.startswith("http://") or link.startswith("https://"):
            abs_links.append(link)
        elif link.startswith("/"):
            m = re.match(r"(https?://[^/]+)", url)
            abs_links.append(m.group(1) + link if m else link)
    return sorted(set(abs_links))


def discover_kmos3d_catalog_url() -> str:
    candidates = []
    try:
        for link in discover_links("https://www.mpe.mpg.de/ir/KMOS3D/data"):
            if "KMOS3D" in link and re.search(r"(table|catalog|fnlsp).*\.(tgz|tar\.gz|fits)", link, re.I):
                candidates.append(link)
    except Exception:
        pass
    candidates += [
        "https://www.mpe.mpg.de/resources/KMOS3D/3d_fnlsp_table_v3.fits.tgz",
        "https://www.mpe.mpg.de/resources/KMOS3D/3d_fnlsp_table_v3.tgz",
    ]
    for url in candidates:
        try:
            r = session().head(url, allow_redirects=True, timeout=20)
            if r.ok:
                return url
        except Exception:
            continue
    return candidates[0]


def discover_kmos3d_all_cubes_url() -> Optional[str]:
    candidates = []
    try:
        for link in discover_links("https://www.mpe.mpg.de/ir/KMOS3D/data"):
            if "KMOS3D" in link and re.search(r"(cube|cubes).*\.(tgz|tar\.gz)", link, re.I):
                candidates.append(link)
    except Exception:
        pass
    guessed = [
        "https://www.mpe.mpg.de/resources/KMOS3D/KMOS3D_cubes.tgz",
        "https://www.mpe.mpg.de/resources/KMOS3D/3d_cubes_v3.tgz",
        "https://www.mpe.mpg.de/resources/KMOS3D/kmos3d_cubes_all.tgz",
    ]
    candidates += guessed
    for url in candidates:
        try:
            r = session().head(url, allow_redirects=True, timeout=20)
            if r.ok:
                return url
        except Exception:
            continue
    return None


def load_fits_table(path: os.PathLike | str, hdu: int | str = 1) -> np.recarray:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        try:
            data = hdul[hdu].data
            if data is not None:
                return data.copy()
        except Exception:
            pass
        for h in hdul:
            data = getattr(h, 'data', None)
            if data is not None and hasattr(data, 'dtype'):
                return data.copy()
    raise RuntimeError(f'No table HDU found in {path}')


def lower_names(table) -> dict[str, str]:
    return {name.lower(): name for name in table.dtype.names}


def find_column_name(table, exact: Sequence[str] = (), contains: Sequence[str] = (), avoid: Sequence[str] = ()) -> Optional[str]:
    names = list(table.dtype.names)
    lname = [n.lower() for n in names]
    for ex in exact:
        for n in names:
            if n.lower() == ex.lower():
                return n
    for token in contains:
        for n in names:
            nl = n.lower()
            if token.lower() in nl and not any(a.lower() in nl for a in avoid):
                return n
    return None


def table_to_dataframe(table) -> pd.DataFrame:
    out = {}
    for n in table.dtype.names:
        arr = table[n]
        if hasattr(arr, "dtype") and arr.dtype.kind == "S":
            out[n] = np.char.decode(arr.astype("S"), errors="ignore")
        else:
            out[n] = arr
    return pd.DataFrame(out)


def comoving_distance_mpc_h(z: np.ndarray, cosmology=DEFAULT_COSMO) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.asarray(cosmology.comoving_distance(z).value * cosmology.h, dtype=float)


def radec_z_to_cartesian(ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray, cosmology=DEFAULT_COSMO) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    chi = comoving_distance_mpc_h(np.asarray(z, dtype=float), cosmology=cosmology)
    x = chi * np.cos(dec) * np.cos(ra)
    y = chi * np.cos(dec) * np.sin(ra)
    zc = chi * np.sin(dec)
    return np.column_stack([x, y, zc])


def skycoord_from_par(par_path: os.PathLike | str) -> Optional[SkyCoord]:
    raj = None
    decj = None
    elong = None
    elat = None
    for line in Path(par_path).read_text(errors='ignore').splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].strip().upper()
        if key == 'RAJ':
            raj = parts[1]
        elif key == 'DECJ':
            decj = parts[1]
        elif key in {'ELONG', 'LAMBDA'}:
            elong = parts[1]
        elif key in {'ELAT', 'BETA'}:
            elat = parts[1]
    if raj and decj:
        try:
            return SkyCoord(raj, decj, unit=(u.hourangle, u.deg), frame='icrs')
        except Exception:
            try:
                return SkyCoord(raj, decj, unit=(u.hourangle, u.deg), frame='fk5')
            except Exception:
                pass
    if elong and elat:
        try:
            from astropy.coordinates import BarycentricTrueEcliptic
            c = SkyCoord(float(elong) * u.deg, float(elat) * u.deg, frame=BarycentricTrueEcliptic())
            return c.icrs
        except Exception:
            pass
    return None


def read_tim_error_proxy(tim_path: os.PathLike | str) -> dict:
    mjd = []
    err_us = []
    with open(tim_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(("C ", "FORMAT", "#", "MODE", "INFO")):
                continue
            parts = line.split()
            # TEMPO2 .tim common format: file freq mjd err obs ...
            if len(parts) < 5:
                continue
            try:
                mjd.append(float(parts[2]))
                err_us.append(float(parts[3]))
            except Exception:
                continue
    if not err_us:
        raise ValueError(f"No parseable TOA rows in {tim_path}")
    mjd = np.asarray(mjd)
    err_us = np.asarray(err_us)
    cadence = np.median(np.diff(np.sort(mjd))) if mjd.size > 3 else np.nan
    # Proxy amplitude: larger scatter + longer span -> larger timing-structure opportunity.
    span_days = float(np.ptp(mjd) if mjd.size else 0.0)
    proxy = float(np.nanmedian(err_us) * np.sqrt(max(1.0, span_days / max(cadence, 30.0))))
    return {
        "n_toa": int(err_us.size),
        "median_toa_err_us": float(np.nanmedian(err_us)),
        "rms_toa_err_us": float(np.sqrt(np.nanmean(err_us**2))),
        "span_days": span_days,
        "cadence_days": float(cadence),
        "proxy_amplitude": proxy,
    }


def try_pint_wrms(par_path: os.PathLike | str, tim_path: os.PathLike | str) -> Optional[float]:
    try:
        from pint.models import get_model
        from pint.toa import get_TOAs
        from pint.residuals import Residuals
        model = get_model(str(par_path))
        toas = get_TOAs(str(tim_path), include_bipm=False, planets=False)
        res = Residuals(toas, model)
        vals = res.time_resids.to_value(u.us)
        if len(vals) == 0:
            return None
        return float(np.sqrt(np.mean(vals**2)))
    except Exception:
        return None


def weighted_rms(x: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    x = np.asarray(x, dtype=float)
    if w is None:
        return float(np.sqrt(np.nanmean(x * x)))
    w = np.asarray(w, dtype=float)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return float("nan")
    return float(np.sqrt(np.sum(w[good] * x[good] ** 2) / np.sum(w[good])))


def bootstrap_stat(values: np.ndarray, func, n_boot: int = 200, rng: Optional[np.random.Generator] = None) -> Tuple[float, Tuple[float, float]]:
    values = np.asarray(values)
    rng = rng or np.random.default_rng(1234)
    if values.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, values.size, values.size)
        boots.append(float(func(values[idx])))
    boots = np.asarray(boots, dtype=float)
    return float(func(values)), (float(np.nanpercentile(boots, 16)), float(np.nanpercentile(boots, 84)))


def cumulative_pair_counts(xyz1: np.ndarray, xyz2: np.ndarray, bins: np.ndarray, self_pairs: bool = False) -> np.ndarray:
    t1 = cKDTree(xyz1)
    t2 = cKDTree(xyz2)
    cum = np.asarray(t1.count_neighbors(t2, bins, cumulative=True), dtype=float)
    diff = np.diff(np.concatenate([[0.0], cum]))
    if self_pairs:
        # count_neighbors(tree, tree) includes self and double counts for distinct pairs.
        diff = np.maximum(0.0, (diff - len(xyz1) * np.r_[1, np.zeros(len(diff)-1)]) / 2.0)
    return diff


def landy_szalay(dd: np.ndarray, dr: np.ndarray, rr: np.ndarray, nd: int, nr: int) -> np.ndarray:
    ddn = dd / max(nd * (nd - 1) / 2, 1)
    drn = dr / max(nd * nr, 1)
    rrn = rr / max(nr * (nr - 1) / 2, 1)
    good = rrn > 0
    xi = np.full_like(rrn, np.nan, dtype=float)
    xi[good] = (ddn[good] - 2.0 * drn[good] + rrn[good]) / rrn[good]
    return xi


def _bao_model(s, a0, a1, a2, amp, mu, sigma):
    x = s - 100.0
    return a0 + a1 * x + a2 * x**2 + amp * np.exp(-0.5 * ((s - mu) / sigma) ** 2)


def fit_bao_peak(s_mid: np.ndarray, xi: np.ndarray) -> dict:
    mask = np.isfinite(s_mid) & np.isfinite(xi) & (s_mid >= 60) & (s_mid <= 150)
    s = np.asarray(s_mid[mask], dtype=float)
    y = np.asarray(xi[mask], dtype=float)
    if s.size < 8:
        raise ValueError("Too few BAO bins for fit")
    p0 = [np.nanmedian(y), 0.0, 0.0, np.nanmax(y) - np.nanmedian(y), 105.0, 10.0]
    bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 80.0, 2.0], [np.inf, np.inf, np.inf, np.inf, 130.0, 30.0])
    popt, pcov = curve_fit(_bao_model, s, y, p0=p0, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    return {
        "rd_mpc_h": float(popt[4]),
        "rd_err_mpc_h": float(perr[4]),
        "sigma_mpc_h": float(popt[5]),
        "amp": float(popt[3]),
        "params": [float(v) for v in popt],
    }


def save_json(obj: dict, path: os.PathLike | str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    return path


def save_xy_plot_png(path: os.PathLike | str, x: np.ndarray, y_series: Sequence[Tuple[str, np.ndarray]], xlabel: str, ylabel: str, title: str) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
        path = Path(path)
        ensure_dir(path.parent)
        plt.figure(figsize=(8, 5))
        for label, y in y_series:
            plt.plot(x, y, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if len(y_series) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path
    except Exception as e:
        log(f"[plot] skipped {path}: {e}")
        return None


def choose_existing(paths: Sequence[os.PathLike | str]) -> Optional[Path]:
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def find_files(root: os.PathLike | str, patterns: Sequence[str]) -> list[Path]:
    root = Path(root)
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        for pat in patterns:
            if re.search(pat, p.name, re.I):
                out.append(p)
                break
    return sorted(out)


def angular_separation_deg(ra1, dec1, ra2, dec2):
    c1 = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame="icrs")
    c2 = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame="icrs")
    return c1.separation(c2).deg


def tangent_plane_xy(ra_deg: np.ndarray, dec_deg: np.ndarray, dec0: Optional[float] = None) -> np.ndarray:
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    if dec0 is None:
        dec0 = np.nanmedian(dec)
    x = (ra - np.nanmedian(ra)) * np.cos(np.deg2rad(dec0))
    y = dec - np.nanmedian(dec)
    return np.column_stack([x, y])


def estimate_local_orientation(ra_deg: np.ndarray, dec_deg: np.ndarray, k: int = 8) -> np.ndarray:
    xy = tangent_plane_xy(ra_deg, dec_deg)
    tree = cKDTree(xy)
    k = min(k, len(xy))
    if k < 3:
        return np.full(len(xy), np.nan)
    _, idx = tree.query(xy, k=k)
    angles = np.full(len(xy), np.nan, dtype=float)
    for i in range(len(xy)):
        pts = xy[idx[i]]
        pts = pts - pts.mean(axis=0, keepdims=True)
        cov = pts.T @ pts
        w, v = np.linalg.eigh(cov)
        vec = v[:, np.argmax(w)]
        angles[i] = float(np.arctan2(vec[1], vec[0]))
    return angles


def bin_orientation_correlation(xyz: np.ndarray, theta: np.ndarray, bins: np.ndarray, max_points: int = 4000, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(123)
    good = np.isfinite(theta)
    xyz = np.asarray(xyz[good], dtype=float)
    theta = np.asarray(theta[good], dtype=float)
    n = len(theta)
    if n < 5:
        mids = 0.5 * (bins[:-1] + bins[1:])
        return mids, np.full_like(mids, np.nan, dtype=float)
    if n > max_points:
        sel = rng.choice(n, size=max_points, replace=False)
        xyz = xyz[sel]
        theta = theta[sel]
        n = len(theta)
    tree = cKDTree(xyz)
    mids = 0.5 * (bins[:-1] + bins[1:])
    accum = [[] for _ in mids]
    for i in range(n):
        neigh = tree.query_ball_point(xyz[i], r=float(bins[-1]))
        for j in neigh:
            if j <= i:
                continue
            d = float(np.linalg.norm(xyz[i] - xyz[j]))
            b = np.searchsorted(bins, d, side="right") - 1
            if 0 <= b < len(mids):
                accum[b].append(np.cos(2.0 * (theta[i] - theta[j])))
    corr = np.array([np.nanmean(a) if a else np.nan for a in accum], dtype=float)
    return mids, corr


def fit_exponential_positive(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size < 3:
        return {"A": float("nan"), "r_texture_mpc_h": float("nan")}
    def model(x, A, r0):
        return A * np.exp(-x / r0)
    p0 = [float(np.nanmax(y)), 300.0]
    bounds = ([0.0, 1.0], [10.0, 1e6])
    popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    return {"A": float(popt[0]), "A_err": float(perr[0]), "r_texture_mpc_h": float(popt[1]), "r_texture_err_mpc_h": float(perr[1])}


def sample_healpix_map_value(map_path: os.PathLike | str, ra_deg: np.ndarray, dec_deg: np.ndarray, field: int = 0) -> np.ndarray:
    map_path = str(map_path)
    try:
        import healpy as hp
        vals = hp.read_map(map_path, field=field, verbose=False)
        theta = np.deg2rad(90.0 - np.asarray(dec_deg))
        phi = np.deg2rad(np.asarray(ra_deg))
        pix = hp.ang2pix(hp.get_nside(vals), theta, phi, nest=False)
        return np.asarray(vals[pix], dtype=float)
    except Exception:
        pass
    try:
        from astropy_healpix import HEALPix
        with fits.open(map_path, memmap=True) as hdul:
            data = np.asarray(hdul[1].data.field(field), dtype=float)
        npix = len(data)
        nside = int(np.sqrt(npix / 12.0))
        hp = HEALPix(nside=nside, order="ring", frame="icrs")
        coords = SkyCoord(np.asarray(ra_deg) * u.deg, np.asarray(dec_deg) * u.deg, frame="icrs")
        pix = hp.skycoord_to_healpix(coords)
        return np.asarray(data[pix], dtype=float)
    except Exception as e:
        raise RuntimeError("Need healpy or astropy-healpix to sample HEALPix maps") from e


def find_planck_pr4_files(extract_root: os.PathLike | str) -> Tuple[Path, Optional[Path]]:
    files = find_files(extract_root, [r"\.fits$"])
    map_candidates = [p for p in files if re.search(r"(mv|convergence|kappa)", p.name, re.I) and not re.search(r"(mean|mf|mask)", p.name, re.I)]
    mask_candidates = [p for p in files if re.search(r"mask", p.name, re.I)]
    if not map_candidates:
        # fall back to first fits file with map-ish size
        map_candidates = files
    if not map_candidates:
        raise FileNotFoundError(f"No FITS maps found under {extract_root}")
    map_path = sorted(map_candidates, key=lambda p: ("mv" not in p.name.lower(), len(p.name)))[0]
    mask_path = sorted(mask_candidates, key=lambda p: len(p.name))[0] if mask_candidates else None
    return map_path, mask_path


def kmos_guess_columns(df: pd.DataFrame) -> dict:
    names = {c.lower(): c for c in df.columns}
    def pick(*tokens):
        for token in tokens:
            for lc, c in names.items():
                if token in lc:
                    return c
        return None
    return {
        "z": pick(" z", "z_", "redshift", "zspec", "z"),
        "re_kpc": pick("re_kpc", "r_e_kpc", "reff_kpc", "re_kpc_circ", "rhalf_kpc", "re"),
        "vrot": pick("vrot", "vmax", "vcirc", "vflat", "v_out"),
        "mstar": pick("mstar", "logm", "log_m", "stellar_mass"),
        "mgas": pick("mgas", "gas_mass", "m_gas"),
        "field": pick("field"),
        "id": pick("id", "name"),
    }


def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def a0_from_gobs_gbar(gobs: np.ndarray, gbar: np.ndarray) -> np.ndarray:
    gobs = np.asarray(gobs, dtype=float)
    gbar = np.asarray(gbar, dtype=float)
    out = gobs * gobs / gbar - gobs
    bad = ~np.isfinite(out) | (gbar <= 0) | (gobs <= 0)
    out[bad] = np.nan
    return out
