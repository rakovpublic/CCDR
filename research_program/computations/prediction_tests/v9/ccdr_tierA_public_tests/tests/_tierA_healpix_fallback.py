#!/usr/bin/env python3
"""Small public-data helpers for CCDR Tier-A hotfixes T04/T05/T16/T22.

Design goals:
- no healpy dependency;
- all data downloaded by script;
- if a full-resolution HEALPix map cannot be used, sample the public ALM file
  through a controlled low-l spherical-harmonic fallback and report that fact.

The low-l fallback is a sign/proxy test, not a substitute for a full-resolution
lensing-map likelihood. Use --alm-lmax to raise/lower the truncation.
"""
from __future__ import annotations

import csv
import gzip
import html
import io
import json
import math
import os
import random
import re
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


USER_AGENT = "ccdr-tierA-hotfix/9.4 (+public-data-script)"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dump(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2, sort_keys=False, default=_json_default))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        x = float(o)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def download_file(url: str, path: os.PathLike[str] | str, *, timeout: int = 1200, force: bool = False) -> Dict[str, Any]:
    path = Path(path)
    ensure_dir(path.parent)
    if path.exists() and path.stat().st_size > 0 and not force:
        return {"ok": True, "url": url, "path": str(path), "size": path.stat().st_size, "cached": True}
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(path)
    return {"ok": True, "url": url, "path": str(path), "size": path.stat().st_size, "cached": False}


def fetch_text(url: str, *, timeout: int = 120) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="replace")


def url_basename(url: str) -> str:
    return Path(urllib.parse.urlparse(url).path).name or "download.dat"


def safe_name(name: str) -> str:
    name = urllib.parse.unquote(name)
    name = name.replace("/", "__").replace("\\", "__")
    return re.sub(r"[^A-Za-z0-9_.+\-]+", "_", name)


def parse_csv_numeric_table(path: os.PathLike[str] | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        rdr = csv.DictReader(f, dialect=dialect)
        for row in rdr:
            rows.append(row)
    return rows


def load_euclid_q1_positions(cache: Path, *, max_rows: int = 8000, force: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Download public Euclid Q1 MER RA/DEC sample through IRSA TAP."""
    data_sources: List[Dict[str, Any]] = []
    ensure_dir(cache / "irsa_tap")
    query_tables = "SELECT table_name, description FROM TAP_SCHEMA.tables WHERE LOWER(table_name) LIKE '%euclid%'"
    url_tables = "https://irsa.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode({
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "CSV", "QUERY": query_tables
    })
    p_tables = cache / "irsa_tap" / "euclid_tables.csv"
    try:
        data_sources.append(download_file(url_tables, p_tables, force=force, timeout=240))
    except Exception as e:
        data_sources.append({"ok": False, "url": url_tables, "error": repr(e)})
    selected = "euclid_q1_mer_catalogue"
    query = f"SELECT TOP {int(max_rows)} ra,dec FROM {selected} WHERE 1=1"
    url = "https://irsa.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode({
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "CSV", "QUERY": query
    })
    p = cache / "irsa_tap" / f"euclid_sample_{selected}_{int(max_rows)}.csv"
    data_sources.append(download_file(url, p, force=force, timeout=600))
    arr = np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding=None)
    if arr.size == 0:
        raise RuntimeError("Euclid TAP returned no rows")
    names = [n.lower() for n in arr.dtype.names or ()]
    ra_name = arr.dtype.names[names.index("ra")] if "ra" in names else arr.dtype.names[0]
    dec_name = arr.dtype.names[names.index("dec")] if "dec" in names else arr.dtype.names[1]
    ra = np.asarray(arr[ra_name], dtype=float)
    dec = np.asarray(arr[dec_name], dtype=float)
    m = np.isfinite(ra) & np.isfinite(dec)
    pos = np.column_stack([ra[m] % 360.0, np.clip(dec[m], -90.0, 90.0)])
    data_sources.append({"ok": True, "selected_table": selected, "n_positions": int(pos.shape[0])})
    return pos, data_sources


def maybe_extract_tar(tar_path: os.PathLike[str] | str, extract_dir: os.PathLike[str] | str) -> Path:
    tar_path = Path(tar_path)
    extract_dir = ensure_dir(extract_dir)
    marker = extract_dir / ".extract_complete"
    if marker.exists():
        return extract_dir
    with tarfile.open(tar_path, "r:*") as tf:
        def is_safe(member: tarfile.TarInfo) -> bool:
            target = (extract_dir / member.name).resolve()
            return str(target).startswith(str(extract_dir.resolve()))
        for m in tf.getmembers():
            if is_safe(m):
                tf.extract(m, extract_dir)
    marker.write_text(now_utc())
    return extract_dir


def download_act_dr6(cache: Path, *, force: bool = False, timeout: int = 2400) -> Tuple[Path, List[Dict[str, Any]]]:
    """Download/extract ACT DR6 lensing release and select a non-curl kappa ALM product."""
    data_sources: List[Dict[str, Any]] = []
    url = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/dr6_lensing_release.tar.gz"
    p = cache / "act_dr6" / "dr6_lensing_release.tar.gz"
    data_sources.append(download_file(url, p, force=force, timeout=timeout))
    extract_dir = cache / "act_dr6" / "dr6_lensing_release_extracted"
    maybe_extract_tar(p, extract_dir)
    candidates = list(extract_dir.rglob("*.fits")) + list(extract_dir.rglob("*.fits.gz"))
    if not candidates:
        raise RuntimeError("No FITS files found in ACT DR6 tarball")

    def score(path: Path) -> int:
        s = str(path).lower().replace("\\", "/")
        val = 0
        if "kappa" in s: val += 120
        if "alm" in s: val += 80
        if "data" in s: val += 40
        if "baseline" in s or "mv" in s: val += 20
        if "curl" in s: val -= 1000
        if "mask" in s or "noise" in s or "sim" in s or "random" in s: val -= 300
        return val

    ranked = sorted(candidates, key=lambda x: score(x), reverse=True)
    chosen = ranked[0]
    data_sources.append({
        "ok": True,
        "selected_map": str(chosen),
        "note": "selected ACT DR6 non-curl kappa FITS/ALM product for low-l fallback sampling",
        "selection_score": score(chosen),
        "top_candidates": [str(p) for p in ranked[:5]],
    })
    return chosen, data_sources


def download_planck_pr4_lensing(cache: Path, *, force: bool = False, timeout: int = 1200) -> Tuple[Path, List[Dict[str, Any]]]:
    """Download public Planck PR4 lensing ALM release from GitHub assets."""
    data_sources: List[Dict[str, Any]] = []
    url = "https://github.com/carronj/planck_PR4_lensing/releases/download/Data/PR42018like_maps.tar"
    p = cache / "planck_lensing" / "PR42018like_maps.tar"
    data_sources.append(download_file(url, p, force=force, timeout=timeout))
    extract_dir = cache / "planck_lensing" / "maps_extracted"
    maybe_extract_tar(p, extract_dir)
    candidates = list(extract_dir.rglob("*.fits")) + list(extract_dir.rglob("*.fits.gz"))
    if not candidates:
        raise RuntimeError("No FITS files found in Planck PR4 lensing tarball")

    def score(path: Path) -> int:
        s = str(path).lower().replace("\\", "/")
        val = 0
        if "klm" in s or "kappa" in s: val += 120
        if "dat" in s or "data" in s: val += 80
        if "mv" in s: val += 30
        if "curl" in s or "mf" in s or "noise" in s or "sim" in s: val -= 300
        return val

    ranked = sorted(candidates, key=lambda x: score(x), reverse=True)
    chosen = ranked[0]
    data_sources.append({
        "ok": True,
        "selected_map": str(chosen),
        "note": "selected Planck PR4 kappa/klm FITS/ALM product for low-l fallback sampling",
        "selection_score": score(chosen),
        "top_candidates": [str(p) for p in ranked[:5]],
    })
    return chosen, data_sources


def _import_fits():
    try:
        from astropy.io import fits  # type: ignore
        return fits
    except Exception as e:
        raise RuntimeError("astropy_required_for_fits_parsing: install with `python -m pip install astropy` or `conda install -c conda-forge astropy`") from e


@dataclass
class AlmData:
    l: np.ndarray
    m: np.ndarray
    alm: np.ndarray
    source_hdu: str
    source_columns: List[str]
    inferred_lmax: int


def infer_lmax_from_nalm(nalm: int) -> int:
    # nalm=(lmax+1)(lmax+2)/2
    val = int((math.sqrt(8 * nalm + 1) - 3) // 2)
    while (val + 1) * (val + 2) // 2 < nalm:
        val += 1
    while val > 0 and (val + 1) * (val + 2) // 2 > nalm:
        val -= 1
    return val


def healpy_order_lm(lmax: int) -> Tuple[np.ndarray, np.ndarray]:
    ls = []
    ms = []
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            ls.append(l)
            ms.append(m)
    return np.asarray(ls, dtype=int), np.asarray(ms, dtype=int)


def read_alm_fits(path: os.PathLike[str] | str, *, lmax_limit: int = 96) -> AlmData:
    fits = _import_fits()
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        best = None
        for hdu_i, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            cols = getattr(getattr(hdu, "columns", None), "names", None)
            if data is None or not cols:
                continue
            names = [c.upper() for c in cols]
            if any(n in names for n in ["REAL", "IMAG", "RE", "IM", "L", "M", "INDEX"]):
                best = (hdu_i, hdu, names, cols)
                break
        if best is None:
            raise RuntimeError("No recognizable ALM binary table HDU found in FITS file")
        hdu_i, hdu, names, cols_original = best
        data = hdu.data

        def col(*candidates: str):
            for cand in candidates:
                if cand.upper() in names:
                    return np.asarray(data[cols_original[names.index(cand.upper())]])
            return None

        real = col("REAL", "RE", "K_REAL", "T_REAL")
        imag = col("IMAG", "IM", "K_IMAG", "T_IMAG")
        if real is None:
            # Some FITS tables store one complex-valued column; use first complex column.
            for cname in cols_original:
                arr = np.asarray(data[cname])
                if np.iscomplexobj(arr):
                    real = np.real(arr)
                    imag = np.imag(arr)
                    break
        if real is None:
            # Last resort: first two numeric columns after optional index/l/m.
            numeric = []
            for cname in cols_original:
                arr = np.asarray(data[cname])
                if np.issubdtype(arr.dtype, np.number) and arr.ndim == 1:
                    numeric.append((cname, arr))
            filtered = [(c, a) for c, a in numeric if c.upper() not in ("INDEX", "L", "M")]
            if len(filtered) >= 2:
                real, imag = filtered[0][1], filtered[1][1]
            elif len(filtered) == 1:
                real, imag = filtered[0][1], np.zeros_like(filtered[0][1], dtype=float)
        if real is None:
            raise RuntimeError("Could not identify ALM real/imag columns")
        if imag is None:
            imag = np.zeros_like(real, dtype=float)
        real = np.asarray(real, dtype=float).reshape(-1)
        imag = np.asarray(imag, dtype=float).reshape(-1)
        nalm = min(len(real), len(imag))
        real, imag = real[:nalm], imag[:nalm]

        l_col = col("L", "ELL")
        m_col = col("M")
        if l_col is not None and m_col is not None:
            l_arr = np.asarray(l_col, dtype=int).reshape(-1)[:nalm]
            m_arr = np.asarray(m_col, dtype=int).reshape(-1)[:nalm]
            inferred_lmax = int(np.nanmax(l_arr))
        else:
            inferred_lmax = infer_lmax_from_nalm(nalm)
            l_arr, m_arr = healpy_order_lm(inferred_lmax)
            n = min(nalm, l_arr.size)
            l_arr, m_arr, real, imag = l_arr[:n], m_arr[:n], real[:n], imag[:n]

        mask = (l_arr <= int(lmax_limit)) & np.isfinite(real) & np.isfinite(imag)
        # remove monopole/dipole for lensing proxy; keep l>=2
        mask &= (l_arr >= 2)
        return AlmData(
            l=l_arr[mask].astype(int),
            m=m_arr[mask].astype(int),
            alm=(real[mask] + 1j * imag[mask]).astype(complex),
            source_hdu=f"HDU {hdu_i}",
            source_columns=[str(c) for c in cols_original],
            inferred_lmax=int(inferred_lmax),
        )


def sph_harm_y_vector(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Compute Y_lm(theta,phi) without scipy/healpy, vectorized over theta/phi.

    Uses Condon-Shortley phase convention. Good enough for low-l proxy maps.
    """
    x = np.cos(theta)
    # P_m^m
    pmm = np.ones_like(x, dtype=float)
    if m > 0:
        somx2 = np.sqrt(np.maximum(0.0, 1.0 - x * x))
        fact = 1.0
        for _ in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        pll = pmm
    else:
        pmmp1 = x * (2 * m + 1) * pmm
        if l == m + 1:
            pll = pmmp1
        else:
            p_lm2 = pmm
            p_lm1 = pmmp1
            pll = pmmp1
            for ll in range(m + 2, l + 1):
                pll = ((2 * ll - 1) * x * p_lm1 - (ll + m - 1) * p_lm2) / (ll - m)
                p_lm2, p_lm1 = p_lm1, pll
    log_norm = 0.5 * (math.log((2 * l + 1) / (4 * math.pi)) + math.lgamma(l - m + 1) - math.lgamma(l + m + 1))
    norm = math.exp(log_norm)
    return norm * pll * np.exp(1j * m * phi)


def sample_alm_at_radec(path: os.PathLike[str] | str, radec_deg: np.ndarray, *, lmax_limit: int = 64) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample an ALM FITS product at RA/DEC using low-l fallback projection."""
    alm = read_alm_fits(path, lmax_limit=lmax_limit)
    if alm.alm.size == 0:
        raise RuntimeError("No ALM coefficients left after lmax/l>=2 filtering")
    ra = np.deg2rad(np.asarray(radec_deg[:, 0], dtype=float) % 360.0)
    dec = np.deg2rad(np.asarray(radec_deg[:, 1], dtype=float))
    theta = np.pi / 2.0 - dec
    phi = ra
    vals = np.zeros_like(theta, dtype=float)
    # Group by (l,m) to keep memory low.
    for l, m, a in zip(alm.l, alm.m, alm.alm):
        y = sph_harm_y_vector(int(l), int(m), theta, phi)
        if m == 0:
            vals += np.real(a * y)
        else:
            vals += 2.0 * np.real(a * y)
    meta = {
        "mode": "low_l_alm_projection_no_healpy",
        "path": str(path),
        "alm_lmax_used": int(max(alm.l) if alm.l.size else 0),
        "alm_lmax_file_inferred": int(alm.inferred_lmax),
        "n_alm_used": int(alm.alm.size),
        "source_hdu": alm.source_hdu,
        "source_columns_first12": alm.source_columns[:12],
        "warning": "Low-l ALM fallback is a sign/proxy sampler, not full-resolution HEALPix map sampling.",
    }
    return vals, meta


def local_density_knn(radec_deg: np.ndarray, *, k: int = 16) -> np.ndarray:
    ra = np.deg2rad(radec_deg[:, 0])
    dec = np.deg2rad(radec_deg[:, 1])
    x = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(x)
        d, _ = tree.query(x, k=min(k + 1, len(x)))
        rk = d[:, -1]
    except Exception:
        # O(N^2) fallback, used only for small samples.
        n = len(x)
        if n > 2500:
            idx = np.linspace(0, n - 1, 2500).astype(int)
            x = x[idx]
            radec_deg = radec_deg[idx]
            n = len(x)
        dist = np.sqrt(np.maximum(0.0, ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)))
        dist.sort(axis=1)
        rk = dist[:, min(k, n - 1)]
    rk = np.maximum(rk, np.nanpercentile(rk, 1) * 1e-3 + 1e-12)
    return k / (math.pi * rk * rk)


def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    vals = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x = x[m] - np.mean(x[m])
    y = y[m] - np.mean(y[m])
    den = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / den) if den > 0 else float("nan")


def spearman_np(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 4:
        return {"n": n, "rho": None, "pvalue_approx": None}
    rho = pearsonr_np(rankdata(x[m]), rankdata(y[m]))
    # Normal approximation for large n; sufficient for screening.
    if not np.isfinite(rho):
        p = None
    else:
        z = abs(rho) * math.sqrt(max(n - 3, 1))
        # two-sided normal tail via erfc
        p = math.erfc(z / math.sqrt(2.0))
    return {"n": n, "rho": float(rho), "pvalue_approx": p}


def density_kappa_metrics(radec: np.ndarray, kappa: np.ndarray, *, rng_seed: int = 12345, n_null: int = 200) -> Dict[str, Any]:
    density = local_density_knn(radec, k=16)
    m = np.isfinite(density) & np.isfinite(kappa)
    density = density[m]
    kappa = np.asarray(kappa, dtype=float)[m]
    radec = radec[m]
    if len(kappa) < 20:
        return {"error": "too_few_valid_samples", "n": int(len(kappa))}
    split = np.nanmedian(density)
    hi = density >= split
    lo = density < split
    delta = float(np.nanmean(kappa[hi]) - np.nanmean(kappa[lo]))
    rng = np.random.default_rng(rng_seed)
    null = []
    for _ in range(int(n_null)):
        dsh = density.copy()
        rng.shuffle(dsh)
        his = dsh >= split
        los = ~his
        null.append(float(np.nanmean(kappa[his]) - np.nanmean(kappa[los])))
    null = np.asarray(null, dtype=float)
    z = (delta - float(np.nanmean(null))) / (float(np.nanstd(null, ddof=1)) + 1e-30)
    return {
        "n_samples": int(len(kappa)),
        "density_median": float(split),
        "kappa_mean_high_density": float(np.nanmean(kappa[hi])),
        "kappa_mean_low_density": float(np.nanmean(kappa[lo])),
        "delta_kappa_high_minus_low": delta,
        "pearson_density_kappa": pearsonr_np(density, kappa),
        "spearman_density_kappa": spearman_np(density, kappa),
        "density_shuffle_null": {
            "n": int(n_null),
            "mean": float(np.nanmean(null)),
            "sigma": float(np.nanstd(null, ddof=1)),
            "z_vs_null": float(z),
            "p_one_sided_high_gt_low": float((np.sum(null >= delta) + 1) / (len(null) + 1)),
        },
    }


def sexa_to_deg(s: str, *, is_ra: bool) -> Optional[float]:
    s = s.strip()
    if not s or s == "*":
        return None
    # Remove uncertainty suffixes and quotes.
    s = s.split()[0].strip().replace("h", ":").replace("m", ":").replace("s", "")
    sign = -1.0 if s.startswith("-") else 1.0
    s2 = s.lstrip("+-")
    parts = re.split(r"[: ]+", s2)
    try:
        nums = [float(p) for p in parts if p != ""]
    except ValueError:
        return None
    if not nums:
        return None
    val = nums[0] + (nums[1] / 60.0 if len(nums) > 1 else 0.0) + (nums[2] / 3600.0 if len(nums) > 2 else 0.0)
    if is_ra:
        return (val * 15.0) % 360.0
    return sign * val


def parse_psrj_position(name: str) -> Optional[Tuple[float, float]]:
    # JHHMM+DDMM or JHHMMSS.s-DDMMSS; approximate if only HHMM/DDMM.
    m = re.search(r"J(\d{2})(\d{2})(\d{0,2}(?:\.\d+)?)?([+\-])(\d{2})(\d{2})(\d{0,2}(?:\.\d+)?)?", name.upper())
    if not m:
        return None
    hh = float(m.group(1)); mm = float(m.group(2)); ss = float(m.group(3) or 0.0)
    sign = -1.0 if m.group(4) == "-" else 1.0
    dd = float(m.group(5)); dm = float(m.group(6)); ds = float(m.group(7) or 0.0)
    ra = 15.0 * (hh + mm / 60.0 + ss / 3600.0)
    dec = sign * (dd + dm / 60.0 + ds / 3600.0)
    return ra % 360.0, dec


def parse_nanograv_par_positions(cache: Path, *, force: bool = False, timeout: int = 1800) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    data_sources: List[Dict[str, Any]] = []
    url = "https://zenodo.org/api/records/16051178/files/NANOGrav15yr_PulsarTiming_v2.1.0.tar.gz/content"
    p = cache / "zenodo_16051178" / "NANOGrav15yr_PulsarTiming_v2.1.0.tar.gz"
    data_sources.append(download_file(url, p, force=force, timeout=timeout))
    positions = []
    n_par = 0
    used_raj = 0
    used_psrj = 0
    with tarfile.open(p, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".par"):
                continue
            n_par += 1
            fh = tf.extractfile(member)
            if fh is None:
                continue
            txt = fh.read().decode("utf-8", errors="replace")
            vals = {}
            for line in txt.splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    vals[parts[0].upper()] = parts[1]
            ra = sexa_to_deg(vals.get("RAJ", ""), is_ra=True) if "RAJ" in vals else None
            dec = sexa_to_deg(vals.get("DECJ", ""), is_ra=False) if "DECJ" in vals else None
            if ra is not None and dec is not None:
                positions.append((ra, dec))
                used_raj += 1
                continue
            for key in ("PSRJ", "PSR", "PSRB"):
                if key in vals:
                    got = parse_psrj_position(vals[key])
                    if got is not None:
                        positions.append(got)
                        used_psrj += 1
                        break
    arr = np.asarray(positions, dtype=float)
    meta = {"n_par_files_scanned": int(n_par), "n_positions": int(arr.shape[0]), "used_raj_decj": int(used_raj), "used_psrj_fallback": int(used_psrj)}
    return arr, meta, data_sources


def rotation_null_for_positions(radec: np.ndarray, values: np.ndarray, *, n_null: int = 500, rng_seed: int = 6789) -> Dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    values = np.asarray(values, dtype=float)
    obs = float(np.nanmean(values))
    null = []
    # preserve declination distribution; randomize RA as a conservative sky-rotation proxy
    for _ in range(int(n_null)):
        null.append(float(np.nanmean(rng.permutation(values))))
    # Pure permutation of sampled values has same mean; use sign/asymmetry metric too.
    # Better: compare mean positive fraction to random sign flips.
    sign_obs = float(np.nanmean(values > np.nanmedian(values)))
    sign_null = []
    centered = values - np.nanmedian(values)
    for _ in range(int(n_null)):
        flips = rng.choice([-1.0, 1.0], size=len(centered))
        sign_null.append(float(np.nanmean(centered * flips > 0)))
    sign_null = np.asarray(sign_null)
    z = (sign_obs - float(np.mean(sign_null))) / (float(np.std(sign_null, ddof=1)) + 1e-30)
    return {
        "n_positions": int(len(values)),
        "mean_kappa_at_pulsars": obs,
        "median_kappa_at_pulsars": float(np.nanmedian(values)),
        "std_kappa_at_pulsars": float(np.nanstd(values, ddof=1)) if len(values) > 1 else None,
        "sign_balance_vs_random_sign_null": {
            "observed_fraction_above_median": sign_obs,
            "null_mean": float(np.mean(sign_null)),
            "null_sigma": float(np.std(sign_null, ddof=1)),
            "z": float(z),
        },
        "note": "Position-only kappa sampler. Without a PTA residual/WRMS vector this is a diagnostic sky-position cross-link, not a full CL2 timing-residual correlation.",
    }


def discover_planck_spectrum_urls() -> List[str]:
    base = "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/"
    urls: List[str] = []
    seen = set()
    try:
        root = fetch_text(base)
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', root, flags=re.I)
    except Exception:
        hrefs = []
    subdirs = [base]
    for h in hrefs:
        h = html.unescape(h)
        if h.startswith("?") or h.startswith("../"):
            continue
        u = urllib.parse.urljoin(base, h)
        if u.endswith("/") and len(subdirs) < 10:
            subdirs.append(u)
        elif re.search(r"PowerSpect.*R3.*\.txt$", u, flags=re.I):
            urls.append(u)
    for d in subdirs:
        try:
            txt = fetch_text(d)
        except Exception:
            continue
        for h in re.findall(r'href=["\']([^"\']+)["\']', txt, flags=re.I):
            h = html.unescape(h)
            u = urllib.parse.urljoin(d, h)
            if re.search(r"PowerSpect.*R3.*\.txt$", u, flags=re.I) and u not in seen:
                urls.append(u); seen.add(u)
    # Known stable candidates; included if directory scraping changes.
    known = [
        base + "cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt",
        base + "cmb_derived_products/COM_PowerSpect_CMB-TT-full_R3.01.txt",
        base + "cmb_derived_products/COM_PowerSpect_CMB-TT-binned_R3.01.txt",
        base + "cmb_derived_products/COM_PowerSpect_CMB-TT-lowT_R3.01.txt",
    ]
    for u in known:
        if u not in urls:
            urls.append(u)
    return urls


def read_numeric_rows(path: os.PathLike[str] | str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            st = line.strip()
            if not st or st.startswith("#") or st.startswith(";"):
                continue
            st = st.replace(",", " ")
            vals = []
            ok = True
            for tok in st.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    ok = False
                    break
            if ok and len(vals) >= 2:
                rows.append(vals)
    if not rows:
        return np.empty((0, 0))
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def download_and_parse_planck_spectra(cache: Path, *, force: bool = False, timeout: int = 240) -> Tuple[List[Tuple[str, Path, np.ndarray]], List[Dict[str, Any]]]:
    data_sources: List[Dict[str, Any]] = []
    parsed = []
    ensure_dir(cache / "planck_spectra")
    urls = discover_planck_spectrum_urls()
    data_sources.append({"ok": True, "n_candidate_urls": len(urls), "note": "Planck PR3 PowerSpect URL discovery, plus known fallbacks"})
    for u in urls:
        name = safe_name(url_basename(u))
        p = cache / "planck_spectra" / name
        try:
            ds = download_file(u, p, force=force, timeout=timeout)
            arr = read_numeric_rows(p)
            ds["n_numeric_rows"] = int(arr.shape[0])
            ds["n_numeric_cols"] = int(arr.shape[1]) if arr.ndim == 2 else 0
            data_sources.append(ds)
            if arr.shape[0] >= 20 and arr.shape[1] >= 2:
                parsed.append((u, p, arr))
        except Exception as e:
            data_sources.append({"ok": False, "url": u, "error": repr(e)})
    return parsed, data_sources


def planck_large_angle_metrics(parsed: List[Tuple[str, Path, np.ndarray]]) -> Dict[str, Any]:
    # Prefer observed TT full/binned table if available; otherwise use best-fit model table.
    candidates = []
    for u, p, arr in parsed:
        lname = str(p).lower() + " " + u.lower()
        score = 0
        if "tt" in lname: score += 100
        if "full" in lname or "binned" in lname or "lowt" in lname: score += 40
        if "base-plik" in lname or "cosmoparams" in lname: score += 10
        if "te" in lname or "ee" in lname or "bb" in lname: score -= 20
        ell = arr[:, 0]
        if np.nanmin(ell) <= 2 and np.nanmax(ell) >= 100:
            score += 30
        candidates.append((score, u, p, arr))
    if not candidates:
        return {"error": "no_parseable_planck_spectrum"}
    candidates.sort(reverse=True, key=lambda x: x[0])
    score, u, p, arr = candidates[0]
    ell = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)
    m = np.isfinite(ell) & np.isfinite(y) & (ell >= 2) & (y != 0)
    ell, y = ell[m], y[m]
    order = np.argsort(ell)
    ell, y = ell[order], y[order]
    low = (ell >= 2) & (ell <= 29)
    high = (ell >= 30) & (ell <= min(250, np.nanmax(ell)))
    if low.sum() < 3 or high.sum() < 10:
        return {"error": "insufficient_low_or_high_ell_rows", "selected_url": u, "n_low": int(low.sum()), "n_high": int(high.sum())}
    # Fit smooth continuation in log D_l from high-l range and extrapolate to low-l.
    # This is a no-map proxy for the large-angle TT suppression, not a C_l likelihood.
    xh = np.log(np.maximum(ell[high], 1.0))
    yh = np.log(np.maximum(np.abs(y[high]), np.nanpercentile(np.abs(y[high]), 5) * 1e-3 + 1e-30))
    deg = min(3, max(1, high.sum() // 30))
    coeff = np.polyfit(xh, yh, deg=deg)
    pred_low = np.exp(np.polyval(coeff, np.log(ell[low])))
    obs_low = y[low]
    ratio = obs_low / pred_low
    residual = ratio - 1.0
    mean_resid = float(np.nanmean(residual))
    std_resid = float(np.nanstd(residual, ddof=1)) if low.sum() > 1 else float("nan")
    z = mean_resid / (std_resid / math.sqrt(int(low.sum())) + 1e-30)
    # Quadrupole/octopole proxy.
    qmask = (ell >= 2) & (ell <= 3)
    q_ratio = float(np.nanmean(y[qmask] / np.exp(np.polyval(coeff, np.log(ell[qmask]))))) if qmask.sum() else float("nan")
    return {
        "selected_spectrum_url": u,
        "selected_spectrum_path": str(p),
        "selected_score": int(score),
        "n_rows_used": int(len(ell)),
        "n_low_ell_2_29": int(low.sum()),
        "n_high_fit_ell_30_250": int(high.sum()),
        "fit_degree_loglog": int(deg),
        "low_ell_mean_obs_over_smooth": float(np.nanmean(ratio)),
        "low_ell_mean_residual_fraction": mean_resid,
        "low_ell_residual_z_proxy": float(z),
        "quadrupole_octopole_obs_over_smooth": q_ratio,
        "interpretation_note": "No-map Planck TT spectrum proxy. Confirms/denies only table-level low-l suppression relative to a smooth high-l continuation; not an official low-l likelihood.",
    }


def status_from_density_kappa(metrics: Dict[str, Any]) -> str:
    if "error" in metrics:
        return "data_limited"
    sp = metrics.get("spearman_density_kappa", {})
    rho = sp.get("rho")
    p = sp.get("pvalue_approx")
    delta = metrics.get("delta_kappa_high_minus_low")
    z = metrics.get("density_shuffle_null", {}).get("z_vs_null")
    if rho is not None and delta is not None and rho > 0 and delta > 0:
        if p is not None and p < 0.01 and z is not None and z > 2:
            return "suggestive"
        return "weak_positive"
    if rho is not None and delta is not None and (rho < 0 or delta < 0):
        return "null"
    return "diagnostic_only"


def status_from_t22(metrics: Dict[str, Any]) -> str:
    if "error" in metrics:
        return "data_limited"
    resid = metrics.get("low_ell_mean_residual_fraction")
    z = metrics.get("low_ell_residual_z_proxy")
    if resid is None or z is None:
        return "diagnostic_only"
    # Negative low-l residual is the known large-angle-suppression direction.
    if resid < -0.10 and abs(z) > 2.0:
        return "suggestive"
    if abs(z) < 2.0:
        return "null"
    return "diagnostic_only"
