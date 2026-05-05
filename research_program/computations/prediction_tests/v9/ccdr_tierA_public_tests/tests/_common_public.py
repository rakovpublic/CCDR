#!/usr/bin/env python3
"""Common helpers for CCDR Tier-A public-data tests.

All network access uses public HTTP endpoints. The helpers are intentionally
conservative: failed downloads become structured JSON warnings instead of
hard crashes whenever possible.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from scipy import optimize, stats, signal
except Exception:  # pragma: no cover
    optimize = None
    stats = None
    signal = None

USER_AGENT = "ccdr-tierA-public-tests/1.0 (+https://github.com/rakovpublic/CCDR)"
C_LIGHT = 299792.458  # km/s
MPC_M = 3.085677581491367e22
KPC_M = 3.085677581491367e19
G_SI = 6.67430e-11
MSUN_KG = 1.98847e30
KSS_ETA_OVER_S = 1.054571817e-34 / (4.0 * math.pi * 1.380649e-23)


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--outdir", default="out", help="Output directory for JSON results")
    p.add_argument("--cache", default=".cache", help="Download cache directory")
    p.add_argument("--max-rows", type=int, default=20000, help="Maximum rows to use where supported")
    p.add_argument("--allow-large", action="store_true", help="Allow large public downloads")
    p.add_argument("--force", action="store_true", help="Redownload files even if cached")
    p.add_argument("--timeout", type=int, default=90, help="Network timeout in seconds")
    p.add_argument("--prefer-healpy", action="store_true", help="Prefer healpy for HEALPix/pixel-map sampling when installed")
    p.add_argument("--no-harmonic", action="store_true", help="Do not reconstruct alm/klm harmonic products; return data_limited if no pixel map exists")
    p.add_argument("--seed", type=int, default=12345)
    return p


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify(text: str, max_len: int = 180) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return s[:max_len] or "download"


def url_basename(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    return name or slugify(url)


def http_get_bytes(url: str, timeout: int = 90, retries: int = 2) -> bytes:
    last = None
    for i in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def http_get_text(url: str, timeout: int = 90, retries: int = 2) -> str:
    data = http_get_bytes(url, timeout=timeout, retries=retries)
    for enc in ("utf-8", "latin1"):
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("utf-8", errors="replace")


def download_file(url: str, cache: Path, *, filename: Optional[str] = None, timeout: int = 90, force: bool = False, max_bytes: Optional[int] = None) -> Path:
    ensure_dir(cache)
    name = filename or url_basename(url)
    path = cache / slugify(name)
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    req = Request(url, headers={"User-Agent": USER_AGENT})
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise RuntimeError(f"refusing to download {url}: exceeds max_bytes={max_bytes}")
                f.write(chunk)
        tmp.replace(path)
        return path
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def download_first(urls: Sequence[str], cache: Path, *, timeout: int = 90, force: bool = False, max_bytes: Optional[int] = None) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    attempts = []
    for url in urls:
        try:
            p = download_file(url, cache, timeout=timeout, force=force, max_bytes=max_bytes)
            attempts.append({"url": url, "ok": True, "path": str(p), "size": p.stat().st_size})
            return p, attempts
        except Exception as e:
            attempts.append({"url": url, "ok": False, "error": str(e)})
    return None, attempts


def discover_links(url: str, pattern: Optional[str] = None, timeout: int = 90) -> List[str]:
    html = http_get_text(url, timeout=timeout)
    hrefs = re.findall(r'''href=["']([^"']+)["']''', html, flags=re.I)
    out = []
    for h in hrefs:
        full = urljoin(url, h)
        if pattern is None or re.search(pattern, full, flags=re.I):
            out.append(full)
    # stable unique
    seen = set(); uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq


def zenodo_files(record_id: str, timeout: int = 90) -> List[Dict[str, Any]]:
    url = f"https://zenodo.org/api/records/{record_id}"
    obj = json.loads(http_get_text(url, timeout=timeout))
    files = []
    for f in obj.get("files", []):
        link = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        files.append({"key": f.get("key") or f.get("filename"), "size": f.get("size"), "url": link})
    return files


def download_zenodo_matching(record_id: str, cache: Path, patterns: Sequence[str], *, timeout: int = 90, force: bool = False, allow_large: bool = False, max_default_mb: int = 350) -> Tuple[List[Path], List[Dict[str, Any]]]:
    attempts = []
    try:
        files = zenodo_files(record_id, timeout=timeout)
    except Exception as e:
        return [], [{"record_id": record_id, "ok": False, "error": str(e)}]
    out = []
    for f in files:
        key = f.get("key") or ""
        url = f.get("url")
        if not url:
            continue
        if patterns and not any(re.search(p, key, flags=re.I) for p in patterns):
            continue
        size = f.get("size") or 0
        if size and size > max_default_mb * 1024 * 1024 and not allow_large:
            attempts.append({"record_id": record_id, "file": key, "url": url, "ok": False, "reason": "large_download_not_enabled", "size": size})
            continue
        try:
            p = download_file(url, cache / f"zenodo_{record_id}", filename=key, timeout=timeout, force=force)
            attempts.append({"record_id": record_id, "file": key, "url": url, "ok": True, "path": str(p), "size": p.stat().st_size})
            out.append(p)
        except Exception as e:
            attempts.append({"record_id": record_id, "file": key, "url": url, "ok": False, "error": str(e)})
    return out, attempts


def github_repo_file_list(owner: str, repo: str, branch: str = "main", timeout: int = 90) -> List[str]:
    api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    obj = json.loads(http_get_text(api, timeout=timeout))
    return [x["path"] for x in obj.get("tree", []) if x.get("type") == "blob"]


def github_raw_url(owner: str, repo: str, path: str, branch: str = "main") -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{quote(path)}".replace("%2F", "/")


def github_download_matching(owner: str, repo: str, cache: Path, patterns: Sequence[str], branch: str = "main", *, timeout: int = 90, force: bool = False, max_files: int = 20) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """Download matching files from a public GitHub repository.

    Data-release repositories are inconsistent about branch names.  Try the
    requested branch, then `master`, then `main`; record every attempt.
    """
    attempts: List[Dict[str, Any]] = []
    branches = []
    for b in [branch, "master", "main"]:
        if b and b not in branches:
            branches.append(b)

    files = None
    used_branch = None
    failed_branches: List[Dict[str, Any]] = []
    for b in branches:
        try:
            files = github_repo_file_list(owner, repo, branch=b, timeout=timeout)
            used_branch = b
            attempts.append({
                "github": f"{owner}/{repo}",
                "ok": True,
                "used_branch": b,
                "tried_branches": branches,
                "n_files": len(files),
                "note": "GitHub branch autodetection succeeded; earlier branch misses are expected and are not data failures." if failed_branches else "GitHub branch autodetection succeeded."
            })
            break
        except Exception as e:
            # Do not expose expected branch-name misses (for example main -> master)
            # as failed data sources if a later branch succeeds. They confused the
            # run log with harmless HTTP 404 records. Keep them only if every
            # branch fails.
            failed_branches.append({"branch": b, "error": str(e)})
    if files is None or used_branch is None:
        attempts.append({"github": f"{owner}/{repo}", "ok": False, "tried_branches": branches, "errors": failed_branches})
        return [], attempts

    chosen = [p for p in files if any(re.search(pat, p, re.I) for pat in patterns)][:max_files]
    if not chosen:
        attempts.append({"github": f"{owner}/{repo}", "branch": used_branch, "ok": False, "error": "no_files_matched_patterns", "patterns": list(patterns)})
    out = []
    for pth in chosen:
        url = github_raw_url(owner, repo, pth, branch=used_branch)
        fname = slugify(str(Path(pth).with_suffix("").as_posix()).replace("/", "__")) + Path(pth).suffix
        try:
            p = download_file(url, cache / f"github_{owner}_{repo}", filename=fname, timeout=timeout, force=force)
            attempts.append({"url": url, "path": str(p), "ok": True})
            out.append(p)
        except Exception as e:
            attempts.append({"url": url, "ok": False, "error": str(e)})
    return out, attempts


def extract_archive(path: Path, outdir: Path) -> Path:
    ensure_dir(outdir)
    marker = outdir / ".extracted"
    if marker.exists():
        return outdir
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(outdir)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            def safe_members(members):
                root = outdir.resolve()
                for m in members:
                    dest = (outdir / m.name).resolve()
                    if not str(dest).startswith(str(root)):
                        raise RuntimeError("unsafe archive path")
                    yield m
            t.extractall(outdir, members=safe_members(t.getmembers()))
    else:
        raise RuntimeError(f"not a supported archive: {path}")
    marker.write_text("ok")
    return outdir


def read_table_any(path: Path, *, max_rows: Optional[int] = None) -> Optional[Any]:
    """Return pandas DataFrame when possible, otherwise None."""
    if pd is None:
        return None
    suffix = path.suffix.lower()
    try:
        if suffix in (".csv",):
            # HEPData CSV files often include comment metadata lines starting
            # with '#:' before the real table.
            for kwargs in [dict(comment="#"), dict(comment="#", header=None), dict()]:
                try:
                    df = pd.read_csv(path, nrows=max_rows, **kwargs)
                    if df.shape[1] >= 1:
                        return df
                except Exception:
                    pass
            return None
        if suffix in (".tsv", ".tab"):
            return pd.read_csv(path, sep="\t", comment="#", nrows=max_rows)
        if suffix in (".txt", ".dat", ".mrt", ".data", ""):
            # Try header-aware parsing first, but many astronomy catalogues are
            # pure numeric whitespace tables.  If pandas treats the first
            # numeric row as a header, re-read with header=None.
            attempts = [
                dict(sep=r"\s+", comment="#"),
                dict(sep=r"\s+", comment="#", header=None),
                dict(sep=","),
                dict(sep=",", header=None),
                dict(sep="\t", comment="#"),
                dict(sep="\t", comment="#", header=None),
            ]
            for kwargs in attempts:
                try:
                    df = pd.read_csv(path, nrows=max_rows, engine="python", **kwargs)
                    if df.shape[1] >= 2 and df.shape[0] >= 1:
                        if "header" not in kwargs:
                            numeric_names = 0
                            for c in df.columns:
                                try:
                                    float(str(c)); numeric_names += 1
                                except Exception:
                                    pass
                            if numeric_names >= max(2, int(0.6 * len(df.columns))):
                                continue
                        return df
                except Exception:
                    pass
            return None
        if suffix in (".fits", ".fit", ".fits.gz"):
            try:
                from astropy.table import Table
                tbl = Table.read(path)
                df = tbl.to_pandas()
                if max_rows:
                    df = df.head(max_rows)
                return df
            except Exception:
                return None
    except Exception:
        return None
    return None


def find_numeric_columns(df: Any) -> List[str]:
    if pd is None or df is None:
        return []
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(s).sum() >= max(3, int(0.1 * len(s))):
            cols.append(c)
    return cols


def numeric_array(df: Any, col: str) -> np.ndarray:
    if pd is None:
        return np.array([])
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def safe_corr(x: Sequence[float], y: Sequence[float], method: str = "spearman") -> Dict[str, Any]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 4 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return {"n": int(len(x)), "rho": None, "pvalue": None, "error": "too_few_or_constant"}
    if stats is not None:
        if method == "pearson":
            r, p = stats.pearsonr(x, y)
        else:
            r, p = stats.spearmanr(x, y)
        return {"n": int(len(x)), "rho": float(r), "pvalue": float(p)}
    # fallback rank corr
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    r = float(np.corrcoef(rx, ry)[0, 1])
    return {"n": int(len(x)), "rho": r, "pvalue": None}


def bootstrap_mean_delta(a: Sequence[float], b: Sequence[float], nboot: int = 1000, seed: int = 12345) -> Dict[str, Any]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {"delta": None, "z": None, "error": "too_few"}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nboot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        vals.append(float(np.mean(aa) - np.mean(bb)))
    vals = np.asarray(vals)
    delta = float(np.mean(a) - np.mean(b))
    sd = float(np.std(vals, ddof=1))
    return {"delta": delta, "boot_sigma": sd, "z": delta / sd if sd > 0 else None, "n_a": int(len(a)), "n_b": int(len(b))}


def angular_separation_deg(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(ra1); dec1 = np.deg2rad(dec1); ra2 = np.deg2rad(ra2); dec2 = np.deg2rad(dec2)
    s = np.sin((dec2-dec1)/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin((ra2-ra1)/2)**2
    return np.rad2deg(2*np.arcsin(np.minimum(1, np.sqrt(s))))


def nearest_density(ra: np.ndarray, dec: np.ndarray, k: int = 10) -> np.ndarray:
    n = len(ra)
    if n < k + 2:
        return np.full(n, np.nan)
    out = np.empty(n)
    for i in range(n):
        d = angular_separation_deg(ra[i], dec[i], ra, dec)
        d.sort()
        rk = max(d[min(k, n-1)], 1e-6)
        out[i] = k / (math.pi * rk * rk)
    return out


def local_orientation_angles(ra: np.ndarray, dec: np.ndarray, k: int = 12) -> np.ndarray:
    n = len(ra)
    ang = np.full(n, np.nan)
    if n < k + 2:
        return ang
    x = np.asarray(ra, float) * np.cos(np.deg2rad(np.nanmedian(dec)))
    y = np.asarray(dec, float)
    for i in range(n):
        dx = x - x[i]; dy = y - y[i]
        d2 = dx*dx + dy*dy
        idx = np.argsort(d2)[1:k+1]
        pts = np.vstack([dx[idx], dy[idx]]).T
        if len(pts) < 3:
            continue
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, np.argmax(vals)]
        ang[i] = math.atan2(v[1], v[0])
    return ang


def orientation_correlation(ra: np.ndarray, dec: np.ndarray, angles: np.ndarray, bins: Sequence[float]) -> List[Dict[str, Any]]:
    rows = []
    n = len(ra)
    for lo, hi in zip(bins[:-1], bins[1:]):
        vals = []
        # sample pairs if large
        max_pairs = 200000
        rng = np.random.default_rng(123)
        if n*(n-1)//2 > max_pairs:
            pairs = zip(rng.integers(0, n, max_pairs), rng.integers(0, n, max_pairs))
        else:
            pairs = ((i, j) for i in range(n) for j in range(i+1, n))
        for i, j in pairs:
            if i == j or not np.isfinite(angles[i]) or not np.isfinite(angles[j]):
                continue
            d = angular_separation_deg(ra[i], dec[i], ra[j], dec[j])
            if lo <= d < hi:
                vals.append(math.cos(2*(angles[i] - angles[j])))
        rows.append({"bin_lo_deg": float(lo), "bin_hi_deg": float(hi), "n_pairs": int(len(vals)), "corr": float(np.mean(vals)) if vals else None})
    return rows


def mad(x: Sequence[float]) -> float:
    a = np.asarray(x, float); a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    med = np.median(a)
    return float(np.median(np.abs(a-med)))


def result_template(test_id: str, prediction_ids: Sequence[str], description: str) -> Dict[str, Any]:
    return {
        "test_id": test_id,
        "prediction_ids": list(prediction_ids),
        "description": description,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "data_limited",
        "data_sources": [],
        "metrics": {},
        "falsification_logic": {},
        "warnings": [],
        "notes": [],
    }


def write_result(res: Dict[str, Any], outdir: Path, filename: Optional[str] = None) -> Path:
    ensure_dir(outdir)
    path = outdir / (filename or f"{res.get('test_id','result')}.json")
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            if np.isnan(o): return None
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)
    path.write_text(json.dumps(res, indent=2, sort_keys=True, default=conv), encoding="utf-8")
    print(json.dumps(res, indent=2, sort_keys=True, default=conv))
    return path


def classify_by_sign(value: Optional[float], *, positive_confirm: bool = True, sigma: Optional[float] = None, confirm_sigma: float = 2.0) -> str:
    if value is None or not np.isfinite(value):
        return "data_limited"
    sign_ok = value > 0 if positive_confirm else value < 0
    if sigma is not None and np.isfinite(sigma):
        if sign_ok and abs(value / sigma) >= confirm_sigma:
            return "confirm_like"
        if sign_ok:
            return "suggestive"
        return "null"
    return "suggestive" if sign_ok else "null"


def safe_polyfit(x, y, deg=1):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < deg + 2:
        return None
    try:
        coef = np.polyfit(x[m], y[m], deg)
        return [float(c) for c in coef]
    except Exception:
        return None


def simple_kurtosis(x: Sequence[float]) -> Optional[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return None
    if stats is not None:
        return float(stats.kurtosis(x, fisher=False, bias=False))
    mu = np.mean(x); sd = np.std(x)
    if sd <= 0:
        return None
    return float(np.mean(((x-mu)/sd)**4))


def load_pantheon(cache: Path, timeout: int = 90, force: bool = False, max_rows: Optional[int] = None) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    urls = [
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",
    ]
    p, attempts = download_first(urls, cache / "pantheon", timeout=timeout, force=force)
    if not p:
        return None, attempts
    if pd is None:
        return None, attempts + [{"ok": False, "error": "pandas_missing"}]
    # Pantheon+ is whitespace-delimited with header.
    for sep in [r"\s+", ",", "\t"]:
        try:
            df = pd.read_csv(p, sep=sep, comment="#", engine="python", nrows=max_rows)
            if df.shape[0] > 10 and df.shape[1] > 4:
                return df, attempts
        except Exception:
            pass
    return None, attempts + [{"ok": False, "error": "could_not_parse_pantheon"}]


def fit_pantheon_nu_like(df: Any, z_col: Optional[str] = None, mu_col: Optional[str] = None, err_col: Optional[str] = None) -> Dict[str, Any]:
    """Lightweight SN-only ν-like audit.

    Model: mu = 5 log10(DL(Om,nu)) + intercept, with
    E(z)^2 = Om(1+z)^3 + (1-Om)*(1 + nu*log(1+z)).
    This is an audit proxy, not a replacement for full RVM chains.
    """
    if pd is None or df is None:
        return {"error": "no_dataframe"}
    cols = {c.lower(): c for c in df.columns}
    zc = z_col or cols.get("zhel") or cols.get("zcmb") or cols.get("z_hd") or cols.get("z")
    mc = mu_col or cols.get("mu_sh0es") or cols.get("mu") or cols.get("m_b_corr") or cols.get("mb_corr")
    ec = err_col or cols.get("mu_sh0es_err_diag") or cols.get("muerr") or cols.get("dmu") or cols.get("err")
    if zc is None or mc is None:
        return {"error": "missing_z_or_mu", "columns": list(df.columns)[:40]}
    z = numeric_array(df, zc); mu = numeric_array(df, mc)
    if ec:
        err = numeric_array(df, ec)
    else:
        err = np.full_like(z, np.nanmedian(np.abs(mu - np.nanmedian(mu))) or 0.15)
    m = np.isfinite(z) & np.isfinite(mu) & (z > 0.005) & (z < 2.5)
    z = z[m]; mu = mu[m]; err = err[m]
    err = np.where(np.isfinite(err) & (err > 0), err, np.nanmedian(err[np.isfinite(err) & (err > 0)]) if np.any(np.isfinite(err)&(err>0)) else 0.15)
    if len(z) < 20:
        return {"error": "too_few_sne", "n": int(len(z))}

    order = np.argsort(z); z = z[order]; mu = mu[order]; err = err[order]
    # precompute integration grid per z, simple cumulative trapezoid over sorted unique z
    def mu_model(om, nu):
        zz = np.r_[0.0, z]
        e2 = om*(1+zz)**3 + (1-om)*(1 + nu*np.log1p(zz))
        if np.any(e2 <= 0):
            return None
        invE = 1/np.sqrt(e2)
        dc = np.zeros_like(zz)
        dc[1:] = np.cumsum(0.5*(invE[1:]+invE[:-1])*np.diff(zz))
        dl = (1+z) * C_LIGHT/70.0 * dc[1:]  # intercept absorbs H0
        if np.any(dl <= 0):
            return None
        base = 5*np.log10(dl)
        # weighted intercept
        w = 1/err**2
        intercept = np.sum(w*(mu-base))/np.sum(w)
        return base + intercept

    def chi2(params):
        om, nu = params
        if om <= 0.05 or om >= 0.6 or nu <= -0.05 or nu >= 0.05:
            return 1e99
        mm = mu_model(om, nu)
        if mm is None:
            return 1e99
        return float(np.sum(((mu-mm)/err)**2))
    if optimize is not None:
        res = optimize.minimize(chi2, x0=[0.3, 0.0], method="Nelder-Mead")
        om, nu = res.x if res.success else (np.nan, np.nan)
    else:
        oms = np.linspace(0.15, 0.45, 41); nus = np.linspace(-0.05, 0.05, 81)
        best = (1e99, np.nan, np.nan)
        for om in oms:
            for nu in nus:
                c = chi2([om, nu])
                if c < best[0]: best = (c, om, nu)
        _, om, nu = best
    # crude sigma by profile around nu at fixed-ish optimized Om
    if np.isfinite(nu):
        grid = np.linspace(max(-0.05, nu-0.02), min(0.05, nu+0.02), 81)
        cs = np.array([chi2([om, n]) for n in grid])
        cmin = np.nanmin(cs)
        ok = grid[cs < cmin + 1.0]
        sig = float((ok.max()-ok.min())/2) if len(ok) >= 2 else None
    else:
        sig = None
    c0 = chi2([om, 0.0]) if np.isfinite(om) else None
    cb = chi2([om, nu]) if np.isfinite(om) and np.isfinite(nu) else None
    return {"n_sne": int(len(z)), "omega_m": float(om), "nu_like": float(nu), "nu_like_sigma_profile": sig, "chi2_best": cb, "chi2_nu0_same_om": c0, "delta_chi2_vs_nu0": (c0-cb if c0 is not None and cb is not None else None), "hit_nu_bound": bool(np.isfinite(nu) and abs(nu) > 0.049), "model_note": "phenomenological SN-only audit: E2=Om(1+z)^3+(1-Om)*(1+nu*log(1+z)); bounded |nu|<0.05; not a full RVM likelihood"}


def load_sparc_rotmods(cache: Path, timeout: int = 90, force: bool = False, allow_large: bool = True) -> Tuple[List[Path], List[Dict[str, Any]]]:
    files, attempts = download_zenodo_matching("16284118", cache, [r"Rotmod.*\.zip", r"rotmod.*\.zip"], timeout=timeout, force=force, allow_large=allow_large)
    if not files:
        p, att2 = download_first(["https://astroweb.case.edu/SPARC/Rotmod_LTG.zip", "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip"], cache / "sparc", timeout=timeout, force=force)
        attempts.extend(att2)
        files = [p] if p else []
    paths = []
    for f in files:
        try:
            ex = extract_archive(f, cache / "sparc" / "rotmods_extracted")
            paths.extend(list(ex.rglob("*.dat")))
            paths.extend(list(ex.rglob("*.txt")))
        except Exception as e:
            attempts.append({"path": str(f), "ok": False, "error": str(e)})
    return paths, attempts


def parse_sparc_file(path: Path) -> Optional[Any]:
    if pd is None:
        return None
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")
        if df.shape[1] < 6 or df.shape[0] < 3:
            return None
        names = ["R_kpc", "Vobs_kms", "eVobs_kms", "Vgas_kms", "Vdisk_kms", "Vbul_kms"] + [f"col{i}" for i in range(6, df.shape[1])]
        df.columns = names[:df.shape[1]]
        df["galaxy"] = path.stem
        return df
    except Exception:
        return None


def fit_sparc_a0(rotmod_paths: Sequence[Path], max_galaxies: Optional[int] = None) -> Dict[str, Any]:
    if pd is None:
        return {"error": "pandas_missing"}
    rows = []
    for i, p in enumerate(rotmod_paths):
        if max_galaxies and i >= max_galaxies:
            break
        df = parse_sparc_file(p)
        if df is not None:
            rows.append(df)
    if not rows:
        return {"error": "no_rotmod_tables"}
    dat = pd.concat(rows, ignore_index=True)
    R = numeric_array(dat, "R_kpc"); Vobs = numeric_array(dat, "Vobs_kms")
    Vgas = numeric_array(dat, "Vgas_kms"); Vdisk = numeric_array(dat, "Vdisk_kms"); Vbul = numeric_array(dat, "Vbul_kms")
    gobs = (Vobs*1000)**2 / (R*KPC_M)
    # Conservative fiducial stellar M/L scalings are not fitted here; signs of Vgas may encode convention.
    vbar2 = np.maximum(0, Vgas*np.abs(Vgas) + 0.5*Vdisk*np.abs(Vdisk) + 0.7*Vbul*np.abs(Vbul))
    gbar = (vbar2*1e6) / (R*KPC_M)
    m = np.isfinite(gobs) & np.isfinite(gbar) & (gobs > 0) & (gbar > 0) & (R > 0)
    gobs = gobs[m]; gbar = gbar[m]
    if len(gobs) < 30:
        return {"error": "too_few_points", "n_points": int(len(gobs))}
    def pred(a0):
        x = np.sqrt(np.maximum(gbar/a0, 1e-300))
        return gbar / np.maximum(1 - np.exp(-x), 1e-30)
    grid = np.logspace(-12, -9, 241)
    rms = []
    for a0 in grid:
        rms.append(np.sqrt(np.mean((np.log10(gobs) - np.log10(pred(a0)))**2)))
    rms = np.asarray(rms)
    j = int(np.argmin(rms))
    best = float(grid[j])
    # bootstrap points for crude uncertainty
    rng = np.random.default_rng(123)
    bs = []
    for _ in range(150):
        idx = rng.integers(0, len(gobs), len(gobs))
        gb = gbar[idx]; go = gobs[idx]
        rr = []
        for a0 in grid[::2]:
            x = np.sqrt(np.maximum(gb/a0, 1e-300))
            pr = gb/np.maximum(1-np.exp(-x), 1e-30)
            rr.append(np.sqrt(np.mean((np.log10(go)-np.log10(pr))**2)))
        bs.append(float(grid[::2][int(np.argmin(rr))]))
    return {"n_points": int(len(gobs)), "n_galaxies_used": int(len(rows)), "a0_best_m_s2": best, "a0_boot_median": float(np.median(bs)), "a0_boot_sigma_dex": float(np.std(np.log10(bs), ddof=1)), "rms_dex": float(rms[j]), "reference_milgrom_m_s2": 1.2e-10}


def try_irsa_tap_query(query: str, cache: Path, name: str, timeout: int = 90, maxrec: Optional[int] = None) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    if pd is None:
        return None, [{"ok": False, "error": "pandas_missing"}]
    q = query
    if maxrec and "TOP" not in q.upper():
        q = re.sub(r"^\s*SELECT\s+", f"SELECT TOP {maxrec} ", q, flags=re.I)
    url = "https://irsa.ipac.caltech.edu/TAP/sync?" + "REQUEST=doQuery&LANG=ADQL&FORMAT=CSV&QUERY=" + quote(q)
    attempts = []
    try:
        p = download_file(url, cache / "irsa_tap", filename=f"{name}.csv", timeout=timeout, force=True)
        attempts.append({"url": url[:500], "ok": True, "path": str(p)})
        df = pd.read_csv(p)
        return df, attempts
    except Exception as e:
        attempts.append({"url": url[:500], "ok": False, "error": str(e)})
        return None, attempts


def load_euclid_q1_sample(cache: Path, timeout: int = 90, max_rows: int = 20000, force: bool = False) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    attempts: List[Dict[str, Any]] = []
    # First discover IRSA table names. Table names change, so this is deliberately dynamic.
    tables, att = try_irsa_tap_query("SELECT table_name, description FROM TAP_SCHEMA.tables WHERE LOWER(table_name) LIKE '%euclid%'", cache, "euclid_tables", timeout=timeout)
    attempts.extend(att)
    if tables is not None and len(tables) > 0 and pd is not None:
        # Prefer merged/source catalog tables.
        name_col = "table_name" if "table_name" in tables.columns else tables.columns[0]
        candidates = [str(x) for x in tables[name_col].dropna().tolist()]
        score = []
        for t in candidates:
            s = 0
            tl = t.lower()
            for kw, val in [("q1", 5), ("mer", 4), ("catalog", 3), ("source", 2), ("object", 2), ("agn", -2), ("image", -2)]:
                if kw in tl: s += val
            score.append((s, t))
        for _, t in sorted(score, reverse=True)[:10]:
            # Try common coordinate column aliases.
            for cols in ["ra,dec", "RA,DEC", "ra,decl", "raj2000,dej2000", "source_id,ra,dec"]:
                q = f"SELECT TOP {max_rows} {cols} FROM {t} WHERE 1=1"
                df, att2 = try_irsa_tap_query(q, cache, f"euclid_sample_{slugify(t)}", timeout=timeout)
                attempts.extend(att2)
                if df is not None and len(df) > 20:
                    # normalize RA/DEC
                    cols_lower = {c.lower(): c for c in df.columns}
                    ra = cols_lower.get("ra") or cols_lower.get("raj2000")
                    dec = cols_lower.get("dec") or cols_lower.get("decl") or cols_lower.get("dej2000")
                    if ra and dec:
                        df = df.rename(columns={ra: "ra", dec: "dec"})
                        return df, attempts + [{"selected_table": t, "ok": True}]
    return None, attempts + [{"ok": False, "error": "euclid_q1_catalog_not_found_via_irsa_tap", "note": "The script queried IRSA TAP dynamically; if ESA/IRSA changed table names, update the discovery scoring only."}]


def sample_map_values_for_points(map_path: Path, ra: np.ndarray, dec: np.ndarray, *, max_points: int = 5000) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Try to sample a FITS image map at sky coordinates without HEALPix.

    Works for ordinary WCS maps. HEALPix maps are intentionally not used because
    the requested bundle skips specialized software.
    """
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except Exception as e:
        return None, {"error": "astropy_missing_or_unavailable", "detail": str(e)}
    try:
        with fits.open(map_path, memmap=True) as hdul:
            hdu = None
            for h in hdul:
                if getattr(h, "data", None) is not None and np.ndim(h.data) >= 2:
                    hdu = h; break
            if hdu is None:
                return None, {"error": "no_2d_image_hdu"}
            data = np.asarray(hdu.data)
            while data.ndim > 2:
                data = data[0]
            w = WCS(hdu.header)
            n = min(len(ra), max_points)
            pix = w.world_to_pixel_values(ra[:n], dec[:n])
            x = np.asarray(pix[0]).round().astype(int); y = np.asarray(pix[1]).round().astype(int)
            vals = np.full(n, np.nan)
            good = (x >= 0) & (y >= 0) & (y < data.shape[-2]) & (x < data.shape[-1])
            vals[good] = data[y[good], x[good]]
            return vals, {"n_requested": int(n), "n_good_pixels": int(np.isfinite(vals).sum()), "map_shape": list(data.shape)}
    except Exception as e:
        return None, {"error": "map_sampling_failed", "detail": str(e)}


def download_act_dr6_lensing_map(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    info = "https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_info.html"
    attempts = []
    try:
        links = discover_links(info, pattern=r"(fits|fits\.gz|\.tar|\.tgz|\.zip|get)", timeout=timeout)
        attempts.append({"url": info, "ok": True, "n_links": len(links)})
    except Exception as e:
        return None, [{"url": info, "ok": False, "error": str(e)}]
    # Prefer non-archive FITS; otherwise use LAMBDA get script page links.
    candidates = [u for u in links if re.search(r"kappa|lensing|convergence", u, re.I) and re.search(r"fits(\.gz)?$", u, re.I)]
    candidates += [u for u in links if re.search(r"fits(\.gz)?$", u, re.I)]
    if not candidates:
        # try product table / get page scraping
        for u in links[:5]:
            try:
                candidates += discover_links(u, pattern=r"fits(\.gz)?$", timeout=timeout)
            except Exception:
                pass
    if not allow_large:
        return None, attempts + [{"ok": False, "reason": "large_download_not_enabled", "note": "ACT DR6 lensing maps are large; rerun with --allow-large."}]
    p, att = download_first(candidates[:5], cache / "act_dr6", timeout=timeout, force=force)
    attempts.extend(att)
    return p, attempts


def download_planck_lensing_or_spectra(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False, prefer_map: bool = False) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    pages = [
        "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/",
        "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/",
    ]
    attempts = []
    candidates = []
    for page in pages:
        try:
            links = discover_links(page, pattern=r"(COM_|base|plik|lensing|cl|Cls|fits|txt|dat)", timeout=timeout)
            attempts.append({"url": page, "ok": True, "n_links": len(links)})
            candidates.extend(links)
        except Exception as e:
            attempts.append({"url": page, "ok": False, "error": str(e)})
    if prefer_map:
        candidates = [u for u in candidates if re.search(r"lensing|kappa|phi", u, re.I)] + candidates
        if not allow_large:
            return None, attempts + [{"ok": False, "reason": "large_download_not_enabled", "note": "Planck maps can be large; rerun with --allow-large."}]
    else:
        candidates = [u for u in candidates if re.search(r"(cl|Cls|plik|base).*\.(txt|dat|fits|fits\.gz)$", u, re.I)] + candidates
    p, att = download_first(candidates[:10], cache / "planck", timeout=timeout, force=force, max_bytes=None if allow_large else 300*1024*1024)
    attempts.extend(att)
    return p, attempts



def choose_highz_acceleration_columns(df: Any) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
    """Choose z, velocity, and radius columns for KMOS3D/KROSS-like catalogues.

    KROSS names the useful columns `Z`, `VC`/`V22`, and `R_IM` or `RD_RPSF`.
    The first bundle used loose substring matching, which could select `Z_AB`
    (a magnitude column) and miss `R_IM`; this exact-priority selector fixes it.
    """
    info: Dict[str, Any] = {}
    if df is None or pd is None:
        return None, None, None, {"error": "no_dataframe_or_pandas"}

    cols = list(df.columns)
    by_lower = {str(c).strip().lower(): c for c in cols}

    def viable(c):
        if c is None:
            return False
        x = numeric_array(df, c)
        return np.isfinite(x).sum() >= max(5, min(30, int(0.05 * len(x))))

    def first_exact(names):
        for name in names:
            c = by_lower.get(name.lower())
            if viable(c):
                return c
        return None

    zc = first_exact(["z", "zspec", "z_spec", "redshift"])
    if zc is None:
        for c in cols:
            cl = str(c).lower()
            if ("redshift" in cl or cl in ("z", "z_spec", "zspec")) and viable(c):
                zc = c; break

    vc = first_exact(["v22", "v22_obs", "vc", "vc_obs", "v_c", "vrot", "v_rot", "vmax", "vmax_obs"])
    if vc is None:
        for c in cols:
            cl = str(c).lower()
            if any(k in cl for k in ["vrot", "v_rot", "velocity", "vmax", "v22", "vc"]) and "err" not in cl and viable(c):
                vc = c; break

    rc = first_exact(["r_im", "rd_rpsf", "r_d", "rd", "re", "r_e", "radius", "r_eff", "reff", "size_kpc"])
    if rc is None:
        for c in cols:
            cl = str(c).lower()
            if any(k in cl for k in ["r_im", "rd_rpsf", "radius", "r_eff", "reff", "size", "scale"]) and "err" not in cl and viable(c):
                rc = c; break

    info["available_columns"] = [str(c) for c in cols[:80]]
    info["selected"] = {"z": str(zc) if zc is not None else None, "velocity": str(vc) if vc is not None else None, "radius": str(rc) if rc is not None else None}
    return zc, vc, rc, info


def acceleration_proxy_from_highz(df: Any, zc: str, vc: str, rc: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return redshift and acceleration proxy g=v^2/r.

    If the selected radius has median <2, treat it as angular arcsec and use a
    conservative 7.5 kpc/arcsec high-z conversion.  Otherwise treat it as kpc.
    """
    z = numeric_array(df, zc)
    v = numeric_array(df, vc)
    r_raw = numeric_array(df, rc)
    med_r = float(np.nanmedian(r_raw[np.isfinite(r_raw)])) if np.isfinite(r_raw).any() else float("nan")
    radius_mode = "kpc"
    r_kpc = r_raw.copy()
    if np.isfinite(med_r) and med_r > 0 and med_r < 2.0:
        r_kpc = r_raw * 7.5
        radius_mode = "arcsec_assumed_7p5_kpc_per_arcsec"
    g = (v * 1000.0) ** 2 / (r_kpc * KPC_M)
    meta = {"radius_mode": radius_mode, "radius_median_raw": med_r}
    return z, g, meta



def load_kmos_or_kross(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False, max_rows: int = 20000) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    attempts = []
    # KROSS catalogue is a compact fallback when KMOS3D only exposes cubes.
    pages = ["https://www.mpe.mpg.de/ir/KMOS3D/data", "https://astro.dur.ac.uk/KROSS/data.html"]
    candidates = []
    for page in pages:
        try:
            links = discover_links(page, pattern=r"(catalog|catalogue|table|fits|csv|dat|txt|xlsx)", timeout=timeout)
            attempts.append({"url": page, "ok": True, "n_links": len(links)})
            candidates.extend(links)
        except Exception as e:
            attempts.append({"url": page, "ok": False, "error": str(e)})
    # Prefer compact tabular files over cubes/social/share links.
    candidates = [u for u in candidates if re.search(r"\.(fits|fit|csv|dat|txt|tsv|xlsx)(\?|$)", u, re.I)]
    candidates = [u for u in candidates if not re.search(r"cube|_3D|share-offsite|intent/compose|data-protection|print=yes|linkedin|bsky", u, re.I)]
    candidates = sorted(set(candidates), key=lambda u: (0 if re.search(r"kross_release|catalog|catalogue|table", u, re.I) else 1, u))
    for idx, u in enumerate(candidates[:30]):
        try:
            fname = f"{idx:02d}_{url_basename(u)}"
            p = download_file(u, cache / "kmos_kross", filename=fname, timeout=timeout, force=force, max_bytes=None if allow_large else 300*1024*1024)
            attempts.append({"url": u, "ok": True, "path": str(p)})
            df = read_table_any(p, max_rows=max_rows)
            if df is not None and len(df) > 10:
                return df, attempts
        except Exception as e:
            attempts.append({"url": u, "ok": False, "error": str(e)})
    return None, attempts + [{"ok": False, "error": "no_parseable_kmos_or_kross_catalog"}]


def find_cols_by_keywords(df: Any, keywords: Sequence[str]) -> Optional[str]:
    if df is None:
        return None
    for c in df.columns:
        cl = str(c).lower()
        if all(k.lower() in cl for k in keywords):
            return c
    # any one keyword in order
    for k in keywords:
        for c in df.columns:
            if k.lower() in str(c).lower():
                return c
    return None


def load_vast_void_tables(cache: Path, timeout: int = 90, force: bool = False, allow_large: bool = True) -> Tuple[List[Any], List[Dict[str, Any]]]:
    paths, attempts = download_zenodo_matching("7406035", cache, [r"\.(csv|dat|txt|fits|zip)$", r"void"], timeout=timeout, force=force, allow_large=allow_large)
    tables = []
    for p in paths:
        try:
            if zipfile.is_zipfile(p) or tarfile.is_tarfile(p):
                ex = extract_archive(p, cache / "vast" / f"extract_{p.stem}")
                for q in list(ex.rglob("*.csv")) + list(ex.rglob("*.dat")) + list(ex.rglob("*.txt")) + list(ex.rglob("*.fits")):
                    df = read_table_any(q)
                    if df is not None and len(df) > 5:
                        tables.append(df)
            else:
                df = read_table_any(p)
                if df is not None and len(df) > 5:
                    tables.append(df)
        except Exception as e:
            attempts.append({"path": str(p), "ok": False, "error": str(e)})
    return tables, attempts


def load_xenon_limit_curves(cache: Path, timeout: int = 90, force: bool = False) -> Tuple[List[Any], List[Dict[str, Any]]]:
    attempts = []
    tables = []
    # XENONnT light WIMP release often stores CSV/NPY/NPZ; also scan public page links.
    paths, att = github_download_matching("XENONnT", "light_wimp_data_release", cache, [r"\.(csv|txt|dat)$", r"limit", r"result"], timeout=timeout, force=force, max_files=40)
    attempts.extend(att)
    for p in paths:
        df = read_table_any(p)
        if df is not None and df.shape[1] >= 2:
            tables.append(df)
    try:
        links = discover_links("https://xenonexperiment.org/public-data/", pattern=r"(github|zenodo|csv|txt|dat|limit)", timeout=timeout)
        attempts.append({"url": "https://xenonexperiment.org/public-data/", "ok": True, "n_links": len(links)})
    except Exception as e:
        links = []; attempts.append({"url": "https://xenonexperiment.org/public-data/", "ok": False, "error": str(e)})
    # Don't follow arbitrary GitHub pages here; raw GitHub API already used.
    for u in [x for x in links if re.search(r"\.(csv|txt|dat)$", x, re.I)][:10]:
        try:
            p = download_file(u, cache / "xenon_public", timeout=timeout, force=force)
            attempts.append({"url": u, "ok": True, "path": str(p)})
            df = read_table_any(p)
            if df is not None and df.shape[1] >= 2:
                tables.append(df)
        except Exception as e:
            attempts.append({"url": u, "ok": False, "error": str(e)})
    return tables, attempts


def choose_mass_limit_columns(df: Any) -> Tuple[Optional[str], Optional[str]]:
    nums = find_numeric_columns(df)
    if len(nums) < 2:
        return None, None
    mass = None; lim = None
    for c in nums:
        cl = str(c).lower()
        if "mass" in cl or "m_dm" in cl or "mchi" in cl or "m_dm" in cl:
            mass = c; break
    for c in nums:
        cl = str(c).lower()
        if c == mass: continue
        if "sigma" in cl or "limit" in cl or "cross" in cl or "xs" in cl:
            lim = c; break
    if mass is None:
        # heuristic: mass is numeric column with positive broad range; often first
        mass = nums[0]
    if lim is None:
        lim = nums[1] if len(nums) > 1 else None
    return mass, lim


def load_firas_spectrum(cache: Path, timeout: int = 90, force: bool = False) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    urls = [
        "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.txt",
        "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.dat",
        "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.txt",
    ]
    p, attempts = download_first(urls, cache / "firas", timeout=timeout, force=force)
    if p:
        df = read_table_any(p)
        if df is not None:
            return df, attempts
        # Try manual whitespace with known FIRAS columns skipping headers
        if pd is not None:
            try:
                df = pd.read_csv(p, sep=r"\s+", comment="#", header=None, engine="python")
                if df.shape[1] >= 3:
                    return df, attempts
            except Exception:
                pass
    return None, attempts + [{"ok": False, "error": "firas_download_or_parse_failed"}]


def get_gwosc_event_json(timeout: int = 90) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    urls = [
        "https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/",
        "https://www.gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/",
        "https://gwosc.org/eventapi/json/GWTC-2.1-confident/GW150914/v3/",
    ]
    attempts = []
    for u in urls:
        try:
            obj = json.loads(http_get_text(u, timeout=timeout))
            attempts.append({"url": u, "ok": True})
            return obj, attempts
        except Exception as e:
            attempts.append({"url": u, "ok": False, "error": str(e)})
    return None, attempts


def extract_gwosc_file_urls(event_json: Dict[str, Any]) -> List[str]:
    urls = []
    def rec(o):
        if isinstance(o, dict):
            for k,v in o.items():
                if isinstance(v, str) and re.search(r"\.h(df5|5)$|\.gwf$", v, re.I):
                    urls.append(v)
                else:
                    rec(v)
        elif isinstance(o, list):
            for v in o: rec(v)
    rec(event_json)
    # prefer H1 4K short files
    urls = sorted(set(urls), key=lambda u: ("H-H1" not in u and "H1" not in u, "32" not in u, len(u)))
    return urls


def load_nanograv_archive(cache: Path, timeout: int = 90, force: bool = False, allow_large: bool = False) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    # v2.1.0 first, older official records as fallback
    attempts = []
    for rid in ["16051178", "8423265", "8104459"]:
        files, att = download_zenodo_matching(rid, cache, [r"\.tar\.gz$", r"\.tgz$", r"\.zip$"], timeout=timeout, force=force, allow_large=allow_large, max_default_mb=350)
        attempts.extend(att)
        if files:
            return files[0], attempts
    return None, attempts


def parse_par_positions(root: Path) -> Tuple[Optional[Any], Dict[str, Any]]:
    if pd is None:
        return None, {"error": "pandas_missing"}
    rows = []
    for p in root.rglob("*.par"):
        ra = dec = None
        name = p.stem
        try:
            for line in p.read_text(errors="ignore").splitlines():
                sp = line.split()
                if len(sp) >= 2:
                    if sp[0].upper() in ("PSR", "PSRJ"):
                        name = sp[1]
                    elif sp[0].upper() == "RAJ":
                        ra = sp[1]
                    elif sp[0].upper() == "DECJ":
                        dec = sp[1]
            if ra and dec:
                rows.append({"pulsar": name, "RAJ": ra, "DECJ": dec})
        except Exception:
            pass
    if not rows:
        return None, {"error": "no_par_positions"}
    # astropy converts sexagesimal if available
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        ras=[]; decs=[]
        for r in rows:
            c = SkyCoord(r["RAJ"], r["DECJ"], unit=(u.hourangle, u.deg))
            ras.append(c.ra.deg); decs.append(c.dec.deg)
        for r, ra, dec in zip(rows, ras, decs):
            r["ra"] = ra; r["dec"] = dec
    except Exception:
        pass
    return pd.DataFrame(rows), {"n_positions": len(rows)}

# ------------------------- v9.3 overrides -------------------------
# Definitions below intentionally override earlier helpers.  They add real map
# extraction/sampling support, robust NANOGrav .par position parsing, and safer
# public-product selection without breaking existing scripts.

def find_files(root: Path, patterns: Sequence[str], limit: int = 20) -> List[Path]:
    root = Path(root)
    found: List[Path] = []
    if root.exists():
        for q in root.rglob("*"):
            if q.is_file() and any(re.search(pat, q.name, re.I) or re.search(pat, str(q), re.I) for pat in patterns):
                found.append(q)
    def score(q: Path):
        n = str(q).lower(); sc = 0
        for kw, val in [("kappa",20),("convergence",18),("lensing",12),("mv",8),("map",4),("mask",-20),("mean",-10),("mf",-10),("alm",-6),("cls",-4)]:
            if kw in n: sc += val
        return (-sc, len(str(q)))
    return sorted(found, key=score)[:limit]


def extract_if_archive(path: Path, outdir: Path) -> Path:
    try:
        if zipfile.is_zipfile(path) or tarfile.is_tarfile(path):
            return extract_archive(path, outdir)
    except Exception:
        pass
    return path.parent


def _choose_fits_map_from_path(path: Path, extract_dir: Path) -> Optional[Path]:
    path = Path(path)
    if str(path).lower().endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        return path
    root = extract_if_archive(path, extract_dir)
    fits = find_files(root, [r"\.fits(\.gz)?$", r"\.fit(\.gz)?$"], limit=100)
    return fits[0] if fits else None


def sample_map_values_for_points(map_path: Path, ra: np.ndarray, dec: np.ndarray, *, max_points: int = 5000, prefer_healpix: bool = True) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except Exception as e:
        return None, {"error": "astropy_missing_or_unavailable", "detail": str(e)}
    map_path = Path(map_path)
    try:
        with fits.open(map_path, memmap=True) as hdul:
            # WCS image maps, e.g. ACT DR6 equatorial maps.
            for hdu_i, hdu in enumerate(hdul):
                data = getattr(hdu, "data", None)
                if data is None:
                    continue
                arr = np.asarray(data)
                if arr.ndim >= 2 and hdu.header.get("CTYPE1"):
                    arr2 = arr
                    while arr2.ndim > 2:
                        arr2 = arr2[0]
                    try:
                        w = WCS(hdu.header)
                        n = min(len(ra), max_points)
                        pix = w.world_to_pixel_values(ra[:n], dec[:n])
                        x = np.asarray(pix[0]).round().astype(int)
                        y = np.asarray(pix[1]).round().astype(int)
                        vals = np.full(n, np.nan)
                        good = (x >= 0) & (y >= 0) & (y < arr2.shape[-2]) & (x < arr2.shape[-1])
                        vals[good] = arr2[y[good], x[good]]
                        if np.isfinite(vals).sum() > 0:
                            return vals, {"mode":"fits_wcs","hdu":int(hdu_i),"n_requested":int(n),"n_good_pixels":int(np.isfinite(vals).sum()),"map_shape":list(arr2.shape),"path":str(map_path)}
                    except Exception:
                        pass
            # HEALPix maps, e.g. Planck/PR4.  Optional dependency only.
            if prefer_healpix:
                try:
                    import healpy as hp
                    from astropy.coordinates import SkyCoord
                    import astropy.units as u
                except Exception as e:
                    return None, {"error":"healpix_map_requires_optional_healpy","detail":str(e),"path":str(map_path),"install_hint":"conda install -c conda-forge healpy"}
                hp_map = None; selected = None
                for fld in [0,1,2]:
                    try:
                        m = hp.read_map(str(map_path), field=fld, verbose=False)
                        if np.asarray(m).ndim == 1 and len(m) >= 12:
                            hp_map = np.asarray(m, float); selected = {"field": int(fld)}; break
                    except Exception:
                        pass
                if hp_map is None:
                    return None, {"error":"no_wcs_or_healpix_map_detected","path":str(map_path)}
                n = min(len(ra), max_points)
                is_planck = bool(re.search(r"planck|pr4|healpix|com_lensing", str(map_path), re.I))
                if is_planck:
                    c = SkyCoord(ra=ra[:n]*u.deg, dec=dec[:n]*u.deg, frame="icrs").galactic
                    theta = 0.5*np.pi - np.deg2rad(c.b.deg); phi = np.deg2rad(c.l.deg)
                    coordsys = "galactic_from_icrs"
                else:
                    theta = 0.5*np.pi - np.deg2rad(dec[:n]); phi = np.deg2rad(ra[:n]); coordsys = "icrs_assumed"
                pix = hp.ang2pix(hp.get_nside(hp_map), theta, phi, nest=False)
                vals = hp_map[pix]
                vals = np.where(np.isfinite(vals) & (np.abs(vals) < 1e30), vals, np.nan)
                return vals, {"mode":"healpix","coordsys":coordsys,"nside":int(hp.get_nside(hp_map)),"n_requested":int(n),"n_good_pixels":int(np.isfinite(vals).sum()),"selected":selected,"path":str(map_path)}
    except Exception as e:
        return None, {"error":"map_sampling_failed","detail":str(e),"path":str(map_path)}
    return None, {"error":"no_usable_map_hdu","path":str(map_path)}


def download_act_dr6_lensing_map(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    info = "https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html"
    direct = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/dr6_lensing_release.tar.gz"
    attempts: List[Dict[str, Any]] = []
    try:
        links = discover_links(info, pattern=r"(dr6_lensing_release|fits|tar|tgz|gz)", timeout=timeout)
        attempts.append({"url": info, "ok": True, "n_links": len(links)})
    except Exception as e:
        links = []; attempts.append({"url": info, "ok": False, "error": str(e)})
    candidates=[]
    for u in [direct]+links:
        if u not in candidates: candidates.append(u)
    if not allow_large:
        return None, attempts + [{"ok":False,"reason":"large_download_not_enabled","note":"ACT DR6 lensing release is ~1.37 GB; rerun with --allow-large."}]
    p, att = download_first(candidates, cache/"act_dr6", timeout=timeout, force=force, max_bytes=None)
    attempts.extend(att)
    if p is None:
        return None, attempts
    mp = _choose_fits_map_from_path(p, cache/"act_dr6"/"dr6_lensing_release_extracted")
    if mp is None:
        attempts.append({"ok":False,"error":"downloaded_ACT_release_but_no_FITS_map_found","path":str(p)})
        return None, attempts
    attempts.append({"ok":True,"selected_map":str(mp),"note":"selected extracted ACT DR6 FITS map for sampling"})
    return mp, attempts


def _github_release_asset_url(owner: str, repo: str, asset_pattern: str, timeout: int = 90, tag_hint: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    urls=[]
    if tag_hint:
        urls.append(f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{quote(tag_hint)}")
    urls.append(f"https://api.github.com/repos/{owner}/{repo}/releases")
    info={"github_releases":f"{owner}/{repo}","asset_pattern":asset_pattern}
    for api in urls:
        try:
            obj=json.loads(http_get_text(api,timeout=timeout))
            releases=obj if isinstance(obj,list) else [obj]
            for rel in releases:
                for a in rel.get("assets",[]):
                    name=a.get("name","")
                    if re.search(asset_pattern,name,re.I):
                        info.update({"ok":True,"api":api,"release":rel.get("name") or rel.get("tag_name"),"asset":name})
                        return a.get("browser_download_url"),info
        except Exception as e:
            info.setdefault("errors",[]).append({"api":api,"error":str(e)})
    info["ok"]=False
    return None,info


def download_planck_lensing_or_spectra(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False, prefer_map: bool = False) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    attempts: List[Dict[str, Any]]=[]
    if prefer_map:
        if not allow_large:
            return None,[{"ok":False,"reason":"large_download_not_enabled","note":"Planck lensing maps are large HEALPix products; rerun with --allow-large. Sampling needs optional healpy."}]
        asset,info=_github_release_asset_url("carronj","planck_PR4_lensing",r"PR42018like_maps\.tar",timeout=timeout,tag_hint="Data")
        attempts.append(info)
        candidates=[]
        if asset: candidates.append(asset)
        candidates += [
            "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/lensing/COM_Lensing_4096_R3.00.tar",
            "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/lensing/COM_Lensing-Szdeproj_4096_R3.00.tar",
        ]
        p,att=download_first(candidates,cache/"planck_lensing",timeout=timeout,force=force,max_bytes=None)
        attempts.extend(att)
        if p is None: return None, attempts
        root=extract_if_archive(p,cache/"planck_lensing"/"maps_extracted")
        fits=find_files(root,[r"\.fits(\.gz)?$"],limit=100)
        if not fits:
            attempts.append({"ok":False,"error":"downloaded_planck_archive_but_no_FITS_map_found","path":str(p)})
            return None, attempts
        attempts.append({"ok":True,"selected_map":str(fits[0]),"note":"selected Planck/PR4 HEALPix FITS map; sampling needs optional healpy"})
        return fits[0], attempts
    pages=["https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/"]
    candidates=[]
    for page in pages:
        try:
            links=discover_links(page,pattern=r"(TT|TE|EE|BB|PowerSpect|Cls|cl).*\.(txt|dat|fits|fits\.gz)$",timeout=timeout)
            attempts.append({"url":page,"ok":True,"n_links":len(links)})
            candidates.extend(links)
        except Exception as e:
            attempts.append({"url":page,"ok":False,"error":str(e)})
    p,att=download_first(candidates[:20],cache/"planck",timeout=timeout,force=force,max_bytes=300*1024*1024)
    attempts.extend(att)
    return p,attempts


def _sexagesimal_to_deg_ra(s: str) -> Optional[float]:
    try:
        if ":" in s:
            h,m,sec=[float(x) for x in s.replace("+","").split(":")[:3]]
            return 15.0*(h + m/60.0 + sec/3600.0)
        return float(s)
    except Exception:
        return None


def _sexagesimal_to_deg_dec(s: str) -> Optional[float]:
    try:
        sign=-1 if str(s).strip().startswith("-") else 1
        ss=str(s).strip().lstrip("+-")
        if ":" in ss:
            d,m,sec=[float(x) for x in ss.split(":")[:3]]
            return sign*(d + m/60.0 + sec/3600.0)
        return float(s)
    except Exception:
        return None


def _psrj_to_approx_radec(name: str) -> Tuple[Optional[float], Optional[float]]:
    m=re.search(r"J(\d{2})(\d{2})(?:\d{0,2}(?:\.\d+)?)?([+-])(\d{2})(\d{2})?", name)
    if not m: return None, None
    hh=int(m.group(1)); mm=int(m.group(2)); ra=15.0*(hh+mm/60.0)
    sign=-1 if m.group(3)=="-" else 1
    dd=int(m.group(4)); dm=int(m.group(5) or 0); dec=sign*(dd+dm/60.0)
    return ra, dec


def parse_par_positions(root: Path) -> Tuple[Optional[Any], Dict[str, Any]]:
    if pd is None: return None,{"error":"pandas_missing"}
    rows=[]; n_files=0; keys_seen={}
    for p in Path(root).rglob("*.par"):
        n_files += 1; vals={}; name=p.stem
        try:
            for raw in p.read_text(errors="ignore").splitlines():
                line=raw.split("#",1)[0].strip()
                if not line: continue
                sp=line.split()
                if len(sp)<2: continue
                k=sp[0].upper(); v=sp[1]
                keys_seen[k]=keys_seen.get(k,0)+1
                if k in ("PSR","PSRJ","PSRB"): name=v
                if k in ("RAJ","DECJ","ELONG","ELAT"):
                    vals[k]=v
            ra=dec=None
            if "RAJ" in vals and "DECJ" in vals:
                ra=_sexagesimal_to_deg_ra(vals["RAJ"]); dec=_sexagesimal_to_deg_dec(vals["DECJ"])
            elif "ELONG" in vals and "ELAT" in vals:
                try:
                    from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
                    import astropy.units as u
                    c=SkyCoord(lon=float(vals["ELONG"])*u.deg,lat=float(vals["ELAT"])*u.deg,frame=BarycentricTrueEcliptic).icrs
                    ra=float(c.ra.deg); dec=float(c.dec.deg)
                except Exception: pass
            if (ra is None or dec is None):
                ra,dec=_psrj_to_approx_radec(name)
            if ra is not None and dec is not None and np.isfinite(ra) and np.isfinite(dec):
                rows.append({"pulsar":name,"ra":float(ra)%360.0,"dec":float(dec),"RAJ":vals.get("RAJ"),"DECJ":vals.get("DECJ"),"source_par":str(p)})
        except Exception:
            pass
    if not rows:
        return None,{"error":"no_par_positions","n_par_files_scanned":int(n_files),"keys_seen_top":sorted(keys_seen.items(),key=lambda kv:-kv[1])[:20]}
    return pd.DataFrame(rows),{"n_positions":len(rows),"n_par_files_scanned":int(n_files),"used_psrj_fallback":int(sum(1 for r in rows if not r.get("RAJ") or not r.get("DECJ")))}

# ---- v9.4 result-quality helpers ----

def is_bound_hit_metric(d: Any) -> bool:
    return isinstance(d, dict) and bool(d.get('hit_nu_bound'))


def desi_dr2_mean_paths(paths: Sequence[Path]) -> List[Path]:
    """Keep only compact DESI DR2 mean-vector files, excluding covariances/grids."""
    out = []
    for p in paths:
        s = str(p).replace('\\', '/').lower()
        name = Path(s).name
        if 'desi_bao_dr2' not in s:
            continue
        if 'mean' not in name:
            continue
        if any(bad in name for bad in ['cov', 'grid', 'likelihood', 'test_']):
            continue
        out.append(p)
    return out


def parse_bao_mean_table(path: Path, max_rows: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Parse a compact BAO mean-vector table into z/y arrays for trend screens.

    Cobaya BAO files are heterogeneous.  This deliberately avoids claiming a
    full DESI likelihood; it extracts only small DR2 mean-vector files and
    normalizes each observable internally.
    """
    df = read_table_any(path, max_rows=max_rows)
    if df is None or df.shape[0] < 1 or df.shape[1] < 2:
        return None
    nums = find_numeric_columns(df)
    if len(nums) < 2:
        return None
    zcol = None
    # Prefer a numeric column with plausible redshifts and multiple distinct values.
    for c in nums:
        x = numeric_array(df, c)
        finite = x[np.isfinite(x)]
        if len(finite) >= 1 and np.nanmin(finite) >= 0 and np.nanmax(finite) <= 5.0 and len(np.unique(np.round(finite, 4))) >= min(2, len(finite)):
            zcol = c
            break
    if zcol is None:
        zcol = nums[0]
    ycol = None
    for c in nums:
        if c == zcol:
            continue
        y = numeric_array(df, c)
        finite = y[np.isfinite(y)]
        if len(finite) >= 1 and np.nanstd(finite) > 0:
            ycol = c
            break
    if ycol is None:
        return None
    z = numeric_array(df, zcol)
    y = numeric_array(df, ycol)
    m = np.isfinite(z) & np.isfinite(y) & (z > 0) & (z < 6)
    if m.sum() < 1:
        return None
    obs = None
    # If a non-numeric column contains strings like DM_over_rs, DH_over_rs, DV_over_rs, preserve it.
    for c in df.columns:
        if c in nums:
            continue
        vals = df[c].astype(str).str.lower().tolist()
        joined = ' '.join(vals[:20])
        if any(k in joined for k in ['dm', 'dh', 'dv', 'bao', 'fsig', 'rs']):
            obs = str(c); break
    yy = y[m]
    # robust normalized residual-like ordinate for a trend screen only
    denom = mad(yy) or float(np.nanstd(yy)) or 1.0
    y_norm = (yy - float(np.nanmedian(yy))) / denom
    return {
        'path': str(path),
        'n': int(m.sum()),
        'z_col': str(zcol),
        'y_col': str(ycol),
        'observable_col': obs,
        'z': z[m].astype(float),
        'y_norm': y_norm.astype(float),
        'z_min': float(np.nanmin(z[m])),
        'z_max': float(np.nanmax(z[m])),
    }


def summarize_dr2_bao_trend(paths: Sequence[Path], max_rows: Optional[int] = None) -> Dict[str, Any]:
    tabs = []
    allz: List[float] = []
    ally: List[float] = []
    for p in desi_dr2_mean_paths(paths):
        t = parse_bao_mean_table(p, max_rows=max_rows)
        if not t:
            continue
        if t['n'] >= 1:
            tabs.append({k:v for k,v in t.items() if k not in ('z','y_norm')})
            allz.extend([float(x) for x in t['z']])
            ally.extend([float(y) for y in t['y_norm']])
    corr = safe_corr(allz, ally) if len(allz) >= 4 else {'n': int(len(allz)), 'rho': None, 'pvalue': None, 'error': 'too_few_dr2_mean_points'}
    coef = safe_polyfit(allz, ally, 1) if len(allz) >= 3 else None
    return {
        'n_tables': len(tabs),
        'n_points': int(len(allz)),
        'tables_used': tabs,
        'spearman': corr,
        'linear_coef': coef,
        'note': 'DESI DR2 mean-vector trend screen only; not a covariance likelihood and not a branch-selection result.'
    }


def highz_quality_mask(df: Any, zc: str, vc: str, rc: str, z: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return a conservative quality mask for KROSS/KMOS-like catalogues."""
    base = np.isfinite(z) & np.isfinite(g) & (z > 0.1) & (z < 4) & (g > 0) & (g < 1e-7)
    info: Dict[str, Any] = {'initial_n': int(np.sum(base)), 'applied_cuts': []}
    if df is None or pd is None:
        return base, info
    cols = {str(c).lower(): c for c in df.columns}
    def cut_flag_zero(names: Sequence[str], label: str):
        nonlocal base
        for nm in names:
            c = cols.get(nm.lower())
            if c is not None:
                arr = numeric_array(df, c)
                before = int(np.sum(base))
                base = base & np.isfinite(arr) & (arr == 0)
                info['applied_cuts'].append({'column': str(c), 'rule': f'{label} == 0', 'before': before, 'after': int(np.sum(base))})
                return
    cut_flag_zero(['AGN_FLAG','AGN'], 'AGN flag')
    cut_flag_zero(['IRR_FLAG','IRREGULAR_FLAG'], 'irregular flag')
    cut_flag_zero(['EXTRAP_FLAG'], 'extrapolation flag')
    # Require finite velocity/radius uncertainties if present but do not cut harshly.
    for exact in [str(vc)+'_ERR', str(rc)+'_ERR', str(rc)+'_ERR_LOW', str(rc)+'_ERR_HIGH']:
        c = cols.get(exact.lower())
        if c is not None:
            arr = numeric_array(df, c)
            before = int(np.sum(base))
            base = base & (np.isfinite(arr) | ~base)
            info['applied_cuts'].append({'column': str(c), 'rule': 'finite where selected', 'before': before, 'after': int(np.sum(base))})
    info['final_n'] = int(np.sum(base))
    return base, info


def highz_acceleration_summary(df: Any) -> Dict[str, Any]:
    zc, vc, rc, info = choose_highz_acceleration_columns(df)
    if not (zc and vc and rc):
        return {'error': 'no_highz_columns', 'column_selection': info}
    z, g, meta = acceleration_proxy_from_highz(df, zc, vc, rc)
    raw = np.isfinite(z) & np.isfinite(g) & (z > 0.1) & (z < 4) & (g > 0) & (g < 1e-7)
    qmask, qinfo = highz_quality_mask(df, zc, vc, rc, z, g)
    def block(mask):
        if int(np.sum(mask)) < 5:
            return {'n': int(np.sum(mask))}
        corr = safe_corr(z[mask], g[mask])
        coef = safe_polyfit(z[mask], np.log10(g[mask]), 1)
        # Split offset between lower and upper redshift halves.
        medz = float(np.nanmedian(z[mask]))
        lo = mask & (z <= medz); hi = mask & (z > medz)
        delta = bootstrap_mean_delta(np.log10(g[hi]), np.log10(g[lo]), nboot=500) if np.sum(lo) >= 5 and np.sum(hi) >= 5 else {'error': 'too_few_halves'}
        return {
            'n': int(np.sum(mask)),
            'z_min': float(np.nanmin(z[mask])),
            'z_max': float(np.nanmax(z[mask])),
            'z_mean': float(np.nanmean(z[mask])),
            'mean_a_proxy_m_s2': float(np.nanmean(g[mask])),
            'median_a_proxy_m_s2': float(np.nanmedian(g[mask])),
            'spearman_z_g': corr,
            'log_g_vs_z_coef': coef,
            'high_minus_low_z_log10g': delta,
        }
    return {
        'cols': {'z': str(zc), 'v': str(vc), 'r': str(rc)},
        'radius_meta': meta,
        'raw': block(raw),
        'quality_cut': block(qmask),
        'quality_cuts': qinfo,
    }


def sparc_point_accelerations(rotmod_paths: Sequence[Path], max_galaxies: Optional[int] = None) -> Dict[str, Any]:
    if pd is None:
        return {'error': 'pandas_missing'}
    rows=[]
    for i,p in enumerate(rotmod_paths):
        if max_galaxies and i >= max_galaxies:
            break
        df=parse_sparc_file(p)
        if df is not None:
            rows.append(df)
    if not rows:
        return {'error': 'no_sparc_rows'}
    dat=pd.concat(rows,ignore_index=True)
    R=numeric_array(dat,'R_kpc'); Vobs=numeric_array(dat,'Vobs_kms')
    gobs=(Vobs*1000)**2/(R*KPC_M)
    m=np.isfinite(gobs)&(gobs>0)&(gobs<1e-7)&np.isfinite(R)&(R>0)
    return {'n': int(np.sum(m)), 'median_gobs_m_s2': float(np.nanmedian(gobs[m])) if np.sum(m) else None, 'mean_gobs_m_s2': float(np.nanmean(gobs[m])) if np.sum(m) else None, 'p16_gobs_m_s2': float(np.nanpercentile(gobs[m],16)) if np.sum(m) else None, 'p84_gobs_m_s2': float(np.nanpercentile(gobs[m],84)) if np.sum(m) else None}


def gaussian_mixture_bic_1d(y: Sequence[float]) -> Dict[str, Any]:
    y=np.asarray(y,float); y=y[np.isfinite(y)]
    if len(y)<50:
        return {'error':'too_few','n':int(len(y))}
    mu=np.mean(y); sig=np.std(y) or 1
    ll1=np.sum(-0.5*((y-mu)/sig)**2-np.log(sig*np.sqrt(2*np.pi))); bic1=2*np.log(len(y))-2*ll1
    qs=np.quantile(y,[0.3,0.7]); m1,m2=qs; s1=s2=sig; w=0.5
    for _ in range(120):
        p1=w*np.exp(-0.5*((y-m1)/s1)**2)/(s1+1e-12); p2=(1-w)*np.exp(-0.5*((y-m2)/s2)**2)/(s2+1e-12)
        r=p1/(p1+p2+1e-300); w=float(np.mean(r))
        if r.sum()>0: m1=float(np.sum(r*y)/r.sum()); s1=float(np.sqrt(np.sum(r*(y-m1)**2)/r.sum())+1e-12)
        if (1-r).sum()>0: m2=float(np.sum((1-r)*y)/(1-r).sum()); s2=float(np.sqrt(np.sum((1-r)*(y-m2)**2)/(1-r).sum())+1e-12)
    p1=w*np.exp(-0.5*((y-m1)/s1)**2)/(s1*np.sqrt(2*np.pi)); p2=(1-w)*np.exp(-0.5*((y-m2)/s2)**2)/(s2*np.sqrt(2*np.pi))
    ll2=np.sum(np.log(p1+p2+1e-300)); bic2=5*np.log(len(y))-2*ll2
    return {'n': int(len(y)), 'bic_single': float(bic1), 'bic_two_component': float(bic2), 'delta_bic_single_minus_mix': float(bic1-bic2), 'components': {'w1': float(w), 'mu1': float(m1), 'mu2': float(m2), 's1': float(s1), 's2': float(s2)}}

# ---- v9.5 result-quality/data-depth helpers ----
# These definitions intentionally override or extend v9.4 helpers while keeping
# the public-data/no-manual-file workflow.  The emphasis is on converting
# previous data_limited screens into computable diagnostics without allowing
# proxy-level outputs to masquerade as confirmations.


def _is_healpix_npix(n: int) -> Optional[int]:
    try:
        ns = int(round(math.sqrt(max(int(n), 0) / 12.0)))
        if ns > 0 and 12 * ns * ns == int(n) and (ns & (ns - 1)) == 0:
            return ns
    except Exception:
        pass
    return None


def _ang2pix_ring_numpy(nside: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Small vectorized HEALPix RING ang2pix replacement.

    This is adapted from the public HEALPix ring-index equations and is used
    only for nearest-neighbour table-map sampling when healpy is unavailable.
    It is not a replacement for harmonic transforms or map reprojection.
    """
    theta = np.asarray(theta, float)
    phi = np.asarray(phi, float) % (2.0 * np.pi)
    z = np.cos(theta)
    za = np.abs(z)
    tt = phi / (0.5 * np.pi)  # in [0,4)
    nside = int(nside)
    nl4 = 4 * nside
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    pix = np.full(theta.shape, -1, dtype=np.int64)

    eq = za <= (2.0 / 3.0)
    if np.any(eq):
        tt_e = tt[eq]
        z_e = z[eq]
        jp = np.floor(nside * (0.5 + tt_e - 0.75 * z_e)).astype(np.int64)
        jm = np.floor(nside * (0.5 + tt_e + 0.75 * z_e)).astype(np.int64)
        ir = nside + 1 + jp - jm  # counted from north pole, 1-based
        kshift = 1 - (ir & 1)
        ip = np.floor((jp + jm - nside + kshift + 1) / 2.0).astype(np.int64) + 1
        ip = ((ip - 1) % nl4) + 1
        pix[eq] = ncap + (ir - 1) * nl4 + ip - 1

    pol = ~eq
    if np.any(pol):
        tt_p = tt[pol]
        z_p = z[pol]
        za_p = za[pol]
        tp = tt_p - np.floor(tt_p)
        tmp = nside * np.sqrt(3.0 * (1.0 - za_p))
        jp = np.floor(tp * tmp).astype(np.int64)
        jm = np.floor((1.0 - tp) * tmp).astype(np.int64)
        ir = jp + jm + 1
        ip = np.floor(tt_p * ir).astype(np.int64) + 1
        ip = ((ip - 1) % (4 * ir)) + 1
        north = z_p > 0
        out = np.empty_like(ir)
        out[north] = 2 * ir[north] * (ir[north] - 1) + ip[north] - 1
        out[~north] = npix - 2 * ir[~north] * (ir[~north] + 1) + ip[~north] - 1
        pix[pol] = out
    return np.clip(pix, 0, npix - 1)


def _maybe_icrs_to_galactic(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg, frame="icrs").galactic
        return np.asarray(c.l.deg, float), np.asarray(c.b.deg, float), "galactic_from_icrs"
    except Exception:
        return np.asarray(ra_deg, float), np.asarray(dec_deg, float), "icrs_assumed_no_astropy_transform"


def _read_healpix_table_map_numpy(hdul: Any, map_path: Path) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Read a 1D HEALPix map from a FITS binary table without healpy."""
    best = None
    best_info: Dict[str, Any] = {}
    for hdu_i, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is None or not hasattr(data, "columns"):
            continue
        names = list(getattr(data.columns, "names", []) or [])
        for name in names:
            try:
                arr = np.asarray(data[name])
            except Exception:
                continue
            # FITS binary tables sometimes store map as one scalar column, and
            # sometimes as one row containing a vector.  Flatten both forms.
            arr = np.asarray(arr, float).reshape(-1)
            arr = arr[np.isfinite(arr) | ~np.isfinite(arr)]
            ns = _is_healpix_npix(len(arr))
            if ns is None:
                continue
            finite = int(np.isfinite(arr).sum())
            if finite <= 12:
                continue
            cl = str(name).lower()
            score = 0
            for kw, val in [("kappa", 30), ("map", 12), ("i_stokes", 8), ("temperature", 6), ("field", 2), ("mask", -50), ("hit", -20), ("noise", -10)]:
                if kw in cl:
                    score += val
            # Prefer larger maps and real finite coverage.
            score += min(20, int(math.log2(ns))) + min(10, finite // max(1, len(arr) // 10))
            if best is None or score > best_info.get("score", -10**9):
                best = arr.astype(float)
                best_info = {"hdu": int(hdu_i), "column": str(name), "nside": int(ns), "ordering": str(hdu.header.get("ORDERING", "RING")).upper(), "score": int(score), "npix": int(len(arr)), "path": str(map_path)}
    if best is None:
        return None, {"error": "no_healpix_binary_table_column", "path": str(map_path)}
    return best, best_info


def sample_map_values_for_points(map_path: Path, ra: np.ndarray, dec: np.ndarray, *, max_points: int = 5000, prefer_healpix: bool = True) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Sample WCS or HEALPix FITS maps at RA/Dec positions.

    v9.5 adds a pure-NumPy HEALPix RING nearest-neighbour path so T04/T05/T16
    no longer hard-require healpy for map-level diagnostics.  If healpy is
    installed it is still used first for validation and nested maps.
    """
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except Exception as e:
        return None, {"error": "astropy_missing_or_unavailable", "detail": str(e), "path": str(map_path)}
    map_path = Path(map_path)
    try:
        with fits.open(map_path, memmap=True) as hdul:
            # Ordinary WCS image maps.
            for hdu_i, hdu in enumerate(hdul):
                data = getattr(hdu, "data", None)
                if data is None:
                    continue
                arr = np.asarray(data)
                if arr.ndim >= 2 and hdu.header.get("CTYPE1"):
                    arr2 = arr
                    while arr2.ndim > 2:
                        arr2 = arr2[0]
                    try:
                        w = WCS(hdu.header)
                        n = min(len(ra), max_points)
                        pix = w.world_to_pixel_values(ra[:n], dec[:n])
                        x = np.asarray(pix[0]).round().astype(int)
                        y = np.asarray(pix[1]).round().astype(int)
                        vals = np.full(n, np.nan)
                        good = (x >= 0) & (y >= 0) & (y < arr2.shape[-2]) & (x < arr2.shape[-1])
                        vals[good] = np.asarray(arr2)[y[good], x[good]]
                        vals = np.where(np.isfinite(vals) & (np.abs(vals) < 1e30), vals, np.nan)
                        if np.isfinite(vals).sum() > 0:
                            return vals, {"mode": "fits_wcs", "hdu": int(hdu_i), "n_requested": int(n), "n_good_pixels": int(np.isfinite(vals).sum()), "map_shape": list(arr2.shape), "path": str(map_path)}
                    except Exception:
                        pass

            # HEALPix maps.  Try healpy, then pure-NumPy RING binary-table path.
            if prefer_healpix:
                n = min(len(ra), max_points)
                try:
                    import healpy as hp
                    hp_map = None; selected = None
                    for fld in [0, 1, 2, None]:
                        try:
                            m = hp.read_map(str(map_path), field=0 if fld is None else fld, verbose=False)
                            if np.asarray(m).ndim == 1 and len(m) >= 12:
                                hp_map = np.asarray(m, float); selected = {"field": int(0 if fld is None else fld)}; break
                        except Exception:
                            pass
                    if hp_map is not None:
                        is_planck = bool(re.search(r"planck|pr4|healpix|com_lensing", str(map_path), re.I))
                        if is_planck:
                            lon, lat, coordsys = _maybe_icrs_to_galactic(ra[:n], dec[:n])
                        else:
                            lon, lat, coordsys = np.asarray(ra[:n], float), np.asarray(dec[:n], float), "icrs_assumed"
                        theta = 0.5 * np.pi - np.deg2rad(lat); phi = np.deg2rad(lon)
                        pix = hp.ang2pix(hp.get_nside(hp_map), theta, phi, nest=False)
                        vals = hp_map[pix]
                        vals = np.where(np.isfinite(vals) & (np.abs(vals) < 1e30), vals, np.nan)
                        return vals, {"mode": "healpix_healpy", "coordsys": coordsys, "nside": int(hp.get_nside(hp_map)), "n_requested": int(n), "n_good_pixels": int(np.isfinite(vals).sum()), "selected": selected, "path": str(map_path)}
                except Exception as e_healpy:
                    healpy_error = str(e_healpy)
                else:
                    healpy_error = None

                hp_map, hp_info = _read_healpix_table_map_numpy(hdul, map_path)
                if hp_map is not None:
                    if hp_info.get("ordering", "RING").startswith("NEST"):
                        return None, {"error": "nested_healpix_requires_healpy", "detail": healpy_error, **hp_info, "install_hint": "conda install -c conda-forge healpy"}
                    is_planck = bool(re.search(r"planck|pr4|healpix|com_lensing", str(map_path), re.I))
                    if is_planck:
                        lon, lat, coordsys = _maybe_icrs_to_galactic(ra[:n], dec[:n])
                    else:
                        lon, lat, coordsys = np.asarray(ra[:n], float), np.asarray(dec[:n], float), "icrs_assumed"
                    theta = 0.5 * np.pi - np.deg2rad(lat); phi = np.deg2rad(lon)
                    pix = _ang2pix_ring_numpy(int(hp_info["nside"]), theta, phi)
                    vals = hp_map[pix]
                    vals = np.where(np.isfinite(vals) & (np.abs(vals) < 1e30), vals, np.nan)
                    hp_info.update({"mode": "healpix_numpy_ring", "coordsys": coordsys, "n_requested": int(n), "n_good_pixels": int(np.isfinite(vals).sum()), "healpy_unavailable_detail": healpy_error})
                    return vals, hp_info
    except Exception as e:
        return None, {"error": "map_sampling_failed", "detail": str(e), "path": str(map_path)}
    return None, {"error": "no_usable_map_hdu", "path": str(map_path)}


def _fits_map_score(q: Path) -> Tuple[int, int]:
    n = str(q).replace("\\", "/").lower()
    score = 0
    for kw, val in [
        ("kappa", 60), ("convergence", 50), ("lensing", 30), ("mv", 12), ("map", 20),
        ("klm", -80), ("alm", -80), ("curl", -60), ("mask", -70), ("noise", -30),
        ("meanfield", -30), ("mf", -20), ("dat_", 5), ("input", -5),
    ]:
        if kw in n:
            score += val
    if re.search(r"(kappa|convergence).*(map|dat)", n):
        score += 20
    return (-score, len(n))


def _choose_fits_map_from_path(path: Path, extract_dir: Path) -> Optional[Path]:
    path = Path(path)
    if str(path).lower().endswith((".fits", ".fit", ".fits.gz", ".fit.gz")) and not re.search(r"alm|klm|curl", path.name, re.I):
        return path
    root = extract_if_archive(path, extract_dir)
    fits = []
    for q in Path(root).rglob("*"):
        if q.is_file() and re.search(r"\.fits(\.gz)?$|\.fit(\.gz)?$", q.name, re.I):
            fits.append(q)
    if not fits:
        return None
    return sorted(fits, key=_fits_map_score)[0]


def _select_euclid_table(cache: Path, timeout: int = 90) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    attempts: List[Dict[str, Any]] = []
    tables, att = try_irsa_tap_query("SELECT table_name, description FROM TAP_SCHEMA.tables WHERE LOWER(table_name) LIKE '%euclid%'", cache, "euclid_tables", timeout=timeout)
    attempts.extend(att)
    if tables is None or pd is None or len(tables) == 0:
        return None, attempts
    name_col = "table_name" if "table_name" in tables.columns else tables.columns[0]
    candidates = [str(x) for x in tables[name_col].dropna().tolist()]
    ranked = []
    for t in candidates:
        tl = t.lower(); s = 0
        for kw, val in [("q1", 7), ("mer", 6), ("catalogue", 4), ("catalog", 4), ("source", 2), ("object", 2), ("agn", -3), ("image", -4), ("tile", -1)]:
            if kw in tl:
                s += val
        ranked.append((s, t))
    return sorted(ranked, reverse=True)[0][1] if ranked else None, attempts


def _euclid_table_columns(table: str, cache: Path, timeout: int = 90) -> Tuple[List[str], List[Dict[str, Any]]]:
    q = f"SELECT column_name, datatype, description FROM TAP_SCHEMA.columns WHERE table_name = '{table}'"
    df, att = try_irsa_tap_query(q, cache, f"euclid_columns_{slugify(table)}", timeout=timeout)
    if df is None or pd is None or len(df) == 0:
        return [], att
    ccol = "column_name" if "column_name" in df.columns else df.columns[0]
    return [str(x) for x in df[ccol].dropna().tolist()], att


def _pick_column_alias(columns: Sequence[str], aliases: Sequence[str], contains: Sequence[str] = ()) -> Optional[str]:
    by_lower = {str(c).lower(): str(c) for c in columns}
    for a in aliases:
        if a.lower() in by_lower:
            return by_lower[a.lower()]
    for c in columns:
        cl = str(c).lower()
        if contains and any(k in cl for k in contains):
            return str(c)
    return None


def load_euclid_q1_sample(cache: Path, timeout: int = 90, max_rows: int = 20000, force: bool = False) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    """Load Euclid Q1 with coordinates plus optional depth/photo-z/quality columns."""
    attempts: List[Dict[str, Any]] = []
    table, att = _select_euclid_table(cache, timeout=timeout)
    attempts.extend(att)
    if not table:
        return None, attempts + [{"ok": False, "error": "euclid_q1_catalog_not_found_via_irsa_tap"}]
    cols, att2 = _euclid_table_columns(table, cache, timeout=timeout)
    attempts.extend(att2)
    if not cols:
        # Fallback to old coordinate-only guesses.
        coords_sets = ["ra,dec", "RA,DEC", "ra,decl", "raj2000,dej2000", "source_id,ra,dec"]
        for coord_expr in coords_sets:
            df, att3 = try_irsa_tap_query(f"SELECT TOP {max_rows} {coord_expr} FROM {table} WHERE 1=1", cache, f"euclid_sample_{slugify(table)}", timeout=timeout)
            attempts.extend(att3)
            if df is not None and len(df) > 20:
                lower = {c.lower(): c for c in df.columns}
                ra = lower.get("ra") or lower.get("raj2000")
                dec = lower.get("dec") or lower.get("decl") or lower.get("dej2000")
                if ra and dec:
                    return df.rename(columns={ra: "ra", dec: "dec"}), attempts + [{"selected_table": table, "ok": True, "enriched": False}]
        return None, attempts + [{"ok": False, "error": "euclid_q1_coordinate_columns_not_found", "selected_table": table}]

    ra_col = _pick_column_alias(cols, ["ra", "RA", "raj2000", "RAJ2000"])
    dec_col = _pick_column_alias(cols, ["dec", "DEC", "decl", "dej2000", "DEJ2000"])
    if not (ra_col and dec_col):
        return None, attempts + [{"ok": False, "error": "euclid_q1_coordinate_columns_not_found", "selected_table": table, "columns_sample": cols[:40]}]
    optional: List[str] = []
    alias_groups = [
        (["source_id", "object_id", "id"], []),
        (["z", "photoz", "phot_z", "z_phot", "redshift"], ["photoz", "phot_z", "redshift"]),
        (["mag", "vis_mag", "mag_vis", "flux_vis", "nisp_h_mag", "h_mag", "i_mag", "r_mag"], ["mag", "flux", "vis", "nisp"]),
        (["flag", "flags", "quality_flag", "mask", "masked"], ["flag", "mask", "quality"]),
        (["snr", "sn", "depth", "limmag", "lim_mag", "exptime"], ["snr", "depth", "lim", "exptime"]),
    ]
    for exacts, contains in alias_groups:
        c = _pick_column_alias(cols, exacts, contains)
        if c and c not in (ra_col, dec_col) and c not in optional:
            optional.append(c)
    select_cols = [ra_col, dec_col] + optional[:10]
    # Quote only simple ADQL identifiers are avoided here because IRSA accepts plain TAP column names.
    q = f"SELECT TOP {max_rows} {', '.join(select_cols)} FROM {table} WHERE {ra_col} IS NOT NULL AND {dec_col} IS NOT NULL"
    df, att3 = try_irsa_tap_query(q, cache, f"euclid_enriched_sample_{slugify(table)}", timeout=timeout)
    attempts.extend(att3)
    if df is None or len(df) <= 20:
        return None, attempts + [{"ok": False, "error": "euclid_q1_enriched_query_failed", "selected_table": table, "selected_columns": select_cols}]
    rename = {ra_col: "ra", dec_col: "dec"}
    df = df.rename(columns=rename)
    return df, attempts + [{"selected_table": table, "ok": True, "enriched": True, "selected_optional_columns": optional[:10], "available_columns_sample": cols[:80]}]


def euclid_depth_proxy(df: Any) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Return a rough depth/completeness proxy from optional Euclid columns."""
    if df is None or pd is None:
        return None, {"error": "no_dataframe"}
    nums = find_numeric_columns(df)
    scored = []
    for c in nums:
        cl = str(c).lower()
        if cl in ("ra", "dec") or cl.startswith("ra") or cl.startswith("dec"):
            continue
        score = 0
        for kw, val in [("lim", 20), ("depth", 20), ("snr", 15), ("sn_", 10), ("mag", 8), ("flux", 6), ("vis", 4), ("nisp", 4), ("flag", -20), ("mask", -20), ("id", -20), ("z", -5)]:
            if kw in cl:
                score += val
        arr = numeric_array(df, c)
        finite = arr[np.isfinite(arr)]
        if len(finite) >= max(20, int(0.02 * len(arr))) and np.nanstd(finite) > 0:
            scored.append((score, c, arr))
    scored = [x for x in scored if x[0] > 0]
    if not scored:
        return None, {"error": "no_depth_or_magnitude_proxy_columns", "numeric_columns_sample": [str(c) for c in nums[:30]]}
    scored.sort(reverse=True, key=lambda x: x[0])
    score, col, arr = scored[0]
    return np.asarray(arr, float), {"column": str(col), "score": int(score), "note": "Used as an automated depth/completeness proxy, not a calibrated survey mask."}


def _cell_patch_table(ra: np.ndarray, dec: np.ndarray, labels: np.ndarray, depth: Optional[np.ndarray] = None, nbin: int = 4) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lab in sorted(set(labels)):
        if lab < 0:
            continue
        m = labels == lab
        if int(np.sum(m)) < nbin * nbin * 3:
            continue
        r = ra[m]; d = dec[m]
        dep = depth[m] if depth is not None and len(depth) == len(ra) else None
        rb = np.linspace(np.nanmin(r), np.nanmax(r), nbin + 1)
        db = np.linspace(np.nanmin(d), np.nanmax(d), nbin + 1)
        for i in range(nbin):
            for j in range(nbin):
                mm = (r >= rb[i]) & ((r < rb[i + 1]) if i < nbin - 1 else (r <= rb[i + 1])) & (d >= db[j]) & ((d < db[j + 1]) if j < nbin - 1 else (d <= db[j + 1]))
                rows.append({"field": int(lab), "i": int(i), "j": int(j), "count": int(np.sum(mm)), "depth_median": float(np.nanmedian(dep[mm])) if dep is not None and np.sum(mm) else None})
    return rows


def patch_count_depth_diagnostic(ra: np.ndarray, dec: np.ndarray, labels: np.ndarray, depth: Optional[np.ndarray]) -> Dict[str, Any]:
    rows = _cell_patch_table(ra, dec, labels, depth=depth)
    if not rows:
        return {"error": "no_patch_cells"}
    counts = np.asarray([r["count"] for r in rows], float)
    depths = np.asarray([r["depth_median"] if r["depth_median"] is not None else np.nan for r in rows], float)
    out: Dict[str, Any] = {"n_cells": int(len(rows)), "count_cv": float(np.nanstd(counts, ddof=1) / np.nanmean(counts)) if np.nanmean(counts) else None, "patch_rows_first20": rows[:20]}
    if np.isfinite(depths).sum() >= 6 and np.nanstd(depths) > 0:
        corr = safe_corr(depths, counts)
        X = np.vstack([np.ones(np.isfinite(depths).sum()), depths[np.isfinite(depths)]]).T
        y = np.log1p(counts[np.isfinite(depths)])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            out.update({"count_depth_spearman": corr, "depth_logcount_slope": float(beta[1]), "depth_residual_cv": float(np.std(resid, ddof=1) / (abs(np.mean(y)) or 1.0))})
        except Exception as e:
            out["depth_fit_error"] = str(e)
    else:
        out["depth_control"] = "unavailable"
    return out


def load_pantheon_covariance(cache: Path, timeout: int = 90, force: bool = False) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
    urls = [
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov",
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STATONLY.cov",
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov",
    ]
    p, attempts = download_first(urls, cache / "pantheon", timeout=timeout, force=force, max_bytes=80 * 1024 * 1024)
    if p is None:
        return None, attempts
    try:
        txt = p.read_text(errors="ignore").split()
        vals = np.asarray([float(x) for x in txt], float)
        n0 = int(round(vals[0])) if len(vals) else 0
        if n0 > 0 and len(vals) == n0 * n0 + 1:
            cov = vals[1:].reshape(n0, n0)
        else:
            n = int(round(math.sqrt(len(vals))))
            cov = vals.reshape(n, n)
        return cov, attempts + [{"ok": True, "path": str(p), "cov_shape": list(cov.shape), "note": "Pantheon+ covariance parsed"}]
    except Exception as e:
        return None, attempts + [{"ok": False, "path": str(p), "error": f"could_not_parse_pantheon_covariance: {e}"}]


def _pantheon_columns(df: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    cols = {str(c).lower(): c for c in df.columns}
    zc = cols.get("zhel") or cols.get("zcmb") or cols.get("z_hd") or cols.get("z")
    mc = cols.get("mu_sh0es") or cols.get("mu") or cols.get("m_b_corr") or cols.get("mb_corr")
    ec = cols.get("mu_sh0es_err_diag") or cols.get("muerr") or cols.get("dmu") or cols.get("err")
    return zc, mc, ec


def fit_pantheon_nu_cov_like(df: Any, cov: Optional[np.ndarray] = None, zmin: float = 0.005, zmax: float = 2.5, nu_bound: float = 0.2) -> Dict[str, Any]:
    """Pantheon+ ν-like fit using the public covariance when available."""
    if pd is None or df is None:
        return {"error": "no_dataframe"}
    zc, mc, ec = _pantheon_columns(df)
    if zc is None or mc is None:
        return {"error": "missing_z_or_mu", "columns": list(map(str, df.columns[:60]))}
    z_all = numeric_array(df, zc); mu_all = numeric_array(df, mc)
    idx = np.where(np.isfinite(z_all) & np.isfinite(mu_all) & (z_all > zmin) & (z_all < zmax))[0]
    if len(idx) < 30:
        return {"error": "too_few_sne", "n": int(len(idx))}
    z = z_all[idx]; mu = mu_all[idx]
    if cov is not None and cov.shape[0] >= len(df) and cov.shape[1] >= len(df):
        C = np.asarray(cov[np.ix_(idx, idx)], float)
        mode = "full_public_covariance"
    else:
        err = numeric_array(df, ec)[idx] if ec else np.full(len(idx), 0.15)
        err = np.where(np.isfinite(err) & (err > 0), err, np.nanmedian(err[np.isfinite(err) & (err > 0)]) if np.any(np.isfinite(err) & (err > 0)) else 0.15)
        C = np.diag(err ** 2)
        mode = "diagonal_error_fallback"
    # Stabilize covariance if needed.
    diag = np.diag(C).copy()
    jitter = max(1e-12, 1e-10 * float(np.nanmedian(diag[diag > 0])) if np.any(diag > 0) else 1e-12)
    C = C + np.eye(len(C)) * jitter
    order = np.argsort(z); z = z[order]; mu = mu[order]; C = C[np.ix_(order, order)]
    try:
        from scipy.linalg import cho_factor, cho_solve
        cf = cho_factor(C, lower=True, check_finite=False)
        solve = lambda b: cho_solve(cf, b, check_finite=False)
    except Exception:
        pinv = np.linalg.pinv(C)
        solve = lambda b: pinv @ b
    one = np.ones_like(mu)
    Cinv_one = solve(one)
    denom_one = float(one @ Cinv_one)

    def base_mu(om: float, nu: float) -> Optional[np.ndarray]:
        if not (0.05 < om < 0.6 and -nu_bound <= nu <= nu_bound):
            return None
        zz = np.r_[0.0, z]
        e2 = om * (1 + zz) ** 3 + (1 - om) * (1 + nu * np.log1p(zz))
        if np.any(e2 <= 0) or not np.all(np.isfinite(e2)):
            return None
        invE = 1.0 / np.sqrt(e2)
        dc = np.zeros_like(zz)
        dc[1:] = np.cumsum(0.5 * (invE[1:] + invE[:-1]) * np.diff(zz))
        dl = (1 + z) * C_LIGHT / 70.0 * dc[1:]
        if np.any(dl <= 0):
            return None
        return 5 * np.log10(dl)

    def chi2_for(om: float, nu: float) -> float:
        b = base_mu(float(om), float(nu))
        if b is None:
            return 1e100
        alpha = float((one @ solve(mu - b)) / denom_one)
        r = mu - b - alpha
        return float(r @ solve(r))

    if optimize is None:
        return {"error": "scipy_optimize_missing", "mode": mode, "n_sne": int(len(z))}
    res = optimize.minimize(lambda p: chi2_for(p[0], p[1]), x0=np.array([0.3, 0.0]), method="Nelder-Mead", options={"maxiter": 500})
    om, nu = [float(x) for x in (res.x if res.success else [np.nan, np.nan])]
    chi_best = chi2_for(om, nu) if np.isfinite(om) and np.isfinite(nu) else float("nan")
    res0 = optimize.minimize(lambda p: chi2_for(p[0], 0.0), x0=np.array([0.3]), method="Nelder-Mead", options={"maxiter": 250})
    om0 = float(res0.x[0]) if res0.success else 0.3
    chi0 = chi2_for(om0, 0.0)
    # Crude profile uncertainty from local curvature in nu with Omega fixed at best.
    grid = np.linspace(max(-nu_bound, nu - 0.05), min(nu_bound, nu + 0.05), 41) if np.isfinite(nu) else np.linspace(-0.05, 0.05, 21)
    chi_grid = np.asarray([chi2_for(om, v) for v in grid], float) if np.isfinite(om) else np.full_like(grid, np.nan)
    sig = None
    try:
        ok = np.isfinite(chi_grid) & (np.abs(grid - nu) < 0.04)
        if np.sum(ok) >= 5:
            co = np.polyfit(grid[ok] - nu, chi_grid[ok] - np.nanmin(chi_grid[ok]), 2)
            if co[0] > 0:
                sig = float(math.sqrt(1.0 / co[0]))
    except Exception:
        pass
    return {
        "mode": mode,
        "n_sne": int(len(z)),
        "omega_m": om,
        "nu_like": nu,
        "nu_like_sigma_profile_approx": sig,
        "chi2_best": chi_best,
        "chi2_nu0_profile_omega_m": chi0,
        "delta_chi2_vs_nu0": float(chi0 - chi_best) if np.isfinite(chi0) and np.isfinite(chi_best) else None,
        "hit_nu_bound": bool(np.isfinite(nu) and abs(abs(nu) - nu_bound) < 1e-3),
        "model_note": "Covariance-aware Pantheon+ ν-like audit; still phenomenological, but no longer diagonal-only SN proxy when covariance is available.",
    }


def _read_numeric_matrix(path: Path) -> Optional[np.ndarray]:
    try:
        rows = []
        for line in Path(path).read_text(errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = []
            ok = True
            for tok in re.split(r"\s+|,", s):
                if not tok:
                    continue
                try:
                    vals.append(float(tok))
                except Exception:
                    ok = False; break
            if ok and vals:
                rows.append(vals)
        if not rows:
            return None
        width = max(len(r) for r in rows)
        if min(len(r) for r in rows) != width:
            return None
        return np.asarray(rows, float)
    except Exception:
        return None


def summarize_dr2_bao_cov_likelihood(paths: Sequence[Path], max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Covariance-aware DESI DR2 mean-vector trend diagnostic.

    This is not a full CLASS/CAMB RVM likelihood.  It is a covariance-aware
    GLS test of a redshift trend after observable-wise normalization, using the
    public DESI DR2 mean/cov pair instead of treating 13 means as independent.
    """
    mean_paths = desi_dr2_mean_paths(paths)
    candidates = [p for p in mean_paths if "ALL_GCcomb" in str(p)] + mean_paths
    for mp in candidates:
        cov_candidates = []
        s = str(mp)
        cov_candidates.append(Path(s.replace("_mean", "_cov")))
        cov_candidates += [p for p in paths if Path(str(p)).name == Path(s.replace("_mean", "_cov")).name]
        covp = next((p for p in cov_candidates if Path(p).exists()), None)
        if covp is None:
            continue
        df = read_table_any(mp, max_rows=max_rows)
        C = _read_numeric_matrix(covp)
        if df is None or C is None:
            continue
        nums = find_numeric_columns(df)
        if len(nums) < 2:
            continue
        z = numeric_array(df, nums[0]); y = numeric_array(df, nums[1])
        obs = None
        if len(nums) >= 3:
            obs = numeric_array(df, nums[2])
        else:
            for c in df.columns:
                if c not in nums:
                    vals = df[c].astype(str).to_numpy()
                    if len(vals) == len(y):
                        obs = vals; break
        m = np.isfinite(z) & np.isfinite(y) & (z > 0) & (z < 6)
        if int(np.sum(m)) < 4:
            continue
        z = z[m]; y = y[m]
        idx = np.where(m)[0]
        if C.shape[0] < max(idx) + 1 or C.shape[1] < max(idx) + 1:
            continue
        C = C[np.ix_(idx, idx)].astype(float)
        if obs is None or len(obs) != len(m):
            groups = np.zeros(len(y), dtype=int)
        else:
            groups = np.asarray(obs)[m]
        yn = np.zeros_like(y, dtype=float)
        scale = np.ones_like(y, dtype=float)
        for g in sorted(set(map(str, groups))):
            gg = np.asarray([str(x) == g for x in groups])
            yy = y[gg]
            med = float(np.nanmedian(yy)); sc = mad(yy) or float(np.nanstd(yy)) or 1.0
            yn[gg] = (yy - med) / sc
            scale[gg] = sc
        Cn = C / np.outer(scale, scale)
        Cn = Cn + np.eye(len(Cn)) * max(1e-12, 1e-10 * float(np.nanmedian(np.diag(Cn)[np.diag(Cn) > 0])) if np.any(np.diag(Cn) > 0) else 1e-12)
        try:
            Ci = np.linalg.pinv(Cn)
            x = z - float(np.mean(z))
            A0 = np.ones((len(z), 1))
            A1 = np.vstack([np.ones_like(x), x]).T
            def gls(A):
                covb = np.linalg.pinv(A.T @ Ci @ A)
                beta = covb @ (A.T @ Ci @ yn)
                r = yn - A @ beta
                chi = float(r @ Ci @ r)
                return beta, covb, chi
            b0, cb0, chi0 = gls(A0)
            b1, cb1, chi1 = gls(A1)
            slope = float(b1[1]); slope_sig = float(math.sqrt(abs(cb1[1, 1]))) if cb1.shape == (2, 2) else None
            return {
                "mode": "desi_dr2_public_covariance_gls_trend",
                "mean_path": str(mp), "cov_path": str(covp), "n_points": int(len(z)),
                "z_min": float(np.nanmin(z)), "z_max": float(np.nanmax(z)),
                "slope_norm_per_z": slope, "slope_sigma": slope_sig,
                "slope_zscore": float(slope / slope_sig) if slope_sig and slope_sig > 0 else None,
                "chi2_constant": chi0, "chi2_linear": chi1,
                "delta_chi2_linear_vs_constant": float(chi0 - chi1),
                "spearman_on_normalized_means": safe_corr(z, yn),
                "note": "Covariance-aware trend screen; improves on v9.4 mean-vector Spearman but is still not a full RVM BAO likelihood.",
            }
        except Exception as e:
            return {"error": "desi_covariance_gls_failed", "detail": str(e), "mean_path": str(mp), "cov_path": str(covp)}
    return {"error": "no_matching_desi_dr2_mean_cov_pair", "n_paths": int(len(paths))}


def load_direct_detection_public_curves(cache: Path, timeout: int = 90, force: bool = False) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Load a wider automated set of public direct-detection curve/data tables."""
    tables, attempts = load_xenon_limit_curves(cache, timeout=timeout, force=force)
    # Additional public sources discovered as lightweight CSV/HEPData endpoints.
    urls = [
        # PandaX-4T light-DM public CSVs; useful mostly below CCDR heavy window but improves null/readiness metadata.
        "https://pandax.sjtu.edu.cn/sites/default/files/run0_data.csv",
        "https://pandax.sjtu.edu.cn/sites/default/files/run1_data.csv",
        # HEPData record for LZ 4.2 tonne-year result: use csv table downloads if exposed by stable record IDs.
        "https://www.hepdata.net/download/record/155182?format=csv",
        "https://www.hepdata.net/download/record/155182?format=yaml",
    ]
    for u in urls:
        try:
            p = download_file(u, cache / "direct_detection_extra", filename=slugify(url_basename(u) or u), timeout=timeout, force=force, max_bytes=80 * 1024 * 1024)
            attempts.append({"url": u, "ok": True, "path": str(p), "source_family": "direct_detection_extra"})
            if zipfile.is_zipfile(p):
                root = extract_archive(p, cache / "direct_detection_extra" / (p.stem + "_extract"))
                files = list(root.rglob("*.csv")) + list(root.rglob("*.txt")) + list(root.rglob("*.dat"))
            else:
                files = [p]
            for q in files[:80]:
                df = read_table_any(q)
                if df is not None and df.shape[1] >= 2:
                    tables.append(df)
        except Exception as e:
            attempts.append({"url": u, "ok": False, "error": str(e), "source_family": "direct_detection_extra"})
    return tables, attempts


def _numeric_column_profile(df: Any, max_cols: int = 40) -> List[Dict[str, Any]]:
    rows = []
    if df is None:
        return rows
    for c in find_numeric_columns(df)[:max_cols]:
        x = numeric_array(df, c); f = x[np.isfinite(x)]
        if len(f):
            rows.append({"column": str(c), "n": int(len(f)), "min": float(np.nanmin(f)), "max": float(np.nanmax(f)), "median": float(np.nanmedian(f))})
    return rows


def firas_unit_audit(freq: np.ndarray, I: np.ndarray, err: np.ndarray) -> Dict[str, Any]:
    return {
        "freq_min": float(np.nanmin(freq)) if len(freq) else None,
        "freq_max": float(np.nanmax(freq)) if len(freq) else None,
        "intensity_median": float(np.nanmedian(I)) if len(I) else None,
        "error_median": float(np.nanmedian(err)) if len(err) else None,
        "warning": "FIRAS public monopole tables have legacy units; official covariance/residual likelihood is still required for a hard P28 bound.",
    }


def highz_group_bootstrap_summary(df: Any, nboot: int = 300, seed: int = 12345) -> Dict[str, Any]:
    """Galaxy/object-level bootstrap for high-z acceleration proxies."""
    zc, vc, rc, info = choose_highz_acceleration_columns(df)
    if not (zc and vc and rc):
        return {"error": "no_highz_columns", "column_selection": info}
    z, g, meta = acceleration_proxy_from_highz(df, zc, vc, rc)
    qmask, qinfo = highz_quality_mask(df, zc, vc, rc, z, g)
    if pd is None or df is None:
        ids = np.arange(len(z))
        id_col = None
    else:
        id_col = None
        for c in df.columns:
            cl = str(c).lower()
            if any(k in cl for k in ["gal", "object", "source", "id", "name"]):
                vals = df[c].astype(str).to_numpy()
                if len(set(vals[:min(len(vals), 200)])) > 5:
                    id_col = c; break
        ids = df[id_col].astype(str).to_numpy() if id_col is not None else np.arange(len(z))
    m = qmask & np.isfinite(z) & np.isfinite(g) & (g > 0)
    if int(np.sum(m)) < 10:
        return {"error": "too_few_quality_points", "n": int(np.sum(m)), "quality_cuts": qinfo}
    z = z[m]; g = g[m]; ids = np.asarray(ids)[m]
    groups = sorted(set(map(str, ids)))
    rng = np.random.default_rng(seed)
    vals = []
    slopes = []
    for _ in range(nboot):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        mm = np.concatenate([np.where(np.asarray([str(x) for x in ids]) == ch)[0] for ch in chosen]) if len(chosen) else np.array([], dtype=int)
        if len(mm) < 5:
            continue
        vals.append(float(np.nanmean(g[mm])))
        co = safe_polyfit(z[mm], np.log10(g[mm]), 1)
        if co:
            slopes.append(float(co[0]))
    co_all = safe_polyfit(z, np.log10(g), 1)
    return {
        "id_column": str(id_col) if id_col is not None else None,
        "n_quality_points": int(len(z)),
        "n_groups": int(len(groups)),
        "mean_a_proxy_m_s2": float(np.nanmean(g)),
        "mean_a_proxy_boot_sigma": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else None,
        "log_g_vs_z_coef": co_all,
        "slope_boot_median": float(np.nanmedian(slopes)) if slopes else None,
        "slope_boot_p16": float(np.nanpercentile(slopes, 16)) if slopes else None,
        "slope_boot_p84": float(np.nanpercentile(slopes, 84)) if slopes else None,
        "quality_cuts": qinfo,
        "radius_meta": meta,
    }


def download_planck_lensing_or_spectra(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False, prefer_map: bool = False) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    """v9.5 Planck downloader with direct spectrum fallbacks and map support."""
    attempts: List[Dict[str, Any]]=[]
    if prefer_map:
        if not allow_large:
            return None,[{"ok":False,"reason":"large_download_not_enabled","note":"Planck lensing maps are large HEALPix products; rerun with --allow-large. v9.5 can sample RING maps without healpy."}]
        asset,info=_github_release_asset_url("carronj","planck_PR4_lensing",r"PR42018like_maps\.tar",timeout=timeout,tag_hint="Data")
        attempts.append(info)
        candidates=[]
        if asset: candidates.append(asset)
        candidates += [
            "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/lensing/COM_Lensing_4096_R3.00.tar",
            "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/lensing/COM_Lensing-Szdeproj_4096_R3.00.tar",
        ]
        p,att=download_first(candidates,cache/"planck_lensing",timeout=timeout,force=force,max_bytes=None)
        attempts.extend(att)
        if p is None: return None, attempts
        root=extract_if_archive(p,cache/"planck_lensing"/"maps_extracted")
        fits=[]
        for q in Path(root).rglob("*"):
            if q.is_file() and re.search(r"\.fits(\.gz)?$", q.name, re.I): fits.append(q)
        if not fits:
            attempts.append({"ok":False,"error":"downloaded_planck_archive_but_no_FITS_map_found","path":str(p)})
            return None, attempts
        best=sorted(fits,key=_fits_map_score)[0]
        attempts.append({"ok":True,"selected_map":str(best),"note":"selected Planck/PR4 HEALPix FITS map; v9.5 samples RING maps without healpy where possible"})
        return best, attempts
    direct = [
        # PLA product-action URLs are public and stable when direct LAMBDA links move.
        "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-full_R3.01.txt",
        "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-binned_R3.01.txt",
        "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt",
    ]
    pages=["https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/", "https://lambda.gsfc.nasa.gov/product/planck/planck_prod_table.html"]
    candidates=[]
    for page in pages:
        try:
            links=discover_links(page,pattern=r"(PowerSpect|COM_Power|TT|TE|EE|BB|Cls|cl).*\.(txt|dat|fits|fits\.gz)$",timeout=timeout)
            attempts.append({"url":page,"ok":True,"n_links":len(links)})
            candidates.extend(links)
        except Exception as e:
            attempts.append({"url":page,"ok":False,"error":str(e)})
    p,att=download_first(direct + candidates[:30],cache/"planck",timeout=timeout,force=force,max_bytes=300*1024*1024)
    attempts.extend(att)
    return p,attempts

# ------------------------- v9.6 overrides -------------------------
# These helpers are appended intentionally so they override previous v9.5
# definitions at import time.  They focus on result-quality and data-limited
# fixes, not cosmetic refactors.


def _is_harmonic_name(path_or_name: Any) -> bool:
    n = str(path_or_name).replace('\\', '/').lower()
    return bool(re.search(r'(^|[/_\-.])(a?lm|klm|alm|curl)([/_\-.]|$)|kappa_alm|klm_', n))


def _fits_product_kind(path: Path) -> Dict[str, Any]:
    """Lightweight FITS inspection for choosing map products.

    kind is one of: wcs_image, healpix_table, harmonic_alm, table, unknown.
    """
    info: Dict[str, Any] = {"path": str(path), "name_harmonic_hint": _is_harmonic_name(path)}
    try:
        from astropy.io import fits
        with fits.open(path, memmap=True) as hdul:
            for hdu_i, hdu in enumerate(hdul):
                data = getattr(hdu, 'data', None)
                if data is None:
                    continue
                hdr = getattr(hdu, 'header', {})
                arr = np.asarray(data)
                if arr.ndim >= 2 and hdr.get('CTYPE1'):
                    info.update({"kind": "wcs_image", "hdu": int(hdu_i), "shape": list(arr.shape)})
                    return info
                if hasattr(data, 'columns'):
                    names = list(getattr(data.columns, 'names', []) or [])
                    low = ' '.join(map(str.lower, names))
                    if re.search(r'alm|klm|real|imag|lmax|ell|emm|m_', low) or _is_harmonic_name(path):
                        # Some Planck/ACT harmonic files have column names such as REAL/IMAG.
                        info.update({"kind": "harmonic_alm", "hdu": int(hdu_i), "columns": names[:20]})
                        # Keep scanning in case the same file also contains a pixel table.
                    hp_map, hp_info = _read_healpix_table_map_numpy(hdul, path)
                    if hp_map is not None:
                        hp_info.pop('score', None)
                        info.update({"kind": "healpix_table", **hp_info})
                        return info
            info.setdefault('kind', 'harmonic_alm' if _is_harmonic_name(path) else 'unknown')
            return info
    except Exception as e:
        info.update({"kind": "unreadable", "error": str(e)})
        return info


def _fits_map_score(q: Path) -> Tuple[int, int]:
    n = str(q).replace('\\', '/').lower()
    info = _fits_product_kind(q)
    kind = info.get('kind')
    score = 0
    if kind == 'wcs_image':
        score += 1000
    elif kind == 'healpix_table':
        score += 900
    elif kind == 'harmonic_alm':
        score -= 500
    elif kind == 'unreadable':
        score -= 800
    for kw, val in [
        ('kappa', 100), ('convergence', 90), ('lensing', 50), ('mv', 20), ('map', 30),
        ('dat_', 10), ('input', -5), ('mask', -120), ('noise', -80), ('sim', -50),
        ('curl', -100), ('alm', -160), ('klm', -160), ('meanfield', -60), ('mf_', -40),
    ]:
        if kw in n:
            score += val
    # Sorting ascending: larger score should come first.
    return (-score, len(n))


def _choose_fits_map_from_path(path: Path, extract_dir: Path) -> Optional[Path]:
    path = Path(path)
    if str(path).lower().endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        info = _fits_product_kind(path)
        if info.get('kind') in ('wcs_image', 'healpix_table'):
            return path
        # harmonic product is allowed only as last resort; sampler will require healpy.
        if info.get('kind') == 'harmonic_alm':
            return path
    root = extract_if_archive(path, extract_dir)
    fits = [q for q in Path(root).rglob('*') if q.is_file() and re.search(r'\.fits(\.gz)?$|\.fit(\.gz)?$', q.name, re.I)]
    if not fits:
        return None
    inspected = [(q, _fits_product_kind(q)) for q in fits]
    usable = [q for q, info in inspected if info.get('kind') in ('wcs_image', 'healpix_table')]
    harmonic = [q for q, info in inspected if info.get('kind') == 'harmonic_alm']
    pool = usable or harmonic or [q for q, _ in inspected]
    return sorted(pool, key=_fits_map_score)[0]


def _alm_table_to_complex(hdul: Any, map_path: Path) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Try to parse simple alm/klm FITS tables into a healpy alm array.

    This handles common REAL/IMAG style tables only.  It intentionally does
    not implement spherical-harmonic transforms without healpy.
    """
    for hdu_i, hdu in enumerate(hdul):
        data = getattr(hdu, 'data', None)
        if data is None or not hasattr(data, 'columns'):
            continue
        names = list(getattr(data.columns, 'names', []) or [])
        low = {str(n).lower(): str(n) for n in names}
        real_col = next((low[k] for k in low if k in ('real', 're', 'alm_real', 'klm_real') or 'real' in k), None)
        imag_col = next((low[k] for k in low if k in ('imag', 'im', 'alm_imag', 'klm_imag') or 'imag' in k), None)
        if real_col is None:
            continue
        try:
            re_arr = np.asarray(data[real_col], float).reshape(-1)
            im_arr = np.asarray(data[imag_col], float).reshape(-1) if imag_col else np.zeros_like(re_arr)
            alm = re_arr + 1j * im_arr
            if len(alm) < 10:
                continue
            return alm, {"hdu": int(hdu_i), "real_col": real_col, "imag_col": imag_col, "n_alm": int(len(alm)), "path": str(map_path)}
        except Exception:
            continue
    return None, {"error": "no_simple_real_imag_alm_columns", "path": str(map_path)}


def _alm_lmax_from_size(nalm: int) -> Optional[int]:
    # healpy.Alm.getlmax equivalent for mmax=lmax: nalm=(lmax+1)(lmax+2)/2
    try:
        l = int((math.sqrt(8 * int(nalm) + 1) - 3) / 2)
        return l if (l + 1) * (l + 2) // 2 == int(nalm) else None
    except Exception:
        return None


def sample_map_values_for_points(map_path: Path, ra: np.ndarray, dec: np.ndarray, *, max_points: int = 5000, prefer_healpix: bool = True) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Sample WCS/HEALPix pixel maps; convert alm/klm only with healpy.

    v9.6 explicitly distinguishes pixel maps from harmonic-space products.
    The pure-NumPy HEALPix path samples RING binary-table maps; it cannot do
    alm2map.  If an alm/klm product is selected, this function tries healpy and
    otherwise returns an accurate data_limited reason instead of `no_usable_hdu`.
    """
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except Exception as e:
        return None, {"error": "astropy_missing_or_unavailable", "detail": str(e), "path": str(map_path)}
    map_path = Path(map_path)
    try:
        with fits.open(map_path, memmap=True) as hdul:
            # 1) WCS images.
            for hdu_i, hdu in enumerate(hdul):
                data = getattr(hdu, 'data', None)
                if data is None:
                    continue
                arr = np.asarray(data)
                if arr.ndim >= 2 and hdu.header.get('CTYPE1'):
                    arr2 = arr
                    while arr2.ndim > 2:
                        arr2 = arr2[0]
                    try:
                        w = WCS(hdu.header)
                        n = min(len(ra), max_points)
                        pix = w.world_to_pixel_values(ra[:n], dec[:n])
                        x = np.asarray(pix[0]).round().astype(int)
                        y = np.asarray(pix[1]).round().astype(int)
                        vals = np.full(n, np.nan)
                        good = (x >= 0) & (y >= 0) & (y < arr2.shape[-2]) & (x < arr2.shape[-1])
                        vals[good] = np.asarray(arr2)[y[good], x[good]]
                        vals = np.where(np.isfinite(vals) & (np.abs(vals) < 1e30), vals, np.nan)
                        if np.isfinite(vals).sum() > 0:
                            return vals, {"mode": "fits_wcs", "hdu": int(hdu_i), "n_requested": int(n), "n_good_pixels": int(np.isfinite(vals).sum()), "map_shape": list(arr2.shape), "path": str(map_path)}
                    except Exception:
                        pass

            # 2) HEALPix pixel tables: healpy first, pure-NumPy RING second.
            n = min(len(ra), max_points)
            healpy_error = None
            if prefer_healpix:
                try:
                    import healpy as hp
                    hp_map = None; selected = None
                    for fld in [0, 1, 2, None]:
                        try:
                            m = hp.read_map(str(map_path), field=0 if fld is None else fld, verbose=False)
                            if np.asarray(m).ndim == 1 and len(m) >= 12:
                                hp_map = np.asarray(m, float); selected = {"field": int(0 if fld is None else fld)}; break
                        except Exception:
                            pass
                    if hp_map is not None:
                        is_planck = bool(re.search(r'planck|pr4|healpix|com_lensing', str(map_path), re.I))
                        if is_planck:
                            lon, lat, coordsys = _maybe_icrs_to_galactic(ra[:n], dec[:n])
                        else:
                            lon, lat, coordsys = np.asarray(ra[:n], float), np.asarray(dec[:n], float), 'icrs_assumed'
                        theta = 0.5 * np.pi - np.deg2rad(lat); phi = np.deg2rad(lon)
                        pix = hp.ang2pix(hp.get_nside(hp_map), theta, phi, nest=False)
                        vals = hp_map[pix]
                        vals = np.where(np.isfinite(vals) & (np.abs(vals) < 1e30), vals, np.nan)
                        return vals, {"mode": "healpix_healpy", "coordsys": coordsys, "nside": int(hp.get_nside(hp_map)), "n_requested": int(n), "n_good_pixels": int(np.isfinite(vals).sum()), "selected": selected, "path": str(map_path)}
                except Exception as e:
                    healpy_error = str(e)

                hp_map, hp_info = _read_healpix_table_map_numpy(hdul, map_path)
                if hp_map is not None:
                    if hp_info.get('ordering', 'RING').startswith('NEST'):
                        return None, {"error": "nested_healpix_requires_healpy", "detail": healpy_error, **hp_info, "install_hint": "conda install -c conda-forge healpy"}
                    is_planck = bool(re.search(r'planck|pr4|healpix|com_lensing', str(map_path), re.I))
                    if is_planck:
                        lon, lat, coordsys = _maybe_icrs_to_galactic(ra[:n], dec[:n])
                    else:
                        lon, lat, coordsys = np.asarray(ra[:n], float), np.asarray(dec[:n], float), 'icrs_assumed'
                    theta = 0.5 * np.pi - np.deg2rad(lat); phi = np.deg2rad(lon)
                    pix = _ang2pix_ring_numpy(int(hp_info['nside']), theta, phi)
                    vals = hp_map[pix]
                    vals = np.where(np.isfinite(vals) & (np.abs(vals) < 1e30), vals, np.nan)
                    hp_info.update({"mode": "healpix_numpy_ring", "coordsys": coordsys, "n_requested": int(n), "n_good_pixels": int(np.isfinite(vals).sum()), "healpy_unavailable_detail": healpy_error})
                    return vals, hp_info

            # 3) Harmonic-space alm/klm fallback with healpy.alm2map only.
            alm, alm_info = _alm_table_to_complex(hdul, map_path)
            if alm is not None or _is_harmonic_name(map_path):
                try:
                    import healpy as hp
                    if alm is None:
                        return None, {"error": "alm_product_not_parseable", **alm_info, "install_hint": "conda install -c conda-forge healpy"}
                    lmax = _alm_lmax_from_size(len(alm))
                    if lmax is None:
                        return None, {"error": "alm_lmax_not_inferred", **alm_info}
                    # Bound output nside for runtime; nearest-neighbour tests do not need native resolution.
                    nside = 2048 if lmax >= 2048 else max(64, 2 ** int(math.floor(math.log2(max(1, lmax / 2)))))
                    hp_map = hp.alm2map(alm, nside=nside, lmax=lmax, verbose=False)
                    is_planck = bool(re.search(r'planck|pr4|healpix|com_lensing', str(map_path), re.I))
                    if is_planck:
                        lon, lat, coordsys = _maybe_icrs_to_galactic(ra[:n], dec[:n])
                    else:
                        lon, lat, coordsys = np.asarray(ra[:n], float), np.asarray(dec[:n], float), 'icrs_assumed'
                    theta = 0.5 * np.pi - np.deg2rad(lat); phi = np.deg2rad(lon)
                    pix = hp.ang2pix(nside, theta, phi, nest=False)
                    vals = np.asarray(hp_map[pix], float)
                    vals = np.where(np.isfinite(vals) & (np.abs(vals) < 1e30), vals, np.nan)
                    return vals, {"mode": "alm2map_healpy", "coordsys": coordsys, "nside": int(nside), "lmax": int(lmax), "n_requested": int(n), "n_good_pixels": int(np.isfinite(vals).sum()), **alm_info}
                except Exception as e:
                    return None, {"error": "harmonic_alm_requires_healpy_alm2map", "detail": str(e), **alm_info, "install_hint": "conda install -c conda-forge healpy"}
    except Exception as e:
        return None, {"error": "map_sampling_failed", "detail": str(e), "path": str(map_path)}
    return None, {"error": "no_pixel_or_harmonic_map_hdu", "path": str(map_path), "product_kind": _fits_product_kind(map_path)}


def load_euclid_q1_sample(cache: Path, timeout: int = 90, max_rows: int = 20000, force: bool = False) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    """Load Euclid Q1 with coordinate, redshift, flux-error/depth, and quality columns.

    v9.6 fixes a practical issue in v9.5: discovered fluxerr/depth columns are
    now prioritised and preserved in the returned dataframe so T06/T07 can use
    them for mask/depth controls.
    """
    attempts: List[Dict[str, Any]] = []
    table, att = _select_euclid_table(cache, timeout=timeout)
    attempts.extend(att)
    if not table:
        return None, attempts + [{"ok": False, "error": "euclid_q1_catalog_not_found_via_irsa_tap"}]
    cols, att2 = _euclid_table_columns(table, cache, timeout=timeout)
    attempts.extend(att2)
    if not cols:
        return None, attempts + [{"ok": False, "error": "euclid_q1_columns_not_discovered", "selected_table": table}]
    ra_col = _pick_column_alias(cols, ["ra", "RA", "raj2000", "RAJ2000"])
    dec_col = _pick_column_alias(cols, ["dec", "DEC", "decl", "dej2000", "DEJ2000"])
    if not (ra_col and dec_col):
        return None, attempts + [{"ok": False, "error": "euclid_q1_coordinate_columns_not_found", "selected_table": table, "columns_sample": cols[:60]}]

    def best_cols(patterns: Sequence[str], maxn: int) -> List[str]:
        scored = []
        for c in cols:
            cl = str(c).lower()
            if c in (ra_col, dec_col):
                continue
            score = 0
            for pat, val in patterns:
                if re.search(pat, cl):
                    score += val
            if score > 0:
                scored.append((score, len(cl), str(c)))
        return [c for _, _, c in sorted(scored, reverse=True)[:maxn]]

    wanted: List[str] = []
    for group in [
        best_cols([(r'object|source|^id$', 10)], 1),
        best_cols([(r'phot.*z|z_phot|photoz|redshift|^z$', 12)], 2),
        best_cols([(r'fluxerr', 25), (r'lim.*mag|depth|exptime', 20), (r'snr', 18), (r'mag', 10), (r'flux_', 7)], 6),
        best_cols([(r'flag_vis|quality|flag|mask', 20)], 3),
    ]:
        for c in group:
            if c not in wanted and c not in (ra_col, dec_col):
                wanted.append(c)
    select_cols = [ra_col, dec_col] + wanted[:12]
    q = f"SELECT TOP {max_rows} {', '.join(select_cols)} FROM {table} WHERE {ra_col} IS NOT NULL AND {dec_col} IS NOT NULL"
    df, att3 = try_irsa_tap_query(q, cache, f"euclid_enriched_sample_{slugify(table)}", timeout=timeout)
    attempts.extend(att3)
    if df is None or len(df) <= 20:
        return None, attempts + [{"ok": False, "error": "euclid_q1_enriched_query_failed", "selected_table": table, "selected_columns": select_cols}]
    df = df.rename(columns={ra_col: "ra", dec_col: "dec"})
    return df, attempts + [{"selected_table": table, "ok": True, "enriched": True, "selected_optional_columns": wanted[:12], "available_columns_sample": cols[:100], "v9_6_note": "fluxerr/depth/quality columns are prioritised and returned for systematics controls"}]


def euclid_depth_proxy(df: Any) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Return depth/completeness proxy; convert flux errors to depth scale."""
    if df is None or pd is None:
        return None, {"error": "no_dataframe"}
    nums = find_numeric_columns(df)
    scored = []
    for c in nums:
        cl = str(c).lower()
        if cl in ('ra', 'dec') or re.search(r'(^|_)id($|_)|object|source', cl):
            continue
        if re.search(r'flag|mask|quality', cl):
            continue
        score = 0
        transform = 'identity'
        if 'fluxerr' in cl:
            score += 80; transform = 'minus_log10_positive_fluxerr'
        if re.search(r'lim.*mag|depth', cl):
            score += 60
        if re.search(r'err', cl) and 'fluxerr' not in cl:
            score += 15; transform = 'minus_log10_positive_error'
        if 'snr' in cl or cl.startswith('sn_'):
            score += 35
        if 'mag' in cl:
            score += 25
        if 'flux' in cl and 'fluxerr' not in cl:
            score += 10
        if cl in ('z', 'photoz', 'phot_z') or 'redshift' in cl:
            score -= 30
        if score <= 0:
            continue
        arr0 = numeric_array(df, c)
        if transform.startswith('minus_log10'):
            arr = np.where(np.isfinite(arr0) & (arr0 > 0), -np.log10(arr0), np.nan)
        else:
            arr = np.asarray(arr0, float)
        finite = arr[np.isfinite(arr)]
        if len(finite) >= max(20, int(0.02 * len(arr))) and np.nanstd(finite) > 0:
            scored.append((score, str(c), arr, transform))
    if not scored:
        return None, {"error": "no_depth_or_magnitude_proxy_columns", "numeric_columns_sample": [str(c) for c in nums[:50]], "v9_6_note": "looked for fluxerr/depth/limmag/snr/mag columns"}
    score, col, arr, transform = sorted(scored, reverse=True, key=lambda x: x[0])[0]
    return np.asarray(arr, float), {"column": col, "score": int(score), "transform": transform, "note": "Automated depth/completeness proxy used only for confounding controls, not calibrated survey depth."}


def desi_dr2_observablewise_gls_stability(paths: Sequence[Path], max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Observable-wise DESI DR2 covariance trend stability guard.

    Prevents a single covariance-weighted all-row slope from becoming support
    when individual observable groups disagree or are unidentified.
    """
    mean_paths = desi_dr2_mean_paths(paths)
    candidates = [p for p in mean_paths if 'ALL_GCcomb' in str(p)] + mean_paths
    for mp in candidates:
        covp = Path(str(mp).replace('_mean', '_cov'))
        if not covp.exists():
            continue
        df = read_table_any(mp, max_rows=max_rows)
        C = _read_numeric_matrix(covp)
        if df is None or C is None:
            continue
        nums = find_numeric_columns(df)
        if len(nums) < 2:
            continue
        z = numeric_array(df, nums[0]); y = numeric_array(df, nums[1])
        m = np.isfinite(z) & np.isfinite(y) & (z > 0) & (z < 6)
        if int(m.sum()) < 4:
            continue
        idx = np.where(m)[0]
        z = z[m]; y = y[m]
        C = C[np.ix_(idx, idx)] if C.shape[0] > max(idx) and C.shape[1] > max(idx) else np.diag(np.ones(len(z)))
        # Identify observable groups from a third column if it has a small number of repeated values.
        groups = np.array(['all'] * len(z), dtype=object)
        group_source = 'none'
        if len(nums) >= 3:
            raw = numeric_array(df, nums[2])[m]
            vals = sorted(set(raw[np.isfinite(raw)]))
            if 1 < len(vals) <= 6:
                groups = np.array([f'{nums[2]}={v:g}' if np.isfinite(v) else 'nan' for v in raw], dtype=object)
                group_source = str(nums[2])
        # If no reliable group labels exist, split by rows belonging to the same rounded z?  Do not fake robustness.
        rows = []
        for g in sorted(set(groups)):
            gg = groups == g
            if int(gg.sum()) < 3:
                continue
            zg = z[gg]; yg = y[gg]
            Cg = C[np.ix_(np.where(gg)[0], np.where(gg)[0])]
            med = float(np.nanmedian(yg)); sc = mad(yg) or float(np.nanstd(yg)) or 1.0
            yn = (yg - med) / sc
            Cn = Cg / (sc * sc) + np.eye(len(zg)) * 1e-10
            try:
                Ci = np.linalg.pinv(Cn)
                x = zg - float(np.mean(zg))
                A = np.vstack([np.ones_like(x), x]).T
                covb = np.linalg.pinv(A.T @ Ci @ A)
                beta = covb @ (A.T @ Ci @ yn)
                slope = float(beta[1]); sig = float(math.sqrt(abs(covb[1, 1]))) if covb.shape == (2, 2) else None
                rows.append({"group": str(g), "n": int(len(zg)), "z_min": float(np.min(zg)), "z_max": float(np.max(zg)), "slope_norm_per_z": slope, "slope_sigma": sig, "slope_zscore": float(slope / sig) if sig and sig > 0 else None})
            except Exception as e:
                rows.append({"group": str(g), "error": str(e), "n": int(gg.sum())})
        valid = [r for r in rows if r.get('slope_norm_per_z') is not None]
        neg = sum(1 for r in valid if r['slope_norm_per_z'] < 0)
        pos = sum(1 for r in valid if r['slope_norm_per_z'] > 0)
        return {"mean_path": str(mp), "cov_path": str(covp), "group_source": group_source, "n_groups_valid": int(len(valid)), "n_negative_slopes": int(neg), "n_positive_slopes": int(pos), "groups": rows, "robust_negative_sign": bool(len(valid) >= 2 and neg == len(valid)), "warning": "If group_source='none', the public mean table did not expose observable labels; all-row trend remains diagnostic only."}
    return {"error": "no_observablewise_desi_dr2_mean_cov_pair"}


def read_astronomy_table_any(path: Path, max_rows: Optional[int] = None) -> Optional[Any]:
    """More permissive parser for VizieR/CDS fixed-width/TSV/VOTable files."""
    df = read_table_any(path, max_rows=max_rows)
    if df is not None and len(df) > 5:
        return df
    if pd is None:
        return None
    try:
        from astropy.table import Table
        tbl = Table.read(path, format='ascii')
        df = tbl.to_pandas()
        return df.head(max_rows) if max_rows else df
    except Exception:
        pass
    try:
        lines = []
        for ln in Path(path).read_text(errors='replace').splitlines():
            if not ln.strip() or ln.startswith('#') or ln.startswith('---') or ln.startswith('==='):
                continue
            lines.append(ln.rstrip())
        if len(lines) < 5:
            return None
        data = '\n'.join(lines)
        for sep in ['\t', r'\s+', ',']:
            try:
                df = pd.read_csv(io.StringIO(data), sep=sep, engine='python', nrows=max_rows)
                if df.shape[0] > 5 and df.shape[1] >= 2:
                    return df
            except Exception:
                pass
    except Exception:
        pass
    return None


def find_sky_coordinate_columns(df: Any) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    if df is None:
        return None, None, {"error": "no_dataframe"}
    cols = list(df.columns)
    # Exact/header aliases first, including CDS/VizieR style.
    ra_alias = ['RAJ2000', 'RA_ICRS', 'RAdeg', 'RA', 'raj2000', 'ra', '_RAJ2000', 'alpha', 'ALPHA_J2000']
    dec_alias = ['DEJ2000', 'DE_ICRS', 'DEdeg', 'DEC', 'Dec', 'dec', '_DEJ2000', 'delta', 'DELTA_J2000']
    lower = {str(c).lower(): c for c in cols}
    ra = next((lower[a.lower()] for a in ra_alias if a.lower() in lower), None)
    de = next((lower[a.lower()] for a in dec_alias if a.lower() in lower), None)
    if ra is not None and de is not None:
        return ra, de, {"mode": "alias"}
    nums = find_numeric_columns(df)
    candidates_ra=[]; candidates_de=[]
    for c in nums:
        cl=str(c).lower()
        arr=numeric_array(df,c); finite=arr[np.isfinite(arr)]
        if len(finite)<10: continue
        if np.nanmin(finite)>=0 and np.nanmax(finite)<=360:
            score = 10 if re.search(r'ra|raj|alpha|glon|lon|x1|x_1|x$', cl) else 1
            candidates_ra.append((score,c))
        if np.nanmin(finite)>=-90 and np.nanmax(finite)<=90:
            score = 10 if re.search(r'dec|dej|delta|glat|lat|y1|y_1|y$', cl) else 1
            candidates_de.append((score,c))
    if candidates_ra and candidates_de:
        return sorted(candidates_ra, reverse=True)[0][1], sorted(candidates_de, reverse=True)[0][1], {"mode": "range_and_name", "ra_candidates": [str(x[1]) for x in candidates_ra[:5]], "dec_candidates": [str(x[1]) for x in candidates_de[:5]]}
    return None, None, {"error": "no_coordinate_columns", "columns": [str(c) for c in cols[:60]]}


def columns_with_keywords(df: Any, keywords: Sequence[str], reject: Sequence[str] = ()) -> List[str]:
    if df is None:
        return []
    nums = find_numeric_columns(df)
    out=[]
    for c in nums:
        cl=str(c).lower()
        if any(r in cl for r in reject):
            continue
        if any(k in cl for k in keywords):
            out.append(c)
    return out


def nanograv_table_identity(df: Any) -> Dict[str, Any]:
    if df is None:
        return {"kind": "unparsed"}
    names = ' '.join(map(lambda x: str(x).lower(), df.columns))
    nums = find_numeric_columns(df)
    kind = 'unknown_numeric_table'
    if re.search(r'resid|residual|mjd|toa|postfit|prefit', names):
        kind = 'residual_or_toa_table'
    elif re.search(r'free.*spectrum|rho_|log10_a|gamma|spectral|hd|gwb|common_red|crn', names):
        kind = 'posterior_parameter_table'
    elif len(nums) >= 2 and any(re.search(r'chain|post|posterior', str(c).lower()) for c in df.columns):
        kind = 'posterior_parameter_table'
    return {"kind": kind, "numeric_columns": [str(c) for c in nums[:20]], "columns": [str(c) for c in list(df.columns)[:40]]}


def parse_spectral_index_constraints(df: Any) -> Dict[str, Any]:
    ident = nanograv_table_identity(df)
    if df is None or pd is None:
        return ident
    cols=[]
    for c in df.columns:
        cl=str(c).lower()
        if re.search(r'gamma|spectral|index|delta.?n|n_t|alpha', cl):
            arr=numeric_array(df,c)
            if np.isfinite(arr).sum() >= 5:
                vals=arr[np.isfinite(arr)]
                cols.append({"column": str(c), "n": int(len(vals)), "median": float(np.nanmedian(vals)), "p16": float(np.nanpercentile(vals,16)), "p84": float(np.nanpercentile(vals,84))})
    ident["spectral_index_columns"] = cols
    ident["has_verified_spectral_index"] = bool(cols)
    return ident


def download_planck_spectrum_candidates(cache: Path, timeout: int = 90, force: bool = False, max_files: int = 8) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """Download several simple public Planck spectrum tables instead of stopping at first URL."""
    attempts: List[Dict[str, Any]]=[]
    urls = [
        "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-full_R3.01.txt",
        "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-binned_R3.01.txt",
        "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt",
    ]
    for page in ["https://lambda.gsfc.nasa.gov/product/planck/planck_prod_table.html", "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/"]:
        try:
            links=discover_links(page,pattern=r"(PowerSpect|COM_Power|TT|TE|EE|BB|Cls|cl).*\.(txt|dat)$",timeout=timeout)
            attempts.append({"url":page,"ok":True,"n_links":len(links)})
            urls.extend(links[:20])
        except Exception as e:
            attempts.append({"url":page,"ok":False,"error":str(e)})
    out=[]; seen=set()
    for i,u in enumerate(urls):
        if u in seen: continue
        seen.add(u)
        try:
            p=download_file(u,cache/"planck_spectra",filename=f"planck_spectrum_{i}_{slugify(url_basename(u) or 'table')}.txt",timeout=timeout,force=force,max_bytes=300*1024*1024)
            attempts.append({"url":u,"ok":True,"path":str(p)})
            out.append(p)
            if len(out)>=max_files: break
        except Exception as e:
            attempts.append({"url":u,"ok":False,"error":str(e)})
    return out, attempts


def bk18_column_label_map(path: Path, n_cols: int) -> Dict[int, str]:
    """Best-effort map of BK18 pair columns to labels from comment headers."""
    labels: Dict[int, str] = {}
    try:
        lines = Path(path).read_text(errors='ignore').splitlines()[:200]
        for line in lines:
            if not line.lstrip().startswith('#'):
                continue
            clean=line.lstrip('#').strip()
            # Patterns like "3: BK18 BB" or column lists.
            for m in re.finditer(r'(?:col(?:umn)?\s*)?(\d+)\s*[:=]\s*([^,;]+)', clean, flags=re.I):
                j=int(m.group(1))
                if 1 <= j <= n_cols:
                    labels[j]=m.group(2).strip()[:80]
            toks=clean.split()
            if len(toks)>=n_cols and any(t.upper() in ('BB','EE','TE','TB','EB') or 'BB' in t.upper() for t in toks):
                for j,t in enumerate(toks[:n_cols], start=1):
                    labels.setdefault(j,t)
    except Exception:
        pass
    return labels


def discover_kss_eta_tables(cache: Path, timeout: int = 90, force: bool = False) -> Tuple[List[Tuple[str, Any]], List[Dict[str, Any]]]:
    """Try a conservative public η/s table discovery without treating flow as viscosity."""
    attempts=[]; parsed=[]
    urls=[
        # Known non-η/s HEPData records are kept to verify rejection by parser.
        'https://www.hepdata.net/download/table/ins1666817/Table%201/1/csv',
        'https://www.hepdata.net/download/table/ins1666817/Figure%201/1/csv',
    ]
    # Search endpoints/pages can change; capture but do not scrape arbitrary PDFs as data.
    pages=['https://www.hepdata.net/search/?q=eta%2Fs', 'https://www.hepdata.net/search/?q=shear%20viscosity%20entropy%20density']
    for page in pages:
        try:
            links=discover_links(page,pattern=r'/download/table/.*/csv',timeout=timeout)
            attempts.append({'url':page,'ok':True,'n_links':len(links),'note':'HEPData search discovery for explicit eta/s-like tables'})
            urls.extend(links[:10])
        except Exception as e:
            attempts.append({'url':page,'ok':False,'error':str(e)})
    seen=set()
    for u in urls:
        if u in seen: continue
        seen.add(u)
        try:
            p=download_file(u,cache/'hepdata_kss',filename=slugify(url_basename(u))+'.csv',timeout=timeout,force=force,max_bytes=50*1024*1024)
            attempts.append({'url':u,'ok':True,'path':str(p)})
            df=read_table_any(p)
            if df is not None and df.shape[1]>=2:
                parsed.append((u,df))
        except Exception as e:
            attempts.append({'url':u,'ok':False,'error':str(e)})
    return parsed, attempts


# ------------------------- v9.7 overrides -------------------------
# Correctness + data-limited fixes requested after the v9.6 run.
# These definitions intentionally appear last so they override previous helpers.


def pantheon_columns(df: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Public wrapper for the Pantheon+ column picker.

    Earlier tests called the private _pantheon_columns helper via `import *`,
    which caused T03 to crash because underscore names are not imported.
    """
    try:
        return _pantheon_columns(df)  # type: ignore[name-defined]
    except Exception:
        if df is None or not hasattr(df, 'columns'):
            return None, None, None
        cols = [str(c) for c in df.columns]
        z = next((c for c in cols if c.lower() in ('z', 'zcmb', 'zhel', 'zhelio')), None)
        mu = next((c for c in cols if c.lower() in ('mu_sh0es', 'mu', 'mures', 'mb_corr')), None)
        err = next((c for c in cols if 'err' in c.lower() or 'sigma' in c.lower()), None)
        return z, mu, err


def is_mask_like_product(path_or_name: Any) -> bool:
    n = str(path_or_name).replace('\\', '/').lower()
    return bool(re.search(r'(^|[/_\-.])(mask|masks|ivar|hit|hits|weight|weights|window|footprint)([/_\-.]|$)', n))


def is_bad_map_product(path_or_name: Any) -> bool:
    n = str(path_or_name).replace('\\', '/').lower()
    return is_mask_like_product(n) or bool(re.search(r'noise|sim|random|curl|meanfield|mf_', n))


def map_signal_label(path_or_name: Any) -> str:
    return 'mask_proxy' if is_mask_like_product(path_or_name) else 'kappa'


def validate_map_sample_values(vals: Optional[np.ndarray], info: Dict[str, Any], selected_path: Any = None, min_finite: int = 20) -> Dict[str, Any]:
    """Classify sampled map values before a test is allowed to call a null.

    Mask products and constant-valued samples are data-limited, never nulls.
    """
    p = str(selected_path or info.get('path') or '')
    out = {'ok': False, 'signal_label': map_signal_label(p), 'reason': None}
    if is_mask_like_product(p):
        out['reason'] = 'selected_mask_not_kappa'
        out['warning'] = 'Selected FITS product is mask/coverage/weight-like, not a κ/convergence map.'
        return out
    if vals is None:
        out['reason'] = 'no_values'
        return out
    arr = np.asarray(vals, float)
    good = np.isfinite(arr)
    out['n_finite'] = int(np.sum(good))
    if int(np.sum(good)) < min_finite:
        out['reason'] = 'too_few_finite_values'
        return out
    finite = arr[good]
    out['std'] = float(np.nanstd(finite))
    out['min'] = float(np.nanmin(finite))
    out['max'] = float(np.nanmax(finite))
    # Constant maps are almost always masks or coverage products in this suite.
    if not np.isfinite(out['std']) or out['std'] <= 0 or np.nanmax(finite) == np.nanmin(finite):
        out['reason'] = 'selected_map_constant_or_mask_like'
        out['warning'] = 'Sampled map values are constant; treating as data_limited instead of null.'
        return out
    out['ok'] = True
    out['reason'] = 'ok'
    return out


def _fits_product_kind(path: Path) -> Dict[str, Any]:
    """v9.7 lightweight FITS inspection with explicit mask/harmonic typing."""
    info: Dict[str, Any] = {"path": str(path), "is_mask_like": is_mask_like_product(path), "name_harmonic_hint": _is_harmonic_name(path)}
    if info['is_mask_like']:
        info['kind'] = 'mask_or_weight'
    try:
        from astropy.io import fits
        with fits.open(path, memmap=True) as hdul:
            for hdu_i, hdu in enumerate(hdul):
                data = getattr(hdu, 'data', None)
                if data is None:
                    continue
                hdr = getattr(hdu, 'header', {})
                arr = np.asarray(data)
                if arr.ndim >= 2 and hdr.get('CTYPE1'):
                    info.update({"kind": "wcs_image", "hdu": int(hdu_i), "shape": list(arr.shape)})
                    return info
                if hasattr(data, 'columns'):
                    names = list(getattr(data.columns, 'names', []) or [])
                    low = ' '.join(map(str.lower, names))
                    if re.search(r'alm|klm|real|imag|lmax|ell|emm|m_', low) or _is_harmonic_name(path):
                        info.update({"kind": "harmonic_alm", "hdu": int(hdu_i), "columns": names[:30]})
                    hp_map, hp_info = _read_healpix_table_map_numpy(hdul, path)
                    if hp_map is not None:
                        info.update({"kind": "healpix_table", **hp_info})
                        # If filename says mask, keep that warning but still describe shape.
                        if is_mask_like_product(path):
                            info['kind'] = 'mask_or_weight'
                        return info
            info.setdefault('kind', 'harmonic_alm' if _is_harmonic_name(path) else ('mask_or_weight' if is_mask_like_product(path) else 'unknown'))
            return info
    except Exception as e:
        info.update({"kind": "unreadable", "error": str(e)})
        return info


def _fits_map_score(q: Path) -> Tuple[int, int]:
    """Prefer exact κ/convergence pixel maps; strongly reject masks."""
    n = str(q).replace('\\', '/').lower()
    info = _fits_product_kind(q)
    kind = info.get('kind')
    score = 0
    if kind == 'wcs_image':
        score += 2000
    elif kind == 'healpix_table':
        score += 1800
    elif kind == 'harmonic_alm':
        score -= 300
    elif kind == 'mask_or_weight':
        score -= 5000
    elif kind == 'unreadable':
        score -= 1000
    exact_good = [
        r'kappa.*map', r'map.*kappa', r'convergence.*map', r'map.*convergence',
        r'klm_dat.*mv', r'kappa.*mv', r'lensing.*mv'
    ]
    if any(re.search(p, n) for p in exact_good):
        score += 500
    for kw, val in [
        ('kappa', 240), ('convergence', 220), ('lensing', 80), ('mv', 50), ('map', 60),
        ('dat_', 20), ('input', -10), ('mask', -6000), ('ivar', -4000), ('weight', -4000),
        ('hit', -2000), ('noise', -1000), ('sim', -1000), ('random', -1000),
        ('curl', -1500), ('alm', -400), ('klm', -400), ('meanfield', -800), ('mf_', -500),
    ]:
        if kw in n:
            score += val
    return (-score, len(n))


def _choose_fits_map_from_path(path: Path, extract_dir: Path) -> Optional[Path]:
    """Choose a real κ/convergence pixel map; masks only if nothing else exists and tests will block it."""
    path = Path(path)
    if str(path).lower().endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        info = _fits_product_kind(path)
        if info.get('kind') in ('wcs_image', 'healpix_table') and not is_mask_like_product(path):
            return path
        if info.get('kind') == 'harmonic_alm' and not is_mask_like_product(path):
            return path
    root = extract_if_archive(path, extract_dir)
    fits = [q for q in Path(root).rglob('*') if q.is_file() and re.search(r'\.fits(\.gz)?$|\.fit(\.gz)?$', q.name, re.I)]
    if not fits:
        return None
    inspected = [(q, _fits_product_kind(q)) for q in fits]
    # Exact ACT/Planck κ pixel products first.
    exact_pixel = []
    for q, info in inspected:
        n = str(q).replace('\\','/').lower()
        if is_mask_like_product(q) or is_bad_map_product(q):
            continue
        if info.get('kind') in ('wcs_image', 'healpix_table') and (('kappa' in n) or ('convergence' in n) or ('lensing' in n and 'mv' in n)):
            exact_pixel.append(q)
    if exact_pixel:
        return sorted(exact_pixel, key=_fits_map_score)[0]
    usable = [q for q, info in inspected if info.get('kind') in ('wcs_image', 'healpix_table') and not is_mask_like_product(q)]
    if usable:
        return sorted(usable, key=_fits_map_score)[0]
    harmonic = [q for q, info in inspected if info.get('kind') == 'harmonic_alm' and not is_mask_like_product(q)]
    if harmonic:
        return sorted(harmonic, key=_fits_map_score)[0]
    # Return a mask only to allow caller to produce explicit data_limited reason.
    return sorted([q for q, _ in inspected], key=_fits_map_score)[0]


def download_act_dr6_lensing_map(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    """ACT DR6 map downloader with exact κ pixel-map discovery."""
    attempts: List[Dict[str, Any]]=[]
    if not allow_large:
        return None, [{"ok": False, "reason": "large_download_not_enabled", "note": "ACT DR6 lensing map archive is large; rerun with --allow-large."}]
    url = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/dr6_lensing_release.tar.gz"
    try:
        links = discover_links("https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html", pattern=r"dr6_lensing_release\.tar\.gz", timeout=timeout)
        attempts.append({"url": "https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html", "ok": True, "n_links": len(links)})
        if links:
            url = links[0]
    except Exception as e:
        attempts.append({"ok": False, "url": "ACT landing page", "error": str(e)})
    p = download_file(url, cache/"act_dr6", filename="dr6_lensing_release.tar.gz", timeout=timeout, force=force, max_bytes=None)
    attempts.append({"ok": True, "path": str(p), "url": url, "size": p.stat().st_size if p.exists() else None})
    root = extract_if_archive(p, cache/"act_dr6"/"dr6_lensing_release_extracted")
    fits = [q for q in Path(root).rglob('*') if q.is_file() and re.search(r'\.fits(\.gz)?$|\.fit(\.gz)?$', q.name, re.I)]
    exact=[]; inspected=[]
    for q in fits:
        info=_fits_product_kind(q); inspected.append({"path": str(q), "kind": info.get('kind'), "is_mask_like": is_mask_like_product(q)})
        n=str(q).replace('\\','/').lower()
        if not is_mask_like_product(q) and not re.search(r'curl|noise|sim|random|meanfield|mf_', n) and (('kappa' in n) or ('convergence' in n)) and info.get('kind') in ('wcs_image','healpix_table'):
            exact.append(q)
    if exact:
        best=sorted(exact,key=_fits_map_score)[0]
        attempts.append({"ok": True, "selected_map": str(best), "selection": "exact_act_kappa_pixel_map", "n_fits_inspected": len(fits)})
        return best, attempts
    best=_choose_fits_map_from_path(p, cache/"act_dr6"/"dr6_lensing_release_extracted")
    attempts.append({"ok": bool(best), "selected_map": str(best) if best else None, "selection": "fallback_best_nonmask_or_harmonic", "n_fits_inspected": len(fits), "inspected_sample": inspected[:20]})
    return best, attempts


def download_planck_lensing_or_spectra(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False, prefer_map: bool = False) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    """Planck PR4/PR3 downloader.

    In map mode, reject masks and prefer pixel maps.  If only klm/alm exists,
    return it as data source, but sample_map_values_for_points will require
    healpy.alm2map and otherwise produce a data_limited reason.
    """
    attempts: List[Dict[str, Any]]=[]
    if prefer_map:
        if not allow_large:
            return None, [{"ok": False, "reason": "large_download_not_enabled", "note": "Planck lensing maps are large; rerun with --allow-large."}]
        candidates=[]
        asset,info=_github_release_asset_url("carronj","planck_PR4_lensing",r"PR42018like_maps\.tar",timeout=timeout,tag_hint="Data")
        attempts.append(info)
        if asset:
            candidates.append(asset)
        candidates += [
            "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/lensing/COM_Lensing_4096_R3.00.tar",
            "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/lensing/COM_Lensing-Szdeproj_4096_R3.00.tar",
        ]
        p,att=download_first(candidates,cache/"planck_lensing",timeout=timeout,force=force,max_bytes=None)
        attempts.extend(att)
        if p is None:
            return None, attempts
        root=extract_if_archive(p, cache/"planck_lensing"/"maps_extracted")
        fits=[q for q in Path(root).rglob('*') if q.is_file() and re.search(r'\.fits(\.gz)?$|\.fit(\.gz)?$', q.name, re.I)]
        pixel=[]; harmonic=[]; masks=[]; inspected=[]
        for q in fits:
            info=_fits_product_kind(q); inspected.append({"path": str(q), "kind": info.get('kind'), "is_mask_like": is_mask_like_product(q)})
            if is_mask_like_product(q):
                masks.append(q); continue
            if info.get('kind') in ('wcs_image','healpix_table') and re.search(r'kappa|convergence|lensing|klm|dat|mv', str(q), re.I):
                pixel.append(q)
            elif info.get('kind') == 'harmonic_alm':
                harmonic.append(q)
        if pixel:
            best=sorted(pixel,key=_fits_map_score)[0]
            attempts.append({"ok": True, "selected_map": str(best), "selection": "planck_pixel_kappa_nonmask", "n_fits_inspected": len(fits)})
            return best, attempts
        if harmonic:
            best=sorted(harmonic,key=_fits_map_score)[0]
            attempts.append({"ok": True, "selected_map": str(best), "selection": "planck_harmonic_klm_requires_healpy_alm2map", "install_hint": "conda install -c conda-forge healpy", "n_fits_inspected": len(fits)})
            return best, attempts
        if masks:
            best=sorted(masks,key=_fits_map_score)[0]
            attempts.append({"ok": True, "selected_map": str(best), "selection": "mask_only_data_limited", "warning": "Only mask/coverage products were found; tests must not call this a κ null.", "n_fits_inspected": len(fits), "inspected_sample": inspected[:20]})
            return best, attempts
        return None, attempts + [{"ok": False, "error": "no_planck_lensing_fits_candidates", "n_fits_inspected": len(fits)}]
    # Spectrum/table mode for T22.
    paths, att = download_planck_spectrum_candidates(cache, timeout=timeout, force=force, max_files=8)
    attempts.extend(att)
    return (paths[0] if paths else None), attempts


def read_planck_lowell_numeric_table(path: Path, max_rows: Optional[int] = None) -> Optional[Any]:
    """Parse Planck low-ell/simple spectra tables: comments + whitespace numeric rows."""
    if pd is None:
        return None
    try:
        rows=[]
        for line in Path(path).read_text(errors='ignore').splitlines():
            s=line.strip()
            if not s or s.startswith('#') or s.startswith('%') or s.startswith('//'):
                continue
            s=re.sub(r'[;,]', ' ', s)
            vals=[]
            for tok in s.split():
                try:
                    vals.append(float(tok.replace('D','E').replace('d','e')))
                except Exception:
                    pass
            if len(vals)>=2:
                rows.append(vals)
            if max_rows and len(rows)>=max_rows:
                break
        if len(rows)<3:
            return None
        n=max(len(r) for r in rows)
        arr=np.full((len(rows), n), np.nan)
        for i,r in enumerate(rows):
            arr[i,:len(r)]=r
        return pd.DataFrame(arr, columns=[f'col{j}' for j in range(n)])
    except Exception:
        return None


def fit_desi_dr2_simplified_rvm_likelihood(paths: Sequence[Path], max_rows: Optional[int] = None) -> Dict[str, Any]:
    """A compact DESI DR2 BAO model likelihood, not just a trend screen.

    It fits public DESI DR2 mean/cov vectors to simple flat cosmology distance
    predictions with a nuisance distance scale.  If observable labels are not
    exposed, it tries permutations of numeric observable codes and reports this
    limitation explicitly.  This is still not CLASS/CAMB, but it is an actual
    model-vector χ² rather than a GLS trend on rows.
    """
    if pd is None:
        return {"error": "pandas_missing"}
    means=[Path(p) for p in desi_dr2_mean_paths(paths) if 'ALL_GCcomb_mean' in str(p)] or desi_dr2_mean_paths(paths)
    if not means:
        return {"error":"no_desi_dr2_mean_paths"}
    mean_path=means[0]
    cov_path=None
    s=str(mean_path)
    for p in paths:
        ps=str(p)
        if 'ALL_GCcomb_cov' in ps and 'desi_bao_dr2' in ps:
            cov_path=Path(p); break
    try:
        df=read_table_any(mean_path, max_rows=max_rows)
        nums=find_numeric_columns(df)
        if len(nums)<2:
            return {"error":"mean_table_has_too_few_numeric_columns", "mean_path": str(mean_path)}
        z=numeric_array(df, nums[0]); y=numeric_array(df, nums[1])
        obs_code=numeric_array(df, nums[2]) if len(nums)>=3 else np.zeros_like(z)
        m=np.isfinite(z)&np.isfinite(y)&(z>0)&(y>0)
        z=z[m]; y=y[m]; obs_code=obs_code[m]
        n=len(y)
        if n<3:
            return {"error":"too_few_desi_rows", "n": int(n), "mean_path": str(mean_path)}
        if cov_path and cov_path.exists():
            C=np.loadtxt(cov_path)
            C=np.asarray(C,float)
            if C.shape[0] != n:
                C=C[:n,:n]
        else:
            C=np.diag(np.maximum(np.abs(y)*0.05, 1.0)**2)
        # condition covariance
        C=C + np.eye(n)*max(1e-12, np.nanmedian(np.diag(C))*1e-10)
        iC=np.linalg.pinv(C)
        codes=sorted(set([int(round(c)) for c in obs_code if np.isfinite(c)]))
        # limit to at most three observable groups; unknown codes are mapped by permutations.
        if len(codes)<2:
            code_maps=[{codes[0] if codes else 0:'DV'}]
            group_source='single_or_unknown'
        else:
            base=['DM','DH','DV'][:len(codes)]
            import itertools
            code_maps=[dict(zip(codes,perm)) for perm in itertools.permutations(['DM','DH','DV'], len(codes))]
            group_source='numeric_observable_code_permutation'
        zg=np.linspace(0, float(np.nanmax(z))*1.05, 512)
        def E_of_z(zz, Om, nu):
            return np.sqrt(np.maximum(Om*(1+zz)**3 + (1-Om)*(1 + nu*np.log1p(zz)), 1e-8))
        def comoving_grid(Om,nu):
            E=E_of_z(zg,Om,nu)
            inv=1.0/E
            integ=np.zeros_like(zg)
            integ[1:]=np.cumsum(0.5*(inv[1:]+inv[:-1])*np.diff(zg))
            return integ
        def model_shape(Om,nu,cmap):
            Dc_grid=comoving_grid(Om,nu)
            Dc=np.interp(z,zg,Dc_grid)
            Ez=E_of_z(z,Om,nu)
            out=np.zeros_like(z)
            for i,c in enumerate(obs_code):
                typ=cmap.get(int(round(c)), 'DV')
                if typ=='DM': out[i]=Dc[i]
                elif typ=='DH': out[i]=1.0/Ez[i]
                else: out[i]=np.maximum((z[i]*Dc[i]*Dc[i]/Ez[i]), 1e-12)**(1/3)
            return out
        def chi2_for(Om,nu,cmap):
            f=model_shape(Om,nu,cmap)
            denom=float(f @ iC @ f)
            if denom<=0 or not np.isfinite(denom): return np.inf, np.nan
            alpha=float(f @ iC @ y)/denom
            r=y-alpha*f
            return float(r @ iC @ r), alpha
        grid_om=np.linspace(0.2,0.45,31)
        grid_nu=np.linspace(-0.2,0.2,41)
        best_lcdm=(np.inf,None,None,None)
        best_rvm=(np.inf,None,None,None,None)
        for cmap in code_maps:
            for Om in grid_om:
                ch,alpha=chi2_for(float(Om),0.0,cmap)
                if ch<best_lcdm[0]: best_lcdm=(ch,float(Om),alpha,cmap)
                for nu in grid_nu:
                    ch,alpha=chi2_for(float(Om),float(nu),cmap)
                    if ch<best_rvm[0]: best_rvm=(ch,float(Om),float(nu),alpha,cmap)
        delta=best_lcdm[0]-best_rvm[0]
        # AIC penalty: RVM adds one parameter over ΛCDM.
        return {
            "mode":"desi_dr2_simplified_bao_model_likelihood",
            "mean_path":str(mean_path), "cov_path":str(cov_path) if cov_path else None,
            "n_points":int(n), "observable_codes":[int(c) for c in codes], "group_source":group_source,
            "chi2_lcdm":float(best_lcdm[0]), "omega_m_lcdm":best_lcdm[1], "alpha_lcdm":best_lcdm[2], "code_map_lcdm":{str(k):v for k,v in (best_lcdm[3] or {}).items()},
            "chi2_rvm_like":float(best_rvm[0]), "omega_m_rvm_like":best_rvm[1], "nu_like":best_rvm[2], "alpha_rvm_like":best_rvm[3], "code_map_rvm_like":{str(k):v for k,v in (best_rvm[4] or {}).items()},
            "delta_chi2_rvm_vs_lcdm":float(delta), "delta_aic_rvm_vs_lcdm":float(delta-2.0),
            "nu_grid_bound_hit": bool(abs(best_rvm[2]) >= 0.199 if best_rvm[2] is not None else False),
            "note":"Model-vector BAO likelihood with nuisance scale and simple RVM-like E(z); still not a full Boltzmann/MCMC analysis."
        }
    except Exception as e:
        return {"error":"desi_model_likelihood_failed", "detail":str(e), "mean_path":str(mean_path)}


def fit_joint_sn_desi_rvm_model_likelihood(df: Any, cov: Optional[np.ndarray], paths: Sequence[Path], max_rows: Optional[int]=None) -> Dict[str, Any]:
    sn=fit_pantheon_nu_cov_like(df,cov) if df is not None else {"error":"no_sn"}
    bao=fit_desi_dr2_simplified_rvm_likelihood(paths,max_rows=max_rows)
    out={"mode":"joint_sn_desi_model_likelihood_approx", "sn":sn, "desi_bao":bao}
    try:
        out["joint_delta_chi2_proxy"] = float((sn.get('delta_chi2_vs_nu0') or 0.0) + (bao.get('delta_chi2_rvm_vs_lcdm') or 0.0))
        out["bound_hit"] = bool(is_bound_hit_metric(sn) or bao.get('nu_grid_bound_hit'))
    except Exception:
        pass
    return out


def sparc_robustness_matrix(rotmod_paths: Sequence[Path], seed: int=12345) -> Dict[str, Any]:
    paths=list(rotmod_paths)
    if not paths:
        return {"error":"no_sparc_paths"}
    rng=np.random.default_rng(seed)
    base=fit_sparc_a0(paths,max_galaxies=None)
    rows=[]
    for label, subset in [
        ('first_half', paths[:max(1,len(paths)//2)]),
        ('second_half', paths[max(1,len(paths)//2):]),
        ('first_50', paths[:50]),
        ('first_100', paths[:100]),
    ]:
        if len(subset)>=5:
            f=fit_sparc_a0(subset,max_galaxies=None); f['subset']=label; rows.append(f)
    boots=[]
    if len(paths)>=10:
        for _ in range(40):
            sub=list(rng.choice(paths,size=min(len(paths), max(20, len(paths)//2)), replace=True))
            f=fit_sparc_a0(sub,max_galaxies=None)
            if isinstance(f,dict) and f.get('a0_best_m_s2'):
                boots.append(float(f['a0_best_m_s2']))
    return {"base":base,"subsets":rows,"bootstrap_subsample_n":len(boots),"a0_subsample_median":float(np.nanmedian(boots)) if boots else None,"a0_subsample_p16":float(np.nanpercentile(boots,16)) if boots else None,"a0_subsample_p84":float(np.nanpercentile(boots,84)) if boots else None,"robust_within_factor2_of_milgrom": bool(base.get('a0_best_m_s2') and 0.5 < base.get('a0_best_m_s2')/1.2e-10 < 2.0)}


def euclid_field_depth_photoz_matched_controls(ra: np.ndarray, dec: np.ndarray, labels: np.ndarray, depth: Optional[np.ndarray], photoz: Optional[np.ndarray]=None, seed: int=12345) -> Dict[str, Any]:
    rng=np.random.default_rng(seed)
    out={"available_depth": depth is not None, "available_photoz": photoz is not None}
    if len(ra)<100:
        out['error']='too_few_points'; return out
    dens=nearest_density(ra,dec,k=10)
    m=np.isfinite(dens)
    if depth is not None: m &= np.isfinite(depth)
    if photoz is not None: m &= np.isfinite(photoz)
    out['n_matched_base']=int(np.sum(m))
    if np.sum(m)<100:
        return out
    covars=[]
    if depth is not None: covars.append(np.asarray(depth,float))
    if photoz is not None: covars.append(np.asarray(photoz,float))
    # residualize log-density against field dummies + depth/photoz.
    X=[np.ones(np.sum(m))]
    for lab in sorted(set(labels[m])):
        X.append((labels[m]==lab).astype(float))
    for c in covars:
        X.append(c[m])
    X=np.vstack(X).T
    y=np.log(np.maximum(dens[m],1e-12))
    try:
        beta=np.linalg.lstsq(X,y,rcond=None)[0]
        resid=np.full(len(ra),np.nan); resid[m]=y-X@beta
        out['residualized_density_std']=float(np.nanstd(resid))
        out['matched_density_median']=float(np.nanmedian(resid[m]))
    except Exception as e:
        out['error']='residualization_failed'; out['detail']=str(e)
    return out


def highz_split_by_survey_model(df: Any, seed: int=12345) -> Dict[str, Any]:
    if df is None or pd is None:
        return {"error":"no_dataframe"}
    candidates=[]
    for c in df.columns:
        cl=str(c).lower()
        if any(k in cl for k in ['survey','field','source','catalog','sample']):
            vals=df[c].astype(str).to_numpy()
            if 1 < len(set(vals[:min(len(vals),300)])) <= 20:
                candidates.append(c)
    split_col=candidates[0] if candidates else None
    if split_col is None:
        return {"split_column":None,"note":"No survey/field column found; using global object-level bootstrap only.","global":highz_group_bootstrap_summary(df,seed=seed)}
    out={"split_column":str(split_col),"groups":{}}
    for val, sub in df.groupby(split_col):
        if len(sub)>=20:
            out['groups'][str(val)]=highz_group_bootstrap_summary(sub,seed=seed)
    out['n_groups']=len(out['groups'])
    return out


def parse_vizier_filament_table(path: Path, max_rows: Optional[int]=None) -> Tuple[Optional[Any], Dict[str,Any]]:
    df=read_astronomy_table_any(path,max_rows=max_rows)
    if df is None or len(df)<5:
        return None,{"error":"unparsed_or_too_few_rows"}
    ra_col,dec_col,cinfo=find_sky_coordinate_columns(df)
    if not (ra_col and dec_col):
        # endpoint aliases can imply first coordinate.
        cols=list(df.columns)
        aliases_ra=[c for c in cols if re.search(r'(^|[^a-z])(ra|raj2000|ra_icrs|ra1|ra_1|x1)([^a-z]|$)',str(c),re.I)]
        aliases_de=[c for c in cols if re.search(r'(^|[^a-z])(de|dec|dej2000|dec_icrs|dec1|dec_1|y1)([^a-z]|$)',str(c),re.I)]
        if aliases_ra and aliases_de:
            ra_col,dec_col=aliases_ra[0],aliases_de[0]
    if not (ra_col and dec_col):
        return df,{"error":"no_coordinate_columns", **cinfo, "columns":[str(c) for c in df.columns[:40]]}
    def valarr(c, is_ra):
        arr=numeric_array(df,c)
        if np.isfinite(arr).sum() < max(3, len(arr)//10):
            out=[]
            for s in df[c].astype(str).to_numpy():
                out.append(_sexagesimal_to_deg_ra(s) if is_ra else _sexagesimal_to_deg_dec(s))
            arr=np.asarray([np.nan if v is None else v for v in out],float)
        return arr
    ra=valarr(ra_col,True); dec=valarr(dec_col,False)
    m=np.isfinite(ra)&np.isfinite(dec)&(ra>=0)&(ra<=360)&(dec>=-90)&(dec<=90)
    # endpoints: RA2/DEC2, ra_end/dec_end, x2/y2.
    cols=list(df.columns)
    ra2_col=next((c for c in cols if re.search(r'(ra2|ra_2|ra.*end|x2|x_2)',str(c),re.I)), None)
    de2_col=next((c for c in cols if re.search(r'(de2|dec2|dec_2|dec.*end|y2|y_2)',str(c),re.I)), None)
    angle=None; angle_mode='knn_reconstructed_angle'
    if ra2_col is not None and de2_col is not None:
        ra2=valarr(ra2_col,True); de2=valarr(de2_col,False)
        mm=m&np.isfinite(ra2)&np.isfinite(de2)
        if np.sum(mm)>=20:
            med_dec=np.nanmedian(dec[mm])
            angle=np.full(len(ra),np.nan)
            angle[mm]=np.arctan2(de2[mm]-dec[mm], (ra2[mm]-ra[mm])*np.cos(np.deg2rad(med_dec)))
            angle_mode='catalogue_endpoint_angle'
    return df,{"ra_col":str(ra_col),"dec_col":str(dec_col),"ra":ra,"dec":dec,"finite_mask":m,"angle":angle,"angle_mode":angle_mode,"coordinate_parse":cinfo,"ra2_col":str(ra2_col) if ra2_col is not None else None,"dec2_col":str(de2_col) if de2_col is not None else None,"n_finite":int(np.sum(m))}


def curated_eta_source_registry() -> List[Dict[str,Any]]:
    return [
        {"label":"HEPData eta/s search", "url":"https://www.hepdata.net/search/?q=%22eta%2Fs%22", "type":"discovery_page", "requires_explicit_column":True},
        {"label":"HEPData shear viscosity entropy density search", "url":"https://www.hepdata.net/search/?q=%22shear%20viscosity%22%20%22entropy%20density%22", "type":"discovery_page", "requires_explicit_column":True},
        {"label":"Reject ALICE/CMS/ATLAS v2 flow tables unless eta/s appears explicitly", "url":None, "type":"guardrail", "requires_explicit_column":True},
    ]


def discover_kss_eta_tables(cache: Path, timeout: int = 90, force: bool = False) -> Tuple[List[Tuple[str, Any]], List[Dict[str, Any]]]:
    attempts=[]; parsed=[]
    attempts.append({"ok":True,"curated_eta_source_registry":curated_eta_source_registry(),"note":"v9.7 registry: tables are used only if columns explicitly identify eta/s or viscosity/entropy."})
    urls=[]
    for item in curated_eta_source_registry():
        if item.get('type')=='discovery_page' and item.get('url'):
            try:
                links=discover_links(item['url'],pattern=r'/download/table/.*/csv',timeout=timeout)
                attempts.append({'url':item['url'],'ok':True,'n_links':len(links),'label':item['label']})
                urls.extend(links[:20])
            except Exception as e:
                attempts.append({'url':item['url'],'ok':False,'error':str(e),'label':item['label']})
    # Known negative controls retained to prove v2/flow rejection.
    urls += ['https://www.hepdata.net/download/table/ins1666817/Table%201/1/csv']
    seen=set()
    for u in urls:
        if not u or u in seen: continue
        seen.add(u)
        try:
            p=download_file(u,cache/'hepdata_kss',filename=slugify(url_basename(u))+'.csv',timeout=timeout,force=force,max_bytes=50*1024*1024)
            df=read_table_any(p)
            attempts.append({'url':u,'ok':True,'path':str(p),'shape':list(df.shape) if df is not None else None})
            if df is not None and df.shape[1]>=2:
                parsed.append((u,df))
        except Exception as e:
            attempts.append({'url':u,'ok':False,'error':str(e)})
    return parsed, attempts


# ------------------------- v9.8 overrides -------------------------
# Correctness/data-limited/science-quality refinements requested after v9.7.


def first_nonempty_dataframe(*dfs: Any) -> Optional[Any]:
    """Return the first object that looks like a non-empty pandas DataFrame/table."""
    for d in dfs:
        if d is not None and hasattr(d, 'shape') and getattr(d, 'shape', (0, 0))[0] > 0 and getattr(d, 'shape', (0, 0))[1] > 0:
            try:
                if not getattr(d, 'empty', False):
                    return d
            except Exception:
                return d
    return None


def healpy_available() -> bool:
    try:
        import healpy  # noqa: F401
        return True
    except Exception:
        return False


def selected_product_is_harmonic(path_or_name: Any) -> bool:
    return _is_harmonic_name(path_or_name) and not is_mask_like_product(path_or_name)


def harmonic_data_limited_info(path: Any, reason: str = 'only_harmonic_product_available') -> Dict[str, Any]:
    return {
        'error': reason,
        'path': str(path) if path is not None else None,
        'requires_healpy': True,
        'install_hint': 'conda install -c conda-forge healpy',
        'note': 'Only alm/klm harmonic products were found; use healpy.alm2map or rerun with a pixel-map product.'
    }


def _exact_pixel_map_candidates(fits: Sequence[Path]) -> List[Path]:
    """Exact κ/convergence pixel-map search before harmonic fallback."""
    out=[]
    for q in fits:
        n=str(q).replace('\\','/').lower()
        if is_mask_like_product(q) or is_bad_map_product(q):
            continue
        if re.search(r'(kappa|convergence)', n) and not re.search(r'(alm|klm|curl|noise|sim|random|meanfield|mf_)', n):
            info=_fits_product_kind(q)
            if info.get('kind') in ('wcs_image','healpix_table'):
                out.append(q)
    return sorted(out, key=_fits_map_score)


def _choose_fits_map_from_path(path: Path, extract_dir: Path) -> Optional[Path]:
    """v9.8: exact pixel κ/convergence products first; harmonic only as controlled fallback."""
    path=Path(path)
    if str(path).lower().endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        info=_fits_product_kind(path)
        if info.get('kind') in ('wcs_image','healpix_table') and not is_mask_like_product(path) and not _is_harmonic_name(path):
            return path
        if info.get('kind') == 'harmonic_alm' and not is_mask_like_product(path):
            return path
    root=extract_if_archive(path, extract_dir)
    fits=[q for q in Path(root).rglob('*') if q.is_file() and re.search(r'\.fits(\.gz)?$|\.fit(\.gz)?$', q.name, re.I)]
    if not fits:
        return None
    exact=_exact_pixel_map_candidates(fits)
    if exact:
        return exact[0]
    inspected=[(q,_fits_product_kind(q)) for q in fits]
    usable=[q for q,info in inspected if info.get('kind') in ('wcs_image','healpix_table') and not is_mask_like_product(q) and not is_bad_map_product(q)]
    if usable:
        return sorted(usable,key=_fits_map_score)[0]
    harmonic=[q for q,info in inspected if info.get('kind') == 'harmonic_alm' and not is_mask_like_product(q)]
    if harmonic:
        return sorted(harmonic,key=_fits_map_score)[0]
    masks=[q for q,_ in inspected if is_mask_like_product(q)]
    return sorted(masks or [q for q,_ in inspected], key=_fits_map_score)[0]


def download_act_dr6_lensing_map(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    """v9.8 ACT DR6 downloader: exact pixel-map search before harmonic fallback."""
    attempts=[]
    if not allow_large:
        return None,[{'ok':False,'reason':'large_download_not_enabled','note':'ACT DR6 lensing archive is large; rerun with --allow-large.'}]
    url='https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/dr6_lensing_release.tar.gz'
    try:
        links=discover_links('https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html', pattern=r'dr6_lensing_release\.tar\.gz', timeout=timeout)
        attempts.append({'url':'https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html','ok':True,'n_links':len(links)})
        if links: url=links[0]
    except Exception as e:
        attempts.append({'ok':False,'url':'ACT landing page','error':str(e)})
    p=download_file(url, cache/'act_dr6', filename='dr6_lensing_release.tar.gz', timeout=timeout, force=force, max_bytes=None)
    attempts.append({'ok':True,'path':str(p),'url':url,'size':p.stat().st_size if p.exists() else None})
    root=extract_if_archive(p, cache/'act_dr6'/'dr6_lensing_release_extracted')
    fits=[q for q in Path(root).rglob('*') if q.is_file() and re.search(r'\.fits(\.gz)?$|\.fit(\.gz)?$', q.name, re.I)]
    exact=_exact_pixel_map_candidates(fits)
    inspected=[]
    for q in fits[:120]:
        info=_fits_product_kind(q)
        inspected.append({'path':str(q),'kind':info.get('kind'),'is_mask_like':is_mask_like_product(q),'harmonic':selected_product_is_harmonic(q)})
    if exact:
        attempts.append({'ok':True,'selected_map':str(exact[0]),'selection':'exact_act_kappa_pixel_map','n_fits_inspected':len(fits)})
        return exact[0], attempts
    harmonic=[q for q in fits if selected_product_is_harmonic(q) and not is_bad_map_product(q)]
    if harmonic:
        best=sorted(harmonic,key=_fits_map_score)[0]
        attempts.append({'ok':True,'selected_map':str(best),'selection':'act_harmonic_requires_healpy_alm2map','requires_healpy':True,'healpy_available':healpy_available(),'n_fits_inspected':len(fits),'inspected_sample':inspected[:20]})
        return best, attempts
    nonmask=[q for q in fits if not is_mask_like_product(q) and _fits_product_kind(q).get('kind') in ('wcs_image','healpix_table')]
    if nonmask:
        best=sorted(nonmask,key=_fits_map_score)[0]
        attempts.append({'ok':True,'selected_map':str(best),'selection':'fallback_nonmask_pixel_map','n_fits_inspected':len(fits)})
        return best, attempts
    attempts.append({'ok':False,'reason':'no_kappa_pixel_or_harmonic_product','requires_healpy':False,'n_fits_inspected':len(fits),'inspected_sample':inspected[:20]})
    return None, attempts


def download_planck_lensing_or_spectra(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False, prefer_map: bool = True) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    """v9.8 Planck PR4/PR3 downloader with pixel-map-first, harmonic-as-healpy-only fallback."""
    attempts=[]
    if prefer_map:
        if not allow_large:
            return None,[{'ok':False,'reason':'large_download_not_enabled','note':'Planck lensing maps are large; rerun with --allow-large.'}]
        api='https://api.github.com/repos/carronj/planck_PR4_lensing/releases/tags/Data'
        try:
            rel=json.loads(http_get_bytes(api,timeout=timeout).decode('utf-8'))
            assets=rel.get('assets',[])
            asset=next((a for a in assets if re.search(r'PR42018like_maps\.tar', a.get('name',''))), None)
            if asset:
                attempts.append({'ok':True,'github_releases':'carronj/planck_PR4_lensing','release':rel.get('name'),'asset':asset.get('name'),'api':api})
                p=download_file(asset['browser_download_url'], cache/'planck_lensing', filename=asset['name'], timeout=timeout, force=force, max_bytes=None)
                attempts.append({'ok':True,'path':str(p),'url':asset['browser_download_url'],'size':p.stat().st_size if p.exists() else None})
                root=extract_if_archive(p, cache/'planck_lensing'/'maps_extracted')
                fits=[q for q in Path(root).rglob('*') if q.is_file() and re.search(r'\.fits(\.gz)?$|\.fit(\.gz)?$', q.name, re.I)]
                exact=_exact_pixel_map_candidates(fits)
                if exact:
                    attempts.append({'ok':True,'selected_map':str(exact[0]),'selection':'planck_exact_kappa_pixel_map','n_fits_inspected':len(fits)})
                    return exact[0], attempts
                harmonic=[q for q in fits if selected_product_is_harmonic(q) and not is_mask_like_product(q)]
                if harmonic:
                    best=sorted(harmonic,key=_fits_map_score)[0]
                    attempts.append({'ok':True,'selected_map':str(best),'selection':'planck_pr4_harmonic_klm_requires_healpy_alm2map','requires_healpy':True,'healpy_available':healpy_available(),'n_fits_inspected':len(fits)})
                    return best, attempts
                pixel=[q for q in fits if _fits_product_kind(q).get('kind') in ('wcs_image','healpix_table') and not is_mask_like_product(q)]
                if pixel:
                    best=sorted(pixel,key=_fits_map_score)[0]
                    attempts.append({'ok':True,'selected_map':str(best),'selection':'planck_fallback_nonmask_pixel_map','n_fits_inspected':len(fits)})
                    return best, attempts
                attempts.append({'ok':False,'reason':'no_planck_kappa_pixel_or_harmonic_product','n_fits_inspected':len(fits)})
        except Exception as e:
            attempts.append({'ok':False,'api':api,'error':str(e)})
    # fallback to spectra for non-map tests
    p, att = download_planck_spectrum_candidates(cache, timeout=timeout, force=force, max_files=1)
    attempts.extend(att)
    if isinstance(p, list):
        return (p[0] if p else None), attempts
    return p, attempts


DESI_DR2_ALL_GCCOMB_SCHEMA = [
    {'row':0, 'tracer':'BGS', 'z':0.295, 'observable':'DV'},
    {'row':1, 'tracer':'LRG', 'z':0.510, 'observable':'DM'}, {'row':2, 'tracer':'LRG', 'z':0.510, 'observable':'DH'},
    {'row':3, 'tracer':'LRG', 'z':0.706, 'observable':'DM'}, {'row':4, 'tracer':'LRG', 'z':0.706, 'observable':'DH'},
    {'row':5, 'tracer':'LRG+ELG', 'z':0.930, 'observable':'DM'}, {'row':6, 'tracer':'LRG+ELG', 'z':0.930, 'observable':'DH'},
    {'row':7, 'tracer':'ELG', 'z':1.317, 'observable':'DM'}, {'row':8, 'tracer':'ELG', 'z':1.317, 'observable':'DH'},
    {'row':9, 'tracer':'QSO', 'z':1.491, 'observable':'DM'}, {'row':10, 'tracer':'QSO', 'z':1.491, 'observable':'DH'},
    {'row':11, 'tracer':'LyA', 'z':2.330, 'observable':'DM'}, {'row':12, 'tracer':'LyA', 'z':2.330, 'observable':'DH'},
]


def _desi_schema_for_n(n: int) -> List[Dict[str, Any]]:
    if n == len(DESI_DR2_ALL_GCCOMB_SCHEMA):
        return DESI_DR2_ALL_GCCOMB_SCHEMA
    return [{'row':i, 'tracer':'unknown', 'z':None, 'observable':'DV'} for i in range(n)]


def _bao_model_vector(z: np.ndarray, obs: Sequence[str], Om: float, nu: float) -> np.ndarray:
    z=np.asarray(z,float)
    zmax=max(float(np.nanmax(z))*1.05, 0.1)
    zg=np.linspace(0,zmax,768)
    E=np.sqrt(np.maximum(Om*(1+zg)**3 + (1-Om)*(1 + nu*np.log1p(zg)), 1e-8))
    inv=1/E; dc=np.zeros_like(zg); dc[1:]=np.cumsum(0.5*(inv[1:]+inv[:-1])*np.diff(zg))
    Dc=np.interp(z,zg,dc)
    Ez=np.sqrt(np.maximum(Om*(1+z)**3 + (1-Om)*(1 + nu*np.log1p(z)), 1e-8))
    out=np.zeros_like(z)
    for i,o in enumerate(obs):
        oo=str(o).upper()
        if oo == 'DM': out[i]=Dc[i]
        elif oo == 'DH': out[i]=1.0/Ez[i]
        else: out[i]=np.maximum(z[i]*Dc[i]*Dc[i]/Ez[i],1e-14)**(1/3)
    return out


def fit_desi_dr2_simplified_rvm_likelihood(paths: Sequence[Path], max_rows: Optional[int] = None) -> Dict[str, Any]:
    """v9.8 schema-based DESI DR2 BAO model-vector likelihood.

    Uses explicit ALL_GCcomb row schema where possible, fits observable-specific
    DM/DH/DV model vectors with one nuisance scale, and reports leave-one-z-bin
    and leave-one-observable-type diagnostics. Still not a Boltzmann/MCMC run.
    """
    if pd is None:
        return {'error':'pandas_missing'}
    means=[Path(p) for p in desi_dr2_mean_paths(paths) if 'ALL_GCcomb_mean' in str(p)] or desi_dr2_mean_paths(paths)
    if not means:
        return {'error':'no_desi_dr2_mean_paths'}
    mean_path=Path(means[0]); cov_path=None
    for p in paths:
        ps=str(p)
        if 'ALL_GCcomb_cov' in ps and 'desi_bao_dr2' in ps:
            cov_path=Path(p); break
    try:
        df=read_table_any(mean_path,max_rows=max_rows)
        nums=find_numeric_columns(df)
        if len(nums)<2:
            return {'error':'mean_table_has_too_few_numeric_columns','mean_path':str(mean_path)}
        z_tab=numeric_array(df,nums[0]); y=numeric_array(df,nums[1])
        m=np.isfinite(z_tab)&np.isfinite(y)&(z_tab>0)&(y>0)
        z_tab=z_tab[m]; y=y[m]; n=len(y)
        if n<3:
            return {'error':'too_few_desi_rows','n':int(n)}
        schema=_desi_schema_for_n(n)
        # Use table redshifts as source of truth, but attach explicit obs labels from schema.
        obs=[schema[i]['observable'] if i < len(schema) else 'DV' for i in range(n)]
        tracers=[schema[i]['tracer'] if i < len(schema) else 'unknown' for i in range(n)]
        z=z_tab
        if cov_path and cov_path.exists():
            C=np.loadtxt(cov_path); C=np.asarray(C,float)
            if C.shape[0] != n: C=C[:n,:n]
        else:
            C=np.diag(np.maximum(np.abs(y)*0.05,1.0)**2)
        C=C+np.eye(n)*max(1e-12,float(np.nanmedian(np.diag(C)))*1e-10)
        def subset_fit(mask, nu_free=True):
            mask=np.asarray(mask,bool)
            yy=y[mask]; zz=z[mask]; oo=[o for o,mm in zip(obs,mask) if mm]
            CC=C[np.ix_(mask,mask)]
            if len(yy)<2: return None
            iC=np.linalg.pinv(CC)
            def chi2(Om,nu):
                f=_bao_model_vector(zz,oo,Om,nu)
                denom=float(f@iC@f)
                if denom<=0 or not np.isfinite(denom): return np.inf,np.nan
                alpha=float(f@iC@yy)/denom
                r=yy-alpha*f
                return float(r@iC@r),alpha
            grid_om=np.linspace(0.18,0.50,41)
            grid_nu=np.linspace(-0.2,0.2,81) if nu_free else [0.0]
            best=(np.inf,None,None,None)
            for Om in grid_om:
                for nu in grid_nu:
                    ch,a=chi2(float(Om),float(nu))
                    if ch<best[0]: best=(ch,float(Om),float(nu),a)
            return {'chi2':float(best[0]),'omega_m':best[1],'nu':best[2],'alpha':best[3],'n':int(len(yy)),'nu_grid_bound_hit':bool(abs(best[2])>=0.199 if best[2] is not None else False)}
        allmask=np.ones(n,dtype=bool)
        lcdm=subset_fit(allmask,nu_free=False); rvm=subset_fit(allmask,nu_free=True)
        delta=(lcdm['chi2']-rvm['chi2']) if lcdm and rvm else None
        loo_z=[]
        for zz in sorted(set([round(float(v),3) for v in z])):
            mask=np.array([round(float(v),3)!=zz for v in z])
            if np.sum(mask)>=3:
                l=subset_fit(mask,False); r=subset_fit(mask,True)
                if l and r: loo_z.append({'left_out_z':zz,'n':int(np.sum(mask)),'delta_chi2':float(l['chi2']-r['chi2']),'nu_like':r['nu'],'nu_grid_bound_hit':r['nu_grid_bound_hit']})
        loo_obs=[]
        for typ in sorted(set(obs)):
            mask=np.array([o!=typ for o in obs])
            if np.sum(mask)>=3:
                l=subset_fit(mask,False); r=subset_fit(mask,True)
                if l and r: loo_obs.append({'left_out_observable':typ,'n':int(np.sum(mask)),'delta_chi2':float(l['chi2']-r['chi2']),'nu_like':r['nu'],'nu_grid_bound_hit':r['nu_grid_bound_hit']})
        return {
            'mode':'desi_dr2_schema_model_likelihood_v9_8',
            'mean_path':str(mean_path),'cov_path':str(cov_path) if cov_path else None,
            'schema':'DESI_DR2_ALL_GCCOMB_SCHEMA' if n==13 else 'fallback_unknown_schema',
            'n_points':int(n),'observables':obs,'tracers':tracers,
            'chi2_lcdm':lcdm['chi2'] if lcdm else None,'omega_m_lcdm':lcdm['omega_m'] if lcdm else None,'alpha_lcdm':lcdm['alpha'] if lcdm else None,
            'chi2_rvm_like':rvm['chi2'] if rvm else None,'omega_m_rvm_like':rvm['omega_m'] if rvm else None,'nu_like':rvm['nu'] if rvm else None,'alpha_rvm_like':rvm['alpha'] if rvm else None,
            'delta_chi2_rvm_vs_lcdm':float(delta) if delta is not None else None,'delta_aic_rvm_vs_lcdm':float(delta-2.0) if delta is not None else None,
            'nu_grid_bound_hit':bool(rvm and rvm.get('nu_grid_bound_hit')),
            'leave_one_redshift_bin_out':loo_z,
            'leave_one_observable_type_out':loo_obs,
            'loo_any_bound_hit':bool(any(x.get('nu_grid_bound_hit') for x in loo_z+loo_obs)),
            'note':'Schema-based DM/DH/DV BAO model-vector likelihood with nuisance scale. Diagnostic if ν hits ±grid boundary or schema is fallback.'
        }
    except Exception as e:
        return {'error':'desi_schema_model_likelihood_failed','detail':str(e),'mean_path':str(mean_path)}


def t06_matched_status(z: Optional[float], depth_corr: Optional[float], matched_controls: Dict[str, Any]) -> str:
    if z is None:
        return 'data_limited'
    if not matched_controls or matched_controls.get('error') or matched_controls.get('n_matched_base',0) < 100:
        return 'matched_controls_data_limited' if z>2 else 'null'
    # v9.8: only matched/statistic-aware status counts; raw z alone never confirms.
    if depth_corr is not None and abs(depth_corr)>0.5:
        return 'depth_confounded'
    if z>2:
        return 'matched_statistic_suggestive'
    return 'null'


def t07_formal_status(mean_diff: Optional[float], p_one_sided: Optional[float], independent_endpoint_support: bool=False) -> str:
    if mean_diff is None:
        return 'data_limited'
    if mean_diff < 0 and not independent_endpoint_support:
        return 'formal_tension_negative_density_split'
    if mean_diff > 0 and p_one_sided is not None and p_one_sided < 0.05:
        return 'suggestive'
    return 'null'


def highz_split_by_survey_model(df: Any, seed: int=12345) -> Dict[str, Any]:
    """v9.8 survey split: KROSS/KMOS3D groups separated when identifiable."""
    if df is None or pd is None:
        return {'error':'no_dataframe'}
    candidates=[]
    for c in df.columns:
        cl=str(c).lower()
        if any(k in cl for k in ['survey','field','source','catalog','sample','instrument']):
            vals=df[c].astype(str).to_numpy()
            if 1 < len(set(vals[:min(len(vals),500)])) <= 30:
                candidates.append(c)
    # Detect survey from table/source labels if available; otherwise global.
    split_col=candidates[0] if candidates else None
    out={'split_column':str(split_col) if split_col is not None else None,'groups':{}}
    if split_col is not None:
        for val,sub in df.groupby(split_col):
            if len(sub)>=15:
                s=highz_acceleration_summary(sub)
                b=highz_group_bootstrap_summary(sub,seed=seed)
                out['groups'][str(val)]={'n_rows':int(len(sub)),'acceleration_summary':s,'object_bootstrap':b,'offset_status':_highz_offset_trend_status(s,b)}
    if not out['groups']:
        s=highz_acceleration_summary(df); b=highz_group_bootstrap_summary(df,seed=seed)
        out['groups']['global_unsplit']={'n_rows':int(len(df)),'acceleration_summary':s,'object_bootstrap':b,'offset_status':_highz_offset_trend_status(s,b)}
        out['note']='No KROSS/KMOS3D survey split column found; global unsplit model only.'
    out['n_groups']=len(out['groups'])
    return out


def _highz_offset_trend_status(summary: Dict[str, Any], boot: Dict[str, Any]) -> Dict[str, Any]:
    q=summary.get('quality_cut',{}) if isinstance(summary,dict) else {}
    slope=q.get('log_g_vs_z_coef',[None])[0] if q.get('log_g_vs_z_coef') else None
    pval=q.get('spearman_z_g',{}).get('pvalue') if isinstance(q.get('spearman_z_g'),dict) else None
    mean=q.get('mean_a_proxy_m_s2')
    boot_p16=boot.get('slope_boot_p16') if isinstance(boot,dict) else None
    return {
        'offset_above_milgrom': bool(mean is not None and mean>1.2e-10),
        'trend_positive_row_level': bool(slope is not None and slope>0 and pval is not None and pval<0.05),
        'trend_positive_object_bootstrap': bool(boot_p16 is not None and boot_p16>0),
        'status': 'offset_and_trend_suggestive' if (mean is not None and mean>1.2e-10 and slope is not None and slope>0 and pval is not None and pval<0.05 and (boot_p16 is None or boot_p16>0)) else ('offset_only' if mean is not None and mean>1.2e-10 else 'null_or_data_limited')
    }


def sparc_robustness_matrix(rotmod_paths: Sequence[Path], seed: int=12345) -> Dict[str, Any]:
    """v9.8 SPARC robustness matrix with simple galaxy-class filename splits when metadata is absent."""
    paths=list(rotmod_paths)
    if not paths:
        return {'error':'no_sparc_paths'}
    rng=np.random.default_rng(seed)
    base=fit_sparc_a0(paths,max_galaxies=None)
    rows=[]
    split_specs=[('first_half',paths[:max(1,len(paths)//2)]),('second_half',paths[max(1,len(paths)//2):]),('first_50',paths[:50]),('first_100',paths[:100])]
    # Crude class splits by filename prefixes/names if SPARC metadata files are not present.
    for label,subset in split_specs:
        if len(subset)>=5:
            f=fit_sparc_a0(subset,max_galaxies=None); f['subset']=label; f['split_type']='order'; rows.append(f)
    # Try gas/LSB-ish name heuristics; harmless if not informative.
    name_groups={
        'dwarf_or_low_mass_name_hint':[p for p in paths if re.search(r'(^|[_-])(ddo|ugc|ic|f\d)', p.stem.lower())],
        'ngc_name_hint':[p for p in paths if p.stem.lower().startswith('ngc')],
    }
    for label,subset in name_groups.items():
        if len(subset)>=10:
            f=fit_sparc_a0(subset,max_galaxies=None); f['subset']=label; f['split_type']='filename_class_hint'; rows.append(f)
    boots=[]
    if len(paths)>=10:
        for _ in range(60):
            sub=list(rng.choice(paths,size=min(len(paths),max(20,len(paths)//2)),replace=True))
            f=fit_sparc_a0(sub,max_galaxies=None)
            if isinstance(f,dict) and f.get('a0_best_m_s2'):
                boots.append(float(f['a0_best_m_s2']))
    return {'base':base,'subsets':rows,'bootstrap_subsample_n':len(boots),'a0_subsample_median':float(np.nanmedian(boots)) if boots else None,'a0_subsample_p16':float(np.nanpercentile(boots,16)) if boots else None,'a0_subsample_p84':float(np.nanpercentile(boots,84)) if boots else None,'robust_within_factor2_of_milgrom': bool(base.get('a0_best_m_s2') and 0.5 < base.get('a0_best_m_s2')/1.2e-10 < 2.0),'metadata_note':'Galaxy-class splits use filename hints unless SPARC metadata are added.'}

# ------------------------- v9.9 overrides -------------------------
# Data-limited repair and stricter science controls after v9.8.

V99_VERSION_NOTE = "v9.9: candidate-map iteration, ACT/Planck harmonic diagnostics, catalogue-angle filament parsing, stricter result schemas."


def _is_nonempty_dataframe_like(d: Any) -> bool:
    try:
        return d is not None and hasattr(d, 'shape') and int(d.shape[0]) > 0 and int(d.shape[1]) > 0 and not bool(getattr(d, 'empty', False))
    except Exception:
        return False


def first_nonempty_dataframe(*dfs: Any) -> Optional[Any]:
    """v9.9 robust DataFrame/table coalescer; never uses pandas truthiness."""
    for d in dfs:
        if _is_nonempty_dataframe_like(d):
            return d
    return None


def validate_result_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    required = ['test_id', 'status', 'generated_utc', 'metrics', 'data_sources']
    missing = [k for k in required if k not in payload]
    return {'ok': not missing, 'missing_required_keys': missing, 'schema': 'ccdr_tierA_result_minimal_v1'}


def _alm_table_to_complex(hdul: Any, map_path: Path) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """v9.9 alm/klm FITS parser.

    Supports packed REAL/IMAG tables and explicit ell/m columns. Non-finite alm
    values are zeroed before healpy.alm2map so one bad coefficient cannot create
    an all-NaN map.
    """
    for hdu_i, hdu in enumerate(hdul):
        data = getattr(hdu, 'data', None)
        if data is None or not hasattr(data, 'columns'):
            continue
        names = list(getattr(data.columns, 'names', []) or [])
        low = {str(n).lower(): str(n) for n in names}
        real_col = next((low[k] for k in low if k in ('real','re','alm_real','klm_real') or 'real' in k), None)
        imag_col = next((low[k] for k in low if k in ('imag','im','alm_imag','klm_imag') or 'imag' in k), None)
        ell_col = next((low[k] for k in low if k in ('ell','l','lval','multipole')), None)
        emm_col = next((low[k] for k in low if k in ('m','emm','mval')), None)
        if real_col is None:
            continue
        try:
            re_arr = np.asarray(data[real_col], float).reshape(-1)
            im_arr = np.asarray(data[imag_col], float).reshape(-1) if imag_col else np.zeros_like(re_arr)
            finite_coeff = np.isfinite(re_arr) & np.isfinite(im_arr)
            re_arr = np.where(np.isfinite(re_arr), re_arr, 0.0)
            im_arr = np.where(np.isfinite(im_arr), im_arr, 0.0)
            if ell_col and emm_col:
                ell = np.asarray(data[ell_col], int).reshape(-1)
                emm = np.asarray(data[emm_col], int).reshape(-1)
                valid = np.isfinite(re_arr) & np.isfinite(im_arr) & (ell >= 0) & (emm >= 0) & (emm <= ell)
                if np.sum(valid) < 10:
                    continue
                lmax = int(np.nanmax(ell[valid]))
                try:
                    import healpy as hp
                    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
                    idx = hp.Alm.getidx(lmax, ell[valid], emm[valid])
                    alm[idx] = re_arr[valid] + 1j * im_arr[valid]
                    return alm, {'hdu': int(hdu_i), 'real_col': real_col, 'imag_col': imag_col, 'ell_col': ell_col, 'm_col': emm_col, 'n_alm': int(len(alm)), 'lmax_explicit': lmax, 'n_coeff_input': int(len(re_arr)), 'n_coeff_finite_input': int(np.sum(finite_coeff)), 'path': str(map_path)}
                except Exception as e:
                    return None, {'error': 'explicit_ell_m_alm_requires_healpy', 'detail': str(e), 'hdu': int(hdu_i), 'path': str(map_path)}
            alm = re_arr + 1j * im_arr
            if len(alm) < 10:
                continue
            return alm, {'hdu': int(hdu_i), 'real_col': real_col, 'imag_col': imag_col, 'n_alm': int(len(alm)), 'n_coeff_finite_input': int(np.sum(finite_coeff)), 'path': str(map_path)}
        except Exception as e:
            continue
    return None, {'error': 'no_simple_real_imag_alm_columns', 'path': str(map_path)}


def _map_candidates_from_archive(path: Path, extract_dir: Path) -> Tuple[List[Path], List[Dict[str, Any]]]:
    root = extract_if_archive(Path(path), extract_dir)
    fits = [q for q in Path(root).rglob('*') if q.is_file() and re.search(r'\.fits(\.gz)?$|\.fit(\.gz)?$', q.name, re.I)]
    inspected = []
    for q in fits[:200]:
        info = _fits_product_kind(q)
        inspected.append({'path': str(q), 'kind': info.get('kind'), 'is_mask_like': is_mask_like_product(q), 'harmonic': selected_product_is_harmonic(q), 'score': _fits_map_score(q)[0]})
    exact = _exact_pixel_map_candidates(fits)
    pixel = [q for q in fits if _fits_product_kind(q).get('kind') in ('wcs_image', 'healpix_table') and not is_mask_like_product(q) and not is_bad_map_product(q)]
    harmonic = [q for q in fits if selected_product_is_harmonic(q) and not is_bad_map_product(q) and not is_mask_like_product(q)]
    ranked = []
    for seq in (exact, sorted(pixel, key=_fits_map_score), sorted(harmonic, key=_fits_map_score)):
        for q in seq:
            if q not in ranked:
                ranked.append(q)
    return ranked, inspected


def download_act_dr6_lensing_map_candidates(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False) -> Tuple[List[Path], List[Dict[str, Any]]]:
    attempts=[]
    if not allow_large:
        return [], [{'ok': False, 'reason': 'large_download_not_enabled', 'note': 'ACT DR6 lensing archive is large; rerun with --allow-large.'}]
    url='https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/dr6_lensing_release.tar.gz'
    try:
        links=discover_links('https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html', pattern=r'dr6_lensing_release\.tar\.gz', timeout=timeout)
        attempts.append({'url':'https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html','ok':True,'n_links':len(links)})
        if links: url=links[0]
    except Exception as e:
        attempts.append({'ok':False,'url':'ACT landing page','error':str(e)})
    try:
        p=download_file(url, cache/'act_dr6', filename='dr6_lensing_release.tar.gz', timeout=timeout, force=force, max_bytes=None)
        attempts.append({'ok':True,'path':str(p),'url':url,'size':p.stat().st_size if p.exists() else None})
        cand, inspected = _map_candidates_from_archive(p, cache/'act_dr6'/'dr6_lensing_release_extracted')
        attempts.append({'ok': bool(cand), 'candidate_count': len(cand), 'candidate_paths_first20': [str(x) for x in cand[:20]], 'inspected_sample': inspected[:25], 'selection': 'v9.9_exact_pixel_then_harmonic_candidates', 'requires_healpy_if_harmonic_only': bool(cand and all(selected_product_is_harmonic(x) for x in cand))})
        return cand, attempts
    except Exception as e:
        attempts.append({'ok':False,'url':url,'error':str(e)})
        return [], attempts


def download_planck_lensing_map_candidates(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False) -> Tuple[List[Path], List[Dict[str, Any]]]:
    attempts=[]
    if not allow_large:
        return [], [{'ok': False, 'reason': 'large_download_not_enabled', 'note': 'Planck lensing maps are large; rerun with --allow-large.'}]
    api='https://api.github.com/repos/carronj/planck_PR4_lensing/releases/tags/Data'
    try:
        rel=json.loads(http_get_bytes(api,timeout=timeout).decode('utf-8'))
        asset=next((a for a in rel.get('assets',[]) if re.search(r'PR42018like_maps\.tar', a.get('name',''))), None)
        if not asset:
            return [], [{'ok':False,'api':api,'reason':'PR42018like_maps_asset_not_found'}]
        attempts.append({'ok':True,'github_releases':'carronj/planck_PR4_lensing','release':rel.get('name'),'asset':asset.get('name'),'api':api})
        p=download_file(asset['browser_download_url'], cache/'planck_lensing', filename=asset['name'], timeout=timeout, force=force, max_bytes=None)
        attempts.append({'ok':True,'path':str(p),'url':asset['browser_download_url'],'size':p.stat().st_size if p.exists() else None})
        cand, inspected = _map_candidates_from_archive(p, cache/'planck_lensing'/'maps_extracted')
        attempts.append({'ok': bool(cand), 'candidate_count': len(cand), 'candidate_paths_first20': [str(x) for x in cand[:20]], 'inspected_sample': inspected[:25], 'selection': 'v9.9_planck_pixel_then_klm_candidates', 'requires_healpy_if_harmonic_only': bool(cand and all(selected_product_is_harmonic(x) for x in cand))})
        return cand, attempts
    except Exception as e:
        attempts.append({'ok':False,'api':api,'error':str(e)})
        return [], attempts


def sample_first_valid_kappa_candidate(candidates: Sequence[Path], ra: np.ndarray, dec: np.ndarray, *, max_points: int = 5000, prefer_healpy: bool = False, no_harmonic: bool = False, min_finite: int = 20) -> Tuple[Optional[Path], Optional[np.ndarray], Dict[str, Any]]:
    attempts=[]
    harmonic_seen=[]
    for mp in candidates:
        mp=Path(mp)
        if is_mask_like_product(mp):
            attempts.append({'path': str(mp), 'skipped': 'mask_or_weight_product'})
            continue
        if no_harmonic and selected_product_is_harmonic(mp):
            harmonic_seen.append(str(mp)); attempts.append({'path': str(mp), 'skipped': 'harmonic_blocked_by_no_harmonic'}); continue
        vals, info = sample_map_values_for_points(mp, ra, dec, max_points=max_points, prefer_healpix=bool(prefer_healpy or healpy_available()))
        vcheck = validate_map_sample_values(vals, info, mp, min_finite=min_finite)
        attempts.append({'path': str(mp), 'sampling': info, 'validation': vcheck})
        if vcheck.get('ok'):
            return mp, vals, {'selected_map': str(mp), 'candidate_attempts': attempts, 'map_sampling': info, 'map_validation': vcheck}
    if harmonic_seen:
        return None, None, {'candidate_attempts': attempts, **harmonic_data_limited_info(harmonic_seen[0], 'only_harmonic_products_available_or_reconstruction_failed')}
    return None, None, {'candidate_attempts': attempts, 'error': 'no_valid_kappa_candidate_after_validation'}


def parse_vizier_filament_table(path: Path, max_rows: Optional[int]=None) -> Tuple[Optional[Any], Dict[str,Any]]:
    """v9.9 VizieR/CDS parser: endpoints first, then catalogue PA/position angle, then kNN fallback."""
    df=read_astronomy_table_any(path,max_rows=max_rows)
    if df is None or len(df)<5:
        return None,{'error':'unparsed_or_too_few_rows'}
    ra_col,dec_col,cinfo=find_sky_coordinate_columns(df)
    cols=list(df.columns)
    if not (ra_col and dec_col):
        aliases_ra=[c for c in cols if re.search(r'(^|[^a-z])(ra|raj2000|ra_icrs|ra1|ra_1|x1)([^a-z]|$)',str(c),re.I)]
        aliases_de=[c for c in cols if re.search(r'(^|[^a-z])(de|dec|dej2000|dec_icrs|dec1|dec_1|y1)([^a-z]|$)',str(c),re.I)]
        if aliases_ra and aliases_de:
            ra_col,dec_col=aliases_ra[0],aliases_de[0]
    if not (ra_col and dec_col):
        return df,{'error':'no_coordinate_columns', **cinfo, 'columns':[str(c) for c in df.columns[:60]]}
    def valarr(c,is_ra):
        arr=numeric_array(df,c)
        if np.isfinite(arr).sum() < max(3, len(arr)//10):
            arr=np.asarray([np.nan if (v:=(_sexagesimal_to_deg_ra(s) if is_ra else _sexagesimal_to_deg_dec(s))) is None else v for s in df[c].astype(str).to_numpy()],float)
        return arr
    ra=valarr(ra_col,True); dec=valarr(dec_col,False)
    m=np.isfinite(ra)&np.isfinite(dec)&(ra>=0)&(ra<=360)&(dec>=-90)&(dec<=90)
    ra2_col=next((c for c in cols if re.search(r'(ra2|ra_2|ra.*end|x2|x_2|raend|ra_end)',str(c),re.I)), None)
    de2_col=next((c for c in cols if re.search(r'(de2|dec2|dec_2|dec.*end|y2|y_2|decend|dec_end)',str(c),re.I)), None)
    angle=None; angle_mode='knn_reconstructed_angle'
    if ra2_col is not None and de2_col is not None:
        ra2=valarr(ra2_col,True); de2=valarr(de2_col,False)
        mm=m&np.isfinite(ra2)&np.isfinite(de2)
        if np.sum(mm)>=20:
            angle=np.full(len(ra),np.nan)
            angle[mm]=np.arctan2(de2[mm]-dec[mm], (ra2[mm]-ra[mm])*np.cos(np.deg2rad(dec[mm])))
            angle_mode='catalogue_endpoint_angle'
    if angle is None:
        pa_col=next((c for c in cols if re.search(r'(^|[^a-z])(pa|posang|position.?angle|theta|angle|phi)([^a-z]|$)',str(c),re.I)), None)
        if pa_col is not None:
            pa=numeric_array(df,pa_col)
            # degrees unless values already look radian-like.
            finite=pa[np.isfinite(pa)]
            if len(finite)>=20:
                angle=np.full(len(ra),np.nan)
                if np.nanmax(np.abs(finite)) > 2*np.pi:
                    angle[np.isfinite(pa)] = np.deg2rad(pa[np.isfinite(pa)])
                else:
                    angle[np.isfinite(pa)] = pa[np.isfinite(pa)]
                angle_mode='catalogue_position_angle'
    return df,{'ra_col':str(ra_col),'dec_col':str(dec_col),'ra':ra,'dec':dec,'finite_mask':m,'angle':angle,'angle_mode':angle_mode,'coordinate_parse':cinfo,'ra2_col':str(ra2_col) if ra2_col is not None else None,'dec2_col':str(de2_col) if de2_col is not None else None,'pa_col':str(pa_col) if 'pa_col' in locals() and pa_col is not None else None,'n_finite':int(np.sum(m))}


def load_kmos_or_kross(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False, max_rows: int = 20000) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    """v9.9 loader: preserve source_catalog so T12 can split KROSS/KMOS3D instead of global-only."""
    attempts=[]
    pages=[('KMOS3D','https://www.mpe.mpg.de/ir/KMOS3D/data'),('KROSS','https://astro.dur.ac.uk/KROSS/data.html')]
    candidates=[]
    for label,page in pages:
        try:
            links=discover_links(page, pattern=r'(catalog|catalogue|table|fits|csv|dat|txt|xlsx)', timeout=timeout)
            attempts.append({'url':page,'ok':True,'n_links':len(links),'source_catalog':label})
            for u in links: candidates.append((label,u))
        except Exception as e:
            attempts.append({'url':page,'ok':False,'error':str(e),'source_catalog':label})
    candidates=[(lab,u) for lab,u in candidates if re.search(r'\.(fits|fit|csv|dat|txt|tsv|xlsx)(\?|$)',u,re.I)]
    candidates=[(lab,u) for lab,u in candidates if not re.search(r'cube|_3D|share-offsite|intent/compose|data-protection|print=yes|linkedin|bsky',u,re.I)]
    candidates=sorted(set(candidates), key=lambda lu: (0 if re.search(r'kross_release|catalog|catalogue|table', lu[1], re.I) else 1, lu[0], lu[1]))
    for idx,(label,u) in enumerate(candidates[:40]):
        try:
            fname=f'{idx:02d}_{url_basename(u)}'
            p=download_file(u, cache/'kmos_kross', filename=fname, timeout=timeout, force=force, max_bytes=None if allow_large else 300*1024*1024)
            attempts.append({'url':u,'ok':True,'path':str(p),'source_catalog':label})
            df=read_table_any(p, max_rows=max_rows)
            if df is not None and len(df)>10:
                try:
                    df=df.copy(); df['source_catalog']=label; df['source_path']=str(p)
                except Exception:
                    pass
                return df, attempts
        except Exception as e:
            attempts.append({'url':u,'ok':False,'error':str(e),'source_catalog':label})
    return None, attempts+[{'ok':False,'error':'no_parseable_kmos_or_kross_catalog'}]


def t10_control_status(best: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    warnings=[]
    if not best:
        return 'data_limited', warnings
    db=best.get('delta_bic_single_minus_mix')
    if db is None:
        return 'data_limited', warnings
    bin_pass=sum(1 for b in best.get('within_redshift_bin_checks',[]) if (b.get('delta_bic_single_minus_mix') or 0)>6)
    sky_pass=sum(1 for b in best.get('sky_region_checks',[]) if (b.get('delta_bic_single_minus_mix') or 0)>6)
    p95=best.get('shuffle_null',{}).get('p95_delta_bic')
    null_ok=(p95 is not None and db>p95)
    best['v9_9_control_summary']={'redshift_bins_passing':int(bin_pass),'sky_regions_passing':int(sky_pass),'shuffle_null_passed':bool(null_ok),'required_for_suggestive':'db>10, >=2 redshift bins, >=2 sky regions, above shuffle p95'}
    if db>10 and bin_pass>=2 and sky_pass>=2 and null_ok:
        return 'suggestive', warnings
    if db>10:
        warnings.append('Mixture is strong after residualization but does not survive required >=2 redshift-bin and >=2 sky-region controls plus shuffle-null.')
        return 'control_limited_mixture_signal', warnings
    return 'null', warnings


def add_claim_strength(res: Dict[str, Any]) -> Dict[str, Any]:
    st=str(res.get('status',''))
    if 'confirm' in st: cs='confirm_like'
    elif 'tension' in st: cs='tension'
    elif 'null' in st: cs='null'
    elif 'data_limited' in st or 'limited' in st: cs='data_limited'
    elif 'diagnostic' in st: cs='diagnostic'
    elif 'suggestive' in st: cs='suggestive'
    else: cs='diagnostic'
    res['claim_strength']=cs
    return res

# Wrap write_result to add claim_strength + minimal schema validation for every output.
_write_result_base = write_result

def write_result(payload: Dict[str, Any], outdir: Path) -> None:
    add_claim_strength(payload)
    payload.setdefault('result_schema_check', validate_result_payload(payload))
    _write_result_base(payload, outdir)


# ------------------------- v10 overrides -------------------------
# Requested after v9.9 analysis: full-run summaries, stronger kappa-map
# diagnostics, endpoint/spine filament preference, decisive control gates, SPARC
# metadata splits, survey-level high-z splits, and conservative posterior/event guards.

V10_VERSION_NOTE = "v10: all-25 summary printing, kappa coordinate validation, ACT/Planck candidate diagnostics, endpoint/spine catalogue priority, SPARC metadata splits, approximate low-ell likelihood."


def kappa_coordinate_validation(selected_map: Any, sampling_info: Optional[Dict[str, Any]], ra: Optional[np.ndarray]=None, dec: Optional[np.ndarray]=None) -> Dict[str, Any]:
    """Report coordinate assumptions for ACT/Planck kappa-product tests.

    This does not certify the science product; it makes the coordinate assumption
    explicit so a later report can distinguish a true κ null from a coordinate-
    ambiguous proxy.
    """
    p = str(selected_map or '')
    info = sampling_info or {}
    name = p.lower()
    is_planck = bool(re.search(r'planck|pr4|com_lensing|klm|plc', name))
    is_act = bool(re.search(r'act|dr6|actadv', name))
    coordsys = str(info.get('coordsys') or '').lower()
    if is_planck:
        expected = 'galactic'
        verified = 'galactic' in coordsys or 'converted' in coordsys
        action = 'ICRS pulsar/Euclid positions should be converted to Galactic before HEALPix sampling.'
    elif is_act:
        expected = 'icrs_or_equatorial_act_release_assumption'
        verified = 'icrs' in coordsys or 'equatorial' in coordsys
        action = 'ACT release coordinate frame is assumed equatorial unless FITS metadata proves otherwise.'
    else:
        expected = 'unknown'
        verified = False
        action = 'Unknown map family; require explicit FITS coordinate metadata.'
    out = {
        'selected_map': p,
        'map_family': 'Planck' if is_planck else ('ACT' if is_act else 'unknown'),
        'sampling_mode': info.get('mode'),
        'sampling_coordsys': info.get('coordsys'),
        'expected_frame': expected,
        'coordsys_verified': bool(verified),
        'coordinate_action': action,
    }
    try:
        if ra is not None and dec is not None:
            out['ra_range_deg'] = [float(np.nanmin(ra)), float(np.nanmax(ra))]
            out['dec_range_deg'] = [float(np.nanmin(dec)), float(np.nanmax(dec))]
    except Exception:
        pass
    return out


def kappa_candidate_diagnostic_summary(selection_info: Dict[str, Any]) -> Dict[str, Any]:
    attempts = selection_info.get('candidate_attempts') or []
    summary = {
        'n_candidates_tried': int(len(attempts)),
        'n_mask_skipped': int(sum(1 for a in attempts if a.get('skipped') == 'mask_or_weight_product')),
        'n_harmonic_attempts': int(sum(1 for a in attempts if selected_product_is_harmonic(a.get('path','')) or (a.get('sampling') or {}).get('mode') == 'alm2map_healpy')),
        'n_validated': int(sum(1 for a in attempts if (a.get('validation') or {}).get('ok'))),
        'requires_healpy': bool(selection_info.get('requires_healpy')),
        'selected_map': selection_info.get('selected_map'),
    }
    if attempts:
        summary['first_failure_reasons'] = [
            {'path': a.get('path'), 'reason': (a.get('validation') or {}).get('reason') or a.get('skipped') or (a.get('sampling') or {}).get('error')}
            for a in attempts[:8]
        ]
    return summary


# Preserve the v9.9 candidate iterator but add an extra diagnostic top-level summary.
_sample_first_valid_kappa_candidate_v99 = sample_first_valid_kappa_candidate

def sample_first_valid_kappa_candidate(candidates: Sequence[Path], ra: np.ndarray, dec: np.ndarray, *, max_points: int = 5000, prefer_healpy: bool = False, no_harmonic: bool = False, min_finite: int = 20) -> Tuple[Optional[Path], Optional[np.ndarray], Dict[str, Any]]:
    mp, vals, info = _sample_first_valid_kappa_candidate_v99(candidates, ra, dec, max_points=max_points, prefer_healpy=prefer_healpy, no_harmonic=no_harmonic, min_finite=min_finite)
    try:
        info['candidate_diagnostic_summary'] = kappa_candidate_diagnostic_summary(info)
        if mp is not None:
            info['coordinate_validation'] = kappa_coordinate_validation(mp, info.get('map_sampling'), ra[:max_points], dec[:max_points])
    except Exception as e:
        info.setdefault('warnings', []).append('v10 candidate diagnostic failed: '+str(e))
    return mp, vals, info


def filament_endpoint_catalogue_registry() -> List[str]:
    """Curated public catalogue attempts where endpoints/spines/PAs may exist.

    The parser still validates endpoint/PA columns before use; these URLs are
    candidates, not trusted evidence.  kNN remains secondary only.
    """
    return [
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/440/2562&-out.all&-out.max=50000',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/A+A/530/A122&-out.all&-out.max=50000',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/A+A/570/A106&-out.all&-out.max=50000',
        # Additional VizieR queries are deliberately broad; the parser rejects
        # tables without coordinate+endpoint/PA evidence.
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/423/3727&-out.all&-out.max=50000',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/445/899&-out.all&-out.max=50000',
    ]


def t08_status_from_mode(mean_corr: Optional[float], angle_mode: Optional[str]) -> str:
    if mean_corr is None:
        return 'data_limited'
    if mean_corr <= 0:
        return 'null'
    if angle_mode == 'catalogue_endpoint_angle':
        return 'endpoint_catalogue_suggestive'
    if angle_mode == 'catalogue_position_angle':
        return 'catalogue_position_angle_suggestive'
    return 'knn_secondary_suggestive'


# Override T10 gate helper with explicit gate breakdown.
def t10_control_status(best: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    warnings=[]
    if not best:
        return 'data_limited', warnings
    db=best.get('delta_bic_single_minus_mix')
    if db is None:
        return 'data_limited', warnings
    bin_checks=best.get('within_redshift_bin_checks',[]) or []
    sky_checks=best.get('sky_region_checks',[]) or []
    bin_pass=sum(1 for b in bin_checks if (b.get('delta_bic_single_minus_mix') or 0)>6)
    sky_pass=sum(1 for b in sky_checks if (b.get('delta_bic_single_minus_mix') or 0)>6)
    p95=best.get('shuffle_null',{}).get('p95_delta_bic')
    null_ok=(p95 is not None and db>p95)
    full_ok=bool(db>10)
    best['v10_control_gate_breakdown']={
        'full_sample_pass': full_ok,
        'delta_bic_single_minus_mix': db,
        'redshift_bins_total': int(len(bin_checks)),
        'redshift_bins_passed': int(bin_pass),
        'redshift_bins_pass': bool(bin_pass>=2),
        'sky_regions_total': int(len(sky_checks)),
        'sky_regions_passed': int(sky_pass),
        'sky_regions_pass': bool(sky_pass>=2),
        'shuffle_null_p95_delta_bic': p95,
        'shuffle_null_pass': bool(null_ok),
        'required_for_suggestive': 'full ΔBIC>10 AND >=2 redshift bins AND >=2 sky regions AND full ΔBIC > shuffle p95',
    }
    if full_ok and bin_pass>=2 and sky_pass>=2 and null_ok:
        return 'suggestive', warnings
    if full_ok:
        failed=[]
        if bin_pass<2: failed.append('redshift_bins')
        if sky_pass<2: failed.append('sky_regions')
        if not null_ok: failed.append('shuffle_null')
        best['v10_control_gate_breakdown']['failed_gates']=failed
        warnings.append('Mixture is strong after residualization but failed control gates: '+', '.join(failed))
        return 'control_limited_mixture_signal', warnings
    return 'null', warnings


def _find_sparc_metadata_candidates(rotmod_paths: Sequence[Path]) -> List[Path]:
    roots=[]
    for p in rotmod_paths:
        try:
            roots.extend([p.parent, p.parent.parent, p.parent.parent.parent])
        except Exception:
            pass
    out=[]; seen=set()
    for r in roots:
        if not r or not Path(r).exists():
            continue
        for q in Path(r).rglob('*'):
            if not q.is_file() or q in seen:
                continue
            if re.search(r'(metadata|sparc|table|galax|mass|phot|info).*\.(csv|tsv|txt|dat|mrt)$', q.name, re.I) and not re.search(r'rotmod', q.name, re.I):
                seen.add(q); out.append(q)
    return out[:20]


def _read_sparc_metadata_table(path: Path) -> Optional[Any]:
    # Try flexible ASCII parsing first; SPARC metadata variants are often whitespace or fixed-width.
    for reader in [read_table_any, read_astronomy_table_any]:
        try:
            df = reader(path, max_rows=5000)
            if df is not None and len(df)>10 and len(df.columns)>=4:
                return df
        except Exception:
            pass
    return None


def sparc_metadata_splits(rotmod_paths: Sequence[Path]) -> Dict[str, Any]:
    """Attempt physical SPARC metadata splits. Falls back cleanly if metadata absent."""
    cands=_find_sparc_metadata_candidates(rotmod_paths)
    result={'metadata_candidates':[str(p) for p in cands], 'tables': [], 'splits': [], 'status':'metadata_not_found'}
    if not cands:
        return result
    galaxy_to_path={p.stem.lower(): p for p in rotmod_paths}
    for p in cands:
        df=_read_sparc_metadata_table(p)
        if df is None:
            continue
        cols=list(df.columns)
        name_col=next((c for c in cols if re.search(r'gal|name|id', str(c), re.I)), cols[0] if cols else None)
        result['tables'].append({'path':str(p),'shape':list(getattr(df,'shape',(0,0))),'columns':[str(c) for c in cols[:30]],'name_col':str(name_col)})
        if name_col is None:
            continue
        names=[str(x).strip().lower() for x in df[name_col].to_numpy()]
        # numeric columns likely to encode inclination, distance, luminosity, surface brightness.
        for c in cols:
            if c == name_col:
                continue
            cl=str(c).lower()
            if not re.search(r'incl|inc|dist|dmpc|sb|mu|lum|mstar|mhi|rdisc|reff|type|morph|gas', cl):
                continue
            arr=numeric_array(df,c)
            good=np.isfinite(arr)
            if good.sum()<20 or np.nanstd(arr[good])==0:
                continue
            med=float(np.nanmedian(arr[good]))
            low_names={names[i] for i,v in enumerate(arr) if np.isfinite(v) and v<=med}
            high_names={names[i] for i,v in enumerate(arr) if np.isfinite(v) and v>med}
            low_paths=[galaxy_to_path[n] for n in low_names if n in galaxy_to_path]
            high_paths=[galaxy_to_path[n] for n in high_names if n in galaxy_to_path]
            if len(low_paths)>=10 and len(high_paths)>=10:
                lf=fit_sparc_a0(low_paths,max_galaxies=None); hf=fit_sparc_a0(high_paths,max_galaxies=None)
                result['splits'].append({'metadata_path':str(p),'column':str(c),'median':med,'low_or_equal_n':len(low_paths),'high_n':len(high_paths),'low_fit':lf,'high_fit':hf})
    result['status']='metadata_splits_available' if result['splits'] else 'metadata_tables_found_no_usable_splits'
    return result


_sparc_robustness_matrix_prev = sparc_robustness_matrix

def sparc_robustness_matrix(rotmod_paths: Sequence[Path], seed: int=12345) -> Dict[str, Any]:
    out=_sparc_robustness_matrix_prev(rotmod_paths, seed=seed)
    try:
        out['physical_metadata_splits']=sparc_metadata_splits(rotmod_paths)
        if out['physical_metadata_splits'].get('status') == 'metadata_splits_available':
            out['metadata_note']='v10 found candidate SPARC metadata splits; filename hints remain secondary.'
        else:
            out['metadata_note']='No usable SPARC metadata split table found locally; filename/order splits remain secondary.'
    except Exception as e:
        out['physical_metadata_splits']={'status':'metadata_split_failed','error':str(e)}
    return out


_load_kmos_or_kross_prev = load_kmos_or_kross

def load_kmos_or_kross(cache: Path, timeout: int = 90, allow_large: bool = False, force: bool = False, max_rows: int = 20000) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    """v10: collect parseable KROSS and KMOS3D tables instead of returning the first one."""
    attempts=[]
    pages=[('KMOS3D','https://www.mpe.mpg.de/ir/KMOS3D/data'),('KROSS','https://astro.dur.ac.uk/KROSS/data.html')]
    candidates=[]
    for label,page in pages:
        try:
            links=discover_links(page, pattern=r'(catalog|catalogue|table|fits|csv|dat|txt|xlsx)', timeout=timeout)
            attempts.append({'url':page,'ok':True,'n_links':len(links),'source_catalog':label})
            for u in links:
                candidates.append((label,u))
        except Exception as e:
            attempts.append({'url':page,'ok':False,'error':str(e),'source_catalog':label})
    candidates=[(lab,u) for lab,u in candidates if re.search(r'\.(fits|fit|csv|dat|txt|tsv|xlsx)(\?|$)',u,re.I)]
    candidates=[(lab,u) for lab,u in candidates if not re.search(r'cube|_3D|share-offsite|intent/compose|data-protection|print=yes|linkedin|bsky',u,re.I)]
    candidates=sorted(set(candidates), key=lambda lu: (0 if re.search(r'kross_release|catalog|catalogue|table', lu[1], re.I) else 1, lu[0], lu[1]))
    tables=[]; seen_labels=set()
    for idx,(label,u) in enumerate(candidates[:60]):
        try:
            fname=f'{idx:02d}_{url_basename(u)}'
            p=download_file(u, cache/'kmos_kross', filename=fname, timeout=timeout, force=force, max_bytes=None if allow_large else 300*1024*1024)
            df=read_table_any(p, max_rows=max_rows)
            ok=df is not None and len(df)>10
            attempts.append({'url':u,'ok':bool(ok),'path':str(p),'source_catalog':label,'shape':list(df.shape) if ok else None})
            if ok:
                try:
                    df=df.copy(); df['source_catalog']=label; df['source_path']=str(p)
                except Exception:
                    pass
                tables.append(df); seen_labels.add(label)
            # Once both surveys have at least one compact table, stop early.
            if len(seen_labels)>=2:
                break
        except Exception as e:
            attempts.append({'url':u,'ok':False,'error':str(e),'source_catalog':label})
    if tables and pd is not None:
        try:
            return pd.concat(tables, ignore_index=True, sort=False), attempts+[{'ok':True,'combined_tables':len(tables),'source_catalogs':sorted(seen_labels)}]
        except Exception:
            return tables[0], attempts+[{'ok':True,'combined_tables':1,'source_catalogs':sorted(seen_labels),'warning':'concat_failed_returned_first'}]
    # fall back to previous loader behavior if discovery structure changed.
    return _load_kmos_or_kross_prev(cache, timeout=timeout, allow_large=allow_large, force=force, max_rows=max_rows)


def planck_lowell_approx_likelihood(ell: np.ndarray, cl: np.ndarray) -> Dict[str, Any]:
    """Approximate low-ell likelihood screen using mid-ell log-linear continuation + cosmic variance.

    This is not a Planck likelihood, but it is stronger than a raw ratio screen
    because it compares low ell to a fitted smooth continuation and an explicit
    cosmic-variance scale.
    """
    ell=np.asarray(ell,float); cl=np.asarray(cl,float)
    m=np.isfinite(ell)&np.isfinite(cl)&(ell>=2)&(cl!=0)
    ell=ell[m]; cl=cl[m]
    low=(ell>=2)&(ell<=30); mid=(ell>30)&(ell<=200)&(cl>0)
    out={'mode':'mid_ell_powerlaw_plus_cosmic_variance_screen','n_low':int(np.sum(low)),'n_mid':int(np.sum(mid)),'official_planck_likelihood':False}
    if np.sum(low)<5 or np.sum(mid)<10:
        out['error']='too_few_low_or_mid_ell_points'; return out
    try:
        x=np.log(ell[mid]); y=np.log(np.abs(cl[mid]))
        beta=np.polyfit(x,y,1)
        pred=np.exp(np.polyval(beta,np.log(ell[low])))
        obs=np.abs(cl[low])
        sigma=np.sqrt(2.0/(2.0*ell[low]+1.0))*np.maximum(pred,1e-300)
        chi2=float(np.sum(((obs-pred)/sigma)**2))
        dof=int(np.sum(low))
        z=float((chi2-dof)/math.sqrt(2*dof)) if dof>0 else None
        p=None
        if stats is not None:
            try: p=float(stats.chi2.sf(chi2,dof))
            except Exception: p=None
        out.update({'chi2_low_vs_mid_continuation':chi2,'dof':dof,'chi2_zscore_approx':z,'p_value_chi2_approx':p,'fit_log_cl_vs_log_ell_slope':float(beta[0]),'fit_log_cl_intercept':float(beta[1]),'mean_low_obs_over_pred':float(np.mean(obs/pred))})
    except Exception as e:
        out['error']='approx_likelihood_failed'; out['detail']=str(e)
    return out


def conservative_evidence_guard(test_family: str, product_identity: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    fam=str(test_family).lower()
    out={'family':test_family,'guard_active':True}
    if fam in ('dm','direct_detection'):
        out['rule']='Limit curves are readiness/window-coverage only; event-level likelihood or binned events required for peak/drift evidence.'
    elif fam in ('pta','nanograv'):
        out['rule']='Only semantically identified posterior/residual/TOA columns are evidence; unlabelled numeric chain columns are rejected.'
    elif fam in ('kss','qgp'):
        out['rule']='Flow v2/dEta tables are not η/s; explicit η/s-like columns or curated extraction required.'
    else:
        out['rule']='No promotion without verified physical column identity.'
    if product_identity is not None:
        out['product_identity']=product_identity
    return out
