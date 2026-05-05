#!/usr/bin/env python3
"""
Shared utilities for CCDR v7.5 Tier-B public-data tests.

Design rules:
- No manual data files are required or accepted by the test scripts.
- Every external datum must be downloaded by URL/API into a cache directory.
- If a public source exists only as a paper/PDF without named physical columns,
  the test returns status='data_limited' and records the attempted sources.
- Generic term-number extraction from article text is disabled for evidence; citation/page/DOI
  numerals must never become partial support.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, urljoin, urlparse

import numpy as np
import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None

USER_AGENT = "CCDR-TierB-PublicTests/1.0 (+https://github.com/rakovpublic/CCDR)"
DEFAULT_TIMEOUT = 45

# ---------------------------------------------------------------------------
# Basic IO/download/cache
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_name(text: str, max_len: int = 100) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    if len(s) > max_len:
        s = s[:max_len] + "_" + sha1_text(text)[:8]
    return s or "item"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    return s


def cache_path_for_url(cache_dir: Path, url: str, suffix: Optional[str] = None) -> Path:
    parsed = urlparse(url)
    base = safe_name(Path(parsed.path).name or parsed.netloc or "download")
    if suffix is None:
        suffix = Path(parsed.path).suffix
        if not suffix or len(suffix) > 12:
            suffix = ".bin"
    if not base.endswith(suffix):
        base = f"{base}{suffix}"
    return cache_dir / f"{sha1_text(url)[:12]}_{base}"


def download_bytes(url: str, cache_dir: Path, timeout: int = DEFAULT_TIMEOUT, force: bool = False) -> Tuple[Optional[bytes], Dict[str, Any]]:
    ensure_dir(cache_dir)
    path = cache_path_for_url(cache_dir, url)
    meta = {"url": url, "cache_path": str(path), "ok": False, "error": None, "status_code": None, "content_type": None}
    if path.exists() and not force:
        try:
            data = path.read_bytes()
            meta.update({"ok": True, "bytes": len(data), "cached": True})
            return data, meta
        except Exception as e:
            meta["error"] = f"cache_read_failed: {e}"
    try:
        resp = session().get(url, timeout=timeout, allow_redirects=True)
        meta["status_code"] = resp.status_code
        meta["content_type"] = resp.headers.get("content-type")
        resp.raise_for_status()
        data = resp.content
        path.write_bytes(data)
        meta.update({"ok": True, "bytes": len(data), "cached": False, "final_url": resp.url})
        return data, meta
    except Exception as e:
        meta["error"] = f"download_failed: {type(e).__name__}: {e}"
        return None, meta


def download_text(url: str, cache_dir: Path, timeout: int = DEFAULT_TIMEOUT, force: bool = False) -> Tuple[Optional[str], Dict[str, Any]]:
    data, meta = download_bytes(url, cache_dir, timeout=timeout, force=force)
    if data is None:
        return None, meta
    enc = "utf-8"
    ctype = (meta.get("content_type") or "").lower()
    m = re.search(r"charset=([^;]+)", ctype)
    if m:
        enc = m.group(1).strip()
    for candidate in [enc, "utf-8", "latin-1"]:
        try:
            return data.decode(candidate, errors="replace"), meta
        except Exception:
            pass
    return data.decode("utf-8", errors="replace"), meta


def get_json(url: str, cache_dir: Path, timeout: int = DEFAULT_TIMEOUT, force: bool = False, params: Optional[dict] = None) -> Tuple[Optional[Any], Dict[str, Any]]:
    if params:
        # Cache full query by appending a stable synthetic suffix.
        full_url = url + ("&" if "?" in url else "?") + "&".join(f"{quote(str(k))}={quote(str(v))}" for k, v in sorted(params.items()))
    else:
        full_url = url
    text, meta = download_text(full_url, cache_dir, timeout=timeout, force=force)
    if text is None:
        return None, meta
    try:
        return json.loads(text), meta
    except Exception as e:
        meta["error"] = f"json_parse_failed: {e}"
        return None, meta


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(obj), indent=2, sort_keys=True), encoding="utf-8")


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, pd.DataFrame):
        return x.to_dict(orient="records")
    if isinstance(x, Path):
        return str(x)
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def emit_result(result: Dict[str, Any], outdir: Path, test_id: str, print_json: bool = True) -> Dict[str, Any]:
    result.setdefault("generated_utc", utc_now())
    result.setdefault("test_id", test_id)
    result.setdefault("schema", "ccdr-tierb-result-v1")
    ensure_dir(outdir)
    path = outdir / f"{test_id.lower()}_result.json"
    write_json(path, result)
    result["result_path"] = str(path)
    if print_json:
        print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
    return result


def base_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--cache", type=Path, default=Path("tierb_cache"), help="Cache directory for downloaded public data")
    p.add_argument("--outdir", type=Path, default=Path("tierb_out"), help="Output directory for JSON results")
    p.add_argument("--force", action="store_true", help="Re-download cached URLs")
    p.add_argument("--max-papers", type=int, default=30, help="Maximum literature/API records to inspect per query")
    p.add_argument("--max-tables", type=int, default=80, help="Maximum HTML/CSV/XLS tables to parse")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout seconds")
    p.add_argument("--mode", choices=["scientific", "discovery"], default="scientific", help="scientific=manifest-only evidence pipeline; discovery=metadata/source scouting only")
    p.add_argument("--manifest-only", action="store_true", default=True, help="Do not use broad arbitrary discovery links as evidence inputs")
    p.add_argument("--allow-broad-discovery", action="store_true", help="Opt-in to broad source discovery; proposed sources are diagnostic only")
    p.add_argument("--max-bytes", type=int, default=50_000_000, help="Maximum bytes for non-manifest full-file download/parse")
    p.add_argument("--header-rows", type=int, default=50, help="Rows to inspect during header-only preflight")
    p.add_argument("--verbose", action="store_true")
    return p

# ---------------------------------------------------------------------------
# v5 source-quality/cache helpers
# ---------------------------------------------------------------------------

def cache_level(cache_dir: Path, level: str) -> Path:
    return ensure_dir(cache_dir / level)

def head_metadata(url: str, cache_dir: Path, timeout: int = DEFAULT_TIMEOUT, force: bool = False) -> Dict[str, Any]:
    ensure_dir(cache_dir)
    path = cache_path_for_url(cache_dir, "HEAD:" + url, suffix=".json")
    if path.exists() and not force:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    meta = {"url": url, "ok": False, "status_code": None, "content_type": None, "content_length": None, "error": None}
    try:
        resp = session().head(url, timeout=timeout, allow_redirects=True)
        meta.update({
            "ok": 200 <= resp.status_code < 400,
            "status_code": resp.status_code,
            "content_type": resp.headers.get("content-type"),
            "content_length": int(resp.headers.get("content-length")) if resp.headers.get("content-length") else None,
            "final_url": resp.url,
        })
    except Exception as e:
        meta["error"] = f"head_failed: {type(e).__name__}: {e}"
    try:
        path.write_text(json.dumps(to_jsonable(meta), indent=2), encoding="utf-8")
    except Exception:
        pass
    return meta

def is_probably_structured_filename(url_or_name: str) -> bool:
    return bool(re.search(r"\.(csv|tsv|tab|txt|dat|xlsx?|json|jsonl|zip)(\?|$)", url_or_name or "", re.I))

def keyword_score(text: str, positive: Sequence[str] = (), negative: Sequence[str] = ()) -> Dict[str, Any]:
    s = (text or "").lower()
    pos_hits = [p for p in positive or [] if re.search(p, s, re.I)]
    neg_hits = [n for n in negative or [] if re.search(n, s, re.I)]
    return {"positive_hits": pos_hits, "negative_hits": neg_hits, "score": len(pos_hits) - 2 * len(neg_hits), "ok": bool(pos_hits) and not bool(neg_hits)}

def guarded_download_bytes(url: str, cache_dir: Path, *, timeout: int = DEFAULT_TIMEOUT, force: bool = False, max_bytes: int = 50_000_000, manifest_approved: bool = False) -> Tuple[Optional[bytes], Dict[str, Any]]:
    meta0 = head_metadata(url, cache_level(cache_dir, "metadata"), timeout=timeout, force=force)
    clen = meta0.get("content_length")
    if not manifest_approved and clen is not None and clen > max_bytes:
        meta0.update({"ok": False, "skipped": True, "skip_reason": f"content_length>{max_bytes}"})
        return None, meta0
    data, meta = download_bytes(url, cache_level(cache_dir, "files"), timeout=timeout, force=force)
    meta["head"] = meta0
    if data is not None and not manifest_approved and len(data) > max_bytes:
        meta.update({"ok": False, "skipped": True, "skip_reason": f"downloaded_bytes>{max_bytes}"})
        return None, meta
    return data, meta

def read_tabular_header_bytes(data: bytes, url_hint: str = "", nrows: int = 50) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    lower = (url_hint or "").lower()
    sample = data[:1_000_000]
    try:
        if lower.endswith('.zip') or data[:4] == b'PK\x03\x04':
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist()[:100]:
                    if info.is_dir() or info.file_size > 100_000_000:
                        continue
                    name = info.filename
                    if not re.search(r'\.(csv|tsv|txt|dat|xlsx?|json|jsonl)$', name, re.I):
                        continue
                    with zf.open(info) as fh:
                        chunk = fh.read(min(info.file_size, 1_000_000))
                    frames.extend(read_tabular_header_bytes(chunk, name, nrows=nrows))
                    if len(frames) > 20:
                        break
            return frames
    except Exception:
        pass
    try:
        if lower.endswith(('.xlsx', '.xls')):
            xls = pd.ExcelFile(io.BytesIO(data))
            for sheet in xls.sheet_names[:20]:
                # Public workbooks often have title/preamble rows before the
                # real header. Scan candidate header rows and let the physical
                # column gate choose the right frame.
                for header in list(range(0, 30)) + [None]:
                    try:
                        df = pd.read_excel(xls, sheet_name=sheet, header=header, nrows=nrows)
                        if df.shape[1] >= 2 and df.shape[0] >= 1:
                            df.attrs['source_sheet'] = sheet
                            df.attrs['header_row'] = header
                            frames.append(df)
                    except Exception:
                        pass
                    if len(frames) >= 80:
                        return frames
            return frames
    except Exception:
        pass
    try:
        stripped = sample[:200].lstrip()
        if lower.endswith('.jsonl'):
            rows = []
            for line in sample.decode('utf-8', errors='replace').splitlines()[:nrows]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
            if rows:
                return [pd.json_normalize(rows)]
        if lower.endswith('.json') or stripped.startswith((b'{', b'[')):
            obj = json.loads(sample.decode('utf-8', errors='replace'))
            if isinstance(obj, list):
                return [pd.json_normalize(obj[:nrows])] if obj and isinstance(obj[0], dict) else [pd.DataFrame({'value': obj[:nrows]})]
            if isinstance(obj, dict):
                for key in ['data', 'results', 'hits', 'records', 'tables']:
                    val = obj.get(key)
                    if isinstance(val, list) and val:
                        return [pd.json_normalize(val[:nrows])]
                return [pd.json_normalize(obj)]
    except Exception:
        pass
    try:
        if b"<table" in sample.lower() or lower.endswith((".html", ".htm")):
            for df in pd.read_html(io.BytesIO(sample))[:10]:
                frames.append(df.head(nrows))
            if frames:
                return frames
    except Exception:
        pass
    for skip in [0, 1, 2, 3, 4, 5, 10, 20]:
        for sep in [None, ',', '\t', ';', r'\s+']:
            try:
                df = pd.read_csv(io.BytesIO(sample), sep=sep, engine='python', comment='#', skiprows=skip, nrows=nrows)
                if df.shape[0] >= 1 and df.shape[1] >= 2:
                    frames.append(df)
                    if len(frames) >= 6:
                        return frames
            except Exception:
                pass
    return frames

def parse_after_header_gate(data: bytes, url: str, required_groups: Sequence[Sequence[str]], *, nrows: int = 50, max_full_bytes: int = 50_000_000, manifest_approved: bool = False) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    header_frames = read_tabular_header_bytes(data, url, nrows=nrows)
    header_reports = []
    gate_ok = False
    for df in header_frames:
        report = column_match_report(df, required_groups)
        header_reports.append({"shape": list(df.shape), "columns": [str(c) for c in df.columns[:60]], "physical_column_match": report})
        if report.get('ok'):
            gate_ok = True
    if not gate_ok:
        return [], {"header_frames": header_reports, "gate_ok": False, "full_parse": False}
    if not manifest_approved and len(data) > max_full_bytes:
        return header_frames, {"header_frames": header_reports, "gate_ok": True, "full_parse": False, "reason": "gate_ok_but_file_too_large"}
    try:
        frames = read_tabular_bytes(data, url)
    except MemoryError:
        frames = header_frames
        return frames, {"header_frames": header_reports, "gate_ok": True, "full_parse": False, "reason": "MemoryError_full_parse_fallback_to_header"}
    return frames, {"header_frames": header_reports, "gate_ok": True, "full_parse": True}

# ---------------------------------------------------------------------------
# Table parsing and generic numeric helpers
# ---------------------------------------------------------------------------

def read_tabular_bytes(data: bytes, url_hint: str = "") -> List[pd.DataFrame]:
    """Parse CSV/TSV/XLSX/HTML/JSON/ZIP bytes into candidate dataframes."""
    frames: List[pd.DataFrame] = []
    lower = url_hint.lower()
    # ZIP archives often host the real CSV/XLSX supplements. Recurse into
    # small tabular members, but ignore PDFs/images so discovery archives can
    # become evidence only through actual structured files.
    try:
        if lower.endswith('.zip') or data[:4] == b'PK\x03\x04':
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist()[:80]:
                    name = info.filename
                    if info.is_dir() or info.file_size > 80_000_000:
                        continue
                    if not re.search(r'\.(csv|tsv|txt|dat|xlsx?|json)$', name, re.I):
                        continue
                    try:
                        frames.extend(read_tabular_bytes(zf.read(info), name))
                    except Exception:
                        pass
                    if len(frames) > 25:
                        break
            if frames:
                return frames
    except Exception:
        pass
    # JSON APIs such as INSPIRE and HEPData records.
    try:
        stripped = data[:200].lstrip()
        if lower.endswith(".json") or stripped.startswith((b"{", b"[")):
            obj = json.loads(data.decode("utf-8", errors="replace"))
            if isinstance(obj, list):
                if obj and isinstance(obj[0], dict):
                    frames.append(pd.json_normalize(obj))
                else:
                    frames.append(pd.DataFrame({"value": obj}))
            elif isinstance(obj, dict):
                # Try common list-bearing keys first, then flatten the object.
                for key in ["data", "results", "hits", "records", "tables"]:
                    val = obj.get(key)
                    if isinstance(val, list) and val:
                        frames.append(pd.json_normalize(val))
                if not frames:
                    frames.append(pd.json_normalize(obj))
            if frames:
                return frames
    except Exception:
        pass
    try:
        if lower.endswith(('.xlsx', '.xls')):
            xls = pd.ExcelFile(io.BytesIO(data))
            for sheet in xls.sheet_names[:30]:
                for header in list(range(0, 40)) + [None]:
                    try:
                        df = pd.read_excel(xls, sheet_name=sheet, header=header)
                        if df.shape[1] >= 2 and df.shape[0] >= 2:
                            df.attrs['source_sheet'] = sheet
                            df.attrs['header_row'] = header
                            frames.append(df)
                    except Exception:
                        pass
                    if len(frames) >= 120:
                        return frames
            return frames
    except Exception:
        pass
    # HTML tables
    try:
        if b"<table" in data[:200000].lower() or lower.endswith((".html", ".htm")):
            for df in pd.read_html(io.BytesIO(data)):
                frames.append(df)
            if frames:
                return frames
    except Exception:
        pass
    # CSV/TSV with flexible delimiter and preamble skip.
    sample = data[:4096].decode("utf-8", errors="replace")
    for skip in [0, 1, 2, 3, 4, 5, 10, 20]:
        for sep in [None, ",", "\t", ";", r"\s+"]:
            try:
                df = pd.read_csv(io.BytesIO(data), sep=sep, engine="python", comment="#", skiprows=skip)
                if df.shape[0] >= 2 and df.shape[1] >= 2:
                    frames.append(df)
                    if len(frames) > 5:
                        return frames
            except Exception:
                pass
    return frames


def numeric_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= max(3, int(0.2 * len(s))):
            cols.append(col)
    return cols


def clean_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        # Extract the first number from strings like "1.23 ± 0.04".
        s = s.astype(str).str.replace("−", "-", regex=False).str.extract(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(s, errors="coerce")


def find_col(df: pd.DataFrame, name_patterns: Sequence[str], exclude_patterns: Sequence[str] = ()) -> Optional[str]:
    lower_cols = [(str(c).lower(), c) for c in df.columns]
    for pat in name_patterns:
        rx = re.compile(pat, re.I)
        for low, orig in lower_cols:
            if rx.search(low) and not any(re.search(e, low, re.I) for e in exclude_patterns):
                return orig
    return None


def select_xy_by_patterns(df: pd.DataFrame, x_patterns: Sequence[str], y_patterns: Sequence[str], *, allow_numeric_fallback: bool = False) -> Optional[pd.DataFrame]:
    """Select x/y only when named physical columns are identifiable.

    v1 allowed falling back to the first two numeric columns. That caused false
    partial results from DOI fragments, page numbers and HTML metadata. v2 keeps
    fallback opt-in only for explicitly structured sources; all public evidence
    paths should use named columns.
    """
    xcol = find_col(df, x_patterns)
    ycol = find_col(df, y_patterns)
    if xcol is None or ycol is None or xcol == ycol:
        if not allow_numeric_fallback:
            return None
        nums = numeric_columns(df)
        if len(nums) >= 2:
            xcol, ycol = nums[0], nums[1]
        else:
            return None
    x = clean_numeric_series(df[xcol])
    y = clean_numeric_series(df[ycol])
    out = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out["x"] > 0) & (out["y"] > 0)]
    if len(out) < 4:
        return None
    return out


def column_match_report(df: pd.DataFrame, required_column_groups: Sequence[Sequence[str]]) -> Dict[str, Any]:
    """Return whether dataframe columns satisfy all required physical groups.

    required_column_groups is an AND of groups; each group is an OR of regexes.
    Example: [[r'E[_ -]?ELM'], [r'pedestal.*pressure|P[_ -]?ped']]
    requires both an ELM-energy-like column and a pedestal-pressure-like column.
    """
    cols = [str(c) for c in df.columns]
    low = [c.lower() for c in cols]
    matched = []
    missing = []
    for group in required_column_groups or []:
        group_hits = []
        for pat in group:
            rx = re.compile(pat, re.I)
            for orig, c in zip(cols, low):
                if rx.search(c):
                    group_hits.append(orig)
        if group_hits:
            matched.append(sorted(set(group_hits))[:8])
        else:
            missing.append(list(group))
    return {"ok": not missing, "matched_groups": matched, "missing_groups": missing, "columns": cols[:40]}


def is_direct_structured_url(url: str, content_type: str = "") -> bool:
    low = url.lower().split("?")[0]
    ctype = (content_type or "").lower()
    if low.endswith((".csv", ".tsv", ".dat", ".txt", ".xlsx", ".xls", ".json", ".zip")):
        return True
    if any(x in ctype for x in ["text/csv", "tab-separated", "excel", "spreadsheet", "json"]):
        return True
    return False


def table_row_count(df: pd.DataFrame) -> int:
    return int(len(df.index))


def pearson(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    arr = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(arr) < 3:
        return {"n": int(len(arr)), "r": None, "pvalue": None}
    if stats:
        r, p = stats.pearsonr(arr["x"], arr["y"])
        return {"n": int(len(arr)), "r": float(r), "pvalue": float(p)}
    r = float(np.corrcoef(arr["x"], arr["y"])[0, 1])
    return {"n": int(len(arr)), "r": r, "pvalue": None}


def spearman(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    arr = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(arr) < 3:
        return {"n": int(len(arr)), "rho": None, "pvalue": None}
    if stats:
        rho, p = stats.spearmanr(arr["x"], arr["y"])
        return {"n": int(len(arr)), "rho": float(rho), "pvalue": float(p)}
    rho = float(np.corrcoef(arr["x"].rank(), arr["y"].rank())[0, 1])
    return {"n": int(len(arr)), "rho": rho, "pvalue": None}


def linfit(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    arr = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(arr) < 3:
        return {"n": int(len(arr)), "slope": None, "intercept": None, "r2": None}
    X = np.vstack([np.ones(len(arr)), np.asarray(arr["x"], float)]).T
    beta, *_ = np.linalg.lstsq(X, np.asarray(arr["y"], float), rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((arr["y"] - pred) ** 2))
    ss_tot = float(np.sum((arr["y"] - np.mean(arr["y"])) ** 2))
    r2 = None if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"n": int(len(arr)), "intercept": float(beta[0]), "slope": float(beta[1]), "r2": r2, "rss": ss_res}


def loglog_fit(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    arr = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    arr = arr[(arr.x > 0) & (arr.y > 0)]
    if len(arr) < 3:
        return {"n": int(len(arr)), "exponent": None, "prefactor": None, "r2": None}
    lf = linfit(np.log(arr.x), np.log(arr.y))
    if lf.get("slope") is None:
        return {"n": int(len(arr)), "exponent": None, "prefactor": None, "r2": None}
    return {"n": int(len(arr)), "exponent": lf["slope"], "prefactor": float(math.exp(lf["intercept"])), "r2": lf["r2"], "rss_log": lf.get("rss")}


def aic_from_rss(rss: float, n: int, k: int) -> Optional[float]:
    if n <= 0 or rss <= 0:
        return None
    return float(n * math.log(rss / n) + 2 * k)


def status_from_counts(n_records: int, min_ok: int = 10, min_partial: int = 3) -> str:
    if n_records >= min_ok:
        return "ok"
    if n_records >= min_partial:
        return "partial"
    return "data_limited"

# ---------------------------------------------------------------------------
# Public metadata APIs and link discovery
# ---------------------------------------------------------------------------

def openalex_search(query: str, cache_dir: Path, per_page: int = 25, timeout: int = DEFAULT_TIMEOUT, force: bool = False) -> Dict[str, Any]:
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": min(max(per_page, 1), 200),
        "mailto": os.environ.get("OPENALEX_MAILTO", "public-tests@example.invalid"),
    }
    data, meta = get_json(url, cache_dir / "openalex", timeout=timeout, force=force, params=params)
    works = []
    if isinstance(data, dict):
        for w in data.get("results", []) or []:
            works.append({
                "id": w.get("id"),
                "doi": w.get("doi"),
                "title": w.get("display_name"),
                "year": w.get("publication_year"),
                "type": w.get("type"),
                "cited_by_count": w.get("cited_by_count"),
                "open_access": w.get("open_access"),
                "primary_location": w.get("primary_location"),
                "landing_page_url": ((w.get("primary_location") or {}).get("landing_page_url")),
                "pdf_url": ((w.get("primary_location") or {}).get("pdf_url")),
            })
    return {"query": query, "meta": meta, "works": works}


def crossref_search(query: str, cache_dir: Path, rows: int = 20, timeout: int = DEFAULT_TIMEOUT, force: bool = False) -> Dict[str, Any]:
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": min(max(rows, 1), 100)}
    data, meta = get_json(url, cache_dir / "crossref", timeout=timeout, force=force, params=params)
    items = []
    if isinstance(data, dict):
        for it in (((data.get("message") or {}).get("items")) or []):
            links = it.get("link") or []
            items.append({
                "doi": it.get("DOI"),
                "title": (it.get("title") or [None])[0],
                "year": (((it.get("published-print") or it.get("published-online") or {}).get("date-parts") or [[None]])[0][0]),
                "URL": it.get("URL"),
                "links": links,
            })
    return {"query": query, "meta": meta, "items": items}


def europepmc_search(query: str, cache_dir: Path, page_size: int = 25, timeout: int = DEFAULT_TIMEOUT, force: bool = False) -> Dict[str, Any]:
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": query, "format": "json", "pageSize": min(max(page_size, 1), 100)}
    data, meta = get_json(url, cache_dir / "europepmc", timeout=timeout, force=force, params=params)
    items = []
    if isinstance(data, dict):
        for it in (((data.get("resultList") or {}).get("result")) or []):
            items.append({"title": it.get("title"), "doi": it.get("doi"), "pmcid": it.get("pmcid"), "year": it.get("pubYear"), "journal": it.get("journalTitle")})
    return {"query": query, "meta": meta, "items": items}


def collect_candidate_urls_from_work(work: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    for key in ["pdf_url", "landing_page_url"]:
        u = work.get(key)
        if u and isinstance(u, str):
            urls.append(u)
    oa = work.get("open_access") or {}
    if isinstance(oa, dict) and oa.get("oa_url"):
        urls.append(oa["oa_url"])
    pl = work.get("primary_location") or {}
    if isinstance(pl, dict):
        for key in ["pdf_url", "landing_page_url"]:
            if pl.get(key):
                urls.append(pl[key])
    # Preserve order unique.
    out = []
    seen = set()
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out


def discover_data_links(html: str, base_url: str) -> List[str]:
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(html, "html.parser")
    out: List[str] = []
    exts = (".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xls", ".json", ".zip")
    keywords = re.compile(r"supplement|data|table|dataset|figshare|zenodo|dryad|osf|github|csv|xlsx|source", re.I)
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        text = (a.get_text(" ") or "")
        full = urljoin(base_url, href)
        low = full.lower()
        if low.endswith(exts) or keywords.search(full) or keywords.search(text):
            out.append(full)
    # unique
    seen, uniq = set(), []
    for u in out:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq[:80]


def extract_numeric_values_from_text(text: str, value_terms: Sequence[str]) -> List[Dict[str, Any]]:
    """Very conservative term-window numeric extraction from paper abstracts/pages."""
    records: List[Dict[str, Any]] = []
    clean = re.sub(r"\s+", " ", text.replace("−", "-"))
    for term in value_terms:
        rx_term = re.compile(term, re.I)
        for m in rx_term.finditer(clean):
            start, end = max(0, m.start() - 180), min(len(clean), m.end() + 180)
            window = clean[start:end]
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:\s*[×x]\s*10\s*[-+]?\d+|(?:e|E)[-+]?\d+)?", window)
            for n in nums[:8]:
                n2 = re.sub(r"\s*[×x]\s*10\s*", "e", n)
                try:
                    val = float(n2)
                    records.append({"term": term, "value": val, "context": window[:350]})
                except Exception:
                    pass
    return records


def literature_probe(
    test_id: str,
    queries: Sequence[str],
    cache_dir: Path,
    max_papers: int = 25,
    max_tables: int = 60,
    timeout: int = DEFAULT_TIMEOUT,
    force: bool = False,
    value_terms: Sequence[str] = (),
    required_column_groups: Sequence[Sequence[str]] = (),
    structured_only: bool = True,
) -> Dict[str, Any]:
    """Search public metadata, discover supplements, and parse only named physical tables.

    v2 evidence rule: a table contributes to status only if it contains named
    physical columns requested by `required_column_groups`. Generic term-window
    numeric extraction from article/PDF/HTML text is disabled as evidence because
    it produced citation years, DOI parts, page numbers and CSS ids in v1.
    """
    searches = []
    candidate_works: List[Dict[str, Any]] = []
    for q in queries:
        oa = openalex_search(q, cache_dir, per_page=max_papers, timeout=timeout, force=force)
        cr = crossref_search(q, cache_dir, rows=min(20, max_papers), timeout=timeout, force=force)
        searches.append({"engine": "OpenAlex", **oa})
        searches.append({"engine": "Crossref", **cr})
        for w in oa.get("works", []):
            candidate_works.append(w)
        for it in cr.get("items", []):
            candidate_works.append({"title": it.get("title"), "doi": it.get("doi"), "landing_page_url": it.get("URL"), "year": it.get("year"), "crossref_links": it.get("links")})

    seen = set(); works = []
    for w in candidate_works:
        key = (w.get("doi") or w.get("title") or w.get("landing_page_url") or "")[:300]
        if key and key not in seen:
            seen.add(key); works.append(w)
    works = works[:max_papers * max(1, len(queries))]

    downloaded = []
    data_links: List[str] = []
    table_summaries: List[Dict[str, Any]] = []
    qualifying_tables: List[Dict[str, Any]] = []
    rejected_tables: List[Dict[str, Any]] = []
    parsed_tables = 0

    def inspect_table(df: pd.DataFrame, url: str, title: Optional[str] = None, source_kind: str = "candidate") -> None:
        nonlocal parsed_tables
        if parsed_tables >= max_tables:
            return
        nums = numeric_columns(df)
        report = column_match_report(df, required_column_groups)
        summary = {
            "source_url": url,
            "source_kind": source_kind,
            "title": title,
            "shape": list(df.shape),
            "numeric_columns": [str(c) for c in nums[:12]],
            "columns": [str(c) for c in list(df.columns)[:24]],
            "physical_column_match": report,
        }
        table_summaries.append(summary)
        parsed_tables += 1
        # Count only named physical tables with at least two numeric columns and enough rows.
        if report["ok"] and len(nums) >= 2 and table_row_count(df) >= 3:
            qualifying_tables.append(summary)
        else:
            rejected_tables.append(summary)

    for w in works[:max_papers]:
        urls = collect_candidate_urls_from_work(w)
        for link in w.get("crossref_links") or []:
            u = link.get("URL")
            if u:
                urls.append(u)
        for url in list(dict.fromkeys(urls))[:5]:
            data, meta = download_bytes(url, cache_dir / test_id / "papers", timeout=timeout, force=force)
            downloaded.append({"title": w.get("title"), "url": url, "meta": meta})
            if data is None:
                continue
            ctype = (meta.get("content_type") or "").lower()
            # Discover supplements from HTML, but do not parse article HTML itself as evidence.
            if b"<html" in data[:1000].lower() or "text/html" in ctype:
                html = data.decode("utf-8", errors="replace")
                data_links.extend(discover_data_links(html, url))
            if not structured_only or is_direct_structured_url(url, ctype):
                for df in read_tabular_bytes(data, url):
                    inspect_table(df, url, title=w.get("title"), source_kind="direct_structured")
                    if parsed_tables >= max_tables:
                        break
            if parsed_tables >= max_tables:
                break
        if parsed_tables >= max_tables:
            break

    # Download discovered data/supplement links and parse only direct structured resources.
    for url in list(dict.fromkeys(data_links))[:max_tables]:
        data, meta = download_bytes(url, cache_dir / test_id / "data_links", timeout=timeout, force=force)
        downloaded.append({"url": url, "meta": meta, "kind": "discovered_data_link"})
        if data is None:
            continue
        ctype = (meta.get("content_type") or "").lower()
        if structured_only and not is_direct_structured_url(url, ctype):
            continue
        for df in read_tabular_bytes(data, url):
            inspect_table(df, url, title=None, source_kind="discovered_structured")
            if parsed_tables >= max_tables:
                break
        if parsed_tables >= max_tables:
            break

    n_qual = len(qualifying_tables)
    # v2 status rule: no named physical columns => data_limited, never partial.
    if n_qual >= 5:
        status = "ok"
    elif n_qual >= 1:
        status = "partial"
    else:
        status = "data_limited"
    return {
        "status": status,
        "queries": list(queries),
        "candidate_works_count": len(works),
        "candidate_works_sample": works[:12],
        "downloaded_sources": downloaded[:80],
        "discovered_data_links_sample": list(dict.fromkeys(data_links))[:50],
        "parsed_tables_count": parsed_tables,
        "qualifying_physical_tables_count": n_qual,
        "required_column_groups": [[str(x) for x in g] for g in (required_column_groups or [])],
        "table_summaries": table_summaries[:60],
        "qualifying_tables_sample": qualifying_tables[:20],
        "rejected_tables_sample": rejected_tables[:20],
        "generic_text_value_extraction": "disabled_in_v2",
        "analysis_note": "Only direct structured data files or discovered supplements are parsed. Tables count as evidence only when named physical columns match the test-specific requirements. Generic term-number text extraction is disabled.",
    }

# ---------------------------------------------------------------------------
# GitHub helpers / domain-specific loaders
# ---------------------------------------------------------------------------

def github_tree(owner: str, repo: str, branch: str, cache_dir: Path, timeout: int = DEFAULT_TIMEOUT, force: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    data, meta = get_json(url, cache_dir / "github", timeout=timeout, force=force)
    if not isinstance(data, dict) or "tree" not in data:
        return [], meta
    return data.get("tree") or [], meta


def raw_github_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"



def classify_cmbs4_material(path: str, columns: Sequence[str] = ()) -> Dict[str, Any]:
    """Heuristic material classification from public repository paths/columns.

    The CMB-S4 repository usually lacks explicit grain-size metadata. This
    classifier separates plausible boundary-dominated subsets from high-purity
    bulk crystals so T31/T32 are serious negative tests rather than broad audits.
    """
    s = (path + " " + " ".join(map(str, columns))).lower()
    composite_terms = ["cfrp", "g10", "garolite", "fiberglass", "carbon_fiber", "carbon fiber", "composite", "epoxy", "kapton", "mylar", "vkl", "dpp", "peek", "vespel", "teflon", "ptfe", "poly", "nylon", "kevlar"]
    crystalline_terms = ["silicon", "sapphire", "diamond", "copper", "aluminum", "aluminium", "niobium", "titanium", "gold", "silver", "brass", "steel", "quartz", "germanium", "beryllium", "molybdenum", "tungsten"]
    amorphous_terms = ["glass", "amorph", "vitreous", "silica"]
    nano_terms = ["nano", "nanocrystal", "nanocrystalline", "nanowire", "thin_film", "thin film"]
    grain_terms = ["grain", "particle", "powder", "sinter", "porous", "porosity", "micron", "micrometer", "um", "µm", "nm"]
    if any(t in s for t in composite_terms):
        mclass = "composite_or_polymer"
    elif any(t in s for t in amorphous_terms):
        mclass = "amorphous"
    elif any(t in s for t in crystalline_terms):
        mclass = "crystalline_or_metal"
    else:
        mclass = "unknown"
    grain_known = any(t in s for t in grain_terms)
    nano = any(t in s for t in nano_terms)
    boundary_dominated = mclass in {"composite_or_polymer", "amorphous"} or grain_known or nano
    return {
        "material_class": mclass,
        "grain_size_known": bool(grain_known),
        "nanocrystalline_yes_no": bool(nano),
        "boundary_dominated_candidate": bool(boundary_dominated),
        "classification_basis": "path_and_column_keyword_heuristic",
    }

def load_cmbs4_thermal_tables(cache_dir: Path, timeout: int = DEFAULT_TIMEOUT, force: bool = False, max_files: int = 250) -> Dict[str, Any]:
    owner, repo = "CMB-S4", "Cryogenic_Material_Properties"
    branch_used = None
    tree = []
    meta = {}
    for br in ["main", "master"]:
        tree, meta = github_tree(owner, repo, br, cache_dir, timeout=timeout, force=force)
        if tree:
            branch_used = br
            break
    files = [x for x in tree if x.get("type") == "blob" and str(x.get("path", "")).lower().endswith(".csv") and "thermal_conductivity" in str(x.get("path", "")).lower()]
    files = files[:max_files]
    tables = []
    downloads = []
    for item in files:
        path = item["path"]
        url = raw_github_url(owner, repo, branch_used or "main", path)
        data, dmeta = download_bytes(url, cache_dir / "cmbs4_thermal", timeout=timeout, force=force)
        downloads.append({"path": path, "url": url, "meta": dmeta})
        if data is None:
            continue
        for df in read_tabular_bytes(data, url):
            xy = select_xy_by_patterns(df, [r"temp", r"temperature", r"^t$", r"\(k\)"], [r"conduct", r"kappa", r"thermal", r"^k$", r"w/m"], allow_numeric_fallback=False)
            if xy is not None:
                tables.append({"path": path, "url": url, "n": len(xy), "xy": xy, "columns": [str(c) for c in df.columns], "classification": classify_cmbs4_material(path, [str(c) for c in df.columns])})
                break
    return {"repo": f"{owner}/{repo}", "branch": branch_used, "tree_meta": meta, "files_seen": len(files), "downloads": downloads, "tables": tables}


def thermal_model_fits(xy: pd.DataFrame) -> Dict[str, Any]:
    """Compare a simple power-law with a CCDR-like boundary-modified power-law.

    The CMB-S4 repository usually lacks grain-size metadata, so B=lambda0/L is fitted
    as a nuisance scale. This is a screening test, not a final material-specific model.
    """
    d = xy.copy().replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d.x > 0) & (d.y > 0)]
    if len(d) < 6:
        return {"n": int(len(d)), "usable": False}
    T = np.asarray(d.x, float)
    kappa = np.asarray(d.y, float)
    logT = np.log(T); logk = np.log(kappa)
    # Baseline log(k)=a+b log(T)
    X = np.vstack([np.ones_like(logT), logT]).T
    beta, *_ = np.linalg.lstsq(X, logk, rcond=None)
    pred = X @ beta
    rss_power = float(np.sum((logk - pred) ** 2))
    aic_power = aic_from_rss(rss_power, len(T), 2)
    # CCDR-like: k=A*T^b*mu(B/T), mu=x/sqrt(1+x), x=B/T.
    # Grid over B because it is poorly constrained in many tables.
    best = None
    for B in np.logspace(-3, 3, 121):
        x = B / T
        mu = x / np.sqrt(1.0 + x)
        if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
            continue
        y = logk - np.log(mu)
        beta2, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred2 = X @ beta2 + np.log(mu)
        rss = float(np.sum((logk - pred2) ** 2))
        if best is None or rss < best["rss_log"]:
            best = {"B_lambda_over_L": float(B), "logA": float(beta2[0]), "exponent": float(beta2[1]), "rss_log": rss}
    if best:
        best["aic"] = aic_from_rss(best["rss_log"], len(T), 3)
    return {
        "n": int(len(T)),
        "usable": True,
        "power_law": {"logA": float(beta[0]), "exponent": float(beta[1]), "rss_log": rss_power, "aic": aic_power},
        "ccdr_mu_model": best,
        "delta_aic_ccdr_minus_power": None if (not best or aic_power is None or best.get("aic") is None) else float(best["aic"] - aic_power),
    }

# ---------------------------------------------------------------------------
# Public API loaders for several Tier-B tests
# ---------------------------------------------------------------------------

def nrel_pv_efficiency_candidates(cache_dir: Path, timeout: int = DEFAULT_TIMEOUT, force: bool = False) -> Dict[str, Any]:
    """Fetch public NREL PV efficiency products and derive a cautious proxy test.

    Uses NREL rows only; no Materials Project key is required. The acoustic-optical
    proxy is a material-class heuristic based on the public NREL material/cell-type
    fields. This is weaker than a first-principles Materials Project join, but it is
    deterministic, public, and avoids manual files.
    """
    roots = [
        "https://www.nrel.gov/pv/cell-efficiency.html",
        "https://www.nrel.gov/pv/interactive-cell-efficiency.html",
        "https://www.nrel.gov/pv/assets/pdfs/best-research-cell-efficiencies.xlsx",
        "https://www.nrel.gov/pv/assets/pdfs/best-research-cell-efficiencies.csv",
    ]
    downloads, frames, links = [], [], []
    for url in roots:
        data, meta = download_bytes(url, cache_dir / "nrel_pv", timeout=timeout, force=force)
        downloads.append({"url": url, "meta": meta})
        if data is None:
            continue
        frames.extend(read_tabular_bytes(data, url))
        if b"<html" in data[:1000].lower():
            links.extend(discover_data_links(data.decode("utf-8", errors="replace"), url))
    for url in list(dict.fromkeys(links))[:30]:
        if not re.search(r"efficien|pv|cell|csv|xls|xlsx|data", url, re.I):
            continue
        data, meta = download_bytes(url, cache_dir / "nrel_pv", timeout=timeout, force=force)
        downloads.append({"url": url, "meta": meta, "kind": "discovered"})
        if data is not None:
            frames.extend(read_tabular_bytes(data, url))

    summaries = []
    candidate_rows = []
    for df in frames:
        nums = numeric_columns(df)
        summaries.append({"shape": list(df.shape), "columns": [str(c) for c in df.columns[:24]], "numeric_columns": [str(c) for c in nums[:12]]})
        cols = {str(c).lower(): c for c in df.columns}
        eff_col = None
        for key, c in cols.items():
            if re.search(r"(^|[^a-z])eff(iciency)?|efficiency", key) and not re.search(r"uncert", key):
                eff_col = c; break
        year_col = None
        for key, c in cols.items():
            if key.strip() == "year" or "measurement date" in key or "month" == key:
                year_col = c; break
        mat_cols = [c for key, c in cols.items() if any(k in key for k in ["material", "cell type", "description", "detailed", "group"])]
        if eff_col is None or not mat_cols:
            continue
        tmp = df.copy()
        tmp["_eff"] = clean_numeric_series(tmp[eff_col])
        if "year" in cols:
            tmp["_year"] = clean_numeric_series(tmp[cols["year"]])
        else:
            tmp["_year"] = clean_numeric_series(tmp[year_col]) if year_col is not None else np.nan
        def row_text(row):
            return " ".join(str(row.get(c, "")) for c in mat_cols)
        for _, row in tmp.dropna(subset=["_eff"]).iterrows():
            txt = row_text(row)
            proxy = pv_material_proxy(txt)
            if proxy is None:
                continue
            candidate_rows.append({"efficiency_pct": float(row["_eff"]), "year": None if pd.isna(row.get("_year")) else float(row.get("_year")), "material_text": txt[:300], **proxy})

    pv_df = pd.DataFrame(candidate_rows)
    metrics = {}
    support_like = None
    status = "data_limited"
    if len(pv_df) >= 20 and pv_df["year"].notna().sum() >= 10:
        # Baseline: efficiency ~ material_class + year. Test residual correlation with proxy.
        d = pv_df.dropna(subset=["efficiency_pct", "year", "ao_proxy"]).copy()
        d = d[(d["year"] >= 1970) & (d["year"] <= 2100) & (d["efficiency_pct"] > 0) & (d["efficiency_pct"] < 80)]
        if len(d) >= 20:
            X_parts = [np.ones(len(d)), d["year"].to_numpy(float)]
            for cls in sorted(d["material_class"].dropna().unique())[1:]:
                X_parts.append((d["material_class"].to_numpy() == cls).astype(float))
            X = np.vstack(X_parts).T
            y = d["efficiency_pct"].to_numpy(float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            d["baseline_residual_eff_pct"] = resid
            metrics = {
                "n_rows_used": int(len(d)),
                "material_classes": sorted(map(str, d["material_class"].dropna().unique())),
                "baseline_model": "efficiency_pct ~ year + material_class_fixed_effects",
                "residual_vs_acoustic_optical_proxy_spearman": spearman(d["ao_proxy"], d["baseline_residual_eff_pct"]),
                "residual_vs_mass_contrast_proxy_spearman": spearman(d["mass_contrast_proxy"], d["baseline_residual_eff_pct"]),
                "sample_rows": d.head(40).to_dict(orient="records"),
            }
            rho = metrics["residual_vs_acoustic_optical_proxy_spearman"].get("rho")
            pval = metrics["residual_vs_acoustic_optical_proxy_spearman"].get("pvalue")
            support_like = (rho is not None and rho > 0 and (pval is None or pval < 0.05))
            status = "ok"
    elif len(pv_df) >= 5:
        status = "partial"

    return {
        "downloads": downloads,
        "tables_count": len(frames),
        "table_summaries": summaries[:20],
        "candidate_rows_count": len(candidate_rows),
        "metrics": metrics,
        "support_like": support_like,
        "status": status,
    }


def pv_material_proxy(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").lower()
    classes = [
        ("perovskite", ["perovskite"], 0.95, 0.75),
        ("iii_v_multijunction", ["iii-v", "gaas", "inp", "gainp", "multijunction", "multi-junction"], 0.90, 0.85),
        ("cigs_cdte", ["cigs", "cigse", "cdte", "cu(in", "cuinse", "thin-film"], 0.75, 0.70),
        ("silicon", ["silicon", "si", "heterojunction", "topcon", "perc"], 0.55, 0.55),
        ("organic_dye_quantum", ["organic", "dye", "quantum dot", "dssc", "polymer"], 0.35, 0.35),
    ]
    for name, keys, ao, mass in classes:
        if any(k in s for k in keys):
            # Crude public-data proxy: higher for strong optical/acoustic separation and symmetry/mass contrast.
            symmetry_bonus = 0.05 if any(k in s for k in ["single crystal", "crystalline", "epitaxial", "monocrystalline"]) else 0.0
            return {"material_class": name, "ao_proxy": float(min(1.0, ao + symmetry_bonus)), "mass_contrast_proxy": float(mass), "proxy_basis": "nrel_material_text_heuristic"}
    return None

def nasa_exoplanet_table(cache_dir: Path, timeout: int = DEFAULT_TIMEOUT, force: bool = False, limit: int = 5000) -> Dict[str, Any]:
    query = (
        "select+top+{limit}+pl_name,hostname,disc_year,pl_orbper,pl_orbpererr1,pl_orbpererr2,"
        "pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,sy_snum,sy_pnum+from+pscomppars+"
        "where+pl_orbper+is+not+null+and+disc_year+is+not+null"
    ).format(limit=limit)
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}&format=csv"
    data, meta = download_bytes(url, cache_dir / "nasa_exoplanet", timeout=timeout, force=force)
    frames = read_tabular_bytes(data or b"", url) if data else []
    return {"url": url, "meta": meta, "tables": frames}


def rcsb_current_entry_ids(cache_dir: Path, timeout: int = DEFAULT_TIMEOUT, force: bool = False, max_ids: int = 1000) -> Dict[str, Any]:
    url = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"
    data, meta = get_json(url, cache_dir / "rcsb", timeout=timeout, force=force)
    ids = data if isinstance(data, list) else []
    return {"url": url, "meta": meta, "ids": ids[:max_ids], "total_ids": len(ids)}


def rcsb_entry(entry_id: str, cache_dir: Path, timeout: int = DEFAULT_TIMEOUT, force: bool = False) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    url = f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
    data, meta = get_json(url, cache_dir / "rcsb_entries", timeout=timeout, force=force)
    return (data if isinstance(data, dict) else None), meta


def hepd_search(query: str, cache_dir: Path, timeout: int = DEFAULT_TIMEOUT, force: bool = False, size: int = 25) -> Dict[str, Any]:
    # HEPData's web search returns JSON with ?format=json; API details have changed over time, so this wrapper is permissive.
    url = "https://www.hepdata.net/search/"
    params = {"q": query, "format": "json", "size": size}
    data, meta = get_json(url, cache_dir / "hepdata", timeout=timeout, force=force, params=params)
    records = []
    if isinstance(data, dict):
        for key in ["results", "hits", "data"]:
            val = data.get(key)
            if isinstance(val, list):
                records.extend(val)
            elif isinstance(val, dict) and isinstance(val.get("hits"), list):
                records.extend(val.get("hits"))
    return {"query": query, "meta": meta, "raw_type": type(data).__name__, "records": records[:size], "raw_keys": list(data.keys()) if isinstance(data, dict) else None}

# ---------------------------------------------------------------------------
# v4 result-quality status helpers
# ---------------------------------------------------------------------------

def evidence_readiness_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Separate data readiness from evidence classification.

    status remains backward-compatible, but v4 adds:
    - readiness_status: how far the public-data pipeline got.
    - evidence_status: confirmed/plausible/null/falsified/data_limited style.
    """
    q = int(result.get("qualifying_table_count") or result.get("qualifying_physical_tables_count") or 0)
    parsed = int(result.get("parsed_tables_count") or 0)
    if result.get("metrics") or result.get("subset_summaries") or result.get("burst_results") or result.get("subtests"):
        readiness = "model_fit_done"
    elif q >= 1:
        readiness = "physical_columns_found"
    elif parsed >= 1 or result.get("table_summaries") or any((r.get("tables") for r in result.get("manifest_records", []) if isinstance(r, dict))):
        readiness = "structured_table_found_no_required_physical_columns"
    elif result.get("downloaded_sources") or result.get("manifest_records") or result.get("nrel_downloads") or result.get("endpoint_records"):
        readiness = "source_found_no_usable_table"
    else:
        readiness = "no_source_found"

    status = result.get("status")
    support = result.get("support_like")
    fals_pressure = result.get("falsification_pressure")
    level = str(result.get("evidence_level") or "")
    if status in {"data_limited", "error"}:
        evidence = status
    elif support is True and "synthetic" in level:
        evidence = "confirm_like_synthetic_only"
    elif support is True:
        evidence = "confirm_like"
    elif fals_pressure is True:
        evidence = "null_with_falsification_pressure"
    elif support is False:
        evidence = "null"
    elif status == "partial":
        evidence = "plausible_but_incomplete"
    elif status == "ok":
        evidence = "diagnostic_ok_no_directional_claim"
    else:
        evidence = status or "unknown"
    return {"readiness_status": readiness, "evidence_status": evidence}


def enrich_result_quality_status(result: Dict[str, Any]) -> Dict[str, Any]:
    result.update(evidence_readiness_from_result(result))
    result.setdefault("status_semantics", {
        "status": "legacy execution/result bucket",
        "readiness_status": "data-pipeline stage reached",
        "evidence_status": "scientific interpretation bucket",
    })
    return result

# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def falsification_block(confirm: str, falsify: str, caveat: Optional[str] = None) -> Dict[str, str]:
    out = {"confirm_like": confirm, "falsify_like": falsify}
    if caveat:
        out["caveat"] = caveat
    return out


def wrap_error(test_id: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "test_id": test_id,
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(limit=20),
    }


def run_main(test_id: str, description: str, func):
    p = base_argparser(description)
    args = p.parse_args()
    try:
        result = func(args)
    except Exception as e:
        result = wrap_error(test_id, e)
    emit_result(result, args.outdir, test_id)

