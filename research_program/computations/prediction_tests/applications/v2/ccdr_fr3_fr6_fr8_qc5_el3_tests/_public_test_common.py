#!/usr/bin/env python3
"""Shared helpers for CCDR engineering public-data tests.

Design rules:
  * no manual input files are required;
  * every data-bearing artifact is downloaded from a public URL by the script;
  * if a paper only exposes plots/ranges, the script returns a conservative
    `data_insufficient_*` result instead of digitising by eye or hard-coding support.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

USER_AGENT = "ccdr-public-proxy-tests/0.1 (+https://github.com/rakovpublic/CCDR)"


@dataclass
class DownloadedSource:
    label: str
    url: str
    path: str
    sha256: str
    bytes: int
    ok: bool = True
    error: Optional[str] = None


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify(value: str, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return s[:max_len] or "download"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_bytes(url: str, timeout: int = 90, retries: int = 2) -> bytes:
    last: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except BaseException as exc:  # noqa: BLE001 - convert all URL failures to clear result
            last = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def download_to_cache(label: str, url: str, cache_dir: Path, force: bool = False) -> DownloadedSource:
    ensure_dir(cache_dir)
    parsed = urllib.parse.urlparse(url)
    ext = Path(parsed.path).suffix
    if not ext or len(ext) > 8:
        ext = ".dat"
    filename = slugify(label) + ext
    path = cache_dir / filename
    try:
        if force or not path.exists() or path.stat().st_size == 0:
            data = download_bytes(url)
            path.write_bytes(data)
        data = path.read_bytes()
        return DownloadedSource(label=label, url=url, path=str(path), sha256=sha256_bytes(data), bytes=len(data))
    except BaseException as exc:  # noqa: BLE001
        return DownloadedSource(label=label, url=url, path=str(path), sha256="", bytes=0, ok=False, error=str(exc))


def read_text_any(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return raw.decode(enc, errors="replace")
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


def pdf_text(path: Path) -> Tuple[str, List[str]]:
    """Extract PDF text using optional libraries.

    pdfminer.six is recommended. pypdf is a fallback. The scripts deliberately
    do not OCR by default because OCR is slow, brittle, and can fabricate digits.
    """
    warnings: List[str] = []
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        return extract_text(str(path)) or "", warnings
    except Exception as exc:
        warnings.append(f"pdfminer extraction failed or unavailable: {exc}")
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        parts: List[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts), warnings
    except Exception as exc:
        warnings.append(f"pypdf extraction failed or unavailable: {exc}")
    return "", warnings


def normalize_text(s: str) -> str:
    repl = {
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ",
        "−": "-", "–": "-", "—": "-", "×": "x", "𝑓": "f",
        "𝜈": "nu", "ν": "nu", "∗": "*", "𝑝": "p", "𝑃": "P",
        "𝑊": "W", "𝐸": "E", "𝑀": "M", "𝑩": "B", "𝑰": "I",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return re.sub(r"[ \t]+", " ", s)


def numbers_in(s: str) -> List[float]:
    out: List[float] = []
    for m in re.finditer(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", normalize_text(s)):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    return out


def parse_range(text: str) -> Optional[Tuple[float, float]]:
    text = normalize_text(text)
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:-|to|-)\s*([-+]?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    return (min(a, b), max(a, b))


def pearsonr(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x)), "r": None, "pvalue": None}
    try:
        from scipy.stats import pearsonr as _pearsonr  # type: ignore

        r, p = _pearsonr(x, y)
        return {"n": int(len(x)), "r": float(r), "pvalue": float(p)}
    except Exception:
        r = float(np.corrcoef(x, y)[0, 1])
        return {"n": int(len(x)), "r": r, "pvalue": None}


def spearmanr(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return {"n": int(len(x)), "rho": None, "pvalue": None}
    try:
        from scipy.stats import spearmanr as _spearmanr  # type: ignore

        rho, p = _spearmanr(x, y)
        return {"n": int(len(x)), "rho": float(rho), "pvalue": float(p)}
    except Exception:
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        return {"n": int(len(x)), "rho": rho, "pvalue": None}


def linfit(x: Sequence[float], y: Sequence[float], through_origin: bool = False) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < (1 if through_origin else 2) or np.std(x) == 0:
        return {"n": int(len(x)), "slope": None, "intercept": None, "rms_frac": None, "r2": None}
    if through_origin:
        slope = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) else float("nan")
        intercept = 0.0
        pred = slope * x
    else:
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(slope)
        intercept = float(intercept)
        pred = slope * x + intercept
    resid = y - pred
    denom = np.where(np.abs(y) > 1e-30, np.abs(y), np.nan)
    rms_frac = float(np.sqrt(np.nanmean((resid / denom) ** 2))) if len(y) else None
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"n": int(len(x)), "slope": slope, "intercept": intercept, "rms_frac": rms_frac, "r2": r2}


def fit_scale_only(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    """Fit y ≈ a*x and report fractional RMS around the fitted scale."""
    return linfit(x, y, through_origin=True)


def write_json(obj: Dict[str, Any], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def source_records(sources: Iterable[DownloadedSource]) -> List[Dict[str, Any]]:
    return [s.__dict__ for s in sources]


def csv_rows_from_url(label: str, url: str, cache_dir: Path) -> Tuple[List[Dict[str, str]], DownloadedSource]:
    src = download_to_cache(label, url, cache_dir)
    if not src.ok:
        return [], src
    txt = read_text_any(Path(src.path))
    # strip HEPData/YAML-style comment metadata if present
    body = "\n".join(line for line in txt.splitlines() if not line.lstrip().startswith("#"))
    try:
        return list(csv.DictReader(io.StringIO(body))), src
    except Exception:
        return [], src


def float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if not s or s.lower() in {"nan", "none", "null", "--"}:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def piecewise_log_slope(
    node_nm: Sequence[float],
    density: Sequence[float],
    min_points: int = 5,
    min_unique_x_per_segment: int = 3,
    break_min_nm: Optional[float] = None,
    break_max_nm: Optional[float] = None,
) -> Dict[str, Any]:
    """Find best two-slope breakpoint in log(density) vs log(node_nm).

    Interpretation: density ∝ node^-alpha. alpha≈3 is volume-like; alpha≈2 is area-like.

    Important orientation note (v2.4): nodes are sorted ascending before fitting.
    Therefore x[:k] is the *small-node / advanced* segment and x[k:] is the
    *large-node / older* segment. Earlier helper versions exposed confusing labels.
    The returned dictionary now includes explicit ``small_node_alpha`` and
    ``large_node_alpha`` fields and keeps the old names as corrected aliases.

    Optional ``break_min_nm`` / ``break_max_nm`` constrain candidate breakpoints.
    This is useful for testing whether the hypothesised 10-100 nm EL3 window is
    competitive with the global optimum, rather than letting a single global
    breakpoint dominate the interpretation.
    """
    x0 = np.asarray(node_nm, dtype=float)
    y0 = np.asarray(density, dtype=float)
    mask = np.isfinite(x0) & np.isfinite(y0) & (x0 > 0) & (y0 > 0)
    x = np.log10(x0[mask])
    y = np.log10(y0[mask])
    raw_x = x0[mask]
    order = np.argsort(x)
    x, y, raw_x = x[order], y[order], raw_x[order]
    n = len(x)
    if n < 2 * min_points + 1:
        return {
            "n": n,
            "break_node_nm": None,
            "large_node_alpha": None,
            "small_node_alpha": None,
            "pre_alpha_large_nodes": None,
            "post_alpha_small_nodes": None,
            "sse": None,
            "skipped_reason": "too_few_points",
        }
    best: Optional[Dict[str, Any]] = None
    skipped_conditioned = 0
    skipped_unique = 0
    skipped_outside_window = 0
    for k in range(min_points, n - min_points):
        br = float(raw_x[k])
        if break_min_nm is not None and br < break_min_nm:
            skipped_outside_window += 1
            continue
        if break_max_nm is not None and br > break_max_nm:
            skipped_outside_window += 1
            continue
        if len(np.unique(x[:k])) < min_unique_x_per_segment or len(np.unique(x[k:])) < min_unique_x_per_segment:
            skipped_unique += 1
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                b_small, a_small = np.polyfit(x[:k], y[:k], 1)   # advanced / smaller nominal nodes
                b_large, a_large = np.polyfit(x[k:], y[k:], 1)   # older / larger nominal nodes
            except Exception:
                skipped_conditioned += 1
                continue
            if caught:
                skipped_conditioned += 1
                continue
        pred = np.concatenate([a_small + b_small * x[:k], a_large + b_large * x[k:]])
        sse = float(np.sum((y - pred) ** 2))
        small_alpha = float(-b_small)
        large_alpha = float(-b_large)
        cand = {
            "n": n,
            "break_index": int(k),
            "break_node_nm": br,
            "small_node_alpha": small_alpha,
            "large_node_alpha": large_alpha,
            # Corrected legacy aliases for older readers.
            "post_alpha_small_nodes": small_alpha,
            "pre_alpha_large_nodes": large_alpha,
            "sse": sse,
            "rmse_log10": float(math.sqrt(sse / n)),
            "break_window_nm": [break_min_nm, break_max_nm] if break_min_nm is not None or break_max_nm is not None else None,
            "skipped_candidate_splits_unique_x": int(skipped_unique),
            "skipped_candidate_splits_conditioned": int(skipped_conditioned),
            "skipped_candidate_splits_outside_window": int(skipped_outside_window),
        }
        if best is None or sse < best["sse"]:
            best = cand
    return best or {
        "n": n,
        "break_node_nm": None,
        "large_node_alpha": None,
        "small_node_alpha": None,
        "pre_alpha_large_nodes": None,
        "post_alpha_small_nodes": None,
        "sse": None,
        "break_window_nm": [break_min_nm, break_max_nm] if break_min_nm is not None or break_max_nm is not None else None,
        "skipped_candidate_splits_unique_x": int(skipped_unique),
        "skipped_candidate_splits_conditioned": int(skipped_conditioned),
        "skipped_candidate_splits_outside_window": int(skipped_outside_window),
        "skipped_reason": "no_stable_split_after_filters",
    }

