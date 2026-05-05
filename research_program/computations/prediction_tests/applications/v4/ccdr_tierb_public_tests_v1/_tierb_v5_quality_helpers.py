from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import requests

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


USER_AGENT = "ccdr-tierb-v5-quality-patch/1.0 (+public-data; contact: local-user)"


def now_utc_iso() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_name(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s[:max_len] or "file"


def http_get(url: str, timeout: int = 45, stream: bool = False) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, stream=stream)
    r.raise_for_status()
    return r


def http_post_json(url: str, payload: Dict[str, Any], timeout: int = 45) -> requests.Response:
    r = requests.post(
        url,
        json=payload,
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r


def download_bytes(url: str, timeout: int = 60) -> Tuple[bytes, Dict[str, Any]]:
    meta = {"url": url, "ok": False, "status_code": None, "content_type": None, "final_url": url, "error": None}
    try:
        r = http_get(url, timeout=timeout)
        meta.update({
            "ok": True,
            "status_code": r.status_code,
            "content_type": r.headers.get("content-type"),
            "final_url": r.url,
            "bytes": len(r.content),
        })
        return r.content, meta
    except Exception as e:
        meta["error"] = f"download_failed: {type(e).__name__}: {e}"
        return b"", meta


# ---------------------------------------------------------------------------
# Improvement 6: correct Figshare search.
# Figshare public article search is POST /v2/articles/search with JSON body.
# ---------------------------------------------------------------------------

def figshare_article_search(search_for: str, page_size: int = 50, page: int = 1) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    url = "https://api.figshare.com/v2/articles/search"
    payload = {"search_for": search_for, "page_size": page_size, "page": page, "order_direction": "desc"}
    meta = {"url": url, "method": "POST", "payload": payload, "ok": False, "error": None}
    try:
        r = http_post_json(url, payload)
        meta.update({"ok": True, "status_code": r.status_code, "content_type": r.headers.get("content-type")})
        data = r.json()
        return data if isinstance(data, list) else [], meta
    except Exception as e:
        meta["error"] = f"figshare_search_failed: {type(e).__name__}: {e}"
        return [], meta


# ---------------------------------------------------------------------------
# Improvement 5: stricter fusion gates and OSF recursive file traversal.
# ---------------------------------------------------------------------------

FUSION_ANCHORS = [
    r"\btokamak\b", r"\bstellarator\b", r"\bplasma\b", r"\bH[- ]?mode\b", r"\bpedestal\b",
    r"\bdivertor\b", r"\bseparatrix\b", r"\bconfinement\b", r"\bDIII[- ]?D\b", r"\bJET\b",
    r"\bASDEX\b", r"\bAUG\b", r"\bKSTAR\b", r"\bEAST\b", r"\bITER\b", r"\bW7[- ]?X\b",
    r"\bLHD\b", r"\bITPA\b",
]
OBSERVABLE_ANCHORS = [
    r"\bE[_ -]?ELM\b", r"\bW[_ -]?ELM\b", r"\bf[_ -]?ELM\b", r"\bELM frequency\b",
    r"\bELM energy\b", r"\bτ[_ -]?E\b", r"\btau[_ -]?E\b", r"\bH[- ]?factor\b",
    r"\bpedestal pressure\b", r"\bP[_ -]?ped\b", r"\bdensity\b", r"\bq95\b", r"\bq_95\b",
    r"\btriangularity\b", r"\belongation\b", r"\bdiffusivity\b", r"\bheat flux\b",
]
NEGATIVE_CONTEXTS = [
    r"\belm tree\b", r"\belm trees\b", r"\bEarth Land Model\b", r"\bDELM\b", r"\bNoDELM\b",
    r"\bELM WD\b", r"\bwhite dwarf\b", r"\bsquirrel\b", r"\bland surface\b", r"\bCOVID\b",
    r"\bdental\b", r"\bquestionnaire\b", r"\bsurvey\b", r"\bSIGFOX\b", r"\bGPS\b", r"\byaw\b", r"\broll\b",
]

T26_REQUIRED = [
    [r"\bELM\b", r"\bedge localized mode\b", r"\bedge localised mode\b"],
    [r"\btokamak\b", r"\bpedestal\b", r"\bH[- ]?mode\b", r"\bplasma\b", r"\bdivertor\b", r"\bJET\b", r"\bDIII[- ]?D\b", r"\bAUG\b", r"\bASDEX\b", r"\bKSTAR\b", r"\bITER\b", r"\bEAST\b"],
    [r"\bE[_ -]?ELM\b", r"\bW[_ -]?ELM\b", r"\bELM energy\b", r"\bpedestal pressure\b", r"\bP[_ -]?ped\b", r"\bΔP\b", r"\bdeltaP\b", r"\bWped\b"],
]
T27_REQUIRED = [
    [r"\bRMP\b", r"\bresonant magnetic perturbation\b", r"\bmagnetic perturbation\b", r"\bcoil phasing\b", r"\bI[- ]?coil\b", r"\bn\s*=\s*2\b", r"\bn\s*=\s*3\b"],
    [r"\bELM frequency\b", r"\bf[_ -]?ELM\b", r"\bELM suppression\b", r"\bELM mitigation\b"],
    [r"\btokamak\b", r"\bDIII[- ]?D\b", r"\bKSTAR\b", r"\bEAST\b", r"\bASDEX\b", r"\bAUG\b", r"\bJET\b"],
]


def _count_matches(patterns: Sequence[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.I))


def fusion_gate(text: str, test_id: str = "generic") -> Dict[str, Any]:
    text = text or ""
    negative_hits = [p for p in NEGATIVE_CONTEXTS if re.search(p, text, flags=re.I)]
    fusion_hits = [p for p in FUSION_ANCHORS if re.search(p, text, flags=re.I)]
    observable_hits = [p for p in OBSERVABLE_ANCHORS if re.search(p, text, flags=re.I)]
    ok = False
    reason = ""
    if negative_hits:
        reason = "negative_context"
    elif test_id == "T26":
        groups = [_count_matches(g, text) for g in T26_REQUIRED]
        ok = groups[0] >= 1 and groups[1] >= 2 and groups[2] >= 1
        reason = f"T26_groups={groups}"
    elif test_id == "T27":
        groups = [_count_matches(g, text) for g in T27_REQUIRED]
        ok = all(x >= 1 for x in groups)
        reason = f"T27_groups={groups}"
    else:
        ok = len(fusion_hits) >= 1 and len(observable_hits) >= 1
        reason = f"fusion_hits={len(fusion_hits)} observable_hits={len(observable_hits)}"
    return {
        "ok": bool(ok),
        "reason": reason,
        "fusion_hits": fusion_hits,
        "observable_hits": observable_hits,
        "negative_hits": negative_hits,
    }


@dataclass
class OSFFile:
    name: str
    kind: str
    path: str
    download_url: Optional[str]
    api_url: Optional[str]
    size: Optional[int] = None


def _osf_item_to_file(item: Dict[str, Any]) -> OSFFile:
    attrs = item.get("attributes", {}) or {}
    links = item.get("links", {}) or {}
    rel = item.get("relationships", {}) or {}
    related = None
    try:
        related = rel.get("files", {}).get("links", {}).get("related", {}).get("href")
    except Exception:
        related = None
    return OSFFile(
        name=attrs.get("name") or item.get("id") or "unknown",
        kind=attrs.get("kind") or item.get("type") or "unknown",
        path=attrs.get("path") or attrs.get("materialized_path") or attrs.get("name") or "",
        download_url=links.get("download"),
        api_url=related,
        size=attrs.get("size"),
    )


def walk_osf_files(api_url: str, max_pages: int = 200, sleep_s: float = 0.1) -> Tuple[List[OSFFile], List[Dict[str, Any]]]:
    """Recursively traverse OSF file API folders and pagination."""
    seen = set()
    queue = [api_url]
    files: List[OSFFile] = []
    diagnostics: List[Dict[str, Any]] = []
    pages = 0
    while queue and pages < max_pages:
        url = queue.pop(0)
        if not url or url in seen:
            continue
        seen.add(url)
        pages += 1
        try:
            r = http_get(url)
            data = r.json()
            diagnostics.append({"url": url, "ok": True, "status_code": r.status_code})
        except Exception as e:
            diagnostics.append({"url": url, "ok": False, "error": f"{type(e).__name__}: {e}"})
            continue
        for item in data.get("data", []) if isinstance(data, dict) else []:
            f = _osf_item_to_file(item)
            if f.kind.lower() == "folder" and f.api_url:
                queue.append(f.api_url)
            else:
                files.append(f)
        next_url = None
        try:
            next_url = data.get("links", {}).get("next")
        except Exception:
            next_url = None
        if next_url:
            queue.append(next_url)
        if sleep_s:
            time.sleep(sleep_s)
    return files, diagnostics


def is_itpa_candidate_file(f: OSFFile) -> bool:
    name = f"{f.name} {f.path}".lower()
    has_keyword = any(k in name for k in ["db5", "db5.2.3", "std5", "itpa", "hmode", "h-mode", "confinement"])
    has_table_ext = re.search(r"\.(csv|tsv|txt|dat|xls|xlsx|zip)$", name) is not None
    return has_keyword and has_table_ext


def classify_readiness(files: Sequence[OSFFile], candidates: Sequence[OSFFile], tables: Sequence[Any]) -> str:
    names = " ".join([f.name for f in files]).lower()
    if tables:
        return "model_fit_done"
    if candidates:
        return "candidate_table_found_missing_required_columns"
    if "variables" in names and ".pdf" in names:
        return "source_found_variables_dictionary_only"
    if files:
        return "source_found_no_usable_table"
    return "no_source_found"


# ---------------------------------------------------------------------------
# Generic structured table parsing.
# ---------------------------------------------------------------------------

TABLE_EXT_RE = re.compile(r"\.(csv|tsv|txt|dat|xls|xlsx)$", re.I)


def read_table_bytes(blob: bytes, name: str):
    if pd is None:
        return None
    low = name.lower()
    bio = io.BytesIO(blob)
    try:
        if low.endswith((".xls", ".xlsx")):
            return pd.read_excel(bio)
        sep = "\t" if low.endswith(".tsv") else None
        if sep:
            return pd.read_csv(bio, sep=sep)
        # Try regular CSV, then whitespace-delimited fallback.
        try:
            return pd.read_csv(bio)
        except Exception:
            bio.seek(0)
            return pd.read_csv(bio, sep=r"\s+", engine="python", comment="#")
    except Exception:
        return None


def extract_tables_from_blob(blob: bytes, name: str) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    if pd is None:
        return out
    low = name.lower()
    if low.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for zi in z.infolist():
                    if zi.is_dir() or not TABLE_EXT_RE.search(zi.filename):
                        continue
                    data = z.read(zi)
                    df = read_table_bytes(data, zi.filename)
                    if df is not None and len(df.columns) > 1:
                        out.append((zi.filename, df))
        except Exception:
            pass
    elif TABLE_EXT_RE.search(name):
        df = read_table_bytes(blob, name)
        if df is not None and len(df.columns) > 1:
            out.append((name, df))
    return out


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def find_col(df, patterns: Sequence[str]) -> Optional[str]:
    cols = list(df.columns)
    normed = {c: norm_col(c) for c in cols}
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in cols:
            if rx.search(str(c)) or rx.search(normed[c]):
                return c
    return None


def numeric_series(df, col: str):
    if pd is None:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def linear_fit_metrics(y, X_cols: Dict[str, Any]) -> Dict[str, Any]:
    """Small numpy-based OLS fit with AIC/BIC/RMS. X_cols should include intercept if wanted."""
    import numpy as np
    keys = list(X_cols.keys())
    X = np.column_stack([np.asarray(X_cols[k], dtype=float) for k in keys])
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    n = int(len(y))
    k = int(X.shape[1])
    if n <= k + 3:
        return {"ok": False, "n": n, "reason": "too_few_rows"}
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    rss = float(np.sum(resid ** 2))
    rms = float(np.sqrt(np.mean(resid ** 2)))
    sigma2 = max(rss / max(n, 1), 1e-300)
    aic = float(n * math.log(sigma2) + 2 * k)
    bic = float(n * math.log(sigma2) + math.log(n) * k)
    return {
        "ok": True,
        "n": n,
        "k": k,
        "columns": keys,
        "beta": {keys[i]: float(beta[i]) for i in range(len(keys))},
        "rss": rss,
        "rms": rms,
        "aic": aic,
        "bic": bic,
    }
