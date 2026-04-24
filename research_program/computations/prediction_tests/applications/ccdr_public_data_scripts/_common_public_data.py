from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Accept": "*/*"}
TIMEOUT = 60


class DownloadError(RuntimeError):
    pass


@dataclass
class FitResult:
    params: dict[str, float]
    yhat: np.ndarray
    sse: float
    rmse: float
    r2: float
    aic: float
    bic: float


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dump(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text(text: str, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    return s


def cached_download(url: str, dest: str | Path, force: bool = False) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    with session() as s:
        r = s.get(url, timeout=TIMEOUT, stream=True)
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    return dest


def download_text(url: str, force: bool = False) -> str:
    with session() as s:
        r = s.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text


def download_json(url: str) -> Any:
    with session() as s:
        r = s.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()


# -------------------------
# Lightweight statistics
# -------------------------

def _base_metrics(y: np.ndarray, yhat: np.ndarray, k: int) -> tuple[float, float, float, float, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse / ss_tot) if ss_tot > 0 else float("nan")
    n = len(y)
    safe_sse = max(sse, 1e-15)
    aic = float(n * math.log(safe_sse / n) + 2 * k)
    bic = float(n * math.log(safe_sse / n) + k * math.log(n))
    return sse, rmse, r2, aic, bic


def fit_linear(x: Iterable[float], y: Iterable[float]) -> FitResult:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    sse, rmse, r2, aic, bic = _base_metrics(y, yhat, 2)
    return FitResult({"a": float(a), "b": float(b)}, yhat, sse, rmse, r2, aic, bic)


def fit_powerlaw(x: Iterable[float], y: Iterable[float]) -> FitResult:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    mask = (x > 0) & (y > 0)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    alpha, loga = np.polyfit(lx, ly, 1)
    a = math.exp(loga)
    yhat = a * x ** alpha
    sse, rmse, r2, aic, bic = _base_metrics(y, yhat, 2)
    return FitResult({"a": float(a), "alpha": float(alpha)}, yhat, sse, rmse, r2, aic, bic)


def fit_custom_feature_linear(feature: Iterable[float], y: Iterable[float], feature_name: str = "x") -> FitResult:
    z = np.asarray(list(feature), dtype=float)
    y = np.asarray(list(y), dtype=float)
    a, b = np.polyfit(z, y, 1)
    yhat = a * z + b
    sse, rmse, r2, aic, bic = _base_metrics(y, yhat, 2)
    return FitResult({f"a_on_{feature_name}": float(a), "b": float(b)}, yhat, sse, rmse, r2, aic, bic)


def spearman_rank_corr(x: Iterable[float], y: Iterable[float]) -> float:
    x = pd.Series(list(x)).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(list(y)).rank(method="average").to_numpy(dtype=float)
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# -------------------------
# Generic text extraction
# -------------------------

def extract_first(patterns: list[str], text: str, flags: int = re.I | re.S) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1)
    return None


def extract_float(patterns: list[str], text: str, flags: int = re.I | re.S) -> float | None:
    raw = extract_first(patterns, text, flags=flags)
    if raw is None:
        return None
    raw = raw.replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


# -------------------------
# Public repository helpers
# -------------------------

def github_raw(url: str, dest: str | Path) -> Path:
    return cached_download(url, dest)


def github_repo_tree(owner: str, repo: str, branch: str = "main", recursive: bool = True) -> Any:
    # Public GitHub API endpoint.
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}"
    if recursive:
        tree_url += "?recursive=1"
    return download_json(tree_url)


def github_find_blob(owner: str, repo: str, name_regex: str, branch: str = "main") -> list[dict[str, str]]:
    data = github_repo_tree(owner, repo, branch=branch, recursive=True)
    out: list[dict[str, str]] = []
    regex = re.compile(name_regex, re.I)
    for item in data.get("tree", []):
        path = item.get("path", "")
        if item.get("type") == "blob" and regex.search(path):
            raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
            out.append({"path": path, "raw_url": raw})
    return out


def osf_list(url: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    while url:
        data = download_json(url)
        items.extend(data.get("data", []))
        url = (data.get("links") or {}).get("next")
    return items


def osf_find_download(node_ids: list[str], filename_regex: str, dest: str | Path) -> Path:
    regex = re.compile(filename_regex, re.I)
    for node_id in node_ids:
        provider_url = f"https://api.osf.io/v2/nodes/{node_id}/files/"
        providers = download_json(provider_url).get("data", [])
        related_urls: list[str] = []
        for prov in providers:
            rel = (((prov.get("relationships") or {}).get("files") or {}).get("links") or {}).get("related") or {}
            href = rel.get("href")
            if href:
                related_urls.append(href)
        if not related_urls:
            related_urls = [f"https://api.osf.io/v2/nodes/{node_id}/files/osfstorage/"]
        for rel_url in related_urls:
            stack = [rel_url]
            while stack:
                folder_url = stack.pop()
                for item in osf_list(folder_url):
                    attrs = item.get("attributes") or {}
                    name = attrs.get("name", "")
                    kind = attrs.get("kind", "")
                    if kind == "folder":
                        sub = (((item.get("relationships") or {}).get("files") or {}).get("links") or {}).get("related") or {}
                        href = sub.get("href")
                        if href:
                            stack.append(href)
                    elif kind == "file" and regex.search(name):
                        dl = (item.get("links") or {}).get("download")
                        if dl:
                            return cached_download(dl, dest)
    raise DownloadError(f"Could not find OSF file matching /{filename_regex}/ in nodes {node_ids}")


def ckan_package_show(base_api: str, dataset_id: str) -> dict[str, Any]:
    url = f"{base_api.rstrip('/')}/action/package_show?id={dataset_id}"
    data = download_json(url)
    if not data.get("success", False):
        raise DownloadError(f"CKAN package_show failed for {dataset_id}")
    return data["result"]


def ckan_download_resource(base_api: str, dataset_id: str, name_regex: str, dest: str | Path) -> Path:
    """Download the first CKAN resource whose name/url/format/mimetype matches name_regex."""
    pkg = ckan_package_show(base_api, dataset_id)
    regex = re.compile(name_regex, re.I)
    available = []
    for res in pkg.get("resources", []):
        name = str(res.get("name", ""))
        fmt = str(res.get("format", ""))
        mime = str(res.get("mimetype", ""))
        url = str(res.get("url") or "")
        descriptor = " ".join([name, fmt, mime, url])
        available.append({"name": name, "format": fmt, "mimetype": mime, "url": url})
        if url and regex.search(descriptor):
            return cached_download(url, dest)
    raise DownloadError(
        f"No CKAN resource matching /{name_regex}/ in dataset {dataset_id}. "
        f"Available resources: {available}"
    )


def ckan_download_matching_resources(base_api: str, dataset_id: str, name_regex: str, out_dir: str | Path, limit: int | None = None) -> list[Path]:
    """Download all matching CKAN resources and return local paths."""
    pkg = ckan_package_show(base_api, dataset_id)
    out = ensure_dir(out_dir)
    regex = re.compile(name_regex, re.I)
    paths: list[Path] = []
    for i, res in enumerate(pkg.get("resources", [])):
        name = str(res.get("name") or f"resource_{i}")
        fmt = str(res.get("format", ""))
        mime = str(res.get("mimetype", ""))
        url = str(res.get("url") or "")
        descriptor = " ".join([name, fmt, mime, url])
        if not url or not regex.search(descriptor):
            continue
        suffix = Path(url.split("?")[0]).suffix or ("." + fmt.lower() if fmt else "")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or f"resource_{i}"
        if suffix and not safe.lower().endswith(suffix.lower()):
            safe += suffix
        try:
            paths.append(cached_download(url, out / safe))
        except Exception:
            continue
        if limit is not None and len(paths) >= limit:
            break
    return paths



def read_public_table(path: str | Path) -> pd.DataFrame:
    """Robustly read a downloaded public table.

    Some public APIs expose HTML/metadata while the caller saved the file as
    .xlsx. Sniff bytes first and return a useful DownloadError instead of a
    pandas engine crash.
    """
    path = Path(path)
    raw = path.read_bytes()
    head = raw[:512].lstrip()
    if not raw:
        raise DownloadError(f"Downloaded table is empty: {path}")
    low = head[:256].lower()
    if low.startswith(b"<!doctype html") or low.startswith(b"<html") or b"<html" in low:
        raise DownloadError(f"Downloaded resource appears to be HTML, not a machine-readable table: {path}")
    if raw[:2] == b"PK":
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            raise DownloadError(f"Could not parse XLSX table {path}: {e!r}") from e
    if raw[:8] == bytes.fromhex("d0cf11e0a1b11ae1"):
        try:
            return pd.read_excel(path)
        except Exception as e:
            raise DownloadError(f"Could not parse XLS table {path}: {e!r}") from e
    text = raw[:4096].decode("utf-8", errors="ignore")
    for sep in [None, ",", "\t", ";", r"\s+"]:
        try:
            df = pd.read_csv(path, engine="python", sep=sep)
            if len(df.columns) >= 2 and len(df) >= 1:
                return df
        except Exception:
            continue
    raise DownloadError(f"Downloaded resource is not a parseable CSV/TSV/Excel table: {path}; first bytes={text[:120]!r}")
def figshare_article(article_id: int) -> dict[str, Any]:
    return download_json(f"https://api.figshare.com/v2/articles/{article_id}")


def figshare_download_file(article_id: int, name_regex: str, dest: str | Path) -> Path:
    article = figshare_article(article_id)
    regex = re.compile(name_regex, re.I)
    for f in article.get("files", []):
        name = str(f.get("name", ""))
        dl = f.get("download_url") or f.get("supplied_md5")
        if regex.search(name) and f.get("download_url"):
            return cached_download(f["download_url"], dest)
    raise DownloadError(f"No Figshare file matching /{name_regex}/ for article {article_id}")


def unzip_one(zip_path: str | Path, out_dir: str | Path) -> list[Path]:
    out = ensure_dir(out_dir)
    created: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            target = out / name
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            created.append(target)
    return created


def find_files(root: str | Path, pattern: str) -> list[Path]:
    regex = re.compile(pattern, re.I)
    out: list[Path] = []
    for p in Path(root).rglob("*"):
        if p.is_file() and regex.search(p.name):
            out.append(p)
    return sorted(out)


def best_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for col_lower, col in lower.items():
            if cand.lower() == col_lower:
                return col
        for col_lower, col in lower.items():
            if cand.lower() in col_lower:
                return col
    return None


def save_plot(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def structured_report(prediction_id: str, status: str, **extra: Any) -> dict[str, Any]:
    out = {"prediction_id": prediction_id, "status": status}
    out.update(extra)
    return out

# -------------------------
# More permissive table utilities used by patched public-data tests
# -------------------------

def _norm_col_name(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def best_matching_column_fuzzy(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column even when public tables use punctuation, units, or aliases.

    Guard against empty/BOM column names. A previous version treated an empty
    normalized column name as matching every alias, creating false FR4/FR7
    results with every field mapped to \"\\ufeff\".
    """
    aliases = [_norm_col_name(c) for c in candidates]
    aliases = [a for a in aliases if len(a) >= 2]
    columns = [(col, _norm_col_name(col)) for col in df.columns]
    columns = [(col, norm) for col, norm in columns if len(norm) >= 2]

    for alias in aliases:
        for col, norm in columns:
            if alias == norm:
                return col
    for alias in aliases:
        for col, norm in columns:
            if len(alias) >= 3 and alias in norm:
                return col
    for alias in aliases:
        for col, norm in columns:
            if len(norm) >= 4 and norm in alias:
                return col
    return None

def repair_header_if_needed(df: pd.DataFrame, candidate_terms: list[str], max_rows: int = 8) -> pd.DataFrame:
    """Some downloaded spreadsheets have real headers in row 0..N, not in columns."""
    if any(best_matching_column_fuzzy(df, [t]) is not None for t in candidate_terms):
        return df
    for i in range(min(max_rows, len(df))):
        vals = [str(v) for v in df.iloc[i].tolist()]
        joined = " ".join(vals).lower()
        if any(t.lower() in joined for t in candidate_terms):
            new = df.iloc[i+1:].copy()
            new.columns = vals
            return new.reset_index(drop=True)
    return df
