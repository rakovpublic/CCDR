#!/usr/bin/env python3
"""
CCDR Tier-A v9.6 quality helpers.

This module is intentionally self-contained and conservative. It improves data-limited
paths without turning weak/proxy data into confirmation.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import re
import tarfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

USER_AGENT = "ccdr-tierA-v9.6-quality/1.0 public-data-test"

# ----------------------------- general utilities -----------------------------

def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x
    if np is not None:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            y = float(x)
            return None if (math.isnan(y) or math.isinf(y)) else y
        if isinstance(x, np.ndarray):
            return [json_safe(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    return str(x)


def write_json(path: str | os.PathLike, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | os.PathLike, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def safe_name(s: str, n: int = 90) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:n].strip("_") or "file"


def http_get_bytes(url: str, cache_dir: str | os.PathLike = ".cache_v96", timeout: int = 45) -> Tuple[bytes, Dict[str, Any]]:
    cache_dir = ensure_dir(cache_dir)
    key = safe_name(urlparse(url).netloc + "_" + urlparse(url).path.replace("/", "_"))
    if not key or key == "file":
        key = safe_name(url)
    cache_path = cache_dir / key
    meta = {"url": url, "cache_path": str(cache_path), "ok": False, "cached": False, "error": None}
    if cache_path.exists() and cache_path.stat().st_size > 0:
        meta.update(ok=True, cached=True, bytes=cache_path.stat().st_size)
        return cache_path.read_bytes(), meta
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as r:
            data = r.read()
            meta.update(ok=True, status_code=getattr(r, "status", None), final_url=getattr(r, "url", url),
                        content_type=r.headers.get("content-type"), bytes=len(data))
            cache_path.write_bytes(data)
            return data, meta
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        return b"", meta


def dataframe_from_bytes(blob: bytes, name: str = "artifact") -> List[Tuple[str, Any]]:
    """Try hard to parse a binary artifact into tables. Returns [(label, df)]."""
    out: List[Tuple[str, Any]] = []
    if pd is None or not blob:
        return out
    low = name.lower()
    try:
        if low.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for zi in z.infolist():
                    if zi.is_dir():
                        continue
                    if re.search(r"\.(csv|tsv|txt|dat|xlsx|xls|json)$", zi.filename, re.I):
                        out += dataframe_from_bytes(z.read(zi), zi.filename)
            return out
        if low.endswith((".tar.gz", ".tgz")):
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
                for m in t.getmembers():
                    if m.isfile() and re.search(r"\.(csv|tsv|txt|dat|tex|json)$", m.name, re.I):
                        f = t.extractfile(m)
                        if f:
                            out += dataframe_from_bytes(f.read(), m.name)
            return out
        if low.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(io.BytesIO(blob))
            for sh in xls.sheet_names:
                # Header scan catches workbooks with title rows.
                for h in range(0, 25):
                    try:
                        df = xls.parse(sh, header=h)
                        if df is not None and len(df.columns) >= 2 and len(df) >= 1:
                            out.append((f"{name}:{sh}:header{h}", df))
                            break
                    except Exception:
                        continue
            return out
        if low.endswith(".json") or (blob[:1] in (b"{", b"[") and len(blob) < 30_000_000):
            try:
                obj = json.loads(blob.decode("utf-8", errors="ignore"))
                if isinstance(obj, list):
                    df = pd.json_normalize(obj)
                else:
                    df = pd.json_normalize(obj)
                out.append((name, df))
                return out
            except Exception:
                pass
        text = blob.decode("utf-8", errors="ignore")
        if low.endswith(".tex") or "\\begin{tabular" in text or "\\begin{deluxetable" in text:
            out += latex_tables_to_dataframes(text, name)
            return out
        if low.endswith(".tsv"):
            out.append((name, pd.read_csv(io.StringIO(text), sep="\t")))
            return out
        if low.endswith((".csv", ".txt", ".dat")) or True:
            # Try CSV, whitespace, fixed comment-heavy formats.
            for kwargs in ({}, {"sep": r"\s+", "engine": "python", "comment": "#"}, {"sep": "|", "engine": "python"}):
                try:
                    df = pd.read_csv(io.StringIO(text), **kwargs)
                    if df is not None and len(df.columns) >= 2:
                        out.append((name, df))
                        return out
                except Exception:
                    pass
    except Exception:
        return out
    return out


def latex_tables_to_dataframes(tex: str, name: str = "tex") -> List[Tuple[str, Any]]:
    if pd is None:
        return []
    out = []
    # capture tabular/longtable/deluxetable-ish blocks conservatively
    pats = [r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", r"\\begin\{longtable\}.*?\\end\{longtable\}", r"\\begin\{deluxetable\}.*?\\end\{deluxetable\}"]
    blocks = []
    for p in pats:
        blocks += re.findall(p, tex, flags=re.S)
    for i, block in enumerate(blocks[:50]):
        lines = []
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            line = re.sub(r"\\(hline|toprule|midrule|bottomrule|tableline)\b", "", line)
            line = re.sub(r"\\colhead\{([^}]*)\}", r"\1", line)
            line = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", line)
            line = re.sub(r"\\[a-zA-Z]+(\[[^]]*\])?(\{[^}]*\})?", "", line)
            line = line.replace("\\", "").strip()
            if "&" in line:
                line = line.rstrip("\\").rstrip(";")
                cells = [re.sub(r"\s+", " ", c).strip(" {}$") for c in line.split("&")]
                if len(cells) >= 2:
                    lines.append(cells)
        if len(lines) >= 2:
            width = max(len(r) for r in lines)
            rows = [r + [""]*(width-len(r)) for r in lines]
            header = rows[0]
            data = rows[1:]
            try:
                out.append((f"{name}:latex_table_{i}", pd.DataFrame(data, columns=header)))
            except Exception:
                pass
    return out

# ----------------------------- T03 Pantheon fix -----------------------------

def pantheon_columns_v96(df) -> Tuple[str, Optional[str], Optional[str]]:
    """Robust Pantheon+/SH0ES column selector used by T03 and other SN tests."""
    cols = list(df.columns)
    norm = {c: re.sub(r"[^a-z0-9]+", "", str(c).lower()) for c in cols}
    def pick(patterns: Sequence[str]) -> Optional[str]:
        for pat in patterns:
            rx = re.compile(pat, re.I)
            for c in cols:
                if rx.search(str(c)) or rx.search(norm[c]):
                    return c
        return None
    z = pick([r"^z_cmb$", r"zcmb", r"zhel", r"redshift", r"^z$"])
    mu = pick([r"^mu$", r"mu_sh0es", r"distmod", r"distance.*mod", r"mub"])
    dmu = pick([r"dmu", r"muerr", r"sigma.*mu", r"err.*mu", r"^dm$"])
    # Pantheon+ often has zHD and MU_SH0ES columns.
    if z is None:
        for candidate in ["zHD", "zCMB", "zHEL", "z"]:
            if candidate in cols:
                z = candidate; break
    if mu is None:
        for candidate in ["MU_SH0ES", "MU", "m_b_corr"]:
            if candidate in cols:
                mu = candidate; break
    if z is None:
        raise KeyError(f"No redshift column found; columns={cols[:30]}")
    return z, mu, dmu

# ----------------------------- kappa map loader -----------------------------

def classify_fits_product_v96(path: str | os.PathLike) -> Dict[str, Any]:
    path = Path(path)
    low = path.name.lower()
    info = {"path": str(path), "exists": path.exists(), "filename": path.name, "product_type": "unknown", "usable_without_transform": False, "reason": None}
    if not path.exists():
        info["reason"] = "missing_file"; return info
    if re.search(r"(^|[_-])(alm|klm|kappa_alm|phi_alm)([_\.-]|$)", low):
        info.update(product_type="alm", usable_without_transform=False, reason="alm_coefficients_need_alm2map")
        return info
    try:
        from astropy.io import fits
        with fits.open(path, memmap=True) as hdul:
            hdus = []
            for i, h in enumerate(hdul):
                shape = getattr(h.data, "shape", None)
                hdr = h.header
                ctype1 = str(hdr.get("CTYPE1", "")); ctype2 = str(hdr.get("CTYPE2", ""))
                hdus.append({"idx": i, "name": h.name, "shape": shape, "ctype1": ctype1, "ctype2": ctype2, "naxis": hdr.get("NAXIS")})
                if shape is not None and len(shape) == 2 and min(shape) > 16 and (ctype1 or ctype2):
                    info.update(product_type="wcs_image", usable_without_transform=True, selected_hdu=i, hdus=hdus)
                    return info
                if shape is not None and len(shape) == 1 and shape[0] >= 12*32*32:
                    n = int(shape[0]); nside = int(round(math.sqrt(n/12)))
                    if 12*nside*nside == n:
                        info.update(product_type="healpix_ring_or_nested", usable_without_transform=True, selected_hdu=i, nside=nside, hdus=hdus)
                        return info
            info.update(hdus=hdus, reason="no_usable_map_hdu")
    except Exception as e:
        info["reason"] = f"fits_inspection_failed:{type(e).__name__}:{e}"
    return info


def find_best_kappa_product_v96(cache_root: str | os.PathLike, prefer_realspace: bool = True) -> Dict[str, Any]:
    root = Path(cache_root)
    candidates = list(root.rglob("*.fits")) + list(root.rglob("*.fits.gz"))
    scored = []
    for p in candidates:
        low = p.name.lower()
        if not any(k in low for k in ["kappa", "lens", "convergence", "phi", "map", "alm", "klm"]):
            continue
        c = classify_fits_product_v96(p)
        score = 0
        if c.get("usable_without_transform"): score += 100
        if c.get("product_type") == "wcs_image": score += 20
        if c.get("product_type") == "healpix_ring_or_nested": score += 10
        if c.get("product_type") == "alm": score -= 20
        if "kappa" in low: score += 5
        if "alm" in low or "klm" in low: score -= 8
        c["score"] = score
        scored.append(c)
    scored.sort(key=lambda d: d.get("score", -999), reverse=True)
    return {"best": scored[0] if scored else None, "candidates": scored[:30], "n_candidates": len(scored)}


def sample_kappa_map_v96(path: str | os.PathLike, ra_deg: Sequence[float], dec_deg: Sequence[float], nested: bool = False, nside_for_alm: int = 2048) -> Dict[str, Any]:
    """Sample WCS/HEALPix kappa maps. ALM sampling requires healpy and runs alm2map."""
    if np is None:
        return {"ok": False, "error": "numpy_unavailable"}
    info = classify_fits_product_v96(path)
    if not info.get("exists"):
        return {"ok": False, "error": "missing_map", "map_info": info}
    try:
        from astropy.io import fits
    except Exception as e:
        return {"ok": False, "error": f"astropy_required:{e}", "map_info": info}
    ra = np.asarray(ra_deg, dtype=float); dec = np.asarray(dec_deg, dtype=float)
    try:
        if info["product_type"] == "wcs_image":
            from astropy.wcs import WCS
            with fits.open(path, memmap=True) as hdul:
                hdu = hdul[info["selected_hdu"]]
                data = np.asarray(hdu.data, dtype=float)
                wcs = WCS(hdu.header)
                x, y = wcs.world_to_pixel_values(ra, dec)
                xi = np.rint(x).astype(int); yi = np.rint(y).astype(int)
                mask = (xi >= 0) & (yi >= 0) & (yi < data.shape[-2]) & (xi < data.shape[-1])
                vals = np.full(len(ra), np.nan)
                vals[mask] = data[yi[mask], xi[mask]]
                return {"ok": True, "product_type": "wcs_image", "n": int(np.isfinite(vals).sum()), "values": vals.tolist(), "map_info": info}
        if info["product_type"] == "healpix_ring_or_nested":
            try:
                import healpy as hp
            except Exception as e:
                return {"ok": False, "error": f"healpy_required_for_healpix:{e}", "map_info": info}
            with fits.open(path, memmap=True) as hdul:
                arr = np.asarray(hdul[info["selected_hdu"]].data, dtype=float).ravel()
            theta = np.radians(90.0 - dec); phi = np.radians(ra)
            pix = hp.ang2pix(info["nside"], theta, phi, nest=nested)
            vals = arr[pix]
            return {"ok": True, "product_type": "healpix", "n": int(np.isfinite(vals).sum()), "values": vals.tolist(), "map_info": info}
        if info["product_type"] == "alm":
            try:
                import healpy as hp
            except Exception as e:
                return {"ok": False, "error": f"alm_product_requires_healpy_alm2map:{e}", "map_info": info}
            alm = hp.read_alm(str(path))
            m = hp.alm2map(alm, nside=nside_for_alm, verbose=False)
            theta = np.radians(90.0 - dec); phi = np.radians(ra)
            pix = hp.ang2pix(nside_for_alm, theta, phi, nest=False)
            vals = m[pix]
            return {"ok": True, "product_type": "alm_reconstructed_healpix", "n": int(np.isfinite(vals).sum()), "values": vals.tolist(), "map_info": info}
        return {"ok": False, "error": info.get("reason", "unsupported_map_product"), "map_info": info}
    except Exception as e:
        return {"ok": False, "error": f"sampling_failed:{type(e).__name__}:{e}", "map_info": info}

# ----------------------------- Euclid depth controls -----------------------------

def add_euclid_depth_proxy_v96(df):
    """Add best-effort survey-depth/quality controls from Euclid Q1 columns."""
    if pd is None or df is None:
        return df, {"ok": False, "reason": "pandas_or_dataframe_missing"}
    d = df.copy()
    cols = list(d.columns)
    fluxerr_cols = [c for c in cols if re.search(r"fluxerr", str(c), re.I)]
    magerr_cols = [c for c in cols if re.search(r"magerr|mag_err", str(c), re.I)]
    flag_cols = [c for c in cols if re.search(r"flag|mask|quality", str(c), re.I)]
    selected = []
    if fluxerr_cols:
        # Pick column with most positive finite values and nonzero variance.
        best = None; bestn = -1; bestvar = -1
        for c in fluxerr_cols:
            x = pd.to_numeric(d[c], errors="coerce")
            good = np.isfinite(x) & (x > 0) if np is not None else x.notna()
            n = int(good.sum())
            var = float(np.nanvar(np.log10(x[good]))) if np is not None and n > 3 else 0
            if n > bestn or (n == bestn and var > bestvar):
                best, bestn, bestvar = c, n, var
        if best is not None:
            x = pd.to_numeric(d[best], errors="coerce")
            d["depth_proxy_v96"] = -np.log10(np.clip(x.astype(float), 1e-300, None))
            selected.append(best)
    elif magerr_cols:
        best = magerr_cols[0]
        x = pd.to_numeric(d[best], errors="coerce")
        d["depth_proxy_v96"] = -np.log10(np.clip(x.astype(float), 1e-300, None))
        selected.append(best)
    else:
        d["depth_proxy_v96"] = np.nan if np is not None else None
    if flag_cols:
        for c in flag_cols[:4]:
            d[f"quality_{safe_name(str(c), 30)}_v96"] = pd.to_numeric(d[c], errors="coerce")
    return d, {"ok": bool(selected), "selected_depth_columns": selected, "n_fluxerr_cols": len(fluxerr_cols), "n_magerr_cols": len(magerr_cols), "flag_cols_sample": flag_cols[:10]}


def residualize_by_depth_v96(y, depth, flags=None):
    if np is None:
        return y, {"ok": False, "reason": "numpy_unavailable"}
    y = np.asarray(y, dtype=float); depth = np.asarray(depth, dtype=float)
    X = [np.ones_like(y), depth]
    if flags is not None:
        for f in flags:
            X.append(np.asarray(f, dtype=float))
    X = np.vstack(X).T
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < X.shape[1] + 5:
        return np.full_like(y, np.nan), {"ok": False, "reason": "too_few_finite_rows", "n": int(mask.sum())}
    beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    resid = np.full_like(y, np.nan)
    resid[mask] = y[mask] - X[mask] @ beta
    return resid, {"ok": True, "n": int(mask.sum()), "beta": beta.tolist()}

# ----------------------------- VizieR/CDS parsers -----------------------------

def read_vizier_like_table_v96(blob: bytes, name: str = "vizier") -> List[Tuple[str, Any]]:
    if pd is None:
        return []
    text = blob.decode("utf-8", errors="ignore")
    out = []
    # Try astropy if available: it handles CDS readme/fixed-width tables well.
    try:
        from astropy.table import Table
        for fmt in ["ascii.cds", "ascii.fixed_width", "ascii.commented_header", "ascii.basic"]:
            try:
                tab = Table.read(io.StringIO(text), format=fmt)
                df = tab.to_pandas()
                if len(df.columns) >= 2:
                    out.append((f"{name}:{fmt}", df))
            except Exception:
                pass
    except Exception:
        pass
    # Fallback: remove separator/header garbage and parse whitespace.
    clean_lines = []
    for line in text.splitlines():
        if not line.strip() or line.startswith(("#", "--", "====", "Byte-by-byte")):
            continue
        if re.match(r"\s*\d+\-\d+\s+", line):
            continue
        clean_lines.append(line)
    clean = "\n".join(clean_lines)
    for kwargs in ({"sep": r"\s+", "engine": "python"}, {"sep": "|", "engine": "python"}):
        try:
            df = pd.read_csv(io.StringIO(clean), **kwargs)
            if len(df.columns) >= 2:
                out.append((f"{name}:fallback", df)); break
        except Exception:
            pass
    # Deduplicate by columns/shape
    seen = set(); uniq = []
    for lbl, df in out:
        key = (tuple(map(str, df.columns)), len(df))
        if key not in seen:
            seen.add(key); uniq.append((lbl, df))
    return uniq

# ----------------------------- metadata/data discovery -----------------------------

DATA_EXT_RE = re.compile(r"\.(csv|tsv|txt|dat|fits|fits\.gz|xlsx|xls|json|yaml|yml|hdf5|h5|zip|tar\.gz|tgz)(\?|$)", re.I)
META_DOMAINS = ["api.crossref.org", "api.openalex.org", "api.datacite.org", "semanticscholar.org"]

def artifact_kind_v96(url: str, content_type: Optional[str] = None) -> str:
    u = url.lower(); ct = (content_type or "").lower(); host = urlparse(url).netloc.lower()
    if any(d in host for d in META_DOMAINS):
        return "metadata_record"
    if "hepdata.net/search" in u or "format=json" in u and "search" in u:
        return "metadata_record"
    if DATA_EXT_RE.search(u):
        return "physical_data_artifact"
    if "application/pdf" in ct or u.endswith(".pdf"):
        return "pdf_article_or_report"
    if "html" in ct or u.endswith(".html") or "/html/" in u:
        return "html_article_or_landing_page"
    if "application/json" in ct:
        return "json_unknown_needs_physical_gate"
    return "unknown"


def extract_links_from_text_v96(text: str, base_url: str = "") -> List[Dict[str, str]]:
    links = []
    for m in re.finditer(r'''(?:href|src)=["']([^"']+)["']''', text, flags=re.I):
        u = urljoin(base_url, m.group(1))
        reason = "html_href"
        if re.search(r"supp|data|source|table|csv|xlsx|xls|zip|figshare|zenodo|osf|dryad|hepdata|fits", u, re.I):
            reason = "html_href_data_like"
        links.append({"url": u, "reason": reason, "label": u[:160]})
    for m in re.finditer(r"https?://[^\s'\"<>]+", text):
        u = m.group(0).rstrip(".,);]")
        if re.search(r"supp|data|source|table|csv|xlsx|xls|zip|figshare|zenodo|osf|dryad|hepdata|fits", u, re.I):
            links.append({"url": u, "reason": "embedded_data_like_url", "label": u[:160]})
    # DOI expansion seeds
    for doi in sorted(set(re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)))[:20]:
        doi = doi.rstrip(".,);]")
        q = quote(doi, safe="")
        links += [
            {"url": f"https://api.crossref.org/works/{q}", "reason": "doi_crossref_metadata", "label": f"Crossref metadata {doi}"},
            {"url": f"https://api.openalex.org/works/doi:{doi}", "reason": "doi_openalex_metadata", "label": f"OpenAlex metadata {doi}"},
            {"url": f"https://api.datacite.org/dois/{q}", "reason": "doi_datacite_metadata", "label": f"DataCite metadata {doi}"},
        ]
    # arXiv source package
    arxiv_ids = set(re.findall(r"arxiv\.org/(?:abs|html|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", base_url + "\n" + text, flags=re.I))
    for aid in sorted(arxiv_ids):
        links.append({"url": f"https://arxiv.org/e-print/{aid}", "reason": "arxiv_source_package", "label": f"arXiv source package {aid}"})
    return dedupe_links_v96(links)


def dedupe_links_v96(links: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []; seen = set()
    for l in links:
        u = l.get("url", "")
        if not u or u in seen or u.startswith("data:image"):
            continue
        seen.add(u); out.append(l)
    return out

# ----------------------------- domain-specific table gates -----------------------------

def norm_col(c: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def match_column_groups_v96(df, groups: Sequence[Sequence[str]]) -> Dict[str, Any]:
    cols = list(df.columns) if df is not None else []
    ncols = {str(c): norm_col(c) for c in cols}
    matched = []; missing = []
    for g in groups:
        hit = []
        for term in g:
            rx = re.compile(re.sub(r"\s+", r".*", re.escape(term)), re.I)
            for c in cols:
                if rx.search(str(c)) or rx.search(ncols[str(c)]):
                    hit.append(str(c))
        if hit: matched.append(sorted(set(hit))[:8])
        else: missing.append(list(g))
    numeric_cols = []
    if pd is not None and df is not None:
        for c in cols:
            try:
                x = pd.to_numeric(df[c], errors="coerce")
                if int(x.notna().sum()) >= max(3, min(10, len(df)//5)):
                    numeric_cols.append(str(c))
            except Exception:
                pass
    return {"ok": len(missing) == 0, "matched_groups": matched, "missing_groups": missing, "numeric_columns": numeric_cols[:50], "n_numeric_columns": len(numeric_cols)}


def score_candidate_table_v96(df, contract: Dict[str, Any], source_url: str, evidence_tier: str) -> Dict[str, Any]:
    groups = contract.get("required_column_groups", [])
    m = match_column_groups_v96(df, groups)
    min_rows = (contract.get("min_rows", {}) or {}).get(evidence_tier, (contract.get("min_rows", {}) or {}).get("primary", 10))
    n_rows = int(len(df)) if df is not None else 0
    kind = artifact_kind_v96(source_url)
    rejection = []
    if kind == "metadata_record": rejection.append("metadata_record_not_physical_table")
    if not m["ok"]: rejection.append("missing_required_physical_column_groups")
    if m["n_numeric_columns"] < 1: rejection.append("too_few_numeric_columns")
    if n_rows < min_rows: rejection.append("below_min_rows_for_contract")
    if evidence_tier.startswith("secondary"): rejection.append("secondary_or_nonprimary_evidence_tier")
    qualifies = not rejection or (rejection == ["secondary_or_nonprimary_evidence_tier"] and m["ok"] and n_rows >= min_rows)
    return {"source_url": source_url, "evidence_tier": evidence_tier, "shape": [n_rows, int(len(df.columns)) if df is not None else 0], "physical_column_match": m, "minimum_rows": min_rows, "qualifies_for_model": bool(qualifies), "rejection_reasons": rejection, "confirmation_allowed": bool(evidence_tier == "primary_structured_public_table" and qualifies), "falsification_allowed": bool(evidence_tier == "primary_structured_public_table" and qualifies)}

# ----------------------------- source manifests / more data seeds -----------------------------

EXTRA_SOURCE_SEEDS_V96 = {
    "T04": ["https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html", "https://pla.esac.esa.int/pla/#cosmology"],
    "T05": ["https://pla.esac.esa.int/pla/#cosmology", "https://irsa.ipac.caltech.edu/TAP"],
    "T08": ["https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/671/A48", "https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/502/2369"],
    "T15": ["https://data.nanograv.org/", "https://zenodo.org/api/records/?q=nanograv%2015-year%20posterior"],
    "T17": ["https://data.nanograv.org/", "https://zenodo.org/api/records/?q=stochastic%20gravitational%20wave%20spectral%20index%20posterior"],
    "T21": ["https://lambda.gsfc.nasa.gov/product/cobe/firas_products.html", "https://lambda.gsfc.nasa.gov/data/cobe/firas/"],
    "T23": ["https://hepdata.net/search/?q=BK18%20B-mode%20bandpower&format=json", "https://bicepkeck.org/"],
    "T24": ["https://www.gw-openscience.org/eventapi/html/GWTC/", "https://zenodo.org/api/records/?q=ringdown%20overtone%20posterior"],
    "T25": ["https://hepdata.net/search/?q=eta%2Fs%20Bayesian%20posterior%20heavy%20ion&format=json", "https://zenodo.org/api/records/?q=QGP%20eta%2Fs%20posterior"],
}

# ----------------------------- posterior/chain readers -----------------------------

def read_posterior_like_artifact_v96(path: str | os.PathLike) -> Dict[str, Any]:
    p = Path(path)
    out = {"path": str(p), "ok": False, "kind": None, "columns": [], "n_rows": 0, "summary": {}}
    if not p.exists():
        out["error"] = "missing"; return out
    try:
        if p.suffix.lower() in [".csv", ".txt", ".dat"] and pd is not None:
            df = pd.read_csv(p, sep=None, engine="python", comment="#")
            out.update(ok=True, kind="table", columns=list(map(str, df.columns)), n_rows=int(len(df)))
            for c in df.columns:
                x = pd.to_numeric(df[c], errors="coerce")
                if x.notna().sum() > 10:
                    out["summary"][str(c)] = {"median": float(x.median()), "q16": float(x.quantile(0.16)), "q84": float(x.quantile(0.84))}
            return out
        if p.suffix.lower() == ".npz" and np is not None:
            z = np.load(p, allow_pickle=True)
            out.update(ok=True, kind="npz", columns=list(z.files))
            for k in z.files:
                arr = np.asarray(z[k]).ravel()
                if arr.size > 10 and np.issubdtype(arr.dtype, np.number):
                    out["summary"][k] = {"median": float(np.nanmedian(arr)), "q16": float(np.nanpercentile(arr, 16)), "q84": float(np.nanpercentile(arr, 84)), "n": int(arr.size)}
            return out
        if p.suffix.lower() in [".h5", ".hdf5"]:
            try:
                import h5py
                with h5py.File(p, "r") as h:
                    keys = []
                    h.visit(lambda x: keys.append(x))
                    out.update(ok=True, kind="hdf5", columns=keys[:200])
                return out
            except Exception as e:
                out["error"] = f"hdf5_reader_failed:{e}"; return out
    except Exception as e:
        out["error"] = f"read_failed:{type(e).__name__}:{e}"
    return out

# ----------------------------- Tier-A specific source contracts -----------------------------

DATA_CONTRACTS_V96 = {
    "T04": {"name": "Euclid density x kappa", "required_column_groups": [["ra", "dec"], ["kappa", "convergence"]], "min_rows": {"primary_structured_public_table": 1000}},
    "T05": {"name": "Euclid density x Planck kappa", "required_column_groups": [["ra", "dec"], ["kappa", "convergence"]], "min_rows": {"primary_structured_public_table": 1000}},
    "T08": {"name": "filament catalogue coordinates", "required_column_groups": [["ra", "dec"], ["filament", "axis", "orientation", "theta", "phi"]], "min_rows": {"primary_structured_public_table": 100}},
    "T21": {"name": "CMB spectral distortion bound", "required_column_groups": [["frequency", "wavenumber"], ["intensity", "residual"], ["uncertainty", "sigma", "covariance"]], "min_rows": {"primary_structured_public_table": 20}},
    "T23": {"name": "B-mode bandpower", "required_column_groups": [["ell", "lmin", "lmax"], ["BB", "bandpower"], ["sigma", "uncertainty", "error"]], "min_rows": {"primary_structured_public_table": 5}},
    "T25": {"name": "eta/s posterior table", "required_column_groups": [["eta/s", "etas", "viscosity"], ["temperature", "T", "centrality"], ["posterior", "median", "credible", "uncertainty"]], "min_rows": {"primary_structured_public_table": 5}},
}

