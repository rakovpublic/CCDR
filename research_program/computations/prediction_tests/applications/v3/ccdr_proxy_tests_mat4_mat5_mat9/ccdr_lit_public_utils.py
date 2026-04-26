#!/usr/bin/env python3
"""
Small public-literature extraction helpers for CCDR prediction tests.

Design rules:
- Every primary datum must come from a public URL downloaded by the script.
- If a paper only exposes plotted values as images/PDF figures, do not invent values.
  Emit status='data_limited' or 'partial'.
- Prefer source-specific regex extractors for stable open HTML/arXiv/PDF text.
- Uses stdlib only; optional bs4/pypdf/pdfminer/fitz improve extraction if installed.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import gzip
import hashlib
import io
import json
import math
import pathlib
import re
import shutil
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 ccdr-public-literature-tests/0.2"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_name(url: str, suffix_hint: str = "") -> str:
    parsed = urlparse(url)
    base = pathlib.Path(parsed.path).name or "download"
    if not pathlib.Path(base).suffix and suffix_hint:
        base += suffix_hint
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return f"{h}_{base}"


def ensure_dir(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download(url: str, cache_dir: str | pathlib.Path, *, timeout: int = 60, force: bool = False) -> pathlib.Path:
    cache = ensure_dir(cache_dir)
    suffix_hint = pathlib.Path(urlparse(url).path).suffix
    out = cache / safe_name(url, suffix_hint=suffix_hint)
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    req = Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
    })
    last_exc: Optional[BaseException] = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError(f"empty download: {url}")
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(out)
            return out
        except BaseException as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last_exc}")


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("μ", "u").replace("µ", "u")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def strip_html(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
    except Exception:
        text = re.sub(r"<script.*?</script>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
    return normalize_text(text)


def pdf_to_text(path: str | pathlib.Path, max_pages: int = 80) -> str:
    p = pathlib.Path(path)
    for modname in ("pypdf", "PyPDF2"):
        try:
            mod = __import__(modname)
            reader = mod.PdfReader(str(p))
            chunks = []
            for page in list(reader.pages)[:max_pages]:
                chunks.append(page.extract_text() or "")
            text = "\n".join(chunks)
            if len(text.strip()) > 200:
                return normalize_text(text)
        except Exception:
            pass
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        text = extract_text(str(p), maxpages=max_pages)
        if len(text.strip()) > 200:
            return normalize_text(text)
    except Exception:
        pass
    try:
        import fitz  # type: ignore
        doc = fitz.open(str(p))
        chunks = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            chunks.append(page.get_text())
        text = "\n".join(chunks)
        if len(text.strip()) > 200:
            return normalize_text(text)
    except Exception:
        pass
    return ""


def path_to_text(path: str | pathlib.Path) -> str:
    p = pathlib.Path(path)
    lower = p.name.lower()
    data = p.read_bytes()
    if lower.endswith((".html", ".htm")) or data[:100].lower().find(b"<html") >= 0 or b"<!doctype html" in data[:200].lower():
        return strip_html(data)
    if lower.endswith(".pdf") or data[:4] == b"%PDF":
        return pdf_to_text(p)
    if lower.endswith(".gz"):
        try:
            return normalize_text(gzip.decompress(data).decode("utf-8", errors="replace"))
        except Exception:
            pass
    return normalize_text(data.decode("utf-8", errors="replace"))


def download_text(url: str, cache_dir: str | pathlib.Path, *, force: bool = False, timeout: int = 60) -> Tuple[pathlib.Path, str]:
    path = download(url, cache_dir, timeout=timeout, force=force)
    return path, path_to_text(path)


def download_arxiv_source(arxiv_id: str, cache_dir: str | pathlib.Path, *, force: bool = False) -> Tuple[pathlib.Path, Dict[str, str]]:
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    path = download(url, cache_dir, force=force, timeout=90)
    work = ensure_dir(pathlib.Path(cache_dir) / f"arxiv_{arxiv_id.replace('/', '_')}_src")
    if force and work.exists():
        shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    data = path.read_bytes()
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            tar.extractall(work)
    except Exception:
        try:
            files["source.tex"] = normalize_text(gzip.decompress(data).decode("utf-8", errors="replace"))
            return path, files
        except Exception:
            files["source.raw"] = normalize_text(data.decode("utf-8", errors="replace"))
            return path, files
    for fp in work.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in {".tex", ".txt", ".csv", ".dat", ".md", ".bib"}:
            try:
                files[str(fp.relative_to(work))] = normalize_text(fp.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
    return path, files


def first_float(s: str) -> Optional[float]:
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else None


def canonical_units_number(s: str) -> float:
    return float(s.strip().replace(",", ""))


def _finite_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def rankdata(a: Sequence[float]) -> List[float]:
    vals = [float(x) for x in a]
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    pairs = []
    for a, b in zip(x, y):
        aa = _finite_float(a)
        bb = _finite_float(b)
        if aa is not None and bb is not None:
            pairs.append((aa, bb))
    if len(pairs) < 2:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((v - mx) ** 2 for v in xs)
    vy = sum((v - my) ** 2 for v in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((a - mx) * (b - my) for a, b in pairs)
    return float(cov / math.sqrt(vx * vy))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    pairs = []
    for a, b in zip(x, y):
        aa = _finite_float(a)
        bb = _finite_float(b)
        if aa is not None and bb is not None:
            pairs.append((aa, bb))
    if len(pairs) < 2:
        return float("nan")
    return pearson(rankdata([p[0] for p in pairs]), rankdata([p[1] for p in pairs]))


def _linear_fit(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need >=2 paired points")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    den = sum((v - mx) ** 2 for v in x)
    if den == 0:
        raise ValueError("x has zero variance")
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / den
    return float(slope), float(my - slope * mx)


def power_law_fit(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    pairs = []
    for a, b in zip(x, y):
        aa = _finite_float(a)
        bb = _finite_float(b)
        if aa is not None and bb is not None and aa > 0 and bb > 0:
            pairs.append((aa, bb))
    n = len(pairs)
    if n < 2:
        return {"n": n, "status": "insufficient"}
    lx = [math.log(a) for a, _ in pairs]
    ly = [math.log(b) for _, b in pairs]
    try:
        b, logA = _linear_fit(lx, ly)
    except ValueError as exc:
        return {"n": n, "status": "insufficient", "reason": str(exc)}
    preds = [math.exp(logA + b * xx) for xx in lx]
    ys = [math.exp(v) for v in ly]
    rel = [(yy - pp) / yy for yy, pp in zip(ys, preds) if yy != 0]
    rms_rel = math.sqrt(sum(r * r for r in rel) / len(rel)) if rel else float("nan")
    return {
        "n": n,
        "A": float(math.exp(logA)),
        "exponent": float(b),
        "rms_rel": float(rms_rel),
        "pearson_log": pearson(lx, ly),
        "spearman": spearman([p[0] for p in pairs], [p[1] for p in pairs]),
    }


def relative_rms(y: Sequence[float], yhat: Sequence[float]) -> float:
    rel = []
    for a, b in zip(y, yhat):
        aa = _finite_float(a)
        bb = _finite_float(b)
        if aa is not None and bb is not None and aa != 0:
            rel.append(((aa - bb) / aa) ** 2)
    if not rel:
        return float("nan")
    return float(math.sqrt(sum(rel) / len(rel)))


def json_dumps(obj: Any) -> str:
    def default(o: Any):
        if isinstance(o, pathlib.Path):
            return str(o)
        if isinstance(o, set):
            return sorted(o)
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return str(o)
    return json.dumps(obj, indent=2, sort_keys=True, default=default, allow_nan=True)


def write_json(obj: Any, outdir: str | pathlib.Path, name: str) -> pathlib.Path:
    od = ensure_dir(outdir)
    path = od / name
    path.write_text(json_dumps(obj) + "\n", encoding="utf-8")
    return path


def common_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--cache", default="public_cache", help="download cache directory")
    p.add_argument("--outdir", default="out_public_tests", help="output directory")
    p.add_argument("--force", action="store_true", help="force re-download")
    p.add_argument("--strict", action="store_true", help="raise if extraction is data-limited")
    return p


def source_record(label: str, url: str, path: Optional[pathlib.Path] = None, notes: Optional[str] = None) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"label": label, "url": url}
    if path is not None:
        rec["downloaded_path"] = str(path)
    if notes:
        rec["notes"] = notes
    return rec


def classify_pass_bool(value: Optional[bool]) -> str:
    if value is None:
        return "partial"
    return "confirm_like" if value else "falsify_like"

# ---- v3 table/text extraction helpers -------------------------------------------------

def binary_strings(path: str | pathlib.Path, *, min_len: int = 4) -> str:
    """Extract printable ASCII/UTF-16-ish strings from binary supplements such as .doc.

    This is intentionally simple and dependency-light. It is not a full Office parser, but
    it often recovers table labels/numbers from legacy Word supplement files.
    """
    p = pathlib.Path(path)
    data = p.read_bytes()
    ascii_chunks = re.findall(rb"[\x20-\x7e]{%d,}" % min_len, data)
    # crude UTF-16LE printable stream: A\x00 B\x00 ...
    utf16_chunks = re.findall((rb"(?:[\x20-\x7e]\x00){%d,}" % min_len), data)
    parts: List[str] = []
    for c in ascii_chunks[:20000]:
        parts.append(c.decode("latin-1", errors="ignore"))
    for c in utf16_chunks[:20000]:
        try:
            parts.append(c.decode("utf-16le", errors="ignore"))
        except Exception:
            pass
    return normalize_text("\n".join(parts))


def _clean_cell_text(s: str) -> str:
    return normalize_text(re.sub(r"\s+", " ", s or "")).strip()


def html_tables_from_path(path: str | pathlib.Path) -> List[Dict[str, Any]]:
    """Return lightweight HTML tables with caption, headers, and rows.

    Works best with BeautifulSoup, but includes a regex fallback. This is used only as an
    extraction aid; callers still apply strict physical/context checks before accepting rows.
    """
    p = pathlib.Path(path)
    data = p.read_bytes()
    text = data.decode("utf-8", errors="replace")
    tables: List[Dict[str, Any]] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(text, "html.parser")
        for ti, tbl in enumerate(soup.find_all("table")):
            caption_tag = tbl.find("caption")
            caption = _clean_cell_text(caption_tag.get_text(" ") if caption_tag else "")
            headers: List[str] = []
            rows: List[List[str]] = []
            for tr in tbl.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                vals = [_clean_cell_text(c.get_text(" ")) for c in cells]
                vals = [v for v in vals if v]
                if not vals:
                    continue
                if tr.find_all("th") and not headers:
                    headers = vals
                else:
                    rows.append(vals)
            if rows or headers:
                tables.append({"index": ti, "caption": caption, "headers": headers, "rows": rows})
        return tables
    except Exception:
        pass

    # regex fallback for simple article tables
    for ti, block in enumerate(re.findall(r"<table\b.*?</table>", text, flags=re.I | re.S)):
        cap_m = re.search(r"<caption\b.*?>(.*?)</caption>", block, flags=re.I | re.S)
        caption = _clean_cell_text(strip_html(cap_m.group(1))) if cap_m else ""
        headers: List[str] = []
        rows: List[List[str]] = []
        for tr in re.findall(r"<tr\b.*?</tr>", block, flags=re.I | re.S):
            ths = [_clean_cell_text(strip_html(x)) for x in re.findall(r"<th\b.*?>(.*?)</th>", tr, flags=re.I | re.S)]
            tds = [_clean_cell_text(strip_html(x)) for x in re.findall(r"<td\b.*?>(.*?)</td>", tr, flags=re.I | re.S)]
            if ths and not headers:
                headers = [h for h in ths if h]
            elif tds:
                rows.append([d for d in tds if d])
        if rows or headers:
            tables.append({"index": ti, "caption": caption, "headers": headers, "rows": rows})
    return tables


def floats_from_text(s: str) -> List[float]:
    out: List[float] = []
    for m in re.finditer(r"[-+]?\d+(?:\.\d+)?(?:\s*[x×]\s*10\s*(?:\^|\*\*)?\s*[-+]?\d+|[eE][-+]?\d+)?", s.replace(",", "")):
        raw = re.sub(r"\s+", "", m.group(0))
        raw = raw.replace("×", "x")
        try:
            if "x10" in raw:
                base, exp = raw.split("x10", 1)
                exp = exp.replace("^", "").replace("**", "")
                out.append(float(base) * (10 ** float(exp)))
            else:
                out.append(float(raw))
        except Exception:
            continue
    return out
