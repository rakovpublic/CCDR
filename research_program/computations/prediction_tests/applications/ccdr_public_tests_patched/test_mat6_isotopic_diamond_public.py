from __future__ import annotations

"""
MAT6 public-source test.

Prediction: isotopically pure diamond thermal conductivity should approach
~3300 W/m/K. This script downloads the public Figshare supplementary material
for the isotopic-diamond paper and performs a conservative public-data audit:

1. Parse machine-readable CSV/TXT/XLSX tables if present.
2. If only a PDF supplement is present, attempt a conservative text scan for
   kappa-like values near thermal-conductivity wording.
3. Do NOT digitize figures automatically. If the source is figure-only, report
   not_executable_pdf_only_no_machine_table instead of pretending this is a
   numeric falsification.
"""

import json
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

from _common_public_data import (
    ensure_dir,
    figshare_article,
    json_dump,
    structured_report,
    unzip_one,
)

ROOT = Path(__file__).resolve().parent
OUT = ensure_dir(ROOT / "outputs" / "mat6")
ARTICLE_ID = 28541759


def _try_read_table(path: Path) -> pd.DataFrame | None:
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".txt", ".dat"}:
            for sep in [",", "\t", ";", None]:
                try:
                    if sep is None:
                        df = pd.read_csv(path, engine="python", sep=r"\s+|,|;|\t")
                    else:
                        df = pd.read_csv(path, sep=sep)
                    if len(df.columns) >= 2 and len(df) >= 3:
                        return df
                except Exception:
                    continue
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
            if len(df.columns) >= 2 and len(df) >= 3:
                return df
    except Exception:
        return None
    return None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, len(df) // 2):
            out.append(str(c))
    return out


def analyse_table(df: pd.DataFrame, source_name: str) -> dict | None:
    num_cols = _numeric_columns(df)
    if len(num_cols) < 2:
        return None
    best = None
    for i, cx in enumerate(num_cols):
        for cy in num_cols[i + 1:]:
            x = pd.to_numeric(df[cx], errors="coerce")
            y = pd.to_numeric(df[cy], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 5:
                continue
            xx = x[mask].to_numpy(float)
            yy = y[mask].to_numpy(float)
            if np.nanmin(xx) >= 0 and np.nanmax(xx) <= 1000 and np.nanmax(yy) > 0:
                idx300 = int(np.argmin(np.abs(xx - 300.0)))
                candidate = {
                    "source_name": source_name,
                    "temperature_column": cx,
                    "conductivity_column": cy,
                    "n_points": int(mask.sum()),
                    "t_min": float(np.min(xx)),
                    "t_max": float(np.max(xx)),
                    "kappa_peak": float(np.max(yy)),
                    "kappa_300K_nearest": float(yy[idx300]),
                    "temperature_nearest_300K": float(xx[idx300]),
                    "distance_to_300K": float(np.min(np.abs(xx - 300.0))),
                }
                if best is None or candidate["kappa_peak"] > best["kappa_peak"]:
                    best = candidate
    return best


def _extract_pdf_text_numbers(path: Path) -> dict:
    result = {"source_name": path.name, "text_extracted": False, "candidate_values_w_m_k": []}
    text = ""
    # Optional dependencies: either pypdf or PyPDF2. If absent, report cleanly.
    for modname in ("pypdf", "PyPDF2"):
        try:
            mod = __import__(modname)
            reader = mod.PdfReader(str(path))
            pages = []
            for page in reader.pages[:30]:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pass
            text = "\n".join(pages)
            if text.strip():
                result["pdf_parser"] = modname
                break
        except Exception as e:
            result.setdefault("pdf_parser_errors", []).append(f"{modname}: {type(e).__name__}: {e}")
    if not text.strip():
        return result
    result["text_extracted"] = True
    result["text_chars"] = len(text)

    vals: list[float] = []
    pattern = r"(?i)(thermal\s+conductivity|conductivity|kappa|κ|W\s*/?\s*m\s*/?\s*K|W\s*m\s*-?1\s*K\s*-?1)"
    for m in re.finditer(pattern, text):
        window = text[max(0, m.start() - 220): min(len(text), m.end() + 220)]
        for num in re.findall(r"(?<![A-Za-z0-9.])([0-9]{2,5}(?:\.[0-9]+)?)(?![A-Za-z0-9.])", window):
            try:
                v = float(num)
            except Exception:
                continue
            if 100.0 <= v <= 10000.0:
                vals.append(v)
    vals = sorted(set(vals))
    result["candidate_values_w_m_k"] = vals
    if vals:
        result["max_candidate_w_m_k"] = max(vals)
    return result


def _parse_env_numeric_urls() -> list[dict]:
    from _common_public_data import cached_download
    out = []
    for i, url in enumerate([u.strip() for u in os.environ.get("MAT6_NUMERIC_URLS", "").split(",") if u.strip()]):
        try:
            suffix = Path(url.split("?")[0]).suffix or ".csv"
            p = cached_download(url, OUT / "env_numeric" / f"mat6_numeric_{i}{suffix}")
            df = _try_read_table(p)
            if df is not None:
                summary = analyse_table(df, p.name)
                if summary:
                    summary["source_url"] = url
                    out.append(summary)
        except Exception as e:
            out.append({"source_url": url, "error": f"{type(e).__name__}: {e}"})
    return out

def main() -> None:
    env_parsed = _parse_env_numeric_urls()
    source_errors = []
    try:
        article = figshare_article(ARTICLE_ID)
    except Exception as e:
        article = {}
        source_errors.append({'source': f'https://figshare.com/articles/{ARTICLE_ID}', 'stage': 'figshare_metadata', 'error': f'{type(e).__name__}: {e}'})
    downloaded = []
    for i, f in enumerate(article.get("files", [])):
        name = str(f.get("name") or f"figshare_file_{i}")
        url = f.get("download_url")
        if not url:
            continue
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or f"figshare_file_{i}"
        try:
            from _common_public_data import cached_download
            downloaded.append(cached_download(url, OUT / "figshare_files" / safe))
        except Exception as e:
            downloaded.append(Path(f"DOWNLOAD_FAILED_{safe}_{type(e).__name__}"))

    extracted = []
    for p in downloaded:
        if not isinstance(p, Path) or str(p).startswith("DOWNLOAD_FAILED"):
            continue
        if p.suffix.lower() == ".zip":
            try:
                extracted.extend(unzip_one(p, OUT / "unzipped" / p.stem))
            except Exception:
                extracted.append(p)
        else:
            extracted.append(p)

    parsed = [r for r in env_parsed if "error" not in r]
    env_errors = [r for r in env_parsed if "error" in r]
    pdf_scans = []
    for p in extracted:
        df = _try_read_table(p)
        if df is not None:
            summary = analyse_table(df, p.name)
            if summary:
                parsed.append(summary)
        elif p.suffix.lower() == ".pdf":
            pdf_scans.append(_extract_pdf_text_numbers(p))

    best = max(parsed, key=lambda r: r["kappa_peak"]) if parsed else None
    pdf_vals = [v for scan in pdf_scans for v in scan.get("candidate_values_w_m_k", [])]
    pdf_max = max(pdf_vals) if pdf_vals else None

    table_pass = bool(best and best.get("kappa_peak", 0.0) >= 3300.0)
    pdf_pass = bool(pdf_max is not None and pdf_max >= 3300.0)
    if best:
        status = "ok_machine_table"
        verdict = "support_like" if table_pass else "no_support_in_parsed_tables"
    elif pdf_max is not None:
        status = "pdf_text_scan_only"
        verdict = "weak_support_like_pdf_text" if pdf_pass else "no_support_in_pdf_text_scan"
    elif extracted:
        status = "not_executable_pdf_only_no_machine_table"
        verdict = "no_physics_verdict"
    elif source_errors:
        status = "source_unavailable"
        verdict = "no_physics_verdict"
    else:
        status = "metadata_only"
        verdict = "no_physics_verdict"

    report = structured_report(
        "MAT6",
        status,
        article_title=article.get("title"),
        article_url=article.get("url_public_html"),
        source_errors=source_errors,
        downloaded_files=[str(p) for p in downloaded],
        extracted_files=[str(p) for p in extracted],
        parsed_tables=parsed,
        env_numeric_errors=env_errors,
        pdf_text_scans=pdf_scans,
        best_table=best,
        max_pdf_text_candidate_w_m_k=pdf_max,
        pass_like=bool(table_pass or pdf_pass),
        verdict=verdict,
        criterion="Peak thermal conductivity at or above 3300 W/m/K in public isotopic-diamond data.",
        note=(
            "The current Figshare source is a PDF supplement. MAT6 is skipped by the default runner unless MAT6_NUMERIC_URLS "
            "is supplied or --include-data-gates is used. The script attempts env numeric tables, then Figshare tables, "
            "then a conservative PDF text scan. It does not digitize plotted curves."
        ),
    )
    json_dump(report, OUT / "mat6_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
