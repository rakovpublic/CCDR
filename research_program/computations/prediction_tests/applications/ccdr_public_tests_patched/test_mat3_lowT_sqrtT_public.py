from __future__ import annotations

"""
MAT3 numeric public-data audit.

Prediction: nanocrystalline/grain-sensitive low-temperature thermal conductivity
should scale approximately as kappa ~ T^0.5.

This script does not just download papers. It downloads public machine-readable
CSV/TXT/DAT files from the CMB-S4 Cryogenic_Material_Properties GitHub
repository, fits low-T power-law exponents for every usable conductivity table,
and reports whether any grain/nano candidate is close to alpha=0.5.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _common_public_data import ensure_dir, github_find_blob, github_raw, json_dump, structured_report

ROOT = Path(__file__).resolve().parent
OUT = ensure_dir(ROOT / "outputs" / "mat3")
OWNER = "CMB-S4"
REPO = "Cryogenic_Material_Properties"

KEYWORDS = re.compile(r"nano|nanocrystal|polycrystal|grain|ceramic|glass|wood|carbon|graphite|composite", re.I)


def read_table(path: Path) -> pd.DataFrame | None:
    for sep in [",", "\t", ";", r"\s+", r",|;|\t|\s+"]:
        try:
            df = pd.read_csv(path, engine="python", sep=sep)
            if df.shape[0] >= 6 and df.shape[1] >= 2:
                return df
        except Exception:
            pass
    return None


def numeric_series(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    out = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= max(6, len(df) // 3):
            out.append((str(col), s))
    return out


def choose_xy(df: pd.DataFrame) -> tuple[str, str, pd.Series, pd.Series] | None:
    nums = numeric_series(df)
    if len(nums) < 2:
        return None
    # Prefer a temperature-looking column as x, otherwise the first monotonic positive column.
    temp_idx = None
    for i, (name, s) in enumerate(nums):
        lname = name.lower()
        if "temp" in lname or lname in {"t", "t/k", "temperature", "temperature(k)"}:
            temp_idx = i
            break
    if temp_idx is None:
        for i, (_, s) in enumerate(nums):
            v = s.dropna().to_numpy(float)
            if len(v) >= 6 and np.all(v > 0) and (np.corrcoef(np.arange(len(v)), v)[0, 1] > 0.5):
                temp_idx = i
                break
    if temp_idx is None:
        temp_idx = 0
    xname, x = nums[temp_idx]
    y_candidates = [(n, s) for j, (n, s) in enumerate(nums) if j != temp_idx]
    # Prefer conductivity-looking y column; otherwise largest dynamic range.
    def y_score(item):
        n, s = item
        lname = n.lower()
        v = s.dropna().to_numpy(float)
        dyn = np.nanmax(v) / max(np.nanmin(v[v > 0]) if np.any(v > 0) else 1e-9, 1e-9)
        keyword = any(k in lname for k in ["kappa", "conduct", "w/m", "tc", "thermal"])
        return (1 if keyword else 0, dyn)
    yname, y = max(y_candidates, key=y_score)
    return xname, yname, x, y


def fit_low_t_alpha(x: pd.Series, y: pd.Series) -> dict | None:
    mask = x.notna() & y.notna() & (x > 0) & (y > 0)
    xx = x[mask].to_numpy(float)
    yy = y[mask].to_numpy(float)
    if len(xx) < 8:
        return None
    order = np.argsort(xx)
    xx, yy = xx[order], yy[order]
    # Use data below 50 K if available; otherwise lowest third.
    low = xx <= 50.0
    if low.sum() < 6:
        low = xx <= np.quantile(xx, 0.33)
    if low.sum() < 6:
        return None
    coeff = np.polyfit(np.log(xx[low]), np.log(yy[low]), 1)
    alpha = float(coeff[0])
    yhat = np.exp(coeff[1]) * xx[low] ** alpha
    ss_res = float(np.sum((np.log(yy[low]) - np.log(yhat)) ** 2))
    ss_tot = float(np.sum((np.log(yy[low]) - np.mean(np.log(yy[low]))) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "alpha": alpha,
        "n_lowT": int(low.sum()),
        "t_min": float(np.min(xx[low])),
        "t_max": float(np.max(xx[low])),
        "loglog_r2": r2,
        "close_to_0p5": bool(abs(alpha - 0.5) <= 0.2),
    }


def main() -> None:
    source_errors = []
    try:
        all_matches = github_find_blob(OWNER, REPO, r"thermal_conductivity/.+\.(csv|txt|dat)$")
    except Exception as e:
        all_matches = []
        source_errors.append({'source': f'https://github.com/{OWNER}/{REPO}', 'stage': 'github_tree', 'error': f'{type(e).__name__}: {e}'})
    # Put grain/nano candidates first, but also include general raw datasets so the result is not empty.
    matches = sorted(all_matches, key=lambda m: (0 if KEYWORDS.search(m["path"]) else 1, m["path"]))[:120]
    downloaded = []
    analyses = []
    for match in matches:
        dest = OUT / "data" / re.sub(r"[^A-Za-z0-9_.-]+", "_", match["path"])
        try:
            github_raw(match["raw_url"], dest)
            downloaded.append({"path": match["path"], "raw_url": match["raw_url"]})
            df = read_table(dest)
            if df is None:
                continue
            xy = choose_xy(df)
            if xy is None:
                continue
            xname, yname, x, y = xy
            fit = fit_low_t_alpha(x, y)
            if fit is None:
                continue
            analyses.append({
                "source_path": match["path"],
                "x_column": xname,
                "y_column": yname,
                "grain_or_nano_candidate": bool(KEYWORDS.search(match["path"])),
                **fit,
            })
        except Exception as e:
            continue

    candidates = [a for a in analyses if a["grain_or_nano_candidate"]]
    close = [a for a in candidates if a["close_to_0p5"]]
    status = "ok" if analyses else ("source_unavailable" if source_errors else "no_machine_readable_tables")
    verdict = (
        "no_physics_verdict" if source_errors and not analyses else
        "support_like" if close else
        "no_support_in_grain_candidates" if candidates else
        "no_grain_candidate_tables_found"
    )
    report = structured_report(
        "MAT3",
        status,
        prediction="low-T nanocrystalline/grain-sensitive thermal conductivity exponent alpha ≈ 0.5",
        repository=f"https://github.com/{OWNER}/{REPO}",
        source_errors=source_errors,
        n_downloaded=len(downloaded),
        n_analysed=len(analyses),
        n_grain_or_nano_candidates=len(candidates),
        verdict=verdict,
        criterion="support-like if any grain/nano candidate has |alpha-0.5| <= 0.2 with at least 6 low-T points",
        close_to_half=close,
        analyses=analyses[:60],
    )
    json_dump(report, OUT / "mat3_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
