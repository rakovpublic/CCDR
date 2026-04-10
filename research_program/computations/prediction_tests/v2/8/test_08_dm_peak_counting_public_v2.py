#!/usr/bin/env python3
"""
Test 08 — Dark-Matter Mass-Peak Counting from Direct Detection

Standalone public-data implementation.

Important note:
This script downloads all inputs from public sources, but it does NOT reconstruct
an exact joint unbinned recoil likelihood for XENONnT, LZ, and PandaX because a
single harmonized public likelihood is not available across all three experiments.
Instead, it performs a public-data peak scan in *DM-mass space* using the best
machine-readable products available from each experiment:

- published SI-WIMP observed/expected limit curves when available;
- public event / analysis tables when available, from which limit-like curves are
  inferred only if the format is recognizable.

The resulting "significances" are therefore proxy significances derived from a
stacked excess-score scan, not official collaboration likelihood significances.
This still gives a reproducible, fully public test that can fire on multi-peak
structure and estimate whether consecutive mass ratios look geometric.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths, savgol_filter

TIMEOUT = 120
UA = "Mozilla/5.0 (compatible; ChatGPT-Test08/1.0)"

# -----------------------------------------------------------------------------
# Public sources
# -----------------------------------------------------------------------------

LZ_HEAVY_SI_CSV_ZIP = "https://www.hepdata.net/download/submission/ins2841863/2/csv"
LZ_LIGHT_SI_CSV_ZIP = "https://www.hepdata.net/download/submission/ins3091049/2/csv"

PANDAX_FIRST_ANALYSIS_XLSX = (
    "https://static.pandax.sjtu.edu.cn/download/data-share/"
    "p4-first-analysis/PandaX4T_Data_ne.xlsx"
)
PANDAX_FIRST_ANALYSIS_EFF_ROOT = (
    "https://static.pandax.sjtu.edu.cn/download/data-share/"
    "p4-first-analysis/eff_RDQ_graph.root"
)
PANDAX_RUN0_CSV = (
    "https://static.pandax.sjtu.edu.cn/download/data-share/"
    "p4-light-dark-matter/run0_data.csv"
)
PANDAX_RUN1_CSV = (
    "https://static.pandax.sjtu.edu.cn/download/data-share/"
    "p4-light-dark-matter/run1_data.csv"
)
PANDAX_S2ONLY_ZIP = (
    "https://static.pandax.sjtu.edu.cn/download/data-share/"
    "p4-s2-only/s2only_data_release.zip"
)

XENONNT_LIGHT_WIMP_ZIP = (
    "https://codeload.github.com/XENONnT/light_wimp_data_release/zip/refs/heads/master"
)

PRED_RATIO_MIN = 10.0
PRED_RATIO_MAX = 100.0


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class Curve:
    experiment: str
    source_name: str
    mass_gev: np.ndarray
    observed: np.ndarray
    expected: np.ndarray | None = None
    provenance: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def cleaned(self) -> "Curve":
        mask = np.isfinite(self.mass_gev) & np.isfinite(self.observed)
        if self.expected is not None:
            mask &= np.isfinite(self.expected)
        mass = np.asarray(self.mass_gev[mask], dtype=float)
        obs = np.asarray(self.observed[mask], dtype=float)
        exp = None if self.expected is None else np.asarray(self.expected[mask], dtype=float)
        order = np.argsort(mass)
        mass = mass[order]
        obs = obs[order]
        exp = None if exp is None else exp[order]
        # drop non-physical or duplicate masses
        keep = (mass > 1e-3) & (mass < 1e8) & (obs > 0)
        if exp is not None:
            keep &= exp > 0
        mass = mass[keep]
        obs = obs[keep]
        exp = None if exp is None else exp[keep]
        if mass.size == 0:
            return Curve(self.experiment, self.source_name, mass, obs, exp, self.provenance, self.metadata)
        uniq_mass, idx = np.unique(mass, return_index=True)
        obs = obs[idx]
        exp = None if exp is None else exp[idx]
        return Curve(self.experiment, self.source_name, uniq_mass, obs, exp, self.provenance, self.metadata)


@dataclass
class Peak:
    mass_gev: float
    width_mev: float
    significance_sigma: float
    amplitude: float


@dataclass
class AnalysisResult:
    n_peaks: int
    peak_masses_gev: list[float]
    peak_significances: list[float]
    consecutive_ratios: list[float]
    ratios_geometric: bool
    inferred_N: int | None
    pass_v6_cascade: bool
    pass_v5_single_step: bool
    m5_lower_limit_gev: float | None
    analysis_mode: str
    sources_loaded: list[str]
    sources_skipped: list[str]
    caveats: list[str]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def download(url: str, dest: Path, overwrite: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        return dest
    with session().get(url, timeout=TIMEOUT, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                if chunk:
                    f.write(chunk)
    return dest


def read_json_url(url: str) -> Any:
    with session().get(url, timeout=TIMEOUT) as r:
        r.raise_for_status()
        return r.json()


def download_zip_and_extract(url: str, dest_zip: Path, extract_dir: Path) -> Path:
    download(url, dest_zip)
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".done"
    if marker.exists():
        return extract_dir
    with zipfile.ZipFile(dest_zip) as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")
    return extract_dir


def to_float(x: Any) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return np.nan
    s = s.replace(",", "")
    s = s.replace("×", "e")
    s = s.replace("−", "-")
    # convert 1e-48-like superscript-ish patterns if present
    m = re.match(r"^([+-]?[0-9.]+)e\^?([+-]?[0-9]+)$", s)
    if m:
        return float(m.group(1)) * (10.0 ** float(m.group(2)))
    # convert 2.2*10^{-48}
    m = re.match(r"^([+-]?[0-9.]+)\s*\*\s*10\^\{?([+-]?[0-9]+)\}?$", s)
    if m:
        return float(m.group(1)) * (10.0 ** float(m.group(2)))
    try:
        return float(s)
    except ValueError:
        return np.nan


def extract_numeric_from_hepdata_value(v: Any) -> float:
    if isinstance(v, dict):
        if "value" in v:
            return to_float(v["value"])
        low = to_float(v.get("low"))
        high = to_float(v.get("high"))
        if np.isfinite(low) and np.isfinite(high):
            # geometric midpoint if both positive, else arithmetic
            if low > 0 and high > 0:
                return float(np.sqrt(low * high))
            return float(0.5 * (low + high))
        for key in ("mid", "x", "y"):
            if key in v:
                return to_float(v[key])
    return to_float(v)


def recursive_find(obj: Any, wanted_keys: set[str]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if wanted_keys.issubset(obj.keys()):
            found.append(obj)
        for value in obj.values():
            found.extend(recursive_find(value, wanted_keys))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(recursive_find(item, wanted_keys))
    return found


def parse_hepdata_table_obj(obj: Any, experiment: str, source_name: str, provenance: str) -> Curve | None:
    tables = recursive_find(obj, {"independent_variables", "dependent_variables"})
    if not tables:
        return None
    table = tables[0]
    indeps = table["independent_variables"]
    deps = table["dependent_variables"]
    if not indeps or not deps:
        return None

    # choose independent variable most likely to be mass
    indep_idx = 0
    for i, var in enumerate(indeps):
        header = str(var.get("header", {}).get("name", "")).lower()
        if "mass" in header or "gev" in header:
            indep_idx = i
            break
    xvar = indeps[indep_idx]
    mass = np.array([extract_numeric_from_hepdata_value(v) for v in xvar.get("values", [])], dtype=float)
    if mass.size == 0:
        return None

    dep_map: dict[str, np.ndarray] = {}
    for dep in deps:
        header = dep.get("header", {})
        name = str(header.get("name", header.get("title", "unnamed")))
        vals = np.array([extract_numeric_from_hepdata_value(v) for v in dep.get("values", [])], dtype=float)
        if vals.size == mass.size:
            dep_map[name] = vals

    if not dep_map:
        return None

    def pick(patterns: Iterable[str]) -> tuple[str, np.ndarray] | None:
        for key, vals in dep_map.items():
            low = key.lower()
            if any(p in low for p in patterns):
                return key, vals
        return None

    observed = pick(["observed", "obs"])
    expected = pick(["expected", "median", "sensitivity"])

    if observed is None:
        # fallback: choose the dependent series with largest spread
        candidates = sorted(dep_map.items(), key=lambda kv: np.nanstd(np.log10(np.clip(kv[1], 1e-300, None))), reverse=True)
        observed = candidates[0]
    observed_name, obs = observed
    exp = expected[1] if expected is not None else None
    curve = Curve(
        experiment=experiment,
        source_name=source_name,
        mass_gev=mass,
        observed=obs,
        expected=exp,
        provenance=provenance,
        metadata={"observed_name": observed_name, "available_columns": list(dep_map.keys())},
    ).cleaned()
    if curve.mass_gev.size < 5:
        return None
    return curve


def parse_hepdata_url(url: str, experiment: str, source_name: str) -> Curve | None:
    obj = read_json_url(url)
    curve = parse_hepdata_table_obj(obj, experiment=experiment, source_name=source_name, provenance=url)
    return curve


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_curve_from_dataframe(df: pd.DataFrame, experiment: str, source_name: str, provenance: str) -> Curve | None:
    df = _normalize_columns(df)
    if df.empty:
        return None
    numeric_cols = []
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        frac = np.isfinite(vals).mean() if len(vals) else 0.0
        if frac > 0.7:
            numeric_cols.append(c)
    if len(numeric_cols) < 2:
        return None

    work = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    best_mass = None
    best_score = -np.inf
    for c in work.columns:
        vals = work[c].to_numpy(dtype=float)
        mask = np.isfinite(vals) & (vals > 0)
        if mask.sum() < 5:
            continue
        v = vals[mask]
        monotonic = float(np.mean(np.diff(v) > 0)) if v.size > 1 else 0.0
        spread = np.log10(np.nanmax(v) / np.nanmin(v)) if np.nanmin(v) > 0 else 0.0
        score = 2.0 * monotonic + spread
        name = c.lower()
        if "mass" in name or "gev" in name or "mχ" in name or "mchi" in name or name == "x":
            score += 3.0
        if score > best_score:
            best_score = score
            best_mass = c
    if best_mass is None:
        return None

    lower_names = {c.lower(): c for c in work.columns if c != best_mass}
    obs_col = None
    exp_col = None
    for low, orig in lower_names.items():
        if any(k in low for k in ["observed", "obs"]):
            obs_col = orig
            break
    for low, orig in lower_names.items():
        if any(k in low for k in ["expected", "median", "sensitivity"]):
            exp_col = orig
            break

    remaining = [c for c in work.columns if c != best_mass]
    if obs_col is None:
        # choose physically plausible positive curve column
        candidates: list[tuple[float, str]] = []
        for c in remaining:
            y = work[c].to_numpy(dtype=float)
            mask = np.isfinite(y) & (y > 0)
            if mask.sum() < 5:
                continue
            yy = y[mask]
            # prefer wide positive dynamic range and same length as mass
            dyn = np.log10(np.nanmax(yy) / np.nanmin(yy)) if np.nanmin(yy) > 0 else 0.0
            candidates.append((dyn, c))
        if not candidates:
            return None
        obs_col = sorted(candidates, reverse=True)[0][1]
    if exp_col == obs_col:
        exp_col = None

    curve = Curve(
        experiment=experiment,
        source_name=source_name,
        mass_gev=work[best_mass].to_numpy(dtype=float),
        observed=work[obs_col].to_numpy(dtype=float),
        expected=None if exp_col is None else work[exp_col].to_numpy(dtype=float),
        provenance=provenance,
        metadata={"mass_col": best_mass, "observed_col": obs_col, "expected_col": exp_col},
    ).cleaned()
    if curve.mass_gev.size < 5:
        return None
    return curve


def parse_tabular_file(path: Path, experiment: str, source_name: str) -> list[Curve]:
    curves: list[Curve] = []
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".txt", ".tsv"}:
            seps = [",", "\t", r"\s+"]
            for sep in seps:
                try:
                    df = pd.read_csv(path, sep=sep, engine="python")
                    curve = infer_curve_from_dataframe(df, experiment, source_name, str(path))
                    if curve is not None:
                        curves.append(curve)
                        break
                except Exception:
                    continue
        elif suffix in {".xlsx", ".xls"}:
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    curve = infer_curve_from_dataframe(df, experiment, f"{source_name}:{sheet}", str(path))
                    if curve is not None:
                        curves.append(curve)
                except Exception:
                    continue
        elif suffix in {".json", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            obj = json.loads(text)
            curve = parse_hepdata_table_obj(obj, experiment, source_name, str(path))
            if curve is not None:
                curves.append(curve)
    except Exception:
        return curves
    return curves


def choose_best(curves: list[Curve]) -> Curve | None:
    if not curves:
        return None
    def score(c: Curve) -> tuple[float, float, float]:
        pts = float(c.mass_gev.size)
        span = float(np.log10(np.nanmax(c.mass_gev) / np.nanmin(c.mass_gev))) if c.mass_gev.size > 1 else 0.0
        has_exp = 1.0 if c.expected is not None else 0.0
        return (has_exp, pts, span)
    return sorted(curves, key=score, reverse=True)[0]


# -----------------------------------------------------------------------------
# Source loaders
# -----------------------------------------------------------------------------

def load_lz(cache_dir: Path) -> tuple[list[Curve], list[str]]:
    curves: list[Curve] = []
    skipped: list[str] = []
    for name, url in [
        ("LZ_4.2tyr_SI", LZ_HEAVY_SI_CSV_ZIP),
        ("LZ_5.7tyr_light_SI", LZ_LIGHT_SI_CSV_ZIP),
    ]:
        try:
            extract_dir = download_zip_and_extract(url, cache_dir / f"{name}.zip", cache_dir / name)
            parsed: list[Curve] = []
            for fp in extract_dir.rglob("*.csv"):
                low = fp.name.lower()
                if not ("si" in low and ("cross" in low or "t1" in low)):
                    continue
                parsed.extend(parse_tabular_file(fp, "LZ", f"{name}/{fp.name}"))
            best = choose_best(parsed)
            if best is None:
                skipped.append(f"{name}: downloaded HEPData CSV bundle but no usable SI curve found")
            else:
                curves.append(best)
        except Exception as e:
            skipped.append(f"{name}: {e}")
    return curves, skipped


def load_pandax(cache_dir: Path) -> tuple[list[Curve], list[str]]:
    curves: list[Curve] = []
    skipped: list[str] = []

    try:
        local = download(PANDAX_FIRST_ANALYSIS_XLSX, cache_dir / "PandaX4T_first_analysis.xlsx")
        found = parse_tabular_file(local, "PandaX", "PandaX4T_first_analysis_xlsx")
        best = choose_best(found)
        if best is not None:
            curves.append(best)
        else:
            skipped.append("PandaX4T_first_analysis_xlsx: downloaded but no usable mass-limit curve found")
    except Exception as e:
        skipped.append(f"PandaX4T_first_analysis_xlsx: {e}")

    for name, url in [
        ("PandaX4T_first_analysis_eff_root", PANDAX_FIRST_ANALYSIS_EFF_ROOT),
        ("PandaX4T_lightDM_run0", PANDAX_RUN0_CSV),
        ("PandaX4T_lightDM_run1", PANDAX_RUN1_CSV),
        ("PandaX4T_run0_S2only_zip", PANDAX_S2ONLY_ZIP),
    ]:
        try:
            local = download(url, cache_dir / Path(url).name)
            if local.suffix.lower() == ".zip":
                skipped.append(f"{name}: downloaded event-level archive; not converted to a mass-limit curve in mass-space scan mode")
            elif local.suffix.lower() == ".root":
                skipped.append(f"{name}: downloaded ROOT efficiency file; not converted to a mass-limit curve in mass-space scan mode")
            else:
                skipped.append(f"{name}: downloaded event-level table; not directly used in mass-space scan mode")
        except Exception as e:
            skipped.append(f"{name}: {e}")
    return curves, skipped


def load_xenonnt(cache_dir: Path) -> tuple[list[Curve], list[str]]:
    curves: list[Curve] = []
    skipped: list[str] = []
    name = "XENONnT_light_wimp_repo"
    try:
        extract_dir = download_zip_and_extract(XENONNT_LIGHT_WIMP_ZIP, cache_dir / f"{name}.zip", cache_dir / name)
        candidate_files: list[Path] = []
        for fp in extract_dir.rglob("*"):
            if not fp.is_file():
                continue
            low = str(fp).lower()
            if fp.suffix.lower() not in {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".json", ".yaml", ".yml"}:
                continue
            if "/limits/" not in low and "\\limits\\" not in low:
                continue
            if any(k in low for k in ["mirror", "mddm", "template", "background", "signal", "notebook", "readme"]):
                continue
            if not any(k in low for k in ["wimp", "si", "limit"]):
                continue
            candidate_files.append(fp)
        parsed: list[Curve] = []
        for fp in candidate_files:
            parsed.extend(parse_tabular_file(fp, "XENONnT", f"{name}/{fp.relative_to(extract_dir)}"))
        parsed = sorted(
            parsed,
            key=lambda c: (("limits" in c.source_name.lower()) + ("si" in c.source_name.lower()) + ("wimp" in c.source_name.lower()), c.mass_gev.size),
            reverse=True,
        )
        best = choose_best(parsed)
        if best is not None:
            curves.append(best)
        else:
            skipped.append(f"{name}: downloaded repo but no usable published limit curve found under limits/")
    except Exception as e:
        skipped.append(f"{name}: {e}")
    return curves, skipped


def smooth_baseline(logy: np.ndarray, window_frac: float = 0.25) -> np.ndarray:
    n = len(logy)
    if n < 7:
        return np.full_like(logy, np.nanmedian(logy))
    w = max(7, int(n * window_frac) | 1)
    if w >= n:
        w = n - 1 if n % 2 == 0 else n
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = max(5, n - 1 if n % 2 == 0 else n)
    poly = 3 if w >= 7 else 2
    return savgol_filter(logy, window_length=w, polyorder=poly, mode="interp")


def curve_score(curve: Curve) -> tuple[np.ndarray, np.ndarray]:
    x = np.log10(curve.mass_gev)
    log_obs = np.log10(curve.observed)
    if curve.expected is not None and np.all(np.isfinite(curve.expected)) and np.all(curve.expected > 0):
        score = log_obs - np.log10(curve.expected)
    else:
        base = smooth_baseline(log_obs)
        score = log_obs - base
    # de-noise lightly
    score = gaussian_filter1d(score, sigma=max(0.8, 0.01 * len(score)))
    return x, score


def combine_scores(curves: list[Curve], ngrid: int = 800) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not curves:
        raise ValueError("No curves to combine")
    xmin = min(np.log10(np.nanmin(c.mass_gev)) for c in curves)
    xmax = max(np.log10(np.nanmax(c.mass_gev)) for c in curves)
    grid = np.linspace(xmin, xmax, ngrid)
    mats = []
    info = []
    for c in curves:
        x, s = curve_score(c)
        f = interp1d(x, s, bounds_error=False, fill_value=np.nan)
        sg = f(grid)
        mats.append(sg)
        info.append({"source": f"{c.experiment}:{c.source_name}", "mass_min": 10**x.min(), "mass_max": 10**x.max()})
    arr = np.vstack(mats)
    combined = np.nanmean(arr, axis=0)
    # fill small gaps from interpolation boundaries
    mask = np.isfinite(combined)
    if mask.sum() < 10:
        raise ValueError("Combined score grid has too few finite points")
    combined = np.interp(np.arange(len(combined)), np.where(mask)[0], combined[mask])
    combined = gaussian_filter1d(combined, sigma=2.0)
    return grid, combined, info


def gaussian_sum(x: np.ndarray, params: np.ndarray, k: int) -> np.ndarray:
    y = np.zeros_like(x)
    for i in range(k):
        amp = params[3 * i + 0]
        mu = params[3 * i + 1]
        sig = params[3 * i + 2]
        y += amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    return y


def fit_k_peaks(x: np.ndarray, y: np.ndarray, peak_idx: np.ndarray, k: int) -> tuple[np.ndarray, float, float]:
    peak_idx = np.asarray(peak_idx, dtype=int)
    if peak_idx.size < k:
        raise ValueError("Not enough candidate peaks")
    order = np.argsort(y[peak_idx])[::-1][:k]
    sel = np.sort(peak_idx[order])
    p0 = []
    lb = []
    ub = []
    for idx in sel:
        amp = max(y[idx], 1e-4)
        mu = x[idx]
        sig = 0.05
        p0.extend([amp, mu, sig])
        lb.extend([0.0, x.min(), 0.01])
        ub.extend([10.0, x.max(), 0.5])
    p0 = np.array(p0, dtype=float)
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    def resid(p: np.ndarray) -> np.ndarray:
        return gaussian_sum(x, p, k) - y

    res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=10000)
    rss = float(np.sum(res.fun ** 2))
    n = len(x)
    p = len(res.x)
    rss = max(rss, 1e-15)
    aic = n * math.log(rss / n) + 2 * p
    bic = n * math.log(rss / n) + p * math.log(n)
    return res.x, aic, bic


def detect_peaks(x: np.ndarray, y: np.ndarray, min_sigma: float = 3.0) -> tuple[list[Peak], dict[str, Any]]:
    # noise estimate from robust residuals
    baseline = smooth_baseline(y, window_frac=0.30)
    resid = y - baseline
    noise = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    noise = max(noise, 1e-4)
    prom = max(0.03, 2.0 * noise)
    peaks_idx, props = find_peaks(y, prominence=prom, distance=max(5, len(y) // 20))
    if len(peaks_idx) == 0:
        return [], {"noise": noise, "best_k": 0, "aic": {}, "bic": {}}

    aics: dict[int, float] = {}
    bics: dict[int, float] = {}
    fits: dict[int, np.ndarray] = {}
    maxk = min(4, len(peaks_idx))
    for k in range(1, maxk + 1):
        try:
            params, aic, bic = fit_k_peaks(x, y, peaks_idx, k)
            fits[k] = params
            aics[k] = aic
            bics[k] = bic
        except Exception:
            continue
    if not bics:
        return [], {"noise": noise, "best_k": 0, "aic": {}, "bic": {}}
    best_k = min(bics, key=bics.get)
    params = fits[best_k]
    # convert fitted peaks to Peak objects
    peaks: list[Peak] = []
    for i in range(best_k):
        amp = float(params[3 * i + 0])
        mu = float(params[3 * i + 1])
        sig = float(params[3 * i + 2])
        mass = 10 ** mu
        # FWHM in log10(m); convert to approximate absolute width in MeV.
        m_lo = 10 ** (mu - math.sqrt(2 * math.log(2)) * sig)
        m_hi = 10 ** (mu + math.sqrt(2 * math.log(2)) * sig)
        width_mev = max(0.0, (m_hi - m_lo) * 1000.0)
        significance = amp / noise
        if significance >= min_sigma:
            peaks.append(Peak(mass_gev=mass, width_mev=width_mev, significance_sigma=significance, amplitude=amp))
    peaks.sort(key=lambda p: p.mass_gev)
    return peaks, {"noise": noise, "best_k": best_k, "aic": aics, "bic": bics, "fit_params": params.tolist()}


def infer_m5_lower_limit(grid_logm: np.ndarray, score: np.ndarray, peaks: list[Peak], threshold_sigma: float, noise: float) -> float | None:
    if len(peaks) == 0:
        # mass below which a first peak is excluded is ill-defined here; return highest scanned mass with no peak.
        return float(10 ** grid_logm.max())
    last_peak_mass = peaks[-1].mass_gev
    mask = 10 ** grid_logm > last_peak_mass
    if not np.any(mask):
        return None
    future_m = 10 ** grid_logm[mask]
    future_s = score[mask]
    thr = threshold_sigma * noise
    above = future_s > thr
    if np.any(above):
        idx = np.argmax(above)
        return float(future_m[idx])
    return float(future_m.max())


# -----------------------------------------------------------------------------
# Plotting and reporting
# -----------------------------------------------------------------------------

def plot_result(outpath: Path, grid_logm: np.ndarray, score: np.ndarray, peaks: list[Peak], curve_grid: list[tuple[str, np.ndarray]], ratios: list[float]) -> None:
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.10)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    mass = 10 ** grid_logm
    for label, y in curve_grid:
        ax.plot(mass, y, alpha=0.35, linewidth=1.2, label=label)
    ax.plot(mass, score, linewidth=2.4, label="Combined excess score")
    for p in peaks:
        ax.axvline(p.mass_gev, linestyle="--", linewidth=1)
        ax.annotate(f"{p.mass_gev:.1f} GeV\n{p.significance_sigma:.1f}σ",
                    xy=(p.mass_gev, np.interp(np.log10(p.mass_gev), grid_logm, score)),
                    xytext=(4, 10), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_ylabel("Excess score")
    ax.set_title("Test 08 public-data stacked DM-mass scan")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    if ratios:
        xrat = [peaks[i].mass_gev for i in range(len(ratios))]
        ax2.scatter(xrat, ratios)
        ax2.axhspan(PRED_RATIO_MIN, PRED_RATIO_MAX, alpha=0.20)
        for xm, r in zip(xrat, ratios):
            ax2.annotate(f"{r:.1f}", (xm, r), textcoords="offset points", xytext=(4, 6), fontsize=8)
        ax2.set_yscale("log")
    else:
        ax2.text(0.5, 0.5, "Need ≥2 peaks for ratio test", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xscale("log")
    ax2.set_xlabel("DM mass [GeV]")
    ax2.set_ylabel("mₖ₊₁ / mₖ")
    ax2.grid(True, alpha=0.25)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, result: AnalysisResult, curves: list[Curve], diag: dict[str, Any]) -> None:
    lines = []
    lines.append("# Test 08 — Dark-Matter Mass-Peak Counting from Direct Detection")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Analysis mode: **{result.analysis_mode}**")
    lines.append(f"- Loaded sources: **{len(result.sources_loaded)}**")
    lines.append(f"- Skipped sources: **{len(result.sources_skipped)}**")
    lines.append(f"- Significant peaks found: **{result.n_peaks}**")
    lines.append(f"- Inferred N: **{result.inferred_N}**")
    lines.append(f"- v6 cascade pass: **{result.pass_v6_cascade}**")
    lines.append(f"- v5 single-step pass: **{result.pass_v5_single_step}**")
    lines.append("")
    lines.append("## Peak summary")
    lines.append("")
    if result.n_peaks:
        for i, (m, s) in enumerate(zip(result.peak_masses_gev, result.peak_significances), start=1):
            lines.append(f"- Peak {i}: {m:.4g} GeV, proxy significance {s:.2f}σ")
    else:
        lines.append("- No ≥3σ proxy peaks in the combined public-data mass scan.")
    lines.append("")
    lines.append("## Consecutive ratios")
    lines.append("")
    if result.consecutive_ratios:
        for i, r in enumerate(result.consecutive_ratios, start=1):
            ok = PRED_RATIO_MIN <= r <= PRED_RATIO_MAX
            lines.append(f"- Ratio {i}: {r:.4g} ({'inside' if ok else 'outside'} 10–100 band)")
    else:
        lines.append("- Fewer than 2 peaks, so no ratio test can fire.")
    lines.append("")
    lines.append("## Loaded curves")
    lines.append("")
    for c in curves:
        lines.append(
            f"- {c.experiment} / {c.source_name}: {len(c.mass_gev)} points, "
            f"mass range {np.min(c.mass_gev):.4g}–{np.max(c.mass_gev):.4g} GeV"
        )
    if result.sources_skipped:
        lines.append("")
        lines.append("## Skipped / partial sources")
        lines.append("")
        for s in result.sources_skipped:
            lines.append(f"- {s}")
    if result.caveats:
        lines.append("")
        lines.append("## Caveats")
        lines.append("")
        for c in result.caveats:
            lines.append(f"- {c}")
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(diag, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Test 08 public-data DM peak counting")
    ap.add_argument("--cache-dir", default="test08_cache", help="Download/cache directory")
    ap.add_argument("--outdir", default="test08_out", help="Output directory")
    ap.add_argument("--min-sigma", type=float, default=3.0, help="Minimum proxy significance for a counted peak")
    ap.add_argument("--strict", action="store_true", help="Fail if fewer than one usable curve per experiment are found")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_curves: list[Curve] = []
    skipped: list[str] = []

    loaders = [load_xenonnt, load_lz, load_pandax]
    for loader in loaders:
        curves, bad = loader(cache_dir)
        all_curves.extend(curves)
        skipped.extend(bad)

    exp_counts = {}
    for c in all_curves:
        exp_counts[c.experiment] = exp_counts.get(c.experiment, 0) + 1

    if args.strict:
        missing = [exp for exp in ["XENONnT", "LZ", "PandaX"] if exp_counts.get(exp, 0) == 0]
        if missing:
            raise RuntimeError(f"Strict mode failed: no usable curve for {missing}")

    if not all_curves:
        raise RuntimeError("No usable public curves were extracted from any source")

    grid_logm, combined_score, info = combine_scores(all_curves)
    peaks, diag = detect_peaks(grid_logm, combined_score, min_sigma=args.min_sigma)

    ratios = [peaks[i + 1].mass_gev / peaks[i].mass_gev for i in range(len(peaks) - 1)]
    ratios_geometric = bool(ratios) and all(PRED_RATIO_MIN <= r <= PRED_RATIO_MAX for r in ratios)
    noise = float(diag.get("noise", np.nan))
    m5_lower_limit = infer_m5_lower_limit(grid_logm, combined_score, peaks, threshold_sigma=args.min_sigma, noise=noise)

    pass_v6 = len(peaks) >= 2 and ratios_geometric
    pass_v5 = (len(peaks) == 1 and m5_lower_limit is not None and m5_lower_limit >= 1.0e4)

    analysis_mode = "stacked public-data mass-space residual scan"
    caveats = [
        "This is a public-data proxy analysis in DM-mass space, not the collaborations' full joint unbinned recoil likelihood.",
        "Peak significances are derived from a stacked excess-score scan built from public observed/expected curves or detrended public limit curves.",
        "If a source did not expose a directly machine-readable SI-WIMP curve, it was downloaded and logged as skipped or partial rather than silently ignored.",
    ]

    result = AnalysisResult(
        n_peaks=len(peaks),
        peak_masses_gev=[float(p.mass_gev) for p in peaks],
        peak_significances=[float(p.significance_sigma) for p in peaks],
        consecutive_ratios=[float(r) for r in ratios],
        ratios_geometric=bool(ratios_geometric),
        inferred_N=(4 + len(peaks)) if len(peaks) > 0 else None,
        pass_v6_cascade=bool(pass_v6),
        pass_v5_single_step=bool(pass_v5),
        m5_lower_limit_gev=None if m5_lower_limit is None else float(m5_lower_limit),
        analysis_mode=analysis_mode,
        sources_loaded=[f"{c.experiment}:{c.source_name}" for c in all_curves],
        sources_skipped=skipped,
        caveats=caveats,
    )

    # Plot individual source curves on same grid
    curve_grid = []
    for c in all_curves:
        x, s = curve_score(c)
        f = interp1d(x, s, bounds_error=False, fill_value=np.nan)
        sg = f(grid_logm)
        curve_grid.append((f"{c.experiment}:{c.source_name}", sg))
    plot_result(outdir / "combined_dm_mass_scan.png", grid_logm, combined_score, peaks, curve_grid, ratios)

    (outdir / "result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    write_report(outdir / "report.md", result, all_curves, diag)

    print(json.dumps(asdict(result), indent=2))
    print(f"Wrote: {outdir / 'result.json'}")
    print(f"Wrote: {outdir / 'report.md'}")
    print(f"Wrote: {outdir / 'combined_dm_mass_scan.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
