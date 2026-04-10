#!/usr/bin/env python3
"""
Test 05 — Pulsar Timing Dark-Matter Spectral Comb

Standalone script that downloads public pulsar-timing products itself,
constructs an aggregate residual-variance time series from the best-timed
pulsars, searches for periodic modulation lines in that variance envelope,
and writes the requested JSON summary plus a diagnostic plot.

Primary public source:
  * NANOGrav 15-year public dataset (Zenodo record 16051178, v2.1.0)
    including ASCII post-fit timing residual tables for narrowband data.

Optional cross-check source:
  * EPTA DR2 (Zenodo record 8300645).

Important caveat:
  The exact internal layout of the public archives may evolve. This script is
  therefore intentionally defensive: it scans the extracted archive for files
  whose names look like timing-residual tables instead of assuming one fixed
  path.

The statistical test implemented here is an operationalization of the user's
specification, not an established PTA dark-matter pipeline. In particular:
  * the search is performed on slow modulation of the residual-variance
    envelope, since direct beat frequencies from particle masses are far above
    PTA timing bands;
  * significance is estimated from circular-shift null simulations of the
    per-pulsar variance series;
  * comparison against RVM cooling predictions requires either explicit input
    ratios (--predicted-ratios) or a user-chosen geometric ratio
    (--geometric-ratio).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

SECONDS_PER_DAY = 86400.0
DEFAULT_NANOGRAV_RECORD_ID = "16051178"
DEFAULT_NANOGRAV_FILENAME = "NANOGrav15yr_PulsarTiming_v2.1.0.tar.gz"
DEFAULT_EPTA_RECORD_ID = "8300645"
DEFAULT_EPTA_FILENAME = "EPTA-DR2.zip"
PSR_RE = re.compile(r"J\d{4}[+-]\d{4}")
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".npz", ".hdf5", ".h5", ".fits", ".fit", ".pickle", ".pkl", ".ps", ".eps", ".gz", ".xz", ".bz2"}
PREFERRED_TEXT_SUFFIXES = {".txt", ".dat", ".csv", ".tim", ".res", ".out", ".asc", ".ecsv"}
SKIP_PATH_TOKENS = {"readme", "overview", "description", "corr", "correlation", "chain", "clock", "template", "profile", "plot", "figure"}


@dataclass
class ResidualSeries:
    pulsar: str
    mjd: np.ndarray
    residual: np.ndarray
    error: np.ndarray
    source_path: Path
    whitened: bool
    epoch_averaged: bool

    @property
    def wrms(self) -> float:
        w = 1.0 / np.maximum(self.error, 1e-30) ** 2
        return float(np.sqrt(np.sum(w * self.residual**2) / np.sum(w)))


@dataclass
class VarianceSeries:
    pulsar: str
    bin_center_mjd: np.ndarray
    variance: np.ndarray
    valid: np.ndarray
    wrms: float


@dataclass
class Coord:
    ra_deg: float
    dec_deg: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("test05_output"))
    p.add_argument("--cache-dir", type=Path, default=Path("test05_cache"))
    p.add_argument("--top-n", type=int, default=30, help="Number of best-timed pulsars to use.")
    p.add_argument(
        "--bin-days",
        type=float,
        default=60.0,
        help="Bin size in days for the variance-envelope time series.",
    )
    p.add_argument(
        "--min-points-per-bin",
        type=int,
        default=2,
        help="Minimum number of residual points required for a within-bin variance estimate.",
    )
    p.add_argument(
        "--min-pulsars-per-bin",
        type=int,
        default=8,
        help="Minimum number of pulsars contributing to an aggregate bin.",
    )
    p.add_argument(
        "--bootstrap-sims",
        type=int,
        default=300,
        help="Number of circular-shift null simulations for 1%% FAP estimation.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=12345,
        help="Random seed for null simulations.",
    )
    p.add_argument(
        "--predicted-ratios",
        type=str,
        default=None,
        help=(
            "Comma-separated expected frequency ratios relative to the lowest line, "
            "e.g. '1,1.8,3.2'. Preferred when you have an explicit RVM cooling prediction."
        ),
    )
    p.add_argument(
        "--geometric-ratio",
        type=float,
        default=None,
        help=(
            "Alternative to --predicted-ratios: assume a geometric comb with ratio q and "
            "compare detected ratios against [1, q, q^2, ...]."
        ),
    )
    p.add_argument(
        "--ratio-tolerance",
        type=float,
        default=0.30,
        help="Allowed fractional mismatch when comparing measured and predicted ratios.",
    )
    p.add_argument(
        "--include-epta-crosscheck",
        action="store_true",
        help="Attempt the same archive scan on EPTA DR2 as an optional cross-check.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def zenodo_download_url(record_id: str, filename: str) -> str:
    return f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"


def download_file(url: str, dest: Path, chunk_size: int = 2**20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logging.info("Using cached download: %s", dest)
        return dest
    logging.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100.0 * written / total
                    logging.info("Downloaded %.1f%% of %s", pct, dest.name)
    return dest


def extract_archive(archive_path: Path, extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    marker = extract_root / ".extracted.ok"
    expected = [extract_root / "README", extract_root / "narrowband", extract_root / "wideband"]
    if marker.exists() and any(p.exists() for p in expected):
        logging.info("Using cached extraction: %s", extract_root)
        return extract_root
    if marker.exists() and not any(p.exists() for p in expected):
        logging.warning("Extraction marker exists but expected contents are missing; re-extracting %s", archive_path)
        marker.unlink(missing_ok=True)
    logging.info("Extracting %s", archive_path)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_root)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_root)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    marker.write_text("ok\n", encoding="utf-8")
    return extract_root


def is_textish_candidate(path: Path) -> bool:
    s = str(path).lower()
    if path.suffix.lower() in BINARY_SUFFIXES:
        return False
    if any(tok in s for tok in SKIP_PATH_TOKENS):
        return False
    if path.stat().st_size == 0 or path.stat().st_size > 25 * 1024 * 1024:
        return False
    return True


def has_residual_name_hint(path: Path) -> bool:
    s = str(path).lower()
    return any(tok in s for tok in ["resid", "residual", "postfit", "post-fit", "timing_model"])


def is_residual_candidate(path: Path) -> bool:
    """Broad first-pass filter for possible residual tables.

    We keep this intentionally permissive and let parse_residual_file do the
    content-based validation, because archive naming/layout varies across
    public PTA releases.
    """
    return is_textish_candidate(path)


def score_residual_candidate(path: Path) -> Tuple[int, int, int, int, str]:
    s = str(path).lower()
    epoch = int(any(tok in s for tok in ["epoch", "avg", "average"]))
    white = int(any(tok in s for tok in ["white", "whiten"]))
    narrow = int("narrowband" in s)
    hint = int(has_residual_name_hint(path))
    suffix_bonus = int(path.suffix.lower() in PREFERRED_TEXT_SUFFIXES)
    return (hint, epoch, white, narrow + suffix_bonus, s)


def detect_pulsar_name(path: Path) -> Optional[str]:
    for piece in [path.name, *path.parts[::-1]]:
        m = PSR_RE.search(piece)
        if m:
            return m.group(0)
    return None


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def parse_table_text(text: str) -> pd.DataFrame:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty file")

    non_comment = [ln for ln in lines if not ln.lstrip().startswith("#")]
    if not non_comment:
        raise ValueError("No non-comment lines")

    header_is_present = bool(re.search(r"[A-Za-z]", non_comment[0]))
    payload = "\n".join(lines)

    for kwargs in [
        dict(sep=r"\s+", engine="python", comment="#", header=0 if header_is_present else None),
        dict(delim_whitespace=True, engine="python", comment="#", header=0 if header_is_present else None),
    ]:
        try:
            df = pd.read_csv(io.StringIO(payload), **kwargs)
            if df.shape[1] >= 2 and len(df) >= 2:
                return df
        except Exception:
            pass

    arr = np.genfromtxt(io.StringIO(payload), comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Unable to parse table")
    return pd.DataFrame(arr)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for col in out.columns:
        try:
            out[col] = pd.to_numeric(out[col])
        except Exception:
            pass
    return out


def choose_column(columns: Sequence[str], preferred_patterns: Sequence[str], excluded_patterns: Sequence[str] = ()) -> Optional[str]:
    lowered = {c: c.lower() for c in columns}
    for pattern in preferred_patterns:
        regex = re.compile(pattern)
        for col, low in lowered.items():
            if regex.search(low) and not any(re.search(ex, low) for ex in excluded_patterns):
                return col
    return None


def infer_time_residual_error_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    cols = list(df.columns)
    time_col = choose_column(cols, [r"\bmjd\b", r"time", r"date", r"epoch", r"day"])
    resid_col = choose_column(cols, [r"resid", r"postfit", r"post_fit", r"dt"], excluded_patterns=[r"err", r"unc", r"sigma", r"var", r"rms"])
    err_col = choose_column(cols, [r"err", r"unc", r"sigma", r"toa.*err", r"resid.*err"])

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if time_col is None:
        monotonic_candidates = []
        for c in numeric_cols:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x) >= 3 and np.all(np.diff(x) >= 0):
                monotonic_candidates.append(c)
        if monotonic_candidates:
            mjd_like = [c for c in monotonic_candidates if np.nanmedian(pd.to_numeric(df[c], errors="coerce")) > 30000]
            time_col = mjd_like[0] if mjd_like else monotonic_candidates[0]

    if resid_col is None:
        candidates = [c for c in numeric_cols if c != time_col]
        resid_col = candidates[0] if candidates else None

    if resid_col is not None and err_col is None:
        candidates = [c for c in numeric_cols if c not in {time_col, resid_col}]
        if candidates:
            # Heuristic: error columns are positive and generally smaller than residual scatter.
            scored: List[Tuple[float, str]] = []
            resid_scale = np.nanstd(pd.to_numeric(df[resid_col], errors="coerce"))
            for c in candidates:
                vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                positivity = np.mean(vals > 0)
                med = np.nanmedian(np.abs(vals))
                score = positivity - abs(math.log10(max(med, 1e-30) / max(resid_scale, 1e-30)))
                scored.append((score, c))
            if scored:
                err_col = max(scored)[1]

    if time_col is None or resid_col is None:
        raise ValueError("Could not infer required time/residual columns")
    return time_col, resid_col, err_col


def looks_like_mjd(values: np.ndarray) -> bool:
    vals = values[np.isfinite(values)]
    if len(vals) < 6:
        return False
    med = float(np.nanmedian(vals))
    return 30000.0 <= med <= 80000.0


def looks_like_residual_series(mjd: np.ndarray, residual: np.ndarray, error: np.ndarray) -> bool:
    if len(mjd) < 6:
        return False
    if not looks_like_mjd(mjd):
        return False
    if np.nanstd(residual) <= 0:
        return False
    if np.any(error <= 0) or not np.all(np.isfinite(error)):
        return False
    increasing_fraction = float(np.mean(np.diff(np.sort(mjd)) >= 0)) if len(mjd) > 1 else 0.0
    return increasing_fraction > 0.95


def parse_residual_file(path: Path) -> Optional[ResidualSeries]:
    pulsar = detect_pulsar_name(path)
    text = read_text_file(path)
    df = sanitize_dataframe(parse_table_text(text))
    if pulsar is None:
        first_col = str(df.columns[0]).lower() if len(df.columns) else ''
        if 'psr' in first_col or 'pulsar' in first_col:
            names = df.iloc[:, 0].astype(str)
            for val in names:
                m = PSR_RE.search(val)
                if m:
                    pulsar = m.group(0)
                    break
    if pulsar is None:
        return None
    time_col, resid_col, err_col = infer_time_residual_error_columns(df)

    mjd = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    residual = pd.to_numeric(df[resid_col], errors="coerce").to_numpy(dtype=float)
    if err_col is not None:
        error = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float)
    else:
        scatter = np.nanstd(residual)
        error = np.full_like(residual, max(scatter, 1.0))

    mask = np.isfinite(mjd) & np.isfinite(residual) & np.isfinite(error) & (error > 0)
    mjd = mjd[mask]
    residual = residual[mask]
    error = error[mask]
    if not looks_like_residual_series(mjd, residual, error):
        return None

    order = np.argsort(mjd)
    return ResidualSeries(
        pulsar=pulsar,
        mjd=mjd[order],
        residual=residual[order],
        error=error[order],
        source_path=path,
        whitened=bool(re.search(r"white|whiten", str(path).lower())),
        epoch_averaged=bool(re.search(r"epoch|avg|average", str(path).lower())),
    )


def collect_best_residual_series(root: Path) -> Dict[str, ResidualSeries]:
    files = sorted([p for p in root.rglob("*") if p.is_file() and is_residual_candidate(p)], key=score_residual_candidate, reverse=True)
    logging.info("Found %d residual-like files under %s", len(files), root)
    by_pulsar: Dict[str, List[ResidualSeries]] = {}
    for path in files:
        try:
            series = parse_residual_file(path)
        except Exception as e:
            logging.debug("Skipping %s: %s", path, e)
            continue
        if series is None:
            continue
        by_pulsar.setdefault(series.pulsar, []).append(series)

    selected: Dict[str, ResidualSeries] = {}
    for pulsar, candidates in by_pulsar.items():
        candidates.sort(
            key=lambda s: (
                int(s.epoch_averaged),
                int(s.whitened),
                -len(s.mjd),
                -s.wrms,
            ),
            reverse=True,
        )
        selected[pulsar] = candidates[0]
    logging.info("Parsed usable residual series for %d pulsars", len(selected))
    return selected


def compute_binned_variance_series(series: ResidualSeries, bin_days: float, min_points_per_bin: int) -> VarianceSeries:
    start = np.floor(series.mjd.min())
    stop = np.ceil(series.mjd.max()) + bin_days
    edges = np.arange(start, stop + 1e-9, bin_days)
    idx = np.digitize(series.mjd, edges) - 1
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    var = np.full(len(bin_centers), np.nan)
    valid = np.zeros(len(bin_centers), dtype=bool)

    for i in range(len(bin_centers)):
        sel = idx == i
        if not np.any(sel):
            continue
        r = series.residual[sel]
        e = series.error[sel]
        w = 1.0 / np.maximum(e, 1e-30) ** 2
        if np.sum(sel) >= min_points_per_bin:
            mean = np.sum(w * r) / np.sum(w)
            var[i] = np.sum(w * (r - mean) ** 2) / np.sum(w)
            valid[i] = True
        else:
            mean = np.sum(w * r) / np.sum(w)
            var[i] = mean**2
            valid[i] = True

    return VarianceSeries(
        pulsar=series.pulsar,
        bin_center_mjd=bin_centers,
        variance=var,
        valid=valid,
        wrms=series.wrms,
    )


def align_variance_series(series_list: Sequence[VarianceSeries]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    if not series_list:
        raise ValueError("No variance series supplied")
    common_start = max(s.bin_center_mjd.min() for s in series_list)
    common_stop = min(s.bin_center_mjd.max() for s in series_list)
    ref = min(series_list, key=lambda s: len(s.bin_center_mjd))
    mask = (ref.bin_center_mjd >= common_start) & (ref.bin_center_mjd <= common_stop)
    grid = ref.bin_center_mjd[mask]
    if len(grid) < 8:
        raise ValueError("Insufficient overlapping baseline between pulsars")

    matrix = np.full((len(series_list), len(grid)), np.nan)
    valid = np.zeros_like(matrix, dtype=bool)
    wrms = np.array([s.wrms for s in series_list], dtype=float)
    names = [s.pulsar for s in series_list]

    for i, s in enumerate(series_list):
        interp = np.interp(grid, s.bin_center_mjd[s.valid], s.variance[s.valid], left=np.nan, right=np.nan) if np.any(s.valid) else np.full(len(grid), np.nan)
        # Mark bins as valid only if they were originally near a true observation bin.
        closest = np.searchsorted(s.bin_center_mjd, grid)
        vmask = np.zeros(len(grid), dtype=bool)
        for j, g in enumerate(grid):
            neighbors = []
            for k in [closest[j] - 1, closest[j]]:
                if 0 <= k < len(s.bin_center_mjd):
                    neighbors.append(k)
            if neighbors:
                best = min(neighbors, key=lambda k: abs(s.bin_center_mjd[k] - g))
                spacing = np.median(np.diff(s.bin_center_mjd)) if len(s.bin_center_mjd) > 1 else np.inf
                if abs(s.bin_center_mjd[best] - g) <= 0.51 * spacing and s.valid[best]:
                    vmask[j] = True
        matrix[i] = interp
        valid[i] = vmask & np.isfinite(interp)

    return grid, matrix, valid, names, wrms


def robust_aggregate_variance(
    grid: np.ndarray,
    matrix: np.ndarray,
    valid: np.ndarray,
    wrms: np.ndarray,
    min_pulsars_per_bin: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = 1.0 / np.maximum(wrms, 1e-30) ** 2
    agg = np.full(len(grid), np.nan)
    counts = np.sum(valid, axis=0)
    for j in range(len(grid)):
        sel = valid[:, j] & np.isfinite(matrix[:, j])
        if np.sum(sel) < min_pulsars_per_bin:
            continue
        vals = matrix[sel, j]
        ww = weights[sel]
        agg[j] = np.sum(ww * vals) / np.sum(ww)

    good = np.isfinite(agg)
    if np.sum(good) < 8:
        raise ValueError(
            f"Too few aggregate bins after requiring at least {min_pulsars_per_bin} pulsars per bin"
        )

    x = grid[good]
    y = agg[good]
    detrended = np.log10(np.maximum(y, 1e-30))
    detrended = detrended - pd.Series(detrended).rolling(window=min(7, len(detrended)), center=True, min_periods=1).median().to_numpy()
    detrended = detrended - np.nanmedian(detrended)
    return x, y, detrended


def autocorrelation(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    acf = np.correlate(x, x, mode="full")
    acf = acf[len(x) - 1 :]
    acf /= acf[0] if acf[0] != 0 else 1.0
    lags = np.arange(len(acf), dtype=float)
    return lags, acf


def periodogram_fft(values: np.ndarray, spacing_days: float) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=float)
    y = y - np.mean(y)
    n = len(y)
    window = np.hanning(n)
    yw = y * window
    fft = np.fft.rfft(yw)
    freq_cpd = np.fft.rfftfreq(n, d=spacing_days)
    power = (np.abs(fft) ** 2) / np.sum(window**2)
    if len(power) > 0:
        power[0] = 0.0
    return freq_cpd / SECONDS_PER_DAY, power


def circular_shift_row(row: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(row) < 2:
        return row.copy()
    shift = int(rng.integers(0, len(row)))
    return np.roll(row, shift)


def null_threshold(
    matrix: np.ndarray,
    valid: np.ndarray,
    wrms: np.ndarray,
    grid: np.ndarray,
    min_pulsars_per_bin: int,
    n_sims: int,
    seed: int,
) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    maxima = np.zeros(n_sims, dtype=float)
    spacing_days = float(np.median(np.diff(grid)))
    for i in range(n_sims):
        shifted = np.full_like(matrix, np.nan)
        shifted_valid = valid.copy()
        for r in range(matrix.shape[0]):
            row = matrix[r].copy()
            row_valid = valid[r].copy()
            finite_row = np.where(row_valid & np.isfinite(row), row, np.nan)
            vals = finite_row[row_valid & np.isfinite(finite_row)]
            if len(vals) == 0:
                continue
            shifted_vals = circular_shift_row(vals, rng)
            target = np.where(row_valid & np.isfinite(finite_row))[0]
            row[:] = np.nan
            row[target] = shifted_vals
            shifted[r] = row
        try:
            _, _, detrended = robust_aggregate_variance(grid, shifted, shifted_valid, wrms, min_pulsars_per_bin)
            _, power = periodogram_fft(detrended, spacing_days)
            maxima[i] = np.nanmax(power) if len(power) else 0.0
        except Exception:
            maxima[i] = 0.0
    threshold = float(np.quantile(maxima, 0.99))
    return threshold, maxima


def parse_predicted_ratios(args: argparse.Namespace, n_lines: int) -> Optional[np.ndarray]:
    if args.predicted_ratios:
        arr = np.array([float(x) for x in args.predicted_ratios.split(",") if x.strip()], dtype=float)
        return arr
    if args.geometric_ratio is not None:
        q = float(args.geometric_ratio)
        return np.array([q**i for i in range(n_lines)], dtype=float)
    return None


def compare_ratios(measured_freqs_hz: np.ndarray, predicted_ratios: Optional[np.ndarray], tolerance: float) -> Tuple[List[float], bool, Optional[List[float]]]:
    if len(measured_freqs_hz) == 0:
        return [], False, None
    base = measured_freqs_hz.min()
    measured = (np.sort(measured_freqs_hz) / base).tolist()
    if predicted_ratios is None:
        return measured, False, None
    pred = np.asarray(predicted_ratios, dtype=float)
    if len(pred) < len(measured):
        pred = pred[: len(measured)]
    else:
        pred = pred[: len(measured)]
    frac_err = np.abs(np.array(measured) - pred) / np.maximum(pred, 1e-30)
    return measured, bool(np.all(frac_err <= tolerance)), frac_err.tolist()


def parse_par_coordinates(root: Path) -> Dict[str, Coord]:
    out: Dict[str, Coord] = {}
    for path in root.rglob("*.par"):
        pulsar = detect_pulsar_name(path)
        if pulsar is None:
            continue
        try:
            ra, dec = read_ra_dec_from_par(path)
        except Exception:
            continue
        if ra is not None and dec is not None:
            out[pulsar] = Coord(ra_deg=ra, dec_deg=dec)
    return out


def read_ra_dec_from_par(path: Path) -> Tuple[Optional[float], Optional[float]]:
    ra = None
    dec = None
    for line in read_text_file(path).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key, val = parts[0].upper(), parts[1]
        if key == "RAJ":
            ra = sexagesimal_to_deg(val, is_ra=True)
        elif key == "DECJ":
            dec = sexagesimal_to_deg(val, is_ra=False)
    return ra, dec


def sexagesimal_to_deg(value: str, is_ra: bool) -> float:
    value = value.strip()
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", value):
        raw = float(value)
        return raw * 15.0 if is_ra and abs(raw) <= 24 else raw
    sign = -1.0 if value.startswith("-") else 1.0
    parts = value.replace("+", "").replace("-", "").split(":")
    nums = [float(p) for p in parts]
    while len(nums) < 3:
        nums.append(0.0)
    deg = nums[0] + nums[1] / 60.0 + nums[2] / 3600.0
    if is_ra:
        return deg * 15.0
    return sign * deg


def hellings_downs(theta_rad: np.ndarray) -> np.ndarray:
    x = (1.0 - np.cos(theta_rad)) / 2.0
    out = np.empty_like(x)
    same = np.isclose(theta_rad, 0.0)
    out[same] = 1.0
    xs = np.clip(x[~same], 1e-12, 1.0)
    out[~same] = 0.5 + 1.5 * xs * np.log(xs) - 0.25 * xs
    return out


def angular_separation_deg(a: Coord, b: Coord) -> float:
    ra1, dec1 = np.radians([a.ra_deg, a.dec_deg])
    ra2, dec2 = np.radians([b.ra_deg, b.dec_deg])
    cosang = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def hd_sanity_statistic(grid: np.ndarray, matrix: np.ndarray, valid: np.ndarray, names: Sequence[str], coords: Dict[str, Coord]) -> Tuple[Optional[float], Optional[bool]]:
    common = [i for i, n in enumerate(names) if n in coords]
    if len(common) < 3:
        return None, None
    corrs: List[float] = []
    hds: List[float] = []
    for i_idx, i in enumerate(common):
        for j in common[i_idx + 1 :]:
            sel = valid[i] & valid[j] & np.isfinite(matrix[i]) & np.isfinite(matrix[j])
            if np.sum(sel) < 4:
                continue
            xi = matrix[i, sel]
            xj = matrix[j, sel]
            if np.nanstd(xi) == 0 or np.nanstd(xj) == 0:
                continue
            corr = np.corrcoef(xi, xj)[0, 1]
            sep = angular_separation_deg(coords[names[i]], coords[names[j]])
            corrs.append(float(corr))
            hds.append(float(hellings_downs(np.array([np.radians(sep)]))[0]))
    if len(corrs) < 3:
        return None, None
    corrcoef = float(np.corrcoef(corrs, hds)[0, 1])
    dark_matter_like = abs(corrcoef) < 0.2
    return corrcoef, dark_matter_like


def detect_lines(freq_hz: np.ndarray, power: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(power) < 3:
        return np.array([]), np.array([])
    peaks, props = find_peaks(power, height=threshold)
    if len(peaks) == 0:
        return np.array([]), np.array([])
    order = np.argsort(props["peak_heights"])[::-1]
    peaks = peaks[order]
    peaks = np.sort(peaks)
    return freq_hz[peaks], power[peaks]


def plot_results(
    outpath: Path,
    lag_days: np.ndarray,
    acf: np.ndarray,
    freq_hz: np.ndarray,
    power: np.ndarray,
    line_freqs_hz: np.ndarray,
    predicted_positions_hz: Optional[np.ndarray],
    threshold: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(lag_days, acf)
    axes[0].axhline(0.0, linewidth=1)
    axes[0].set_xlabel("Lag (days)")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].set_title("Aggregate residual-variance autocorrelation")

    axes[1].plot(freq_hz, power)
    axes[1].axhline(threshold, linestyle="--", linewidth=1, label="1% global FAP threshold")
    for i, f in enumerate(line_freqs_hz):
        label = "Detected lines" if i == 0 else None
        axes[1].axvline(f, linestyle="--", linewidth=1, label=label)
    if predicted_positions_hz is not None and len(predicted_positions_hz) > 0:
        for i, f in enumerate(predicted_positions_hz):
            label = "Predicted comb" if i == 0 else None
            axes[1].axvline(f, linestyle=":", linewidth=1, label=label)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[1].set_title("Variance-envelope periodogram")
    axes[1].legend(loc="best")

    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def run_pipeline_on_archive(
    extract_root: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, float, Optional[np.ndarray]]:
    residual_by_pulsar = collect_best_residual_series(extract_root)
    if not residual_by_pulsar:
        raise RuntimeError(f"No usable residual files found under {extract_root}")

    chosen = sorted(residual_by_pulsar.values(), key=lambda s: s.wrms)[: args.top_n]
    logging.info("Selected %d best-timed pulsars", len(chosen))
    for s in chosen[: min(10, len(chosen))]:
        logging.info("  %s wrms=%.6g from %s", s.pulsar, s.wrms, s.source_path.name)

    var_series = [compute_binned_variance_series(s, args.bin_days, args.min_points_per_bin) for s in chosen]
    grid, matrix, valid, names, wrms = align_variance_series(var_series)
    agg_grid, agg_variance, agg_detrended = robust_aggregate_variance(
        grid, matrix, valid, wrms, args.min_pulsars_per_bin
    )

    lag_idx, acf = autocorrelation(agg_detrended)
    lag_days = lag_idx * float(np.median(np.diff(agg_grid)))
    spacing_days = float(np.median(np.diff(agg_grid)))
    freq_hz, power = periodogram_fft(agg_detrended, spacing_days)
    threshold, null_maxima = null_threshold(
        matrix=matrix,
        valid=valid,
        wrms=wrms,
        grid=grid,
        min_pulsars_per_bin=args.min_pulsars_per_bin,
        n_sims=args.bootstrap_sims,
        seed=args.random_seed,
    )
    line_freqs_hz, line_power = detect_lines(freq_hz, power, threshold)

    predicted_ratios = parse_predicted_ratios(args, max(len(line_freqs_hz), 2))
    measured_ratios, ratios_match_rvm, frac_err = compare_ratios(
        line_freqs_hz, predicted_ratios, args.ratio_tolerance
    )
    predicted_positions_hz = None
    if predicted_ratios is not None and len(line_freqs_hz) >= 1:
        predicted_positions_hz = line_freqs_hz.min() * predicted_ratios

    coords = parse_par_coordinates(extract_root)
    hd_corrcoef, dark_matter_like = hd_sanity_statistic(grid, matrix, valid, names, coords)

    result: Dict[str, object] = {
        "n_lines_detected": int(len(line_freqs_hz)),
        "line_frequencies_hz": [float(x) for x in np.sort(line_freqs_hz)],
        "line_amplitudes": [float(x) for x in line_power[np.argsort(line_freqs_hz)]],
        "frequency_ratios": [float(x) for x in measured_ratios],
        "ratios_match_rvm": bool(ratios_match_rvm),
        "pass_v6": bool(len(line_freqs_hz) >= 2 and ratios_match_rvm),
        "selected_pulsars": [s.pulsar for s in chosen],
        "selected_wrms": {s.pulsar: float(s.wrms) for s in chosen},
        "bin_days": float(args.bin_days),
        "min_points_per_bin": int(args.min_points_per_bin),
        "min_pulsars_per_bin": int(args.min_pulsars_per_bin),
        "false_alarm_threshold_1pct": float(threshold),
        "bootstrap_sims": int(args.bootstrap_sims),
        "predicted_ratios_input": None if predicted_ratios is None else [float(x) for x in predicted_ratios],
        "ratio_tolerance": float(args.ratio_tolerance),
        "ratio_fractional_errors": frac_err,
        "hd_correlation_with_hd_curve": hd_corrcoef,
        "dark_matter_like_not_hd": dark_matter_like,
        "notes": [
            "Search performed on the slow modulation of the aggregate residual-variance envelope.",
            "Null threshold estimated from circular-shift simulations of per-pulsar variance series.",
            "If no explicit RVM ratio prediction is provided, ratios_match_rvm is forced false.",
        ],
    }
    return result, lag_days, acf, freq_hz, power, names, wrms, threshold, predicted_positions_hz


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    nanograv_archive = download_file(
        zenodo_download_url(DEFAULT_NANOGRAV_RECORD_ID, DEFAULT_NANOGRAV_FILENAME),
        args.cache_dir / DEFAULT_NANOGRAV_FILENAME,
    )
    nanograv_root = extract_archive(nanograv_archive, args.cache_dir / "nanograv15yr")

    try:
        result, lag_days, acf, freq_hz, power, names, wrms, threshold, predicted_positions_hz = run_pipeline_on_archive(
            nanograv_root, args
        )
    except RuntimeError as e:
        if "No usable residual files found" not in str(e):
            raise
        logging.warning("Initial archive scan failed; forcing a clean re-extraction and retry")
        shutil.rmtree(nanograv_root, ignore_errors=True)
        nanograv_root = extract_archive(nanograv_archive, args.cache_dir / "nanograv15yr")
        result, lag_days, acf, freq_hz, power, names, wrms, threshold, predicted_positions_hz = run_pipeline_on_archive(
            nanograv_root, args
        )
    result["dataset"] = {
        "name": "NANOGrav 15-year public release",
        "record_id": DEFAULT_NANOGRAV_RECORD_ID,
        "filename": DEFAULT_NANOGRAV_FILENAME,
    }

    if args.include_epta_crosscheck:
        try:
            epta_archive = download_file(
                zenodo_download_url(DEFAULT_EPTA_RECORD_ID, DEFAULT_EPTA_FILENAME),
                args.cache_dir / DEFAULT_EPTA_FILENAME,
            )
            epta_root = extract_archive(epta_archive, args.cache_dir / "epta_dr2")
            epta_result, *_ = run_pipeline_on_archive(epta_root, args)
            result["epta_crosscheck"] = epta_result
        except Exception as e:
            logging.warning("EPTA cross-check failed: %s", e)
            result["epta_crosscheck"] = {"error": str(e)}

    result_path = args.outdir / "result.json"
    plot_path = args.outdir / "variance_comb_search.png"

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    plot_results(
        outpath=plot_path,
        lag_days=lag_days,
        acf=acf,
        freq_hz=freq_hz,
        power=power,
        line_freqs_hz=np.array(result["line_frequencies_hz"], dtype=float),
        predicted_positions_hz=predicted_positions_hz,
        threshold=threshold,
    )

    logging.info("Wrote %s", result_path)
    logging.info("Wrote %s", plot_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
