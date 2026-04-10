#!/usr/bin/env python3
"""BA2_spiral_chirality.py — Spiral galaxy handedness asymmetry analysis.

Implements the BA2 note as a runnable CLI for Galaxy Zoo 2 style catalogues.

Main features
-------------
- Reads Galaxy Zoo 2 CSV/CSV.GZ catalogues.
- Detects likely column names automatically.
- Optional merge with a metadata table (e.g. gz2sample.csv) when RA/Dec are absent.
- Filters galaxies by spiral-feature confidence and optional edge-on exclusion.
- Assigns handedness from clockwise vs anticlockwise vote fractions.
- Computes global handedness asymmetry and simple sky-split dipoles.
- Optionally fits a 3D dipole on the sphere and estimates a permutation p-value.
- Saves results to JSON and optional plots.

Examples
--------
python BA2_spiral_chirality.py analyse gz2_hart16.csv.gz
python BA2_spiral_chirality.py analyse gz2_hart16.csv.gz --metadata gz2sample.csv.gz \
    --output ba2_result.json --plot ba2_summary.png
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


# -----------------------------
# Public Galaxy Zoo 2 download URLs
# -----------------------------
DEFAULT_DATASETS: dict[str, str] = {
    "hart16": "https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz",
    "hart16_fits": "https://gz2hart.s3.amazonaws.com/gz2_hart16.fits.gz",
    "mainspecz": "https://zooniverse-data.s3.amazonaws.com/zoo2MainSpecz.csv.gz",
}
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (BA2_spiral_chirality.py)"}


# -----------------------------
# Column detection helpers
# -----------------------------
FEATURES_CANDIDATES = (
    "t01_smooth_or_features_a02_features_or_disk_fraction",
    "t01_smooth_or_features_a02_features_or_disk_debiased",
    "features_or_disk_fraction",
    "features_or_disk_debiased",
)
CW_CANDIDATES = (
    "t04_spiral_a08_cw_fraction",
    "t04_spiral_a08_spiral_clockwise_fraction",
    "spiral_clockwise_fraction",
    "cw_fraction",
)
ACW_CANDIDATES = (
    "t04_spiral_a09_acw_fraction",
    "t04_spiral_a09_spiral_anticlockwise_fraction",
    "spiral_anticlockwise_fraction",
    "acw_fraction",
)
EDGEON_NO_CANDIDATES = (
    "t02_edgeon_a05_no_fraction",
    "t02_edgeon_a05_no_debiased",
    "edgeon_no_fraction",
)
RA_CANDIDATES = ("ra", "RA", "objra")
DEC_CANDIDATES = ("dec", "DEC", "objdec")
OBJID_CANDIDATES = ("dr7objid", "objid", "specobjid", "OBJID")


@dataclass
class ColumnMap:
    features: str
    cw: str
    acw: str
    ra: str | None = None
    dec: str | None = None
    edgeon_no: str | None = None
    objid: str | None = None


@dataclass
class BA2Result:
    input_rows: int
    used_rows: int
    ambiguous_rows_dropped: int
    feature_threshold: float
    exclude_edgeon: bool
    edgeon_threshold: float | None
    n_cw: int
    n_acw: int
    n_total: int
    cw_fraction: float
    asymmetry: float
    asymmetry_error: float
    significance_sigma: float
    target_prediction: float
    target_sigma_offset: float
    ra_median_dipole: float | None
    dec_median_dipole: float | None
    spherical_dipole_amplitude: float | None
    spherical_dipole_direction_ra_deg: float | None
    spherical_dipole_direction_dec_deg: float | None
    spherical_dipole_intercept: float | None
    spherical_dipole_permutation_pvalue: float | None
    columns: dict[str, str | None]


def find_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    column_set = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        key = cand.lower()
        if key in column_set:
            return column_set[key]
    return None


def detect_columns(df: pd.DataFrame) -> ColumnMap:
    features = find_first_existing(df.columns, FEATURES_CANDIDATES)
    cw = find_first_existing(df.columns, CW_CANDIDATES)
    acw = find_first_existing(df.columns, ACW_CANDIDATES)
    if not (features and cw and acw):
        raise RuntimeError(
            "Could not detect required Galaxy Zoo columns for features/cw/acw. "
            f"Available columns include: {list(df.columns[:20])}..."
        )
    ra = find_first_existing(df.columns, RA_CANDIDATES)
    dec = find_first_existing(df.columns, DEC_CANDIDATES)
    edgeon_no = find_first_existing(df.columns, EDGEON_NO_CANDIDATES)
    objid = find_first_existing(df.columns, OBJID_CANDIDATES)
    return ColumnMap(features=features, cw=cw, acw=acw, ra=ra, dec=dec, edgeon_no=edgeon_no, objid=objid)


def read_table(path: str | Path, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    compression = "gzip" if path.suffix.lower() == ".gz" else "infer"
    return pd.read_csv(path, compression=compression, usecols=usecols, low_memory=False)


def merge_metadata_if_needed(df: pd.DataFrame, metadata_path: str | Path | None, cmap: ColumnMap) -> tuple[pd.DataFrame, ColumnMap]:
    if cmap.ra and cmap.dec:
        return df, cmap
    if metadata_path is None:
        return df, cmap

    if cmap.objid is None:
        raise RuntimeError("RA/Dec missing in main file and no object-id column available for metadata merge.")

    meta_preview = read_table(metadata_path)
    meta_objid = find_first_existing(meta_preview.columns, OBJID_CANDIDATES)
    meta_ra = find_first_existing(meta_preview.columns, RA_CANDIDATES)
    meta_dec = find_first_existing(meta_preview.columns, DEC_CANDIDATES)
    if not (meta_objid and meta_ra and meta_dec):
        raise RuntimeError("Metadata file does not contain a detectable object-id + RA/Dec set.")

    meta = meta_preview[[meta_objid, meta_ra, meta_dec]].copy()
    merged = df.merge(meta, left_on=cmap.objid, right_on=meta_objid, how="left", suffixes=("", "_meta"))
    cmap = ColumnMap(
        features=cmap.features,
        cw=cmap.cw,
        acw=cmap.acw,
        ra=meta_ra,
        dec=meta_dec,
        edgeon_no=cmap.edgeon_no,
        objid=cmap.objid,
    )
    return merged, cmap


def infer_output_path(url: str, output: str | Path | None, directory: str | Path | None) -> Path:
    if output is not None:
        return Path(output)
    filename = url.rstrip('/').split('/')[-1]
    if not filename:
        raise RuntimeError(f"Could not infer filename from URL: {url}")
    base_dir = Path('.') if directory is None else Path(directory)
    return base_dir / filename


def download_file(url: str, destination: str | Path, *, overwrite: bool = False, chunk_size: int = 1 << 20) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"[exists] {destination}")
        return destination

    req = Request(url, headers=DEFAULT_HEADERS)
    tmp = destination.with_suffix(destination.suffix + '.part')
    try:
        with urlopen(req, timeout=120) as resp, tmp.open('wb') as fout:
            total = resp.headers.get('Content-Length')
            total_bytes = int(total) if total and total.isdigit() else None
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fout.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = 100.0 * downloaded / total_bytes
                    print(f"\r[download] {destination.name}: {downloaded / 1e6:.1f}/{total_bytes / 1e6:.1f} MB ({pct:.1f}%)", end='')
                else:
                    print(f"\r[download] {destination.name}: {downloaded / 1e6:.1f} MB", end='')
        tmp.replace(destination)
        print()
        print(f"[saved] {destination}")
        return destination
    except (HTTPError, URLError, TimeoutError) as e:
        print()
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Download failed for {url}: {e}") from e


def resolve_dataset_url(dataset: str | None, url: str | None) -> str:
    if url:
        return url
    if dataset is None:
        raise RuntimeError('Either a named dataset or a URL must be provided.')
    if dataset not in DEFAULT_DATASETS:
        raise RuntimeError(f"Unknown dataset '{dataset}'. Choices: {', '.join(sorted(DEFAULT_DATASETS))}")
    return DEFAULT_DATASETS[dataset]


# -----------------------------
# Analysis helpers
# -----------------------------
def validate_probabilities(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    return v.replace([np.inf, -np.inf], np.nan)


def unit_vectors_from_radec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cos_dec = np.cos(dec)
    x = cos_dec * np.cos(ra)
    y = cos_dec * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack((x, y, z))


@dataclass
class DipoleFit:
    intercept: float
    dx: float
    dy: float
    dz: float
    amplitude: float
    direction_ra_deg: float
    direction_dec_deg: float


def fit_spherical_dipole(ra_deg: np.ndarray, dec_deg: np.ndarray, handed_sign: np.ndarray) -> DipoleFit:
    # y ∈ {-1, +1}; fit y ≈ a + d·n by least squares.
    n = unit_vectors_from_radec(ra_deg, dec_deg)
    design = np.column_stack((np.ones(len(n)), n))
    coeffs, *_ = np.linalg.lstsq(design, handed_sign.astype(float), rcond=None)
    intercept, dx, dy, dz = coeffs
    amplitude = float(np.linalg.norm([dx, dy, dz]))
    if amplitude == 0.0:
        ra_dir = 0.0
        dec_dir = 0.0
    else:
        ra_dir = math.degrees(math.atan2(dy, dx)) % 360.0
        dec_dir = math.degrees(math.asin(dz / amplitude))
    return DipoleFit(
        intercept=float(intercept),
        dx=float(dx),
        dy=float(dy),
        dz=float(dz),
        amplitude=amplitude,
        direction_ra_deg=float(ra_dir),
        direction_dec_deg=float(dec_dir),
    )


def permutation_pvalue(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    handed_sign: np.ndarray,
    observed_amplitude: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    if n_perm <= 0:
        return float("nan")
    n = unit_vectors_from_radec(ra_deg, dec_deg)
    design = np.column_stack((np.ones(len(n)), n))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(handed_sign)
        coeffs, *_ = np.linalg.lstsq(design, perm.astype(float), rcond=None)
        amp = float(np.linalg.norm(coeffs[1:4]))
        if amp >= observed_amplitude:
            count += 1
    return (count + 1) / (n_perm + 1)


def median_split_dipole(coord: np.ndarray, is_cw: np.ndarray) -> float:
    median = np.median(coord)
    mask1 = coord < median
    mask2 = ~mask1
    if mask1.sum() == 0 or mask2.sum() == 0:
        return float("nan")
    a1 = float(np.mean(is_cw[mask1]) - 0.5)
    a2 = float(np.mean(is_cw[mask2]) - 0.5)
    return a1 - a2


def analyse_dataframe(
    df: pd.DataFrame,
    cmap: ColumnMap,
    *,
    feature_threshold: float = 0.5,
    exclude_edgeon: bool = False,
    edgeon_threshold: float = 0.5,
    permutations: int = 0,
    seed: int = 42,
) -> BA2Result:
    input_rows = len(df)

    features = validate_probabilities(df[cmap.features])
    cw = validate_probabilities(df[cmap.cw])
    acw = validate_probabilities(df[cmap.acw])

    mask = features > feature_threshold
    if exclude_edgeon and cmap.edgeon_no is not None:
        edgeon_no = validate_probabilities(df[cmap.edgeon_no])
        mask &= edgeon_no > edgeon_threshold
    elif exclude_edgeon and cmap.edgeon_no is None:
        raise RuntimeError("Requested --exclude-edgeon but no edge-on 'no' column was detected.")

    base = df.loc[mask].copy()
    cw = cw.loc[mask].to_numpy()
    acw = acw.loc[mask].to_numpy()

    tie_mask = np.isfinite(cw) & np.isfinite(acw) & (cw != acw)
    ambiguous_rows_dropped = int(len(base) - int(tie_mask.sum()))
    base = base.loc[tie_mask].copy()
    cw = cw[tie_mask]
    acw = acw[tie_mask]

    is_cw = cw > acw
    n_cw = int(np.sum(is_cw))
    n_total = int(len(is_cw))
    n_acw = n_total - n_cw
    if n_total == 0:
        raise RuntimeError("No galaxies survived the filters. Relax thresholds or inspect the detected columns.")

    asymmetry = float((n_cw - n_acw) / n_total)
    asymmetry_error = float(1.0 / math.sqrt(n_total))
    significance = float(abs(asymmetry) / asymmetry_error)
    target_prediction = 1e-3
    target_sigma_offset = float(abs(asymmetry - target_prediction) / asymmetry_error)

    ra_dipole = None
    dec_dipole = None
    spherical_fit = None
    pvalue = None
    if cmap.ra and cmap.dec:
        ra = pd.to_numeric(base[cmap.ra], errors="coerce").to_numpy()
        dec = pd.to_numeric(base[cmap.dec], errors="coerce").to_numpy()
        coord_mask = np.isfinite(ra) & np.isfinite(dec)
        ra = ra[coord_mask]
        dec = dec[coord_mask]
        is_cw_coord = is_cw[coord_mask]
        if len(ra) > 3:
            ra_dipole = float(median_split_dipole(ra, is_cw_coord))
            dec_dipole = float(median_split_dipole(dec, is_cw_coord))
            handed_sign = np.where(is_cw_coord, 1.0, -1.0)
            spherical_fit = fit_spherical_dipole(ra, dec, handed_sign)
            if permutations > 0:
                rng = np.random.default_rng(seed)
                pvalue = float(permutation_pvalue(ra, dec, handed_sign, spherical_fit.amplitude, permutations, rng))

    result = BA2Result(
        input_rows=int(input_rows),
        used_rows=int(n_total),
        ambiguous_rows_dropped=int(ambiguous_rows_dropped),
        feature_threshold=float(feature_threshold),
        exclude_edgeon=bool(exclude_edgeon),
        edgeon_threshold=float(edgeon_threshold) if exclude_edgeon else None,
        n_cw=n_cw,
        n_acw=n_acw,
        n_total=n_total,
        cw_fraction=float(n_cw / n_total),
        asymmetry=asymmetry,
        asymmetry_error=asymmetry_error,
        significance_sigma=significance,
        target_prediction=target_prediction,
        target_sigma_offset=target_sigma_offset,
        ra_median_dipole=ra_dipole,
        dec_median_dipole=dec_dipole,
        spherical_dipole_amplitude=None if spherical_fit is None else spherical_fit.amplitude,
        spherical_dipole_direction_ra_deg=None if spherical_fit is None else spherical_fit.direction_ra_deg,
        spherical_dipole_direction_dec_deg=None if spherical_fit is None else spherical_fit.direction_dec_deg,
        spherical_dipole_intercept=None if spherical_fit is None else spherical_fit.intercept,
        spherical_dipole_permutation_pvalue=pvalue,
        columns={
            "features": cmap.features,
            "cw": cmap.cw,
            "acw": cmap.acw,
            "ra": cmap.ra,
            "dec": cmap.dec,
            "edgeon_no": cmap.edgeon_no,
            "objid": cmap.objid,
        },
    )
    return result


def print_result(result: BA2Result) -> None:
    print("BA2 Spiral Chirality Analysis")
    print("=" * 50)
    print(f"Input rows:                {result.input_rows}")
    print(f"Rows used:                 {result.used_rows}")
    print(f"Ambiguous rows dropped:    {result.ambiguous_rows_dropped}")
    print(f"N_CW:                      {result.n_cw}")
    print(f"N_ACW:                     {result.n_acw}")
    print(f"N_total:                   {result.n_total}")
    print(f"CW fraction:               {result.cw_fraction:.6f}")
    print(f"Asymmetry:                 {result.asymmetry:.6f} ± {result.asymmetry_error:.6f}")
    print(f"Significance:              {result.significance_sigma:.2f}σ")
    print(f"Target prediction:         {result.target_prediction:.6f}")
    print(f"Target offset:             {result.target_sigma_offset:.2f}σ")
    if result.ra_median_dipole is not None:
        print(f"Median-split dipole (RA):  {result.ra_median_dipole:.6f}")
    if result.dec_median_dipole is not None:
        print(f"Median-split dipole (Dec): {result.dec_median_dipole:.6f}")
    if result.spherical_dipole_amplitude is not None:
        print(f"Spherical dipole amplitude:{result.spherical_dipole_amplitude:.6f}")
        print(
            "Spherical dipole dir:      "
            f"RA={result.spherical_dipole_direction_ra_deg:.2f}°, "
            f"Dec={result.spherical_dipole_direction_dec_deg:.2f}°"
        )
        print(f"Dipole intercept:          {result.spherical_dipole_intercept:.6f}")
        if result.spherical_dipole_permutation_pvalue is not None:
            print(f"Permutation p-value:       {result.spherical_dipole_permutation_pvalue:.6g}")
    print("Detected columns:")
    for key, value in result.columns.items():
        print(f"  {key:<12} {value}")


def maybe_make_plot(df: pd.DataFrame, cmap: ColumnMap, result: BA2Result, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    features = pd.to_numeric(df[cmap.features], errors="coerce")
    cw = pd.to_numeric(df[cmap.cw], errors="coerce")
    acw = pd.to_numeric(df[cmap.acw], errors="coerce")
    mask = (features > result.feature_threshold) & np.isfinite(cw) & np.isfinite(acw) & (cw != acw)
    work = df.loc[mask].copy()
    cw = cw.loc[mask].to_numpy()
    acw = acw.loc[mask].to_numpy()
    is_cw = cw > acw

    fig = plt.figure(figsize=(11, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(features.dropna().to_numpy(), bins=40)
    ax1.axvline(result.feature_threshold, linestyle="--")
    ax1.set_xlabel(cmap.features)
    ax1.set_ylabel("Count")
    ax1.set_title("Spiral-feature confidence")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(["CW", "ACW"], [result.n_cw, result.n_acw])
    ax2.set_title(f"Asymmetry = {result.asymmetry:.4e} ± {result.asymmetry_error:.4e}")
    ax2.set_ylabel("Galaxies")

    if cmap.ra and cmap.dec and cmap.ra in work.columns and cmap.dec in work.columns:
        ra = pd.to_numeric(work[cmap.ra], errors="coerce").to_numpy()
        dec = pd.to_numeric(work[cmap.dec], errors="coerce").to_numpy()
        good = np.isfinite(ra) & np.isfinite(dec)
        ra = ra[good]
        dec = dec[good]
        is_cw_plot = is_cw[good]

        ax3 = fig.add_subplot(2, 1, 2)
        ax3.scatter(ra[~is_cw_plot], dec[~is_cw_plot], s=4, alpha=0.4, label="ACW")
        ax3.scatter(ra[is_cw_plot], dec[is_cw_plot], s=4, alpha=0.4, label="CW")
        ax3.set_xlabel("RA [deg]")
        ax3.set_ylabel("Dec [deg]")
        ax3.set_title("Sky positions of classified spirals")
        ax3.legend(markerscale=3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse Galaxy Zoo spiral handedness asymmetry.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyse", help="Run the BA2 analysis on a Galaxy Zoo style table.")
    p.add_argument("catalog", nargs='?', help="Path to gz2_hart16.csv(.gz) or similar catalogue.")
    p.add_argument("--metadata", help="Optional metadata table with RA/Dec for merging.")
    p.add_argument("--download-dataset", choices=sorted(DEFAULT_DATASETS), help="Download an official Galaxy Zoo 2 dataset if the catalogue path is omitted or missing.")
    p.add_argument("--download-dir", default='.', help="Directory used by --download-dataset.")
    p.add_argument("--feature-threshold", type=float, default=0.5, help="Minimum spiral-feature confidence.")
    p.add_argument("--exclude-edgeon", action="store_true", help="Require the 'not edge-on' vote fraction to exceed a threshold.")
    p.add_argument("--edgeon-threshold", type=float, default=0.5, help="Threshold for the detected edge-on=no column.")
    p.add_argument("--permutations", type=int, default=0, help="Number of dipole-label permutations for a p-value.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for permutations.")
    p.add_argument("--output", help="Optional JSON output path.")
    p.add_argument("--plot", help="Optional PNG plot path.")

    p2 = sub.add_parser("self-test", help="Run on a synthetic catalogue to verify the pipeline.")
    p2.add_argument("--rows", type=int, default=5000)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--output", help="Optional JSON output path.")

    p3 = sub.add_parser("download", help="Download official public Galaxy Zoo 2 tables.")
    p3.add_argument("dataset", nargs='?', choices=sorted(DEFAULT_DATASETS), default='hart16', help="Named dataset to download.")
    p3.add_argument("--url", help="Override the source URL with a custom CSV/CSV.GZ URL.")
    p3.add_argument("--output", help="Explicit output filename.")
    p3.add_argument("--directory", default='.', help="Directory for downloaded files when --output is not given.")
    p3.add_argument("--overwrite", action='store_true', help="Overwrite an existing local file.")
    return parser


def make_synthetic_catalog(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    features = rng.uniform(0.4, 1.0, n)
    # Inject a tiny chirality signal and a mild dipole.
    signal = 1e-3 + 0.02 * np.cos(np.deg2rad(ra - 180.0)) * np.cos(np.deg2rad(dec))
    p_cw = np.clip(0.5 + 0.5 * signal, 0.0, 1.0)
    is_cw = rng.random(n) < p_cw
    margin = rng.uniform(0.01, 0.5, n)
    cw = np.where(is_cw, 0.5 + margin / 2, 0.5 - margin / 2)
    acw = 1.0 - cw
    edgeon_no = rng.uniform(0.5, 1.0, n)
    return pd.DataFrame(
        {
            "dr7objid": np.arange(n),
            "ra": ra,
            "dec": dec,
            "t01_smooth_or_features_a02_features_or_disk_fraction": features,
            "t04_spiral_a08_cw_fraction": cw,
            "t04_spiral_a09_acw_fraction": acw,
            "t02_edgeon_a05_no_fraction": edgeon_no,
        }
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "self-test":
        df = make_synthetic_catalog(args.rows, args.seed)
        cmap = detect_columns(df)
        result = analyse_dataframe(df, cmap, exclude_edgeon=True, permutations=100, seed=args.seed)
        print_result(result)
        if args.output:
            Path(args.output).write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return 0

    if args.command == "download":
        url = resolve_dataset_url(args.dataset, args.url)
        out = infer_output_path(url, args.output, args.directory)
        download_file(url, out, overwrite=args.overwrite)
        print(f"Ready: {out}")
        return 0

    catalog_path = None if args.catalog is None else Path(args.catalog)
    if catalog_path is None or not catalog_path.exists():
        if args.download_dataset:
            url = resolve_dataset_url(args.download_dataset, None)
            catalog_path = infer_output_path(url, None, args.download_dir)
            download_file(url, catalog_path, overwrite=False)
        elif catalog_path is None:
            raise SystemExit("analyse requires a catalogue path, or use --download-dataset to fetch one.")
        else:
            raise SystemExit(f"Catalogue not found: {catalog_path}. Use --download-dataset to fetch an official table.")

    df = read_table(catalog_path)
    cmap = detect_columns(df)
    df, cmap = merge_metadata_if_needed(df, args.metadata, cmap)
    result = analyse_dataframe(
        df,
        cmap,
        feature_threshold=args.feature_threshold,
        exclude_edgeon=args.exclude_edgeon,
        edgeon_threshold=args.edgeon_threshold,
        permutations=args.permutations,
        seed=args.seed,
    )
    print_result(result)

    if args.output:
        Path(args.output).write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    if args.plot:
        maybe_make_plot(df, cmap, result, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
