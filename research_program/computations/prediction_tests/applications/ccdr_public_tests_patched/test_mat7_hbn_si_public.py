from __future__ import annotations
"""MAT7 public-source test with robust Bristol static fallback.

Prediction audited here: hBN/Si public data should show very high in-plane
thermal conductivity for hBN-side tables. This is only the high-k half of
MAT7; phononic gap verification needs phonon-spectrum data.
"""
import json
import os
from pathlib import Path
import pandas as pd

from _common_public_data import (
    cached_download, ckan_download_matching_resources, ckan_package_show,
    ensure_dir, github_find_blob, github_raw, json_dump, read_public_table,
    unzip_one, structured_report,
)

ROOT = Path(__file__).resolve().parent
OUT = ensure_dir(ROOT / "outputs" / "mat7")
CKAN_API = "https://data.bris.ac.uk/data/api/3"
DATASET_ID = "16v9rfpzb3pl221yzel7x5u5ce"
BASE_STATIC = "https://data-bris.acrc.bris.ac.uk/datasets/16v9rfpzb3pl221yzel7x5u5ce"
STATIC_NAMES = [
    "Figure1a.csv", "Figure1b.csv", "Figure1c.csv", "Figure1d.csv",
    "Figure2a.csv", "Figure2b.csv", "Figure2c.csv", "Figure2d.csv",
    "Figure3a.csv", "Figure3b.csv", "Figure3c.csv", "readme.txt",
]


def _read_numeric_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() in {".csv", ".txt", ".dat", ".xlsx", ".xls"}:
            df = read_public_table(path)
            if len(df.columns) >= 2 and len(df) >= 3:
                return df
    except Exception:
        pass
    return None


def _best_xy(df: pd.DataFrame) -> tuple[str, str] | None:
    numeric = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, len(df) // 2):
            numeric.append(c)
    if len(numeric) < 2:
        return None
    # Prefer a temperature-like x column and a conductivity-like y column.
    temp = next((c for c in numeric if any(k in str(c).lower() for k in ["temp", "temperature", "t/k", " t ", "simulation"])), numeric[0])
    y_candidates = [c for c in numeric if c != temp]
    kappa = next((c for c in y_candidates if any(k in str(c).lower() for k in ["kappa", "conduct", "w/m", "thermal", "unnamed"])), y_candidates[0])
    return str(temp), str(kappa)


def _download_static(names: list[str] | None = None, only_cached: bool = False) -> list[Path]:
    out = []
    for name in (names or STATIC_NAMES):
        target = OUT / "bris_direct" / name
        if target.exists() and target.stat().st_size > 0:
            out.append(target)
            continue
        if only_cached:
            continue
        try:
            out.append(cached_download(f"{BASE_STATIC}/{name}", target))
        except Exception:
            continue
    return out


def _download_bristol_resources(resources_meta: list[dict]) -> tuple[list[Path], list[str]]:
    errors = []
    downloaded: list[Path] = []
    try:
        downloaded = ckan_download_matching_resources(
            CKAN_API, DATASET_ID,
            r"zip|csv|xlsx|xls|txt|dat|octet|complete|download|data|figure",
            OUT / "bris_resources", limit=80,
        ) or []
    except Exception as e:
        errors.append(f"ckan_download_matching_resources: {type(e).__name__}: {e}")

    # Important fix: some runs expose CKAN resources but downloader returns []
    # because resource metadata/mimetype is non-standard. Download resource URLs directly.
    if not downloaded and resources_meta:
        for r in resources_meta:
            url = r.get("url")
            name = r.get("name") or Path(str(url)).name
            if not url or not name:
                continue
            if not any(str(name).lower().endswith(ext) for ext in [".csv", ".txt", ".dat", ".xlsx", ".xls", ".zip"]):
                continue
            target = OUT / "bris_direct_from_ckan" / str(name)
            # Avoid repeated slow timeouts if a previous run already cached the file.
            if target.exists() and target.stat().st_size > 0:
                downloaded.append(target)
                continue
            static_target = OUT / "bris_direct" / str(name)
            if static_target.exists() and static_target.stat().st_size > 0:
                downloaded.append(static_target)
                continue
            try:
                downloaded.append(cached_download(str(url), target))
            except Exception as e:
                errors.append(f"direct {name}: {type(e).__name__}: {e}")

    if not downloaded:
        # If CKAN metadata was unavailable, do not burn many HTTP timeouts on
        # static guesses unless explicitly requested. Cached files are still used.
        ckan_failed_without_metadata = bool(errors) and not resources_meta
        if ckan_failed_without_metadata and not os.environ.get("MAT7_TRY_STATIC_ON_CKAN_FAIL"):
            downloaded = _download_static(only_cached=True)
            if not downloaded:
                errors.append("ckan unavailable and no cached Bristol static files; set MAT7_TRY_STATIC_ON_CKAN_FAIL=1 to probe static URLs anyway")
        else:
            downloaded = _download_static()
            if not downloaded:
                errors.append("static fallback also downloaded no files")
    return downloaded, errors


def main() -> None:
    resources_meta: list[dict] = []
    resource_errors: list[str] = []
    try:
        pkg = ckan_package_show(CKAN_API, DATASET_ID)
        resources_meta = [
            {"name": r.get("name"), "format": r.get("format"), "mimetype": r.get("mimetype"), "url": r.get("url")}
            for r in pkg.get("resources", [])
        ]
    except Exception as e:
        resource_errors.append(f"ckan_package_show: {type(e).__name__}: {e}")

    downloaded, dl_errors = _download_bristol_resources(resources_meta)
    resource_errors.extend(dl_errors)

    expanded: list[Path] = []
    for p in downloaded:
        if p.suffix.lower() == ".zip":
            try:
                expanded.extend(unzip_one(p, OUT / "bris_unzipped" / p.stem))
            except Exception:
                expanded.append(p)
        else:
            expanded.append(p)

    hbn_tables = []
    for p in expanded:
        if p.suffix.lower() not in {".csv", ".txt", ".dat", ".xlsx", ".xls"}:
            continue
        df = _read_numeric_table(p)
        if df is None:
            continue
        cols = _best_xy(df)
        if cols is None:
            continue
        cx, cy = cols
        x = pd.to_numeric(df[cx], errors="coerce")
        y = pd.to_numeric(df[cy], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 5:
            continue
        yy = y[mask]
        hbn_tables.append({
            "file": str(p.relative_to(OUT)) if p.is_relative_to(OUT) else str(p),
            "x_column": cx,
            "y_column": cy,
            "n_points": int(mask.sum()),
            "x_min": float(x[mask].min()),
            "x_max": float(x[mask].max()),
            "y_peak": float(yy.max()),
            "y_median": float(yy.median()),
        })

    silicon_samples = []
    # Silicon references are contextual only. Skip this secondary network probe
    # when the primary Bristol source did not download anything.
    if downloaded:
        try:
            for match in github_find_blob("CMB-S4", "Cryogenic_Material_Properties", r"silicon.*(csv|txt)$")[:5]:
                try:
                    github_raw(match["raw_url"], OUT / "cryo_refs" / Path(match["path"]).name)
                    silicon_samples.append({"path": match["path"], "raw_url": match["raw_url"]})
                except Exception:
                    pass
        except Exception as e:
            silicon_samples.append({"error": repr(e)})

    high_k_peak = max((t["y_peak"] for t in hbn_tables), default=None)
    source_unavailable = (not downloaded) and bool(resource_errors)
    verdict = "support_like_high_k_side" if high_k_peak is not None and high_k_peak >= 1000 else ("no_physics_verdict" if source_unavailable else "no_support_or_no_numeric_hbn_tables")
    status = "ok" if hbn_tables else ("source_unavailable" if source_unavailable else "source_available_but_no_numeric_tables")
    report = structured_report(
        "MAT7", status,
        ckan_resources=resources_meta,
        resource_errors=resource_errors,
        downloaded_files=[str(p) for p in downloaded],
        expanded_files=[str(p) for p in expanded],
        hbn_tables=hbn_tables,
        hbn_peak_observed=high_k_peak,
        silicon_reference_files=silicon_samples,
        verdict=verdict,
        partial_test_only=True,
        note=("Tests only high thermal-conductivity side of MAT7. "
              "Fallback now reuses cached Bristol Figure CSVs before retrying network downloads, then directly downloads "
              "resource URLs/static Figure CSVs when CKAN metadata is odd."),
    )
    json_dump(report, OUT / "mat7_report.json")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
