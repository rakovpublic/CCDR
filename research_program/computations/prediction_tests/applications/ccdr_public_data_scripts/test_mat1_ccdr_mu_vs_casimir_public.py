from __future__ import annotations

"""MAT1 public-data audit with stricter confirmation levels.

MAT1 in the note predicts kappa(T)=kappa0*mu(lambda/L_grain), differing
5-15% from Casimir. Public cryogenic tables rarely include grain size and
mean-free-path metadata, so this script separates:
  1) full numeric confirmation (requires explicit grain-size metadata; usually absent),
  2) proxy support: grain/nano/composite candidates show low-T exponents clearly
     below a simple boundary-limited phonon alpha~3 control,
  3) no support for the stronger MAT3-like alpha~0.5 claim.
It also attempts to discover better public nanocrystalline thermal-conductivity
resources from the ACS Figshare collection for nanocrystalline silicon.
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from _common_public_data import ensure_dir, github_find_blob, github_raw, json_dump, structured_report, cached_download

ROOT = Path(__file__).resolve().parent
OUT = ensure_dir(ROOT / 'outputs' / 'mat1')
OWNER = 'CMB-S4'
REPO = 'Cryogenic_Material_Properties'
GRAIN_RE = re.compile(r'nano|nanocrystal|polycrystal|grain|ceramic|composite|porous|wood|graphite|cfrp|g10|fr4', re.I)
CLEAN_CONTROL_RE = re.compile(r'Aluminum|Al_1100|Al_6061|Silicon/RAW|Cu_OFHC', re.I)

# Public Figshare collection noted by ACS for thermal conductivity of nanocrystalline silicon.
# The script uses it only when files are machine-readable tables; PDFs are metadata only.
ACS_NANO_SI_COLLECTION_ID = 2565958


def read_table(path: Path) -> pd.DataFrame | None:
    for sep in [',', '\t', ';', r'\s+', r',|;|\t|\s+']:
        try:
            df = pd.read_csv(path, engine='python', sep=sep)
            if df.shape[0] >= 6 and df.shape[1] >= 2:
                return df
        except Exception:
            continue
    try:
        if path.suffix.lower() in {'.xlsx', '.xls'}:
            df = pd.read_excel(path)
            if df.shape[0] >= 6 and df.shape[1] >= 2:
                return df
    except Exception:
        pass
    return None


def numeric_cols(df: pd.DataFrame):
    out = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() >= max(6, len(df)//3):
            out.append((str(c), s))
    return out


def choose_xy(df):
    nums = numeric_cols(df)
    if len(nums) < 2:
        return None
    ti = next((i for i,(n,_) in enumerate(nums) if 'temp' in n.lower() or n.lower() in {'t','t/k','temperature'}), 0)
    xname, x = nums[ti]
    ylist = [(n,s) for i,(n,s) in enumerate(nums) if i != ti]
    yname, y = max(ylist, key=lambda ns: (1 if any(k in ns[0].lower() for k in ['cond','kappa','w/m','tc','lambda']) else 0, pd.to_numeric(ns[1], errors='coerce').max()))
    return xname, yname, x, y


def fit_alpha(x, y):
    mask = x.notna() & y.notna() & (x > 0) & (y > 0)
    xx = x[mask].to_numpy(float); yy = y[mask].to_numpy(float)
    if len(xx) < 8:
        return None
    idx = np.argsort(xx); xx=xx[idx]; yy=yy[idx]
    low = xx <= 50
    if low.sum() < 6:
        low = xx <= np.quantile(xx, 0.33)
    if low.sum() < 6:
        return None
    alpha, loga = np.polyfit(np.log(xx[low]), np.log(yy[low]), 1)
    pred = loga + alpha*np.log(xx[low])
    residual = float(np.sqrt(np.mean((np.log(yy[low])-pred)**2)))
    ss = float(np.sum((np.log(yy[low])-np.mean(np.log(yy[low])))**2))
    r2 = float(1 - np.sum((np.log(yy[low])-pred)**2)/ss) if ss > 0 else float('nan')
    return {'alpha': float(alpha), 'n_lowT': int(low.sum()), 't_min': float(xx[low].min()), 't_max': float(xx[low].max()), 'log_rmse': residual, 'loglog_r2': r2}


def analyse_file(path: Path, source_label: str, source_url: str | None = None) -> dict | None:
    df = read_table(path)
    if df is None:
        return None
    xy = choose_xy(df)
    if xy is None:
        return None
    xname, yname, x, y = xy
    f = fit_alpha(x, y)
    if f is None:
        return None
    text = f'{source_label} {xname} {yname}'
    return {
        'source_path': source_label, 'source_url': source_url,
        'x_column': xname, 'y_column': yname,
        'grain_or_nano_candidate': bool(GRAIN_RE.search(text)),
        'clean_casimir_like_control_candidate': bool(CLEAN_CONTROL_RE.search(text)),
        **f,
        'closer_to_half_than_3': bool(abs(f['alpha']-0.5) < abs(f['alpha']-3.0)),
        'near_half': bool(abs(f['alpha']-0.5) <= 0.2 and f.get('loglog_r2',0) >= 0.7),
        'strong_deviation_from_alpha3': bool(f['alpha'] < 2.0 and f.get('loglog_r2',0) >= 0.7),
    }


def figshare_collection_files(collection_id: int) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    try:
        meta = requests.get(f'https://api.figshare.com/v2/collections/{collection_id}', timeout=60).json()
        articles_url = f'https://api.figshare.com/v2/collections/{collection_id}/articles'
        articles = requests.get(articles_url, timeout=60).json()
        for art in articles if isinstance(articles, list) else []:
            aid = art.get('id') or art.get('article_id')
            if not aid:
                continue
            detail = requests.get(f'https://api.figshare.com/v2/articles/{aid}', timeout=60).json()
            for f in detail.get('files', []) or []:
                files.append({'article_id': aid, 'article_title': detail.get('title'), 'name': f.get('name'), 'download_url': f.get('download_url'), 'size': f.get('size')})
    except Exception as e:
        files.append({'error': repr(e)})
    return files


def main():
    analyses=[]; downloaded=[]; extra_sources=[]

    matches = github_find_blob(OWNER, REPO, r'thermal_conductivity/.+\.(csv|txt|dat)$')
    # Favor likely boundary/grain-sensitive entries but include enough clean controls.
    matches = sorted(matches, key=lambda m: (0 if GRAIN_RE.search(m['path']) else 1 if CLEAN_CONTROL_RE.search(m['path']) else 2, m['path']))[:180]
    for m in matches:
        dest = OUT/'data'/re.sub(r'[^A-Za-z0-9_.-]+','_',m['path'])
        try:
            github_raw(m['raw_url'], dest)
            downloaded.append({'path': m['path'], 'raw_url': m['raw_url']})
            a = analyse_file(dest, m['path'], m['raw_url'])
            if a:
                analyses.append(a)
        except Exception:
            continue

    # Attempt better nanocrystalline data discovery from ACS Figshare collection.
    for f in figshare_collection_files(ACS_NANO_SI_COLLECTION_ID):
        extra_sources.append(f)
        url = f.get('download_url') if isinstance(f, dict) else None
        name = str(f.get('name','')) if isinstance(f, dict) else ''
        if not url or not re.search(r'\.(csv|txt|dat|xlsx|xls)$', name, re.I):
            continue
        try:
            dest = OUT/'figshare_nano_si'/re.sub(r'[^A-Za-z0-9_.-]+','_',name)
            cached_download(url, dest)
            a = analyse_file(dest, f'figshare:{name}', url)
            if a:
                a['grain_or_nano_candidate'] = True
                analyses.append(a)
        except Exception as e:
            extra_sources[-1]['download_parse_error'] = repr(e)

    candidates = [a for a in analyses if a['grain_or_nano_candidate']]
    clean_controls = [a for a in analyses if a['clean_casimir_like_control_candidate'] and a['loglog_r2'] >= 0.7]
    near_half = [a for a in candidates if a['near_half']]
    deviation_support = [a for a in candidates if a['strong_deviation_from_alpha3']]
    alpha3_controls = [a for a in clean_controls if abs(a['alpha']-3.0) <= 0.75]

    def median(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else None

    # Conservative verdict hierarchy.
    if near_half:
        verdict = 'support_like_near_half_proxy'
    elif deviation_support and alpha3_controls:
        verdict = 'weak_support_boundary_deviation_vs_alpha3_controls'
    elif deviation_support:
        verdict = 'weak_support_boundary_deviation_no_clean_control'
    elif candidates:
        verdict = 'no_support_in_grain_candidates'
    else:
        verdict = 'no_grain_metadata_candidates'

    report = structured_report(
        'MAT1','ok' if analyses else 'no_machine_readable_tables',
        prediction='MAT1: kappa=kappa0*mu(lambda/L_grain), 5-15% deviation from Casimir; public proxy uses low-T exponent shifts.',
        repository=f'https://github.com/{OWNER}/{REPO}',
        better_data_attempt={'acs_figshare_collection_id': ACS_NANO_SI_COLLECTION_ID, 'files_seen': extra_sources[:40]},
        n_downloaded=len(downloaded), n_analysed=len(analyses), n_grain_or_nano_candidates=len(candidates),
        n_clean_control_candidates=len(clean_controls), n_alpha3_clean_controls=len(alpha3_controls),
        median_alpha_grain_or_nano=median([a['alpha'] for a in candidates]),
        median_alpha_clean_controls=median([a['alpha'] for a in clean_controls]),
        verdict=verdict,
        confirmation_level='proxy only; full MAT1 needs grain size L_grain and kappa0/lambda metadata, not just kappa(T)',
        support_near_half=near_half[:20], support_boundary_deviation=deviation_support[:20], alpha3_controls=alpha3_controls[:20], analyses=analyses[:100],
    )
    json_dump(report, OUT/'mat1_report.json')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
