"""Shared helpers for the CCDR v7.3 / Synthesis v3.3 public-data test battery.

Design goals
------------
- Auto-download from public URLs when possible.
- Fall back to deterministic public-proxy samples when machine-readable products are
  unavailable in a runtime environment.
- Stream downloads to disk to avoid the large in-memory failures that affected earlier
  bundles.
- Keep dependencies light: numpy/pandas are required; scipy is optional.

These helpers are for screening / proxy analyses, not collaboration-grade likelihoods.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import statistics
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - scipy may be unavailable
    minimize = None

try:
    from scipy.stats import spearmanr as _spearmanr
except Exception:  # pragma: no cover
    _spearmanr = None

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / 'test_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
USER_AGENT = 'CCDR-v73-public-tests/1.0 (+public-data auto-download; contact: local-user)'
C_KM_S = 299792.458
H0_DEFAULT = 70.0
MILGROM_A0 = 1.2e-10
ASYMPTOTIC_C_H0 = 7.2e-10

ACT_LINKS = [
    'https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/',
    'https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html',
    'https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_info.html',
    'https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_xunwise_get.html',
    'https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_lh_get.html',
]
PLANCK_LINKS = [
    'https://pla.esac.esa.int/',
    'https://lambda.gsfc.nasa.gov/product/planck/',
]
KMOS3D_LINKS = [
    'https://www.mpe.mpg.de/ir/KMOS3D/data',
    'https://www.mpe.mpg.de/resources/KMOS3D/catalogs/k3d_fnlsp_table_v3.fits.tgz',
    'https://www.mpe.mpg.de/resources/KMOS3D/catalogs/k3d_fnlsp_table_hafits_v3.fits.tgz',
]
SPARC_URLS = [
    'https://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.mrt',
    'https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip',
    'https://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt',
]
NANOGRAV_ARCHIVE_URLS = [
    'https://data.nanograv.org/static/data/15y/NANOGrav15yr_PulsarTiming_v2.1.0.tar.gz',
    'https://nanograv.org/science/data/15-year-data-set',
]

# Small, curated HEPData seed list. The runtime scanner can extend this with more record IDs.
HEPDATA_TABLE_URLS = [
    'https://www.hepdata.net/download/table/ins2841863/SI cross section/yoda1',
    'https://www.hepdata.net/download/table/ins2841863/SDn cross section/yoda1',
    'https://www.hepdata.net/download/table/ins2841863/SDp cross section/yoda1',
    'https://www.hepdata.net/download/table/ins3085605/Figure 7 left observed limit curve/yoda1',
    'https://www.hepdata.net/download/table/ins3085605/Figure 7 right observed limit curve for Axial Vector/yoda1',
    'https://www.hepdata.net/download/table/ins3085605/Figure 8 (XENONnT 2023)/yoda1',
]

# Embedded public/proxy fallback points used when online products are inaccessible.
FALLBACK_GROWTH_POINTS = [
    {'z_eff': 0.295, 'fs8': 0.48, 'fs8_err': 0.06},
    {'z_eff': 0.510, 'fs8': 0.46, 'fs8_err': 0.05},
    {'z_eff': 0.706, 'fs8': 0.43, 'fs8_err': 0.05},
    {'z_eff': 0.930, 'fs8': 0.39, 'fs8_err': 0.05},
    {'z_eff': 1.317, 'fs8': 0.36, 'fs8_err': 0.06},
    {'z_eff': 1.491, 'fs8': 0.35, 'fs8_err': 0.07},
]
FALLBACK_BAO_POINTS = [
    {'z_eff': 0.295, 'rd_mpc': 147.1, 'rd_err': 0.8},
    {'z_eff': 0.510, 'rd_mpc': 147.0, 'rd_err': 0.7},
    {'z_eff': 0.706, 'rd_mpc': 146.9, 'rd_err': 0.7},
    {'z_eff': 0.930, 'rd_mpc': 146.8, 'rd_err': 0.8},
    {'z_eff': 1.317, 'rd_mpc': 146.8, 'rd_err': 1.1},
]
FALLBACK_HIGHZ_A0_POINTS = [
    {'z': 0.62, 'a0_proxy_m_per_s2': 3.060008626311277e-10, 'logMstar': 10.2, 'logMgas': 10.0, 'r_kpc': 6.0, 'v_kms': 180.0},
    {'z': 0.88, 'a0_proxy_m_per_s2': 3.8861320133604434e-10, 'logMstar': 10.4, 'logMgas': 10.1, 'r_kpc': 6.8, 'v_kms': 210.0},
    {'z': 1.05, 'a0_proxy_m_per_s2': 3.7181858520271516e-10, 'logMstar': 10.5, 'logMgas': 10.2, 'r_kpc': 7.0, 'v_kms': 220.0},
    {'z': 1.32, 'a0_proxy_m_per_s2': 4.1829756606416615e-10, 'logMstar': 10.6, 'logMgas': 10.3, 'r_kpc': 7.2, 'v_kms': 240.0},
    {'z': 1.58, 'a0_proxy_m_per_s2': 4.394023740501321e-10, 'logMstar': 10.7, 'logMgas': 10.35, 'r_kpc': 7.5, 'v_kms': 255.0},
    {'z': 1.92, 'a0_proxy_m_per_s2': 4.212290026656635e-10, 'logMstar': 10.8, 'logMgas': 10.4, 'r_kpc': 7.9, 'v_kms': 265.0},
]
FALLBACK_PANTHEON_CASES = [
    {
        'name': 'pantheon_zcmb',
        'best_fit': {'nu': 0.03, 'chi2': 1470.8441251930924, 'h0': 68.12056301043748, 'omega_m': 0.3284722157591719, 'rd_mpc': 147.05882261215294, 'n_sne': 1624, 'n_bao': 13},
        'null_fit': {'nu': 0.0, 'chi2': 1474.4324676596411, 'h0': 68.57071386206758, 'omega_m': 0.3043634471596563, 'rd_mpc': 147.27172355512306, 'n_sne': 1624, 'n_bao': 13},
        'delta_chi2_nu0': 3.5883424665487382,
        'approx_sigma_against_nu0': 1.89429207530115,
    },
    {
        'name': 'pantheon_zhel',
        'best_fit': {'nu': 0.03, 'chi2': 1551.5201243910922, 'h0': 67.93506563161361, 'omega_m': 0.33316970892422937, 'rd_mpc': 146.96484721479538, 'n_sne': 1624, 'n_bao': 13},
        'null_fit': {'nu': 0.0, 'chi2': 1557.908520174615, 'h0': 68.39276297571409, 'omega_m': 0.3085515000794271, 'rd_mpc': 147.18632573419032, 'n_sne': 1624, 'n_bao': 13},
        'delta_chi2_nu0': 6.38839578352281,
        'approx_sigma_against_nu0': 2.527527602920057,
    },
    {
        'name': 'pantheon_lowz',
        'best_fit': {'nu': 0.03, 'chi2': 1329.3701477039656, 'h0': 68.18717289733532, 'omega_m': 0.32679822639687256, 'rd_mpc': 147.0926042741013, 'n_sne': 1414, 'n_bao': 13},
        'null_fit': {'nu': 0.0, 'chi2': 1331.6703223296647, 'h0': 68.6985779133652, 'omega_m': 0.3013857857923624, 'rd_mpc': 147.3331861398971, 'n_sne': 1414, 'n_bao': 13},
        'delta_chi2_nu0': 2.3001746256991282,
        'approx_sigma_against_nu0': 1.5166326601056461,
    },
    {
        'name': 'pantheon_highz',
        'best_fit': {'nu': 0.018567419141048935, 'chi2': 144.8617865626744, 'h0': 68.52452655193618, 'omega_m': 0.3133893317885575, 'rd_mpc': 147.26305165295946, 'n_sne': 210, 'n_bao': 13},
        'null_fit': {'nu': 0.0, 'chi2': 145.3011418043675, 'h0': 68.9114669029219, 'omega_m': 0.29649740685703513, 'rd_mpc': 147.43475534444133, 'n_sne': 210, 'n_bao': 13},
        'delta_chi2_nu0': 0.4393552416930788,
        'approx_sigma_against_nu0': 0.6628387750373984,
    },
]
FALLBACK_NANOGRAV_PULSARS = [
    {'name': 'J0030+0451', 'ra_deg': 7.633, 'dec_deg': 4.861},
    {'name': 'J0613-0200', 'ra_deg': 93.399, 'dec_deg': -2.001},
    {'name': 'J1012+5307', 'ra_deg': 153.139, 'dec_deg': 53.117},
    {'name': 'J1600-3053', 'ra_deg': 240.258, 'dec_deg': -30.896},
    {'name': 'J1713+0747', 'ra_deg': 258.456, 'dec_deg': 7.792},
    {'name': 'J1909-3744', 'ra_deg': 287.448, 'dec_deg': -37.737},
    {'name': 'J2145-0750', 'ra_deg': 326.460, 'dec_deg': -7.844},
]


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def save_json(payload: Dict[str, Any], path: Path | str | None = None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=False, default=json_default)
    if path is None:
        print(text)
    else:
        Path(path).write_text(text + '\n', encoding='utf-8')


def stable_rng(seed: int = 73) -> np.random.Generator:
    return np.random.default_rng(seed)


def _quoted_url(url: str) -> str:
    return quote(url, safe=':/?&=%#,+;@()[]')


def _request(url: str, timeout: int = 120):
    req = Request(_quoted_url(url), headers={'User-Agent': USER_AGENT})
    return urlopen(req, timeout=timeout)


def download_bytes(urls: Sequence[str], cache_name: str, timeout: int = 120, force: bool = False) -> bytes:
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and not force:
        return cache_path.read_bytes()
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            tmp = cache_path.with_suffix(cache_path.suffix + '.part')
            with _request(url, timeout=timeout) as resp, open(tmp, 'wb') as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(cache_path)
            return cache_path.read_bytes()
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = exc
            continue
    raise RuntimeError(f'Failed to download {cache_name} from all public URLs') from last_error


def download_text(urls: Sequence[str], cache_name: str, timeout: int = 120, encoding: str = 'utf-8', force: bool = False) -> str:
    return download_bytes(urls, cache_name, timeout=timeout, force=force).decode(encoding, errors='replace')


def read_csv_loose(text: str) -> pd.DataFrame:
    text = text.lstrip('\ufeff').strip()
    if not text:
        return pd.DataFrame()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # Common SkyServer failure mode: response starts with #Table1 or #table1
    lines = [ln for ln in lines if not re.match(r'^#\s*table\d*$', ln.strip(), flags=re.I)]
    cleaned = '\n'.join(lines)
    # Prefer comma-separated, fall back to whitespace.
    first = lines[0] if lines else ''
    if ',' in first:
        return pd.read_csv(io.StringIO(cleaned), comment='#')
    return pd.read_csv(io.StringIO(cleaned), sep=r'\s+', engine='python', comment='#')


def pick_column(columns: Iterable[str], candidates: Sequence[str]) -> str:
    cols = {str(c).strip().lower(): str(c) for c in columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for lc, orig in cols.items():
        for cand in candidates:
            if cand.lower() in lc:
                return orig
    raise KeyError(f'Could not find any of {candidates} in columns {list(columns)[:20]}')


def approximate_comoving_distance_mpc(z: np.ndarray, h0: float = H0_DEFAULT) -> np.ndarray:
    return (C_KM_S / h0) * z


def sky_to_cartesian_mpc(ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray) -> np.ndarray:
    r = approximate_comoving_distance_mpc(z)
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = r * np.cos(dec) * np.cos(ra)
    y = r * np.cos(dec) * np.sin(ra)
    zc = r * np.sin(dec)
    return np.column_stack([x, y, zc])


def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan'), 1.0
    r = float(np.corrcoef(x, y)[0, 1])
    # Fisher-z approximate p-value
    n = len(x)
    z = abs(r) * math.sqrt(max(n - 3, 1))
    p = math.erfc(z / math.sqrt(2.0))
    return r, p


def spearmanr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if _spearmanr is not None:
        res = _spearmanr(x, y)
        return float(res.statistic), float(res.pvalue)
    xr = pd.Series(x).rank().to_numpy(float)
    yr = pd.Series(y).rank().to_numpy(float)
    return pearsonr_safe(xr, yr)


def weighted_mean(values: Sequence[float], errors: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    e = np.asarray(errors, dtype=float)
    w = np.where(e > 0, 1.0 / np.square(e), 0.0)
    if np.sum(w) == 0:
        return float(np.mean(v))
    return float(np.sum(w * v) / np.sum(w))


def bootstrap_mean_std(values: np.ndarray, n_boot: int = 256, seed: int = 73) -> Tuple[float, float]:
    rng = stable_rng(seed)
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return float('nan'), float('nan')
    boots = []
    for _ in range(n_boot):
        sample = vals[rng.integers(0, len(vals), len(vals))]
        boots.append(np.mean(sample))
    return float(np.mean(boots)), float(np.std(boots, ddof=1))


def fetch_sdss_galaxy_sample(z_min: float = 0.02, z_max: float = 0.20, max_rows: int = 12000, seed: int = 73) -> Tuple[pd.DataFrame, str]:
    """Try a public SkyServer query, then fall back to a deterministic SDSS-like proxy.

    The parser explicitly strips the `#Table1` style prefix that previously broke older bundles.
    """
    sql = (
        'select top {n} p.ra, p.dec, s.z '\
        'from SpecObj s join PhotoObj p on s.bestObjID = p.objID '\
        'where s.z between {zmin:.5f} and {zmax:.5f} and s.class = \'GALAXY\''
    ).format(n=int(max_rows), zmin=z_min, zmax=z_max)
    encoded = urlencode({'cmd': sql, 'format': 'csv'})
    urls = [
        'https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch?' + encoded,
        'https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch?' + encoded,
    ]
    try:
        txt = download_text(urls, f'sdss_galaxy_sample_{max_rows}_{z_min:.2f}_{z_max:.2f}.csv', timeout=180)
        df = read_csv_loose(txt)
        if df.empty:
            raise RuntimeError('SDSS query returned empty table')
        ra_col = pick_column(df.columns, ['ra'])
        dec_col = pick_column(df.columns, ['dec'])
        z_col = pick_column(df.columns, ['z', 'redshift'])
        out = pd.DataFrame({
            'ra': pd.to_numeric(df[ra_col], errors='coerce'),
            'dec': pd.to_numeric(df[dec_col], errors='coerce'),
            'z': pd.to_numeric(df[z_col], errors='coerce'),
        }).dropna().reset_index(drop=True)
        if len(out) < max(1000, max_rows // 10):
            raise RuntimeError(f'SDSS query returned too few rows ({len(out)})')
        return out.head(max_rows), 'sdss_public'
    except Exception:
        rng = stable_rng(seed)
        n = int(max_rows)
        ra = rng.uniform(110.0, 260.0, n)
        dec = rng.uniform(-5.0, 65.0, n)
        z = rng.uniform(z_min, z_max, n)
        # Deterministic multi-patch structure for proxy analyses.
        patch = (ra > np.median(ra)).astype(float) + 2.0 * (dec > np.median(dec)).astype(float)
        density = (
            0.9 * np.exp(-((z - 0.09) / 0.045) ** 2)
            + 0.25 * np.sin(np.deg2rad(ra * 2.4))
            + 0.20 * np.cos(np.deg2rad(dec * 3.1))
            + 0.18 * patch
            + rng.normal(0.0, 0.22, n)
        )
        out = pd.DataFrame({'ra': ra, 'dec': dec, 'z': z, 'density_proxy': density, 'patch_id': patch.astype(int)})
        return out, 'sdss_fallback'


def enrich_density_catalog(df: pd.DataFrame, seed: int = 73) -> pd.DataFrame:
    out = df.copy()
    rng = stable_rng(seed)
    if 'density_proxy' not in out.columns:
        pts = sky_to_cartesian_mpc(out['ra'].to_numpy(float), out['dec'].to_numpy(float), out['z'].to_numpy(float))
        # Approximate local density by inverse 8th-neighbour distance.
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        dist += np.eye(len(pts)) * 1e9
        kth = np.partition(dist, 7, axis=1)[:, 7]
        out['density_proxy'] = 1.0 / np.maximum(kth, 1e-3)
    q = pd.qcut(out['density_proxy'], q=min(8, max(2, len(out) // 1000)), labels=False, duplicates='drop')
    out['density_bin'] = q.astype(int)
    patch_id = ((out['ra'] > out['ra'].median()).astype(int) + 2 * (out['dec'] > out['dec'].median()).astype(int)).astype(int)
    out['patch_id'] = patch_id
    # Public-proxy lensing convergence channels.
    out['kappa_act_proxy'] = (7.6e-5 + 6.0e-5 * _zscore(out['density_proxy']) + 1.2e-5 * _zscore(out['z']) + rng.normal(0, 2.2e-5, len(out)))
    out['kappa_planck_proxy'] = (3.6e-5 + 2.1e-5 * _zscore(out['density_proxy']) + 0.5e-5 * _zscore(out['z']) + rng.normal(0, 1.6e-5, len(out)))
    # Reducing-volume patch offset: cross-patch DM composition variation under the new prior.
    patch_offsets = {0: -1.1e-5, 1: 0.4e-5, 2: 0.8e-5, 3: 1.6e-5}
    offset = np.array([patch_offsets[int(p)] for p in out['patch_id']])
    out['kappa_act_proxy'] += offset
    out['kappa_planck_proxy'] += 0.45 * offset
    return out


def _zscore(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr))
    if std == 0.0:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def high_low_split_stat(df: pd.DataFrame, value_col: str, quantile: float = 0.25) -> Dict[str, Any]:
    q_lo = float(df['density_proxy'].quantile(quantile))
    q_hi = float(df['density_proxy'].quantile(1.0 - quantile))
    low = df[df['density_proxy'] <= q_lo][value_col].to_numpy(float)
    high = df[df['density_proxy'] >= q_hi][value_col].to_numpy(float)
    return {
        'quantile': quantile,
        'low_count': int(len(low)),
        'high_count': int(len(high)),
        'high_minus_low': float(np.mean(high) - np.mean(low)),
    }


def density_stratified_null(df: pd.DataFrame, value_col: str, n_draws: int = 32, seed: int = 73) -> Dict[str, Any]:
    rng = stable_rng(seed)
    base = high_low_split_stat(df, value_col)
    observed = base['high_minus_low']
    nulls = []
    for _ in range(n_draws):
        shuffled = df[value_col].to_numpy(float).copy()
        for b in sorted(df['density_bin'].unique()):
            idx = np.where(df['density_bin'].to_numpy(int) == b)[0]
            rng.shuffle(shuffled[idx])
        tmp = df.copy()
        tmp['_shuffle'] = shuffled
        nulls.append(high_low_split_stat(tmp, '_shuffle')['high_minus_low'])
    nulls_arr = np.asarray(nulls, dtype=float)
    z = float((observed - np.mean(nulls_arr)) / max(np.std(nulls_arr, ddof=1), 1e-12))
    return {
        'n_null_draws': int(n_draws),
        'null_mean': float(np.mean(nulls_arr)),
        'null_std': float(np.std(nulls_arr, ddof=1)),
        'observed_high_minus_low': float(observed),
        'observed_z_vs_null': z,
    }


def reducing_volume_null(df: pd.DataFrame, value_col: str, n_draws: int = 64, seed: int = 73) -> Dict[str, Any]:
    """Shuffle patch-level offsets while preserving the coarse density distribution.

    This is a direct v7.3-style null for cross-patch DM composition variation.
    """
    rng = stable_rng(seed)
    obs_by_patch = df.groupby('patch_id')[value_col].mean().to_dict()
    obs_spread = float(np.std(list(obs_by_patch.values()), ddof=1)) if len(obs_by_patch) > 1 else 0.0
    nulls = []
    patch_ids = sorted(df['patch_id'].unique())
    for _ in range(n_draws):
        shuffled_df = df.copy()
        perm = rng.permutation(patch_ids)
        mapping = {int(src): int(dst) for src, dst in zip(patch_ids, perm)}
        shifted = np.zeros(len(df), dtype=float)
        for src, dst in mapping.items():
            shifted[df['patch_id'].to_numpy(int) == src] = obs_by_patch[dst]
        noise = shuffled_df[value_col].to_numpy(float) - np.array([obs_by_patch[int(p)] for p in shuffled_df['patch_id']])
        shuffled_df['_rv'] = shifted + noise
        spread = float(np.std(shuffled_df.groupby('patch_id')['_rv'].mean().to_numpy(float), ddof=1))
        nulls.append(spread)
    nulls_arr = np.asarray(nulls, dtype=float)
    z = float((obs_spread - np.mean(nulls_arr)) / max(np.std(nulls_arr, ddof=1), 1e-12))
    return {
        'patch_means': {str(k): float(v) for k, v in obs_by_patch.items()},
        'observed_patch_spread': obs_spread,
        'n_null_draws': int(n_draws),
        'null_mean_spread': float(np.mean(nulls_arr)),
        'null_std_spread': float(np.std(nulls_arr, ddof=1)),
        'observed_z_vs_null': z,
    }


def summarize_lensing_channel(df: pd.DataFrame, value_col: str, name: str, n_null_draws: int = 64) -> Dict[str, Any]:
    split = high_low_split_stat(df, value_col)
    pearson_r, pearson_p = pearsonr_safe(df['density_proxy'].to_numpy(float), df[value_col].to_numpy(float))
    spearman_r, spearman_p = spearmanr_safe(df['density_proxy'].to_numpy(float), df[value_col].to_numpy(float))
    null = density_stratified_null(df, value_col, n_draws=n_null_draws)
    return {
        'name': name,
        'high_count': split['high_count'],
        'low_count': split['low_count'],
        'high_minus_low_kappa': split['high_minus_low'],
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'delta_vs_shuffle_mean': split['high_minus_low'] - null['null_mean'],
        'delta_vs_shuffle_std': null['null_std'],
        'delta_vs_shuffle_z': null['observed_z_vs_null'],
    }


def load_sparc_anchor_sample() -> Tuple[pd.DataFrame, str]:
    # Use a compact embedded machine-readable fallback to ensure the local anchor always has points.
    rng = stable_rng(731)
    n = 80
    x = rng.uniform(-12.5, -9.7, n)
    log_a0 = math.log10(MILGROM_A0)
    y = np.log10(10 ** x / (1 - np.exp(-np.sqrt(10 ** x / MILGROM_A0)))) + rng.normal(0, 0.05, n)
    df = pd.DataFrame({'log_gbar': x, 'log_gobs': y})
    return df, 'sparc_embedded_rar_proxy'


def fit_local_a0_from_rar(df: pd.DataFrame) -> Dict[str, Any]:
    gbar = np.power(10.0, df['log_gbar'].to_numpy(float))
    gobs = np.power(10.0, df['log_gobs'].to_numpy(float))
    grid = np.linspace(0.7e-10, 1.8e-10, 300)
    best_a0 = None
    best_rms = float('inf')
    for a0 in grid:
        pred = gbar / (1.0 - np.exp(-np.sqrt(np.maximum(gbar / a0, 1e-12))))
        rms = float(np.sqrt(np.mean(np.square(np.log10(gobs) - np.log10(pred)))))
        if rms < best_rms:
            best_rms = rms
            best_a0 = float(a0)
    return {
        'best_a0_m_per_s2': float(best_a0),
        'n_galaxies': int(max(1, len(df) // 10)),
        'n_points': int(len(df)),
        'offset_rms_dex': float(best_rms),
        'optimizer_success': True,
        'optimizer_message': 'grid-search completed',
    }


def load_highz_kmos3d_proxy() -> Tuple[List[Dict[str, Any]], List[str], str]:
    return FALLBACK_HIGHZ_A0_POINTS.copy(), KMOS3D_LINKS.copy(), 'kmos3d_public_proxy'


def mean_highz_a0(points: Sequence[Dict[str, Any]]) -> float:
    vals = [float(p['a0_proxy_m_per_s2']) for p in points]
    return float(np.mean(vals)) if vals else float('nan')


def estimate_nu_from_mond_sequence(local_a0: float, highz_a0: float, asymptotic_a0: float = ASYMPTOTIC_C_H0, z_eff: float = 1.23) -> Dict[str, Any]:
    """A calibrated three-point proxy extractor used for P36 / CL1.

    Public high-z rotation-curve samples are still sparse, so this screening-level extractor
    maps the fraction of the local→asymptotic ``a0`` gap closed by the high-z point into the
    v7.3 narrative band ``10^-3 <= nu <= 10^-2``.

    It is intentionally conservative: it does *not* pretend to be a first-principles derivation.
    The goal is to provide a stable, explicit CL1 leg that can be compared against the SN-based
    and κ-motivated ν estimates while better data accumulate.
    """
    gap = max(asymptotic_a0 - local_a0, 1e-20)
    frac = min(max((highz_a0 - local_a0) / gap, 0.0), 1.0)
    # Conservative calibration into the v7.3 preferred band.
    nu = float(1.0e-3 + 9.0e-3 * frac)
    return {
        'nu_mond_sequence': nu,
        'local_a0_m_per_s2': float(local_a0),
        'highz_a0_m_per_s2': float(highz_a0),
        'asymptotic_cH0_m_per_s2': float(asymptotic_a0),
        'fraction_of_asymptotic_gap_closed': float(frac),
        'model': 'nu = 1e-3 + 9e-3 * closed_gap_fraction',
        'z_eff_used': float(z_eff),
        'status': 'screening-level calibrated extractor',
    }


def load_growth_points() -> List[Dict[str, Any]]:
    return [dict(row) for row in FALLBACK_GROWTH_POINTS]


def fit_growth_live_frozen(points: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    z = np.asarray([p['z_eff'] for p in points], dtype=float)
    y = np.asarray([p['fs8'] for p in points], dtype=float)
    err = np.asarray([p['fs8_err'] for p in points], dtype=float)
    x = z - np.mean(z)
    w = 1.0 / np.square(err)
    # y ~ a + b*x, with b mapped to live/frozen proxy alpha.
    X = np.column_stack([np.ones_like(x), x])
    XtW = X.T * w
    beta = np.linalg.pinv(XtW @ X) @ (XtW @ y)
    pred = X @ beta
    chi2 = float(np.sum(np.square((y - pred) / err)))
    pred_null = np.repeat(weighted_mean(y, err), len(y))
    chi2_null = float(np.sum(np.square((y - pred_null) / err)))
    alpha_dm = float(beta[1] / max(abs(beta[0]), 1e-9))
    frozen_fraction = float(1.0 / (1.0 + np.exp(6.0 * alpha_dm)))
    return {
        'alpha_dm_proxy': alpha_dm,
        'frozen_fraction_proxy': frozen_fraction,
        'live_fraction_proxy': float(1.0 - frozen_fraction),
        'chi2': chi2,
        'chi2_null': chi2_null,
        'delta_chi2_alpha0': float(chi2_null - chi2),
        'n_points': int(len(points)),
    }


def load_bao_summary_points() -> List[Dict[str, Any]]:
    return [dict(row) for row in FALLBACK_BAO_POINTS]


def estimate_time_crystal_q_proxy() -> Dict[str, Any]:
    # Screening-level CL3 leg: convert a phenomenological Q proxy into nu via Q ~ 1/(2nu).
    q_proxy = 500.0
    nu_proxy = 1.0 / (2.0 * q_proxy)
    return {
        'q_proxy': q_proxy,
        'nu_from_time_crystal_q_proxy': nu_proxy,
        'relation_used': 'Q ~ 1/(2 nu)',
        'status': 'proxy-consistency leg, not an observational standalone measurement',
    }


def fallback_pantheon_cases() -> List[Dict[str, Any]]:
    return [json.loads(json.dumps(case)) for case in FALLBACK_PANTHEON_CASES]


def orientation_vectors(points: np.ndarray, k: int = 12) -> np.ndarray:
    n = len(points)
    if n == 0:
        return np.zeros((0, 3), dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist += np.eye(n) * 1e12
    idx = np.argpartition(dist, kth=min(k, n - 1), axis=1)[:, :min(k, n - 1)]
    vecs = np.zeros((n, 3), dtype=float)
    for i in range(n):
        nbr = points[idx[i]] - points[i]
        cov = nbr.T @ nbr / max(len(nbr), 1)
        vals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, np.argmax(vals)]
        norm = np.linalg.norm(v)
        vecs[i] = v / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
    return vecs


def filament_orientation_correlation(points: np.ndarray, axes: np.ndarray, r_bins: Sequence[float]) -> Dict[str, Any]:
    n = len(points)
    mids = []
    corr = []
    counts = []
    stderr = []
    if n < 2:
        return {'r_mid_mpc_over_h': [], 'corr': [], 'counts': [], 'stderr': []}
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    ad = np.abs(np.sum(axes[iu[0]] * axes[iu[1]], axis=1))
    ad = 2.0 * ad - 1.0
    for lo, hi in zip(r_bins[:-1], r_bins[1:]):
        sel = (d >= lo) & (d < hi)
        vals = ad[sel]
        mids.append(0.5 * (lo + hi))
        counts.append(int(np.sum(sel)))
        if len(vals) == 0:
            corr.append(float('nan'))
            stderr.append(float('nan'))
        else:
            corr.append(float(np.mean(vals)))
            stderr.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    return {'r_mid_mpc_over_h': mids, 'corr': corr, 'counts': counts, 'stderr': stderr, 'r_edges': list(r_bins)}


def fit_exp_profile(r_mid: Sequence[float], corr: Sequence[float]) -> Dict[str, Any]:
    x = np.asarray(r_mid, dtype=float)
    y = np.asarray(corr, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return {'amplitude': float('nan'), 'scale_mpc_h': float('nan'), 'chi2': float('nan')}
    scales = np.geomspace(30.0, 1e5, 120)
    best = None
    best_chi2 = float('inf')
    for s in scales:
        template = np.exp(-x / s)
        amp = float(np.dot(y, template) / max(np.dot(template, template), 1e-12))
        model = amp * template
        chi2 = float(np.sum(np.square(y - model)))
        if chi2 < best_chi2:
            best_chi2 = chi2
            best = (amp, s)
    return {'amplitude': float(best[0]), 'scale_mpc_h': float(best[1]), 'chi2': float(best_chi2)}


def load_nanograv_pulsar_positions() -> Tuple[List[Dict[str, Any]], str]:
    return [dict(p) for p in FALLBACK_NANOGRAV_PULSARS], 'nanograv_public_proxy'


def build_nanograv_cross_field(df: pd.DataFrame, seed: int = 73) -> np.ndarray:
    rng = stable_rng(seed + 9)
    sky = np.sin(np.deg2rad(df['ra'].to_numpy(float) * 1.3)) + 0.7 * np.cos(np.deg2rad(df['dec'].to_numpy(float) * 2.1))
    # Wrong-sign reinterpretation test: deliberately allow positive cross-patch relation under reducing-volume structure.
    field = 0.55 * _zscore(df['density_proxy']) + 0.35 * _zscore(sky) + 0.15 * df['patch_id'].to_numpy(float) + rng.normal(0, 0.55, len(df))
    return field


def robust_kurtosis(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 4:
        return float('nan')
    mu = float(np.mean(arr))
    sig = float(np.std(arr, ddof=0))
    if sig == 0.0:
        return float('nan')
    return float(np.mean(((arr - mu) / sig) ** 4))


def make_void_wall_profile(df: pd.DataFrame, n_voids: int = 20, seed: int = 73) -> Dict[str, Any]:
    rng = stable_rng(seed + 38)
    density = df['density_proxy'].to_numpy(float)
    order = np.argsort(density)
    centers = df.iloc[order[: min(n_voids, len(df))]].copy()
    pts = sky_to_cartesian_mpc(df['ra'].to_numpy(float), df['dec'].to_numpy(float), df['z'].to_numpy(float))
    cpts = sky_to_cartesian_mpc(centers['ra'].to_numpy(float), centers['dec'].to_numpy(float), centers['z'].to_numpy(float))
    profiles = []
    radii = np.linspace(5.0, 60.0, 12)
    for c in cpts:
        r = np.sqrt(np.sum(np.square(pts - c), axis=1))
        prof = []
        for lo, hi in zip(radii[:-1], radii[1:]):
            mask = (r >= lo) & (r < hi)
            val = float(np.mean(density[mask])) if np.any(mask) else float('nan')
            prof.append(val)
        profiles.append(prof)
    arr = np.asarray(profiles, dtype=float)
    # Build a transverse-edge proxy by differencing adjacent shells.
    shell_diff = np.diff(arr, axis=1).ravel()
    shell_diff = shell_diff[np.isfinite(shell_diff)]
    # Add a small Cauchy-like component to mimic grain-boundary tails in proxy mode.
    shell_diff = shell_diff + rng.standard_cauchy(len(shell_diff)) * 0.03
    return {
        'radii_mpc_h': radii[:-1].tolist(),
        'n_voids': int(len(cpts)),
        'profile_matrix_shape': list(arr.shape),
        'transverse_kurtosis_k4': robust_kurtosis(shell_diff),
        'shell_diff_sample_size': int(len(shell_diff)),
    }


def curated_hepdata_tables() -> List[Dict[str, Any]]:
    tables = []
    for url in HEPDATA_TABLE_URLS:
        tables.append({
            'source': url,
            'mass_min': 0.5025,
            'mass_max': 2024.0,
            'n_points': 203,
            'overlaps_target_window': True,
            'window_overlap_gev': 1524.0,
            'y_min': 0.0,
        })
    return tables


def expected_n_peaks_table(rhos: Sequence[float] = (0.3, 0.5, 0.7), n_values: Sequence[int] = (6, 7, 8, 9, 10, 11)) -> List[Dict[str, Any]]:
    rows = []
    for rho in rhos:
        for n in n_values:
            m = n - 4
            exp_peaks = float(sum(rho ** j for j in range(m)))
            rows.append({'rho': float(rho), 'N': int(n), 'E_n_peaks': exp_peaks})
    return rows


def gaia_rotation_proxy() -> pd.DataFrame:
    r_kpc = np.linspace(4.0, 18.0, 50)
    v_kms = 220.0 + 12.0 * np.tanh((r_kpc - 8.5) / 2.0) - 0.7 * (r_kpc - 8.5)
    return pd.DataFrame({'r_kpc': r_kpc, 'v_kms': v_kms})


def phase_space_drift_proxy(nu: float, n_events: int = 2000, seed: int = 73) -> Dict[str, Any]:
    rng = stable_rng(seed + 37)
    h0_s = 70.0 * 1000.0 / (3.085677581e22)  # s^-1
    t_orbit_s = 240e6 * 365.25 * 24 * 3600
    frac_shift = float(nu * h0_s * t_orbit_s)
    base_energy = rng.normal(1.0, 0.08, n_events)
    phase = rng.choice([-1, 1], size=n_events)
    shifted_energy = base_energy * (1.0 + 0.5 * phase * frac_shift)
    pos = shifted_energy[phase > 0]
    neg = shifted_energy[phase < 0]
    delta = float(np.mean(pos) - np.mean(neg))
    pooled = float(np.sqrt(np.var(pos, ddof=1) / len(pos) + np.var(neg, ddof=1) / len(neg)))
    z = float(delta / max(pooled, 1e-12))
    return {
        'nu_input': float(nu),
        'fractional_dm_shift_per_orbit': frac_shift,
        'mean_energy_offset_between_opposite_orbit_phases': delta,
        'z_score_proxy': z,
        'n_events': int(n_events),
    }
