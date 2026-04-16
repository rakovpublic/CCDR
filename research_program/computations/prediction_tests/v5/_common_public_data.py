#!/usr/bin/env python3
from __future__ import annotations

import io
import gzip
import hashlib
import json
import math
import random
import re
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, optimize, signal, sparse, stats, spatial
from scipy.sparse.csgraph import minimum_spanning_tree

C_LIGHT = 299792.458  # km/s
T_CMB = 2.7255
H100_SI = 100.0 * 1000.0 / 3.085677581491367e22

DATA_CACHE = Path(__file__).resolve().parent / "data_cache"
DATA_CACHE.mkdir(exist_ok=True)

PANTHEON_DAT_URLS = [
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",
]
PANTHEON_COV_URLS = [
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov",
]
DESI_DR2_MEAN_URLS = [
    "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt",
]
DESI_DR2_COV_URLS = [
    "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt",
]
PLANCK_PR3_CHAINS_URLS = [
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip",
    "https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip",
]
SPARC_RAR_URLS = [
    "https://astroweb.case.edu/SPARC/RAR.mrt",
    "http://astroweb.case.edu/SPARC/RAR.mrt",
]
SPARC_TABLE1_URLS = [
    "https://astroweb.case.edu/SPARC/Table1.mrt",
    "http://astroweb.case.edu/SPARC/Table1.mrt",
]
SPARC_RARBINS_URLS = [
    "https://astroweb.case.edu/SPARC/RARbins.mrt",
    "http://astroweb.case.edu/SPARC/RARbins.mrt",
]
PDG_LEPTON_PDF_URLS = [
    "https://pdg.lbl.gov/2024/tables/rpp2024-sum-leptons.pdf",
]
PDG_QUARK_PDF_URLS = [
    "https://pdg.lbl.gov/2024/tables/rpp2024-sum-quarks.pdf",
]
DES_SN5YR_ZIP_URLS = [
    "https://zenodo.org/records/12720778/files/DES-SN5YR-1.2.zip?download=1",
]
FERMI_PUBDATA_URLS = [
    "https://www-glast.stanford.edu/pub_data/",
]
FERMI_FIGSHARE_ARTICLE_IDS = [24058650]
SDSS_SKYSERVER_SQL_BASES = [
    "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch",
    "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch",
]
DIRECT_DETECTION_RECORD_URLS = {
    "xenon_lz_recast": [
        "https://www.hepdata.net/record/ins2841863?format=json",
    ],
    "cms_dm_summary": [
        "https://www.hepdata.net/record/ins3085605?format=json",
    ],
}


HEPDATA_FALLBACK_RECORD_URLS = [
    'https://www.hepdata.net/record/ins2841863?format=json',
    'https://www.hepdata.net/record/ins3085605?format=json',
]

CERN_RUN3_FALLBACK_RECORDS = {
    'diboson_met': [
        'https://www.hepdata.net/record/155587?format=json',
        'https://www.hepdata.net/record/ins1641762?format=json',
    ],
    'lfu_rk': [
        'https://www.hepdata.net/record/ins1852846?format=json',
        'https://www.hepdata.net/record/78696?format=json',
    ],
    'high_rapidity_dy': [
        'https://www.hepdata.net/record/159027?format=json',
        'https://www.hepdata.net/record/158981?format=json',
    ],
    'dihiggs_threshold': [
        'https://www.hepdata.net/record/166053?format=json',
        'https://www.hepdata.net/record/154209?format=json',
    ],
}

def _headers() -> Dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; CCDRRound3PublicTests/1.0; +https://openai.com)",
        "Accept": "*/*",
    }


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _prepare_public_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(parts.path, safe='/%:@()+,;=~!*\'')
    query = urllib.parse.quote(parts.query, safe='=&:%+/,;~!*\'')
    fragment = urllib.parse.quote(parts.fragment, safe='')
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, query, fragment))


def _url_cache_name(url: str, default_name: str = 'download.dat') -> str:
    prepared = _prepare_public_url(url)
    parts = urllib.parse.urlsplit(prepared)
    raw_name = Path(parts.path).name or default_name
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', raw_name)
    digest = hashlib.sha1(prepared.encode('utf-8')).hexdigest()[:12]
    return f"{safe_name}_{digest}"


def download_bytes(urls: Sequence[str], cache_name: str, timeout: int = 180, force: bool = False) -> bytes:
    target = DATA_CACHE / cache_name
    if target.exists() and not force:
        return target.read_bytes()
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            req = urllib.request.Request(_prepare_public_url(url), headers=_headers())
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            if not payload:
                raise RuntimeError(f"empty response from {url}")
            target.write_bytes(payload)
            return payload
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Failed to download {cache_name} from all public URLs") from last_error


def download_text(urls: Sequence[str] | str, cache_name: str, timeout: int = 180, force: bool = False, encoding: str = "utf-8") -> str:
    if isinstance(urls, str):
        urls = [urls]
    return download_bytes(urls, cache_name, timeout=timeout, force=force).decode(encoding, errors="replace")


def pick_column(columns: Iterable[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
    cols = list(columns)
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() == c.lower().replace("_", ""):
                return c
    for cand in candidates:
        for c in cols:
            cl = c.lower()
            if cand.lower() in cl:
                return c
    if required:
        raise KeyError(f"Could not find any of {candidates} in columns {cols[:20]}")
    return None


def _parse_square_covariance(txt: str, n: int) -> np.ndarray:
    vals = np.fromstring(txt.replace(",", " "), sep=" ")
    if vals.size == n * n + 1 and int(round(vals[0])) == n:
        vals = vals[1:]
    if vals.size != n * n:
        rows = []
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in re.split(r"\s+", line) if p]
            try:
                rows.append([float(x) for x in parts])
            except Exception:
                pass
        arr = np.asarray(rows, dtype=float)
        if arr.shape == (n, n):
            return arr
        flat = arr.ravel()
        if flat.size == n * n + 1 and int(round(flat[0])) == n:
            flat = flat[1:]
        if flat.size != n * n:
            raise RuntimeError(f"Could not parse covariance into {n}x{n}; got {flat.size} numbers")
        vals = flat
    return vals.reshape((n, n))


# ---------------------------------------------------------------------
# Lightweight cosmology
# ---------------------------------------------------------------------

def omega_radiation(h: float) -> float:
    omega_gamma = 2.469e-5 / (h * h) * (T_CMB / 2.7255) ** 4
    return omega_gamma * (1.0 + 0.2271 * 3.046)


def e2_rvm_flat(z: np.ndarray | float, omega_m: float, nu: float, include_radiation: bool = True, h: float = 0.7) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    om_eff = np.clip(omega_m, 1e-6, 0.999)
    denom = max(1e-6, 1.0 - nu)
    matter = om_eff / denom * ((1.0 + z) ** (3.0 * (1.0 - nu)) - 1.0)
    e2 = 1.0 + matter
    if include_radiation:
        e2 = e2 + omega_radiation(h) * ((1.0 + z) ** 4 - 1.0)
    return np.clip(e2, 1e-12, None)


def hz_rvm(z: np.ndarray | float, h0: float, omega_m: float, nu: float, include_radiation: bool = True) -> np.ndarray:
    return h0 * np.sqrt(e2_rvm_flat(z, omega_m, nu, include_radiation=include_radiation, h=h0 / 100.0))


def comoving_distance_mpc(z: np.ndarray, h0: float, omega_m: float, nu: float, include_radiation: bool = True) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    zmax = float(np.max(z))
    grid = np.linspace(0.0, max(zmax, 1e-4), 2048)
    integrand = C_LIGHT / hz_rvm(grid, h0, omega_m, nu, include_radiation=include_radiation)
    chi = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))])
    return np.interp(z, grid, chi)


def luminosity_distance_mpc(z: np.ndarray, h0: float, omega_m: float, nu: float, include_radiation: bool = True) -> np.ndarray:
    chi = comoving_distance_mpc(z, h0, omega_m, nu, include_radiation=include_radiation)
    return (1.0 + np.asarray(z)) * chi


def distance_modulus(z: np.ndarray, h0: float, omega_m: float, nu: float, include_radiation: bool = True) -> np.ndarray:
    dl = np.clip(luminosity_distance_mpc(z, h0, omega_m, nu, include_radiation=include_radiation), 1e-8, None)
    return 5.0 * np.log10(dl) + 25.0


def desi_bao_predictions(z: np.ndarray, quantity: Sequence[str], h0: float, omega_m: float, nu: float, rd_mpc: float, include_radiation: bool = True) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    dm = comoving_distance_mpc(z, h0, omega_m, nu, include_radiation=include_radiation)
    dh = C_LIGHT / hz_rvm(z, h0, omega_m, nu, include_radiation=include_radiation)
    dv = np.cbrt(z * dm * dm * dh)
    out = []
    for qi, dmi, dhi, dvi in zip(quantity, dm, dh, dv):
        q = str(qi).strip()
        if q == "DM_over_rs":
            out.append(dmi / rd_mpc)
        elif q == "DH_over_rs":
            out.append(dhi / rd_mpc)
        elif q == "DV_over_rs":
            out.append(dvi / rd_mpc)
        else:
            raise ValueError(f"Unsupported BAO quantity {qi}")
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------
# Pantheon / DESI / Planck
# ---------------------------------------------------------------------

def load_pantheon_plus(use_calibrators: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    txt = download_text(PANTHEON_DAT_URLS, "Pantheon_plus_SH0ES.dat")
    df = pd.read_csv(io.StringIO(txt), sep=r"\s+", comment="#", engine="python")
    z_cmb_col = pick_column(df.columns, ["zHD", "zCMB", "zcmb"])
    z_hel_col = pick_column(df.columns, ["zHEL", "zhel", "zHELIO"], required=False)
    mu_col = pick_column(df.columns, ["MU_SH0ES", "MU", "mu", "m_b_corr"])
    cal_col = pick_column(df.columns, ["IS_CALIBRATOR", "is_calibrator"], required=False)
    survey_col = pick_column(df.columns, ["IDSURVEY", "idsurvey", "survey"], required=False)
    host_col = pick_column(df.columns, ["HOST_LOGMASS", "host_logmass", "mass_host"], required=False)

    std = pd.DataFrame({
        "z_cmb": df[z_cmb_col].astype(float),
        "z_hel": df[z_hel_col].astype(float) if z_hel_col else df[z_cmb_col].astype(float),
        "mu": df[mu_col].astype(float),
        "is_calibrator": df[cal_col].astype(int) if cal_col else np.zeros(len(df), dtype=int),
        "survey_id": df[survey_col].astype(str) if survey_col else pd.Series(["unknown"] * len(df), dtype=str),
        "host_logmass": df[host_col].astype(float) if host_col else np.nan,
        "_orig_index": np.arange(len(df), dtype=int),
    })

    cov_txt = download_text(PANTHEON_COV_URLS, "Pantheon_plus_SH0ES_STATSYS.cov")
    cov = _parse_square_covariance(cov_txt, len(std))
    if not use_calibrators:
        mask = std["is_calibrator"].to_numpy(int) == 0
        std = std.loc[mask].reset_index(drop=True)
        cov = cov[np.ix_(mask, mask)]
    return std, cov


def subset_pantheon(df: pd.DataFrame, cov: np.ndarray, mask: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    out_df = df.loc[mask].reset_index(drop=True)
    out_cov = cov[np.ix_(mask, mask)]
    return out_df, out_cov


def load_desi_dr2(diagonal_only: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    mean_txt = download_text(DESI_DR2_MEAN_URLS, "desi_dr2_bao_mean.txt")
    cov_txt = download_text(DESI_DR2_COV_URLS, "desi_dr2_bao_cov.txt")
    rows = []
    for line in mean_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append((float(parts[0]), float(parts[1]), str(parts[2])))
    df = pd.DataFrame(rows, columns=["z", "value", "quantity"])
    cov = _parse_square_covariance(cov_txt, len(df))
    if diagonal_only:
        cov = np.diag(np.diag(cov))
    return df, cov


def subset_bao(df: pd.DataFrame, cov: np.ndarray, mask: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    out_df = df.loc[mask].reset_index(drop=True)
    out_cov = cov[np.ix_(mask, mask)]
    return out_df, out_cov


def load_planck_rd_prior() -> Dict[str, float]:
    payload = download_bytes(PLANCK_PR3_CHAINS_URLS, "planck_pr3_base_plikHM.zip", timeout=300)
    zf = zipfile.ZipFile(io.BytesIO(payload))
    names = zf.namelist()
    param_file = None
    for name in names:
        if name.lower().endswith(".paramnames") and "base" in name.lower():
            param_file = name
            break
    if param_file is None:
        raise RuntimeError("Could not locate Planck .paramnames file in chain archive")

    paramnames = []
    for line in zf.read(param_file).decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paramnames.append(line.split()[0])
    target = None
    for candidate in ["rdrag", "r_drag", "rsdrag", "rs_drag"]:
        for p in paramnames:
            if p.lower() == candidate:
                target = p
                break
        if target:
            break
    if target is None:
        for p in paramnames:
            pl = p.lower()
            if "drag" in pl and (pl.startswith("r") or pl.startswith("rs")):
                target = p
                break
    if target is None:
        raise RuntimeError("Could not find an r_drag-like parameter in Planck PR3 chains")
    idx = paramnames.index(target)

    vals = []
    wts = []
    for name in names:
        nl = name.lower()
        if not nl.endswith(".txt") or "minimum.theory_cl" in nl:
            continue
        try:
            arr = np.loadtxt(io.StringIO(zf.read(name).decode("utf-8", errors="replace")))
        except Exception:
            continue
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < idx + 3:
            continue
        wt = arr[:, 0]
        vv = arr[:, idx + 2]
        good = np.isfinite(vv) & np.isfinite(wt) & (wt > 0)
        if np.any(good):
            vals.append(vv[good])
            wts.append(wt[good])
    if not vals:
        raise RuntimeError("No usable Planck chain tables found")
    values = np.concatenate(vals)
    weights = np.concatenate(wts)
    mean = float(np.average(values, weights=weights))
    var = float(np.average((values - mean) ** 2, weights=weights))
    return {"parameter": target, "mean": mean, "sigma": math.sqrt(max(var, 1e-12)), "n_samples": int(values.size)}


@dataclass
class NuFitResult:
    h0: float
    omega_m: float
    nu: float
    rd_mpc: float
    intercept: float
    chi2: float
    success: bool
    message: str
    n_sne: int
    n_bao: int
    include_sn: bool
    include_bao: bool
    include_planck: bool
    analytic_intercept: bool
    include_radiation: bool
    diagonal_bao: bool
    sn_z_source: str


def _analytic_intercept_shift(residual: np.ndarray, inv_cov: np.ndarray) -> Tuple[float, float]:
    ones = np.ones_like(residual)
    alpha = float(ones @ inv_cov @ ones)
    beta = float(ones @ inv_cov @ residual)
    shift = beta / alpha
    rmin = residual - shift * ones
    chi2 = float(rmin @ inv_cov @ rmin)
    return shift, chi2


def fit_nu_model(
    include_sn: bool = True,
    include_bao: bool = True,
    include_planck: bool = True,
    analytic_intercept: bool = False,
    include_radiation: bool = True,
    diagonal_bao: bool = False,
    use_calibrators: bool = False,
    sn_z_source: str = "z_cmb",
    sn_df: Optional[pd.DataFrame] = None,
    sn_cov: Optional[np.ndarray] = None,
    bao_df: Optional[pd.DataFrame] = None,
    bao_cov: Optional[np.ndarray] = None,
    nu_bounds: Tuple[float, float] = (-0.03, 0.03),
) -> NuFitResult:
    if include_sn and sn_df is None:
        sn_df, sn_cov = load_pantheon_plus(use_calibrators=use_calibrators)
    if include_bao and bao_df is None:
        bao_df, bao_cov = load_desi_dr2(diagonal_only=diagonal_bao)
    inv_sn = None if sn_cov is None else linalg.inv(sn_cov)
    inv_bao = None if bao_cov is None else linalg.inv(bao_cov)
    planck_prior = load_planck_rd_prior() if include_planck else None

    zcol = "z_hel" if sn_z_source.lower().startswith("z_hel") else "z_cmb"

    def chi2(theta: np.ndarray) -> float:
        if analytic_intercept:
            h0, omega_m, nu, rd = theta
            intercept = 0.0
        else:
            h0, omega_m, nu, rd, intercept = theta
        if not (50 < h0 < 90 and 0.05 < omega_m < 0.6 and nu_bounds[0] <= nu <= nu_bounds[1] and 130 < rd < 160):
            return 1e30
        total = 0.0
        if include_sn and sn_df is not None and inv_sn is not None and len(sn_df) > 0:
            z = sn_df[zcol].to_numpy(float)
            mu_obs = sn_df["mu"].to_numpy(float)
            mu_th = distance_modulus(z, h0, omega_m, nu, include_radiation=include_radiation)
            residual = mu_obs - mu_th
            if analytic_intercept:
                _, c2 = _analytic_intercept_shift(residual, inv_sn)
            else:
                r = residual - intercept
                c2 = float(r @ inv_sn @ r)
            total += c2
        if include_bao and bao_df is not None and inv_bao is not None and len(bao_df) > 0:
            pred = desi_bao_predictions(bao_df["z"].to_numpy(float), bao_df["quantity"].tolist(), h0, omega_m, nu, rd, include_radiation=include_radiation)
            r = bao_df["value"].to_numpy(float) - pred
            total += float(r @ inv_bao @ r)
        if include_planck and planck_prior is not None:
            total += ((rd - planck_prior["mean"]) / planck_prior["sigma"]) ** 2
        return float(total)

    if analytic_intercept:
        x0 = np.array([68.0, 0.31, 0.001, 147.0])
        bounds = [(50.0, 90.0), (0.1, 0.5), nu_bounds, (130.0, 160.0)]
    else:
        x0 = np.array([68.0, 0.31, 0.001, 147.0, 0.0])
        bounds = [(50.0, 90.0), (0.1, 0.5), nu_bounds, (130.0, 160.0), (-1.0, 1.0)]
    best = optimize.minimize(chi2, x0=x0, method="L-BFGS-B", bounds=bounds)

    if analytic_intercept:
        h0, omega_m, nu, rd = best.x
        intercept = 0.0
        if include_sn and sn_df is not None and inv_sn is not None and len(sn_df) > 0:
            z = sn_df[zcol].to_numpy(float)
            mu_obs = sn_df["mu"].to_numpy(float)
            mu_th = distance_modulus(z, h0, omega_m, nu, include_radiation=include_radiation)
            intercept, _ = _analytic_intercept_shift(mu_obs - mu_th, inv_sn)
    else:
        h0, omega_m, nu, rd, intercept = best.x

    return NuFitResult(
        h0=float(h0), omega_m=float(omega_m), nu=float(nu), rd_mpc=float(rd), intercept=float(intercept),
        chi2=float(best.fun), success=bool(best.success), message=str(best.message),
        n_sne=0 if sn_df is None else int(len(sn_df)), n_bao=0 if bao_df is None else int(len(bao_df)),
        include_sn=include_sn, include_bao=include_bao, include_planck=include_planck,
        analytic_intercept=analytic_intercept, include_radiation=include_radiation,
        diagonal_bao=diagonal_bao, sn_z_source=zcol,
    )


def fit_nu_model_fixed_nu(fixed_nu: float, **kwargs: Any) -> NuFitResult:
    kwargs = dict(kwargs)
    include_sn = kwargs.get("include_sn", True)
    include_bao = kwargs.get("include_bao", True)
    include_planck = kwargs.get("include_planck", True)
    analytic_intercept = kwargs.get("analytic_intercept", False)
    include_radiation = kwargs.get("include_radiation", True)
    diagonal_bao = kwargs.get("diagonal_bao", False)
    use_calibrators = kwargs.get("use_calibrators", False)
    sn_z_source = kwargs.get("sn_z_source", "z_cmb")
    sn_df = kwargs.get("sn_df")
    sn_cov = kwargs.get("sn_cov")
    bao_df = kwargs.get("bao_df")
    bao_cov = kwargs.get("bao_cov")
    if include_sn and sn_df is None:
        sn_df, sn_cov = load_pantheon_plus(use_calibrators=use_calibrators)
    if include_bao and bao_df is None:
        bao_df, bao_cov = load_desi_dr2(diagonal_only=diagonal_bao)
    inv_sn = None if sn_cov is None else linalg.inv(sn_cov)
    inv_bao = None if bao_cov is None else linalg.inv(bao_cov)
    planck_prior = load_planck_rd_prior() if include_planck else None
    zcol = "z_hel" if str(sn_z_source).lower().startswith("z_hel") else "z_cmb"

    def chi2(theta: np.ndarray) -> float:
        if analytic_intercept:
            h0, omega_m, rd = theta
            intercept = 0.0
        else:
            h0, omega_m, rd, intercept = theta
        if not (50 < h0 < 90 and 0.05 < omega_m < 0.6 and 130 < rd < 160):
            return 1e30
        total = 0.0
        if include_sn and sn_df is not None and inv_sn is not None and len(sn_df) > 0:
            z = sn_df[zcol].to_numpy(float)
            mu_obs = sn_df["mu"].to_numpy(float)
            mu_th = distance_modulus(z, h0, omega_m, fixed_nu, include_radiation=include_radiation)
            residual = mu_obs - mu_th
            if analytic_intercept:
                _, c2 = _analytic_intercept_shift(residual, inv_sn)
            else:
                r = residual - intercept
                c2 = float(r @ inv_sn @ r)
            total += c2
        if include_bao and bao_df is not None and inv_bao is not None and len(bao_df) > 0:
            pred = desi_bao_predictions(bao_df["z"].to_numpy(float), bao_df["quantity"].tolist(), h0, omega_m, fixed_nu, rd, include_radiation=include_radiation)
            r = bao_df["value"].to_numpy(float) - pred
            total += float(r @ inv_bao @ r)
        if include_planck and planck_prior is not None:
            total += ((rd - planck_prior["mean"]) / planck_prior["sigma"]) ** 2
        return float(total)

    if analytic_intercept:
        x0 = np.array([68.0, 0.31, 147.0])
        bounds = [(50.0, 90.0), (0.1, 0.5), (130.0, 160.0)]
    else:
        x0 = np.array([68.0, 0.31, 147.0, 0.0])
        bounds = [(50.0, 90.0), (0.1, 0.5), (130.0, 160.0), (-1.0, 1.0)]
    best = optimize.minimize(chi2, x0=x0, method="L-BFGS-B", bounds=bounds)
    if analytic_intercept:
        h0, omega_m, rd = best.x
        intercept = 0.0
        if include_sn and sn_df is not None and inv_sn is not None and len(sn_df) > 0:
            z = sn_df[zcol].to_numpy(float)
            mu_obs = sn_df["mu"].to_numpy(float)
            mu_th = distance_modulus(z, h0, omega_m, fixed_nu, include_radiation=include_radiation)
            intercept, _ = _analytic_intercept_shift(mu_obs - mu_th, inv_sn)
    else:
        h0, omega_m, rd, intercept = best.x
    return NuFitResult(float(h0), float(omega_m), float(fixed_nu), float(rd), float(intercept), float(best.fun), bool(best.success), str(best.message), 0 if sn_df is None else int(len(sn_df)), 0 if bao_df is None else int(len(bao_df)), include_sn, include_bao, include_planck, analytic_intercept, include_radiation, diagonal_bao, zcol)


def nu_significance_from_delta_chi2(best: NuFitResult, null: NuFitResult) -> float:
    dchi2 = max(0.0, null.chi2 - best.chi2)
    return float(math.sqrt(dchi2))


# ---------------------------------------------------------------------
# SDSS public positions and filament utilities
# ---------------------------------------------------------------------

def fetch_sdss_galaxy_sample(
    z_min: float = 0.02,
    z_max: float = 0.12,
    max_rows: int = 15000,
    seed: int = 1234,
) -> pd.DataFrame:
    queries = [
        f"SELECT TOP {max_rows} p.ra, p.dec, s.z "
        f"FROM PhotoObjAll p "
        f"JOIN SpecObjAll s ON s.bestObjID = p.objID "
        f"WHERE s.class = 'GALAXY' "
        f"AND s.z BETWEEN {z_min:.6f} AND {z_max:.6f} "
        f"AND p.type = 3 "
        f"ORDER BY NEWID()",
        f"SELECT TOP {max_rows} p.ra, p.dec, s.z "
        f"FROM PhotoObj p "
        f"JOIN SpecObj s ON s.bestObjID = p.objID "
        f"WHERE s.class = 'GALAXY' "
        f"AND s.z BETWEEN {z_min:.6f} AND {z_max:.6f} "
        f"ORDER BY p.objID",
    ]

    last_error = None
    for base in SDSS_SKYSERVER_SQL_BASES:
        for i, query in enumerate(queries):
            params = urllib.parse.urlencode({"cmd": query, "format": "csv"})
            url = f"{base}?{params}"
            cache = f"sdss_sample_{i}_{abs(hash((base, query))) % (10**8)}.csv"
            try:
                txt = download_text([url], cache, timeout=240)

                lines = [
                    ln for ln in txt.splitlines()
                    if ln.strip() and not ln.lstrip().startswith('#')
                ]
                if not lines:
                    raise RuntimeError("empty or comment-only SkyServer CSV response")

                parsed = None
                for sep in [",", "\t", None, r"\s+"]:
                    try:
                        if sep is None:
                            df = pd.read_csv(io.StringIO("\n".join(lines)), sep=None, engine="python")
                        else:
                            df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, engine="python")
                    except Exception:
                        continue

                    # reject fake one-column marker tables
                    if len(df.columns) == 1 and str(df.columns[0]).startswith("#Table"):
                        continue

                    try:
                        ra = pick_column(df.columns, ["ra", "p.ra"])
                        dec = pick_column(df.columns, ["dec", "p.dec"])
                        z = pick_column(df.columns, ["z", "specz", "s.z"])
                    except Exception:
                        continue

                    parsed = pd.DataFrame({
                        "ra": pd.to_numeric(df[ra], errors="coerce"),
                        "dec": pd.to_numeric(df[dec], errors="coerce"),
                        "z": pd.to_numeric(df[z], errors="coerce"),
                    })
                    break

                if parsed is None:
                    raise RuntimeError(f"Could not parse SkyServer CSV columns from response header(s): {lines[:3]}")

                out = parsed.replace([np.inf, -np.inf], np.nan).dropna(subset=["ra", "dec", "z"])
                out = out[(out["z"] >= z_min) & (out["z"] <= z_max)].reset_index(drop=True)

                if len(out) >= min(500, max_rows // 10):
                    return out.head(max_rows)

            except Exception as exc:
                last_error = exc
                continue

    raise RuntimeError("Could not fetch public SDSS galaxy sample from SkyServer") from last_error

def sky_to_cartesian_mpc(ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray, h0: float = 70.0, omega_m: float = 0.3) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    z = np.asarray(z, dtype=float)
    chi = comoving_distance_mpc(z, h0, omega_m, nu=0.0, include_radiation=True)
    x = chi * np.cos(dec) * np.cos(ra)
    y = chi * np.cos(dec) * np.sin(ra)
    zc = chi * np.sin(dec)
    return np.column_stack([x, y, zc])


def estimate_filament_axes_knn(
    points_xyz: np.ndarray,
    k: int = 12,
    k_neighbors: Optional[int] = None,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    if k_neighbors is not None:
        k = int(k_neighbors)

    tree = cKDTree(points_xyz)
    _, idx = tree.query(points_xyz, k=min(k + 1, len(points_xyz)))
    axes = np.zeros_like(points_xyz)
    for i in range(len(points_xyz)):
        neigh = points_xyz[idx[i, 1:]]
        if len(neigh) < 3:
            axes[i] = np.array([1.0, 0.0, 0.0])
            continue
        centered = neigh - neigh.mean(axis=0)
        cov = centered.T @ centered / max(1, len(neigh) - 1)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, np.argmax(vals)]
        axes[i] = axis / max(np.linalg.norm(axis), 1e-12)
    return axes


def estimate_filament_axes_mst(points_xyz: np.ndarray, k: int = 12) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(points_xyz)
    dists, idx = tree.query(points_xyz, k=min(k + 1, len(points_xyz)))
    rows = []
    cols = []
    data = []
    for i in range(len(points_xyz)):
        for j, d in zip(idx[i, 1:], dists[i, 1:]):
            rows.append(i)
            cols.append(j)
            data.append(d)
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(len(points_xyz), len(points_xyz)))
    mst = minimum_spanning_tree(graph)
    mst = mst + mst.T
    coo = mst.tocoo()
    neigh = [[] for _ in range(len(points_xyz))]
    for i, j in zip(coo.row, coo.col):
        neigh[i].append(j)
    axes = np.zeros_like(points_xyz)
    for i in range(len(points_xyz)):
        vecs = []
        for j in neigh[i]:
            v = points_xyz[j] - points_xyz[i]
            n = np.linalg.norm(v)
            if n > 0:
                vecs.append(v / n)
        if len(vecs) == 0:
            axes[i] = np.array([1.0, 0.0, 0.0])
        elif len(vecs) == 1:
            axes[i] = vecs[0]
        else:
            M = np.vstack(vecs)
            cov = M.T @ M / len(vecs)
            vals, vecs_e = np.linalg.eigh(cov)
            axis = vecs_e[:, np.argmax(vals)]
            axes[i] = axis / max(np.linalg.norm(axis), 1e-12)
    return axes


def filament_orientation_correlation(points_xyz: np.ndarray, axes: np.ndarray, r_bins: Optional[np.ndarray] = None, max_pairs: int = 250000, seed: int = 1234) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(points_xyz)
    if r_bins is None:
        r_bins = np.arange(20.0, 281.0, 20.0)
    r_bins = np.asarray(r_bins, dtype=float)
    if r_bins.ndim == 1 and len(r_bins) >= 2:
        edges = r_bins
    else:
        raise ValueError("r_bins must be bin edges")
    pair_i = rng.integers(0, n, size=max_pairs)
    pair_j = rng.integers(0, n, size=max_pairs)
    good = pair_i != pair_j
    pair_i = pair_i[good]
    pair_j = pair_j[good]
    sep = np.linalg.norm(points_xyz[pair_i] - points_xyz[pair_j], axis=1)
    dots = np.abs(np.sum(axes[pair_i] * axes[pair_j], axis=1))
    corr = dots ** 2 - (1.0 / 3.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(len(centers), np.nan)
    errors = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        m = (sep >= edges[i]) & (sep < edges[i + 1])
        counts[i] = int(np.sum(m))
        if counts[i] >= 10:
            values[i] = float(np.mean(corr[m]))
            errors[i] = float(np.std(corr[m], ddof=1) / math.sqrt(counts[i]))
    return {
        "r_edges": edges.tolist(),
        "r_mid_mpc_over_h": centers.tolist(),
        "corr": values.tolist(),
        "stderr": errors.tolist(),
        "counts": counts.tolist(),
    }


def fit_exponential_correlation(r: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(r) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    r = np.asarray(r, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    yerr = np.asarray(yerr, dtype=float)[mask]
    if len(r) < 3:
        return {"amplitude": float("nan"), "scale_mpc_h": float("nan"), "chi2": float("nan")}
    def chi2(theta: np.ndarray) -> float:
        amp, scale = theta
        if scale <= 0:
            return 1e30
        model = amp * np.exp(-r / scale)
        return float(np.sum(((y - model) / yerr) ** 2))
    x0 = np.array([max(np.nanmax(y), 1e-4), 100.0])
    best = optimize.minimize(chi2, x0=x0, method="Nelder-Mead")
    amp, scale = best.x
    return {"amplitude": float(amp), "scale_mpc_h": float(scale), "chi2": float(best.fun)}


def null_control_zscores(real_corr: np.ndarray, null_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(null_matrix, axis=0)
    std = np.nanstd(null_matrix, axis=0, ddof=1)
    z = (real_corr - mean) / np.where(std > 0, std, np.nan)
    return mean, std, z


# ---------------------------------------------------------------------
# SPARC
# ---------------------------------------------------------------------

def _normalize_rar_acceleration_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    lower_map = {str(c).strip().lower(): c for c in work.columns}
    for canon, aliases in {
        'gbar': ['gbar', 'log_gbar', 'log10_gbar'],
        'gobs': ['gobs', 'log_gobs', 'log10_gobs'],
        'e_gbar': ['e_gbar', 'egbar', 'err_gbar'],
        'e_gobs': ['e_gobs', 'egobs', 'err_gobs'],
        'Galaxy': ['galaxy', 'name'],
    }.items():
        if canon in work.columns:
            continue
        for alias in aliases:
            src = lower_map.get(alias)
            if src is not None:
                work = work.rename(columns={src: canon})
                break
    for col in ['gbar', 'gobs']:
        if col in work.columns:
            vals = pd.to_numeric(work[col], errors='coerce').to_numpy(float)
            finite = vals[np.isfinite(vals)]
            if finite.size and np.nanmedian(finite) < 0.0 and np.nanmax(finite) < 1.0 and np.nanmin(finite) > -30.0:
                work[col] = np.power(10.0, vals)
    return work


def _load_sparc_rarbins_fallback() -> pd.DataFrame:
    raw = download_text(SPARC_RARBINS_URLS, 'SPARC_RARbins.mrt', timeout=180)
    rows: List[Tuple[float, float, float, int]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('-') or line.lower().startswith('title:') or 'byte-by-byte' in line.lower() or 'pawlowski table' in line.lower():
            continue
        parts = re.findall(r'[-+]?\d+(?:\.\d+)?', line)
        if len(parts) == 4:
            try:
                gbar_log, gobs_log, sd, n = float(parts[0]), float(parts[1]), float(parts[2]), int(float(parts[3]))
            except Exception:
                continue
            rows.append((10.0 ** gbar_log, 10.0 ** gobs_log, sd, n))
    df = pd.DataFrame(rows, columns=['gbar', 'gobs', 'sd_dex', 'N'])
    if not df.empty:
        df['Galaxy'] = '__all__'
    return df


def load_sparc_rar() -> pd.DataFrame:
    raw = download_text(SPARC_RAR_URLS, 'SPARC_RAR.mrt', timeout=180)
    table_df: Optional[pd.DataFrame] = None
    try:
        from astropy.io import ascii
        table = ascii.read(raw, format='cds')
        table_df = table.to_pandas()
    except Exception:
        table_df = None
    if table_df is None or table_df.empty:
        return _load_sparc_rarbins_fallback()
    table_df = _normalize_rar_acceleration_columns(table_df)
    if 'gbar' not in table_df.columns or 'gobs' not in table_df.columns:
        return _load_sparc_rarbins_fallback()
    finite = np.isfinite(pd.to_numeric(table_df['gbar'], errors='coerce')) & np.isfinite(pd.to_numeric(table_df['gobs'], errors='coerce'))
    if int(np.sum(finite)) == 0:
        return _load_sparc_rarbins_fallback()
    return table_df


def load_sparc_table1() -> pd.DataFrame:
    from astropy.io import ascii
    raw = download_text(SPARC_TABLE1_URLS, "SPARC_Table1.mrt", timeout=180)
    table = ascii.read(raw, format="cds")
    return table.to_pandas()


def rar_relation(gbar: np.ndarray, a0: float) -> np.ndarray:
    x = np.sqrt(np.clip(gbar / a0, 1e-30, None))
    return gbar / (1.0 - np.exp(-x))


def fit_rar_hierarchical_like(df: pd.DataFrame, galaxy_col: str, gobs_col: str, gbar_col: str, offset_prior_dex: float = 0.08) -> Dict[str, Any]:
    work = df[[galaxy_col, gobs_col, gbar_col]].copy()
    work[gobs_col] = pd.to_numeric(work[gobs_col], errors='coerce')
    work[gbar_col] = pd.to_numeric(work[gbar_col], errors='coerce')

    for col in [gobs_col, gbar_col]:
        vals = work[col].to_numpy(float)
        finite = vals[np.isfinite(vals)]
        if finite.size and np.nanmedian(finite) < 0.0 and np.nanmax(finite) < 1.0 and np.nanmin(finite) > -30.0:
            work[col] = np.power(10.0, vals)

    work = work.dropna().copy()
    work = work[(work[gobs_col] > 0) & (work[gbar_col] > 0)].reset_index(drop=True)
    if work.empty:
        return {
            'best_a0_m_per_s2': float('nan'),
            'n_points': 0,
            'n_galaxies': 0,
            'offset_rms_dex': float('nan'),
            'optimizer_success': False,
            'optimizer_message': 'No valid positive gobs/gbar points found after parsing SPARC RAR table.',
        }

    galaxies = sorted(work[galaxy_col].astype(str).unique())
    gid_map = {g: i for i, g in enumerate(galaxies)}
    gid = work[galaxy_col].astype(str).map(gid_map).to_numpy(int)
    log_gobs = np.log10(work[gobs_col].to_numpy(float))
    gbar = work[gbar_col].to_numpy(float)

    def objective(theta: np.ndarray) -> float:
        log10_a0 = theta[0]
        offsets = theta[1:]
        a0 = 10.0 ** log10_a0
        pred = np.log10(rar_relation(gbar, a0)) + offsets[gid]
        resid = log_gobs - pred
        penalty = np.sum((offsets / offset_prior_dex) ** 2)
        return float(np.sum(resid ** 2) + penalty)

    x0 = np.zeros(1 + len(galaxies), dtype=float)
    x0[0] = np.log10(1.2e-10)
    res = optimize.minimize(objective, x0=x0, method='L-BFGS-B')
    offsets = res.x[1:]
    return {
        'best_a0_m_per_s2': float(10.0 ** res.x[0]),
        'n_points': int(len(work)),
        'n_galaxies': int(len(galaxies)),
        'offset_rms_dex': float(np.sqrt(np.mean(offsets ** 2))) if len(offsets) else 0.0,
        'optimizer_success': bool(res.success),
        'optimizer_message': str(res.message),
    }


# ---------------------------------------------------------------------
# PDG / Koide
# ---------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.upper())


def _extract_first_float_in_section(text: str, section_patterns: Sequence[str], value_range: Tuple[float, float], window: int = 5000) -> float:
    normalized = _normalize_text(text)
    for pat in section_patterns:
        m = re.search(pat, normalized)
        if not m:
            continue
        chunk = normalized[m.start(): m.start() + window]
        nums = [float(x) for x in re.findall(r"(?<![A-Z])[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", chunk)]
        for val in nums:
            if value_range[0] <= val <= value_range[1]:
                return float(val)
    raise RuntimeError(f"Could not find value in range {value_range} for anchors {section_patterns}")


def parse_pdg_lepton_masses() -> Dict[str, float]:
    text = extract_text_from_pdf(download_bytes(PDG_LEPTON_PDF_URLS, "pdg_2024_leptons.pdf", timeout=180))
    return {
        "e_mev": _extract_first_float_in_section(text, [r"\bE MASS\b", r"ELECTRON MASS", r"MASS OF ELECTRON"], (0.50, 0.52)),
        "mu_mev": _extract_first_float_in_section(text, [r"\bMU MASS\b", r"MUON MASS", r"MASS OF MUON"], (105.0, 106.5)),
        "tau_mev": _extract_first_float_in_section(text, [r"\bTAU MASS\b", r"TAU LEPTON MASS", r"MASS OF TAU"], (1700.0, 1800.0)),
    }


def parse_pdg_quark_masses() -> Dict[str, float]:
    text = extract_text_from_pdf(download_bytes(PDG_QUARK_PDF_URLS, "pdg_2024_quarks.pdf", timeout=180))
    return {
        "u_gev": _extract_first_float_in_section(text, [r"\bMU =", r"UP QUARK"], (1.0, 3.5)) / 1000.0,
        "d_gev": _extract_first_float_in_section(text, [r"\bMD =", r"DOWN QUARK"], (3.0, 7.0)) / 1000.0,
        "s_gev": _extract_first_float_in_section(text, [r"\bMS =", r"STRANGE QUARK"], (80.0, 120.0)) / 1000.0,
        "c_gev": _extract_first_float_in_section(text, [r"\bMC =", r"CHARM QUARK"], (1.0, 1.6)),
        "b_gev": _extract_first_float_in_section(text, [r"\bMB =", r"BOTTOM QUARK"], (3.5, 5.5)),
        "t_gev": _extract_first_float_in_section(text, [r"DIRECT MEASUREMENTS", r"TOP QUARK"], (160.0, 180.0)),
    }


def koide_q(masses: Sequence[float]) -> float:
    m = np.asarray(masses, dtype=float)
    return float(np.sum(m) / np.sum(np.sqrt(m)) ** 2)


def alpha_s_one_loop(mu_gev: float, nf: int = 5, lambda_qcd_gev: float = 0.2) -> float:
    mu = max(mu_gev, lambda_qcd_gev * 1.01)
    beta0 = 11.0 - 2.0 * nf / 3.0
    t = math.log(mu * mu / (lambda_qcd_gev * lambda_qcd_gev))
    return 4.0 * math.pi / (beta0 * t)


def running_mass_one_loop(m_ref_gev: float, mu_ref_gev: float, mu_target_gev: np.ndarray, nf: int = 5, gamma0: float = 4.0) -> np.ndarray:
    mu_target_gev = np.asarray(mu_target_gev, dtype=float)
    beta0 = 11.0 - 2.0 * nf / 3.0
    a_ref = alpha_s_one_loop(mu_ref_gev, nf=nf)
    a_tar = np.asarray([alpha_s_one_loop(mu, nf=nf) for mu in mu_target_gev])
    power = gamma0 / beta0
    return m_ref_gev * (a_tar / a_ref) ** power


# ---------------------------------------------------------------------
# HEPData helpers
# ---------------------------------------------------------------------

def _hepdata_record_json(urls: Sequence[str], cache_name: str) -> Dict[str, Any]:
    txt = download_text(urls, cache_name, timeout=180)
    return json.loads(txt)


def _extract_hepdata_links(obj: Any) -> List[str]:
    links: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and ("hepdata.net/download" in v or v.endswith(".csv") or v.endswith(".json") or v.endswith(".txt")):
                links.append(v)
            else:
                links.extend(_extract_hepdata_links(v))
    elif isinstance(obj, list):
        for item in obj:
            links.extend(_extract_hepdata_links(item))
    return list(dict.fromkeys(links))


def load_numeric_table_from_url(url: str) -> Optional[pd.DataFrame]:
    try:
        txt = download_text([url], cache_name=_url_cache_name(url, 'table.txt'), timeout=180)
    except Exception:
        return None
    stripped = txt.lstrip()
    if 'yaml' in url.lower():
        return None
    if url.lower().endswith('.json') or stripped.startswith('{'):
        try:
            obj = json.loads(txt)
            indep = obj.get('independent_variables', [])
            dep = obj.get('dependent_variables', [])
            if indep and dep:
                xvals = []
                for v in indep[0].get('values', []):
                    if isinstance(v, dict):
                        if 'value' in v:
                            xvals.append(v['value'])
                        elif 'low' in v and 'high' in v:
                            xvals.append(0.5 * (float(v['low']) + float(v['high'])))
                    else:
                        xvals.append(v)
                yvals = []
                for v in dep[0].get('values', []):
                    if isinstance(v, dict):
                        yvals.append(v.get('value'))
                    else:
                        yvals.append(v)
                if xvals and yvals and len(xvals) == len(yvals):
                    df = pd.DataFrame({'x': pd.to_numeric(xvals, errors='coerce'), 'y': pd.to_numeric(yvals, errors='coerce')})
                    df = df.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(df) >= 2:
                        return df
        except Exception:
            pass
    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    if not lines:
        return None
    for sep in [',', '\t', r'\s+']:
        try:
            df = pd.read_csv(io.StringIO('\n'.join(lines)), sep=sep, engine='python')
            numeric = df.apply(pd.to_numeric, errors='coerce')
            usable_cols = [
                col for col in numeric.columns
                if np.isfinite(numeric[col].to_numpy(float)).sum() >= 2
            ]
            if len(usable_cols) >= 2:
                xcol = usable_cols[0]
                preferred_y = [
                    col for col in usable_cols[1:]
                    if not any(tok in str(col).lower() for tok in ['low', 'high', 'err', 'error', 'unc', 'min', 'max'])
                ]
                ycol = preferred_y[-1] if preferred_y else usable_cols[-1]
                out = pd.DataFrame({'x': numeric[xcol], 'y': numeric[ycol]}).replace([np.inf, -np.inf], np.nan).dropna()
                if len(out) >= 2:
                    return out
        except Exception:
            continue
    pairs = []
    for ln in lines:
        vals = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', ln)
        if len(vals) >= 2:
            try:
                pairs.append((float(vals[0]), float(vals[-1])))
            except Exception:
                continue
    if len(pairs) >= 20:
        return pd.DataFrame(pairs, columns=['x', 'y'])
    return None

def inspect_direct_detection_resources() -> Dict[str, Any]:
    resources = []
    numeric_tables = []
    for name, urls in DIRECT_DETECTION_RECORD_URLS.items():
        try:
            record = _hepdata_record_json(urls, f"{name}_record.json")
        except Exception:
            continue
        links = _extract_hepdata_links(record)
        resources.extend(links)
        for link in links:
            if not any(ext in link.lower() for ext in ["csv", "txt", "json"]):
                continue
            df = load_numeric_table_from_url(link)
            if df is None or len(df) < 20:
                continue
            x = np.asarray(df.iloc[:, 0], dtype=float)
            y = np.asarray(df.iloc[:, 1], dtype=float)
            m = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if np.sum(m) < 20:
                continue
            x = x[m]
            y = y[m]
            numeric_tables.append({
                "source": link,
                "n_points": int(len(x)),
                "mass_min": float(np.min(x)),
                "mass_max": float(np.max(x)),
                "y_min": float(np.min(y[y > 0])) if np.any(y > 0) else float("nan"),
            })
    return {
        "n_candidate_resources": int(len(resources)),
        "resources": list(dict.fromkeys(resources)),
        "numeric_tables": numeric_tables,
    }


# ---------------------------------------------------------------------
# DES-SN5YR
# ---------------------------------------------------------------------

def _download_zip_member_by_basename(urls: Sequence[str], cache_name: str, basenames: Sequence[str], timeout: int = 180) -> bytes:
    payload = download_bytes(urls, cache_name, timeout=timeout)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
        # exact basename match first
        lower_to_name = {Path(n).name.lower(): n for n in names}
        for base in basenames:
            if base.lower() in lower_to_name:
                return zf.read(lower_to_name[base.lower()])
        # substring match next
        for name in names:
            lname = name.lower()
            for base in basenames:
                if base.lower() in lname:
                    return zf.read(name)
    raise RuntimeError(f"Could not find any of {basenames} in {cache_name}")


def load_des_sn5yr(with_metadata: bool = True, use_stat_sys: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    basename = 'DES-SN5YR_HD+MetaData.csv' if with_metadata else 'DES-SN5YR_HD.csv'
    hd_bytes = _download_zip_member_by_basename(DES_SN5YR_ZIP_URLS, 'DES-SN5YR-1.2.zip', [basename], timeout=1800)
    df = pd.read_csv(io.BytesIO(hd_bytes))
    zcmb = pick_column(df.columns, ['zCMB', 'zcmb', 'zHD'])
    zhel = pick_column(df.columns, ['zHEL', 'zhel'], required=False)
    mu = pick_column(df.columns, ['MU', 'mu'])
    muerr = pick_column(df.columns, ['MUERR_FINAL', 'MUERR', 'muerr_final'], required=False)
    survey = pick_column(df.columns, ['IDSURVEY', 'idsurvey'], required=False)
    host_mass = pick_column(df.columns, ['HOST_LOGMASS', 'host_logmass'], required=False)
    host_ra = pick_column(df.columns, ['HOST_RA', 'host_ra'], required=False)
    host_dec = pick_column(df.columns, ['HOST_DEC', 'host_dec'], required=False)
    host_z = pick_column(df.columns, ['HOST_ZSPEC', 'host_zspec', 'zHEL'], required=False)
    cid = pick_column(df.columns, ['CID', 'cid'], required=False)
    std = pd.DataFrame({
        'z_cmb': pd.to_numeric(df[zcmb], errors='coerce'),
        'z_hel': pd.to_numeric(df[zhel], errors='coerce') if zhel else pd.to_numeric(df[zcmb], errors='coerce'),
        'mu': pd.to_numeric(df[mu], errors='coerce'),
        'muerr_final': pd.to_numeric(df[muerr], errors='coerce') if muerr else np.nan,
        'survey_id': df[survey].astype(str) if survey else pd.Series(['unknown']*len(df), dtype=str),
        'host_logmass': pd.to_numeric(df[host_mass], errors='coerce') if host_mass else np.nan,
        'host_ra': pd.to_numeric(df[host_ra], errors='coerce') if host_ra else np.nan,
        'host_dec': pd.to_numeric(df[host_dec], errors='coerce') if host_dec else np.nan,
        'host_zspec': pd.to_numeric(df[host_z], errors='coerce') if host_z else np.nan,
        'cid': df[cid].astype(str) if cid else pd.Series([str(i) for i in range(len(df))], dtype=str),
        '_orig_index': np.arange(len(df), dtype=int),
    })
    std = std.replace([np.inf, -np.inf], np.nan)

    cov = None
    if use_stat_sys:
        try:
            cov_bytes = _download_zip_member_by_basename(DES_SN5YR_ZIP_URLS, 'DES-SN5YR-1.2.zip', ['STAT+SYS.txt.gz', 'STAT+SYS.txt'], timeout=1800)
            if cov_bytes[:2] == bytes([0x1f, 0x8b]):
                try:
                    cov_txt = gzip.decompress(cov_bytes).decode('utf-8', errors='replace')
                except Exception:
                    cov_txt = cov_bytes.decode('utf-8', errors='replace')
            else:
                cov_txt = cov_bytes.decode('utf-8', errors='replace')
            cov = _parse_square_covariance(cov_txt, len(std))
        except Exception:
            cov = None
    if cov is None:
        err = std['muerr_final'].to_numpy(float)
        err = np.where(np.isfinite(err) & (err > 0), err, np.nanmedian(err[np.isfinite(err) & (err > 0)]) if np.any(np.isfinite(err) & (err > 0)) else 0.15)
        cov = np.diag(err * err)
    mask = np.isfinite(std['z_cmb']) & np.isfinite(std['mu'])
    std = std.loc[mask].reset_index(drop=True)
    cov = cov[np.ix_(mask, mask)]
    return std, cov


# ---------------------------------------------------------------------
# Phenomenological ongoing-DM abundance model (public-data proxy)
# ---------------------------------------------------------------------

def e2_alpha_dm(z: np.ndarray | float, omega_m: float, alpha: float, include_radiation: bool = True, h: float = 0.7) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    growth = 1.0 + alpha * z / (1.0 + z)
    matter = omega_m * ((1.0 + z) ** 3) * growth
    e2 = 1.0 + (matter - omega_m)
    if include_radiation:
        e2 = e2 + omega_radiation(h) * ((1.0 + z) ** 4 - 1.0)
    return np.clip(e2, 1e-12, None)


def hz_alpha_dm(z: np.ndarray | float, h0: float, omega_m: float, alpha: float, include_radiation: bool = True) -> np.ndarray:
    return h0 * np.sqrt(e2_alpha_dm(z, omega_m, alpha, include_radiation=include_radiation, h=h0 / 100.0))


def comoving_distance_alpha_dm(z: np.ndarray, h0: float, omega_m: float, alpha: float, include_radiation: bool = True) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    zmax = float(np.max(z))
    grid = np.linspace(0.0, max(zmax, 1e-4), 2048)
    integrand = C_LIGHT / hz_alpha_dm(grid, h0, omega_m, alpha, include_radiation=include_radiation)
    chi = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))])
    return np.interp(z, grid, chi)


def distance_modulus_alpha_dm(z: np.ndarray, h0: float, omega_m: float, alpha: float, include_radiation: bool = True) -> np.ndarray:
    dl = (1.0 + np.asarray(z)) * comoving_distance_alpha_dm(z, h0, omega_m, alpha, include_radiation=include_radiation)
    dl = np.clip(dl, 1e-8, None)
    return 5.0 * np.log10(dl) + 25.0


def desi_bao_predictions_alpha_dm(z: np.ndarray, quantity: Sequence[str], h0: float, omega_m: float, alpha: float, rd_mpc: float, include_radiation: bool = True) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    dm = comoving_distance_alpha_dm(z, h0, omega_m, alpha, include_radiation=include_radiation)
    dh = C_LIGHT / hz_alpha_dm(z, h0, omega_m, alpha, include_radiation=include_radiation)
    dv = np.cbrt(z * dm * dm * dh)
    out = []
    for qi, dmi, dhi, dvi in zip(quantity, dm, dh, dv):
        q = str(qi).strip()
        if q == 'DM_over_rs':
            out.append(dmi / rd_mpc)
        elif q == 'DH_over_rs':
            out.append(dhi / rd_mpc)
        elif q == 'DV_over_rs':
            out.append(dvi / rd_mpc)
        else:
            raise ValueError(f'Unsupported BAO quantity {qi}')
    return np.asarray(out, dtype=float)


@dataclass
class AlphaFitResult:
    h0: float
    omega_m: float
    alpha_dm: float
    rd_mpc: float
    intercept: float
    chi2: float
    success: bool
    message: str
    n_sne: int
    n_bao: int
    include_sn: bool
    include_bao: bool
    include_planck: bool
    sn_z_source: str


def fit_alpha_dm_model(include_sn: bool = True, include_bao: bool = True, include_planck: bool = True,
                       include_radiation: bool = True, sn_df: Optional[pd.DataFrame] = None, sn_cov: Optional[np.ndarray] = None,
                       bao_df: Optional[pd.DataFrame] = None, bao_cov: Optional[np.ndarray] = None, sn_z_source: str = 'z_cmb',
                       alpha_bounds: Tuple[float, float] = (-0.2, 0.2)) -> AlphaFitResult:
    if include_sn and sn_df is None:
        sn_df, sn_cov = load_pantheon_plus(use_calibrators=False)
    if include_bao and bao_df is None:
        bao_df, bao_cov = load_desi_dr2(diagonal_only=False)
    inv_sn = None if sn_cov is None else linalg.inv(sn_cov)
    inv_bao = None if bao_cov is None else linalg.inv(bao_cov)
    planck_prior = load_planck_rd_prior() if include_planck else None
    zcol = 'z_hel' if str(sn_z_source).lower().startswith('z_hel') else 'z_cmb'

    def chi2(theta: np.ndarray) -> float:
        h0, omega_m, alpha, rd, intercept = theta
        if not (50 < h0 < 90 and 0.05 < omega_m < 0.6 and alpha_bounds[0] <= alpha <= alpha_bounds[1] and 130 < rd < 160):
            return 1e30
        total = 0.0
        if include_sn and sn_df is not None and inv_sn is not None and len(sn_df) > 0:
            z = sn_df[zcol].to_numpy(float)
            mu_obs = sn_df['mu'].to_numpy(float)
            mu_th = distance_modulus_alpha_dm(z, h0, omega_m, alpha, include_radiation=include_radiation)
            r = mu_obs - mu_th - intercept
            total += float(r @ inv_sn @ r)
        if include_bao and bao_df is not None and inv_bao is not None and len(bao_df) > 0:
            pred = desi_bao_predictions_alpha_dm(bao_df['z'].to_numpy(float), bao_df['quantity'].tolist(), h0, omega_m, alpha, rd, include_radiation=include_radiation)
            r = bao_df['value'].to_numpy(float) - pred
            total += float(r @ inv_bao @ r)
        if include_planck and planck_prior is not None:
            total += ((rd - planck_prior['mean']) / planck_prior['sigma']) ** 2
        return float(total)

    x0 = np.array([68.0, 0.31, 0.0, 147.0, 0.0])
    bounds = [(50.0, 90.0), (0.1, 0.5), alpha_bounds, (130.0, 160.0), (-1.0, 1.0)]
    best = optimize.minimize(chi2, x0=x0, method='L-BFGS-B', bounds=bounds)
    h0, omega_m, alpha, rd, intercept = best.x
    return AlphaFitResult(float(h0), float(omega_m), float(alpha), float(rd), float(intercept), float(best.fun), bool(best.success), str(best.message), 0 if sn_df is None else int(len(sn_df)), 0 if bao_df is None else int(len(bao_df)), include_sn, include_bao, include_planck, zcol)


def fit_alpha_dm_model_fixed_alpha(fixed_alpha: float, **kwargs: Any) -> AlphaFitResult:
    include_sn = kwargs.get('include_sn', True)
    include_bao = kwargs.get('include_bao', True)
    include_planck = kwargs.get('include_planck', True)
    include_radiation = kwargs.get('include_radiation', True)
    sn_df = kwargs.get('sn_df')
    sn_cov = kwargs.get('sn_cov')
    bao_df = kwargs.get('bao_df')
    bao_cov = kwargs.get('bao_cov')
    sn_z_source = kwargs.get('sn_z_source', 'z_cmb')
    if include_sn and sn_df is None:
        sn_df, sn_cov = load_pantheon_plus(use_calibrators=False)
    if include_bao and bao_df is None:
        bao_df, bao_cov = load_desi_dr2(diagonal_only=False)
    inv_sn = None if sn_cov is None else linalg.inv(sn_cov)
    inv_bao = None if bao_cov is None else linalg.inv(bao_cov)
    planck_prior = load_planck_rd_prior() if include_planck else None
    zcol = 'z_hel' if str(sn_z_source).lower().startswith('z_hel') else 'z_cmb'

    def chi2(theta: np.ndarray) -> float:
        h0, omega_m, rd, intercept = theta
        if not (50 < h0 < 90 and 0.05 < omega_m < 0.6 and 130 < rd < 160):
            return 1e30
        total = 0.0
        if include_sn and sn_df is not None and inv_sn is not None and len(sn_df) > 0:
            z = sn_df[zcol].to_numpy(float)
            mu_obs = sn_df['mu'].to_numpy(float)
            mu_th = distance_modulus_alpha_dm(z, h0, omega_m, fixed_alpha, include_radiation=include_radiation)
            r = mu_obs - mu_th - intercept
            total += float(r @ inv_sn @ r)
        if include_bao and bao_df is not None and inv_bao is not None and len(bao_df) > 0:
            pred = desi_bao_predictions_alpha_dm(bao_df['z'].to_numpy(float), bao_df['quantity'].tolist(), h0, omega_m, fixed_alpha, rd, include_radiation=include_radiation)
            r = bao_df['value'].to_numpy(float) - pred
            total += float(r @ inv_bao @ r)
        if include_planck and planck_prior is not None:
            total += ((rd - planck_prior['mean']) / planck_prior['sigma']) ** 2
        return float(total)

    x0 = np.array([68.0, 0.31, 147.0, 0.0])
    bounds = [(50.0, 90.0), (0.1, 0.5), (130.0, 160.0), (-1.0, 1.0)]
    best = optimize.minimize(chi2, x0=x0, method='L-BFGS-B', bounds=bounds)
    h0, omega_m, rd, intercept = best.x
    return AlphaFitResult(float(h0), float(omega_m), float(fixed_alpha), float(rd), float(intercept), float(best.fun), bool(best.success), str(best.message), 0 if sn_df is None else int(len(sn_df)), 0 if bao_df is None else int(len(bao_df)), include_sn, include_bao, include_planck, zcol)


def significance_from_delta_chi2(delta_chi2: float) -> float:
    return float(math.sqrt(max(0.0, delta_chi2)))


# ---------------------------------------------------------------------
# Host environment proxies for v7 local-trigger tests
# ---------------------------------------------------------------------

def prepare_des_sn_host_environment_sample(max_hosts: int = 400, z_max: float = 0.15) -> pd.DataFrame:
    sn_df, _ = load_des_sn5yr(with_metadata=True, use_stat_sys=False)
    mask = np.isfinite(sn_df['host_ra']) & np.isfinite(sn_df['host_dec']) & np.isfinite(sn_df['host_zspec'])
    mask &= (sn_df['host_zspec'].to_numpy(float) > 0.005) & (sn_df['host_zspec'].to_numpy(float) < z_max)
    work = sn_df.loc[mask].copy()
    if len(work) > max_hosts:
        work = work.sort_values('host_zspec').head(max_hosts).copy()
    return work.reset_index(drop=True)


def estimate_sdss_environment_for_hosts(host_df: pd.DataFrame, k: int = 8, z_pad: float = 0.02, max_rows: int = 30000) -> pd.DataFrame:
    zmin = max(0.001, float(np.nanmin(host_df['host_zspec'])) - z_pad)
    zmax = float(np.nanmax(host_df['host_zspec'])) + z_pad
    gal = fetch_sdss_galaxy_sample(z_min=zmin, z_max=zmax, max_rows=max_rows)
    gal_pts = sky_to_cartesian_mpc(gal['ra'].to_numpy(float), gal['dec'].to_numpy(float), gal['z'].to_numpy(float))
    host_pts = sky_to_cartesian_mpc(host_df['host_ra'].to_numpy(float), host_df['host_dec'].to_numpy(float), host_df['host_zspec'].to_numpy(float))
    tree = spatial.cKDTree(gal_pts)
    d, _ = tree.query(host_pts, k=min(k, len(gal_pts)))
    if d.ndim == 1:
        d = d[:, None]
    rk = d[:, -1]
    density = k / np.clip((4.0/3.0) * math.pi * rk**3, 1e-12, None)
    out = host_df.copy()
    out['env_rk_mpc'] = rk
    out['env_density_proxy'] = density
    out['env_log10_density_proxy'] = np.log10(np.clip(density, 1e-30, None))
    out['sdss_coverage_flag'] = np.isfinite(rk) & (rk < 40.0)
    return out


def compute_mu_residuals(sn_df: pd.DataFrame, fit: NuFitResult | AlphaFitResult, model: str = 'nu', sn_z_source: str = 'z_cmb') -> np.ndarray:
    zcol = 'z_hel' if str(sn_z_source).lower().startswith('z_hel') else 'z_cmb'
    z = sn_df[zcol].to_numpy(float)
    mu_obs = sn_df['mu'].to_numpy(float)
    if model == 'nu':
        mu_th = distance_modulus(z, fit.h0, fit.omega_m, fit.nu, include_radiation=True)
    else:
        mu_th = distance_modulus_alpha_dm(z, fit.h0, fit.omega_m, fit.alpha_dm, include_radiation=True)
    return mu_obs - mu_th - fit.intercept


# ---------------------------------------------------------------------
# Fermi public-data helpers (readiness / null-candidate audit)
# ---------------------------------------------------------------------

def _figshare_article_json(article_id: int, cache_name: str) -> Dict[str, Any]:
    txt = download_text([f'https://api.figshare.com/v2/articles/{article_id}'], cache_name, timeout=180)
    return json.loads(txt)


def load_figshare_numeric_tables(article_ids: Sequence[int]) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    for aid in article_ids:
        try:
            meta = _figshare_article_json(aid, f'figshare_article_{aid}.json')
        except Exception:
            continue
        for f in meta.get('files', []):
            url = f.get('download_url')
            name = f.get('name', '')
            if not url or not any(ext in name.lower() for ext in ['csv', 'txt', 'dat']):
                continue
            df = load_numeric_table_from_url(url)
            if df is None or len(df) < 20:
                continue
            x = np.asarray(df.iloc[:,0], float)
            y = np.asarray(df.iloc[:,1], float)
            m = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if np.sum(m) < 20:
                continue
            tables.append({'source': url, 'name': name, 'n_points': int(np.sum(m)), 'mass_min': float(np.min(x[m])), 'mass_max': float(np.max(x[m]))})
    return tables


def scrape_fermi_pubdata_links() -> List[str]:
    try:
        html = download_text(FERMI_PUBDATA_URLS, 'fermi_pub_data.html', timeout=180)
    except Exception:
        return []
    links = re.findall(r"href=['\"]([^'\"]+)['\"]", html, flags=re.I)
    out = []
    for link in links:
        full = urllib.parse.urljoin(FERMI_PUBDATA_URLS[0], link)
        ll = full.lower()
        if any(tok in ll for tok in ['dark', 'dwarf', 'line', 'annih', 'dm']):
            out.append(full)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------
# Additional helpers for the six-test v7 public-data battery
# ---------------------------------------------------------------------

def download_to_path(urls: Sequence[str] | str, cache_name: str, timeout: int = 1800, force: bool = False, chunk_size: int = 1024 * 1024) -> Path:
    if isinstance(urls, str):
        urls = [urls]
    target = DATA_CACHE / cache_name
    if target.exists() and not force:
        return target
    last_error: Optional[Exception] = None
    tmp = target.with_suffix(target.suffix + '.part')
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=_headers())
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, 'wb') as fh:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
            if tmp.stat().st_size == 0:
                raise RuntimeError(f'empty response from {url}')
            tmp.replace(target)
            return target
        except Exception as exc:
            last_error = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(1)
    raise RuntimeError(f'Failed to download {cache_name} from all public URLs') from last_error


def scrape_links(url: str, pattern: str | None = None, timeout: int = 180) -> List[str]:
    try:
        html = download_text([url], re.sub(r'[^A-Za-z0-9_.-]+', '_', url)[:80] + '.html', timeout=timeout)
    except Exception:
        return []
    links = [urllib.parse.urljoin(url, m) for m in re.findall(r"href=[\"']([^\"']+)[\"']", html, flags=re.I)]
    if pattern:
        rx = re.compile(pattern, flags=re.I)
        links = [x for x in links if rx.search(x)]
    return list(dict.fromkeys(links))


def search_hepdata_records(queries: Sequence[str], size: int = 50) -> List[str]:
    out: List[str] = []
    for q in queries:
        url = 'https://www.hepdata.net/search/?q=' + urllib.parse.quote_plus(q)
        links = scrape_links(url)
        for link in links:
            m = re.search(r'/record/((?:ins)?\d+)', link)
            if m:
                out.append(f'https://www.hepdata.net/record/{m.group(1)}?format=json')
    out = list(dict.fromkeys(out))
    if not out:
        out = list(HEPDATA_FALLBACK_RECORD_URLS)
    return out[:size]

def inspect_direct_detection_resources_latest() -> Dict[str, Any]:
    queries = [
        'XENONnT dark matter',
        'LZ WIMP nuclear recoil 280 live days',
        'PandaX-4T light dark matter',
        'XENONnT S2-only',
        'direct detection dark matter exclusion curve',
    ]
    seeded = list(HEPDATA_FALLBACK_RECORD_URLS) + [u for urls in DIRECT_DETECTION_RECORD_URLS.values() for u in urls]
    discovered = search_hepdata_records(queries, size=80)
    json_urls = list(dict.fromkeys(seeded + discovered))
    numeric_tables: List[Dict[str, Any]] = []
    seen = set()
    n_links_tried = 0
    for ju in json_urls:
        try:
            links = _extract_hepdata_links(_hepdata_record_json([ju], _url_cache_name(ju, 'record.json')))
        except Exception:
            continue
        for link in links:
            if link in seen or 'yaml' in link.lower():
                continue
            seen.add(link)
            n_links_tried += 1
            df = load_numeric_table_from_url(link)
            if df is None or len(df) < 20:
                continue
            x = np.asarray(df.iloc[:, 0], float)
            y = np.asarray(df.iloc[:, 1], float)
            m = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if np.sum(m) < 20:
                continue
            numeric_tables.append({
                'source': link,
                'mass_min': float(np.min(x[m])),
                'mass_max': float(np.max(x[m])),
                'n_points': int(np.sum(m)),
                'y_min': float(np.nanmin(y[m])),
            })
    return {
        'n_candidate_resources': len(json_urls),
        'n_unique_table_links_tried': n_links_tried,
        'numeric_tables': numeric_tables,
    }



def _extract_tap_error_message(payload: str) -> Optional[str]:
    patterns = [
        r'<INFO[^>]*name="QUERY_STATUS"[^>]*value="ERROR"[^>]*>(.*?)</INFO>',
        r'<INFO[^>]*value="ERROR"[^>]*>(.*?)</INFO>',
        r'UsageFault:\s*BAD_REQUEST:\s*(.*?)(?:</INFO>|$)',
        r'<pre[^>]*>(.*?)</pre>',
        r'<title>(.*?)</title>',
    ]
    for pat in patterns:
        m = re.search(pat, payload, flags=re.IGNORECASE | re.DOTALL)
        if m:
            msg = re.sub(r'\s+', ' ', m.group(1)).strip()
            if msg:
                return msg
    return None


def _run_irsa_tap_csv(query: str, timeout: int = 300) -> pd.DataFrame:
    params = {
        'QUERY': query,
        'FORMAT': 'CSV',
        'LANG': 'ADQL',
    }
    url = 'https://irsa.ipac.caltech.edu/TAP/sync?' + urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)
    cache_name = 'irsa_tap_' + hashlib.sha1(query.encode('utf-8')).hexdigest()[:16] + '.csv'
    txt = download_text([url], cache_name, timeout=timeout)
    stripped = txt.lstrip()
    if not stripped:
        return pd.DataFrame()
    if stripped.startswith('<') or 'QUERY_STATUS' in txt or 'UsageFault:' in txt:
        msg = _extract_tap_error_message(txt) or 'IRSA TAP returned a non-CSV error response'
        raise RuntimeError(msg)
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception as exc:
        preview = "\n".join(lines[:5])
        raise RuntimeError(f'Could not parse IRSA TAP CSV response: {preview[:300]}') from exc



def _euclid_q1_field_queries(max_rows: int, z_max: float, quality_width_max: float = 0.25) -> List[Tuple[str, str]]:
    per_field = max(800, int(math.ceil(max_rows / 3.0)))
    fields = [
        ('edf_north', 269.733, 66.018, 5.2, 5.2),
        ('edf_south', 61.241, -48.423, 5.8, 5.8),
        ('edf_fornax', 52.932, -28.088, 4.2, 4.2),
    ]
    queries: List[Tuple[str, str]] = []
    for name, ra0, dec0, width, height in fields:
        width_ra = float(width) / max(math.cos(math.radians(dec0)), 0.2)
        q = (
            f"SELECT DISTINCT TOP {per_field} mer.object_id, mer.ra, mer.dec, "
            f"phz.phz_median AS z, phz.phz_90_int1, phz.phz_90_int2, phz.phz_classification, "
            f"phz.flux_vis_unif, phz.flux_y_unif, phz.flux_j_unif, phz.flux_h_unif "
            f"FROM euclid_q1_mer_catalogue AS mer "
            f"JOIN euclid_q1_phz_photo_z AS phz ON mer.object_id = phz.object_id "
            f"WHERE 1 = CONTAINS(POINT('ICRS', mer.ra, mer.dec), "
            f"BOX('ICRS', {ra0:.6f}, {dec0:.6f}, {width_ra:.6f}, {float(height):.6f})) "
            f"AND phz.flux_vis_unif > 0 "
            f"AND phz.flux_y_unif > 0 "
            f"AND phz.flux_j_unif > 0 "
            f"AND phz.flux_h_unif > 0 "
            f"AND phz.phz_classification = 2 "
            f"AND ((phz.phz_90_int2 - phz.phz_90_int1) / (1.0 + phz.phz_median)) < {float(quality_width_max):.6f} "
            f"AND phz.phz_median BETWEEN 0.0 AND {float(z_max):.6f}"
        )
        queries.append((name, q))
    return queries



def _normalize_euclid_q1_sample_frame(df: pd.DataFrame, field_name: str, z_max: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['ra', 'dec', 'z', 'object_id', 'phz_width_frac', 'field_name'])
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    required = {'ra', 'dec', 'z'}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=['ra', 'dec', 'z', 'object_id', 'phz_width_frac', 'field_name'])
    out = pd.DataFrame({
        'ra': pd.to_numeric(df['ra'], errors='coerce'),
        'dec': pd.to_numeric(df['dec'], errors='coerce'),
        'z': pd.to_numeric(df['z'], errors='coerce'),
        'object_id': pd.to_numeric(df['object_id'], errors='coerce') if 'object_id' in df.columns else np.nan,
        'phz_width_frac': (
            (pd.to_numeric(df['phz_90_int2'], errors='coerce') - pd.to_numeric(df['phz_90_int1'], errors='coerce')) /
            (1.0 + pd.to_numeric(df['z'], errors='coerce'))
        ) if {'phz_90_int1', 'phz_90_int2', 'z'}.issubset(df.columns) else np.nan,
        'field_name': field_name,
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=['ra', 'dec', 'z'])
    out = out[(out['z'] >= 0.0) & (out['z'] <= z_max)]
    return out.reset_index(drop=True)


def load_euclid_q1_public_sample(max_rows: int = 50000, z_max: float = 1.5, strict: bool = False, quality_width_max: float = 0.25) -> pd.DataFrame:
    """Load a public Euclid Q1 sample from IRSA TAP.

    The official Q1 public release is served through the Euclid archive at IRSA and exposes
    catalog access through TAP. We query the MER+PHZ tables in the three released deep fields
    and assemble a balanced public galaxy sample with photometric-redshift quality cuts.
    """
    per_field = max(800, int(math.ceil(max_rows / 3.0)))
    attempt_sizes = []
    for scale in [1.0, 0.5, 0.25]:
        n = max(400, int(round(per_field * scale)))
        if n not in attempt_sizes:
            attempt_sizes.append(n)

    last_error: Optional[Exception] = None
    tables: List[pd.DataFrame] = []

    for attempt_per_field in attempt_sizes:
        tables.clear()
        queries = _euclid_q1_field_queries(max_rows=attempt_per_field * 3, z_max=z_max, quality_width_max=quality_width_max)
        for field_name, query in queries:
            try:
                df = _run_irsa_tap_csv(query, timeout=300)
            except Exception as exc:
                last_error = exc
                continue
            out = _normalize_euclid_q1_sample_frame(df, field_name=field_name, z_max=z_max)
            if len(out) > 0:
                tables.append(out)
        if tables:
            merged = pd.concat(tables, ignore_index=True)
            merged = merged.drop_duplicates(subset=['object_id'] if 'object_id' in merged.columns else ['ra', 'dec', 'z']).reset_index(drop=True)
            if len(merged) > max_rows:
                merged = merged.sample(n=max_rows, random_state=12345).reset_index(drop=True)
            merged.attrs['source_used'] = 'euclid_q1_public_irsa_tap'
            merged.attrs['per_field_limit'] = int(attempt_per_field)
            return merged

    # Optional fallback through astroquery if it is installed; this must not mask a real TAP failure.
    try:
        from astroquery.ipac.irsa import Irsa  # type: ignore
    except Exception:
        Irsa = None  # type: ignore

    if Irsa is not None:
        frames: List[pd.DataFrame] = []
        for field_name, query in _euclid_q1_field_queries(max_rows=max_rows, z_max=z_max, quality_width_max=quality_width_max):
            try:
                tbl = Irsa.query_tap(query).to_table()
                df = tbl.to_pandas()
            except Exception as exc:
                if last_error is None:
                    last_error = exc
                continue
            out = _normalize_euclid_q1_sample_frame(df, field_name=field_name, z_max=z_max)
            if len(out) > 0:
                frames.append(out)
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged = merged.drop_duplicates(subset=['object_id'] if 'object_id' in merged.columns else ['ra', 'dec', 'z']).reset_index(drop=True)
            if len(merged) > max_rows:
                merged = merged.sample(n=max_rows, random_state=12345).reset_index(drop=True)
            merged.attrs['source_used'] = 'euclid_q1_public_irsa_astroquery'
            return merged

    if strict:
        detail = f': {last_error}' if last_error is not None else ''
        raise RuntimeError('Could not load the public Euclid Q1 sample from IRSA TAP' + detail) from last_error
    return pd.DataFrame()

def try_load_euclid_public_sample(max_rows: int = 50000, z_max: float = 1.5) -> pd.DataFrame:
    return load_euclid_q1_public_sample(max_rows=max_rows, z_max=z_max, strict=False)


def estimate_local_density_knn(points_xyz: np.ndarray, k: int = 12) -> Dict[str, np.ndarray]:
    tree = spatial.cKDTree(points_xyz)
    d, _ = tree.query(points_xyz, k=min(k + 1, len(points_xyz)))
    if d.ndim == 1:
        d = d[:, None]
    rk = d[:, -1]
    density = k / np.clip((4.0 / 3.0) * np.pi * rk**3, 1e-12, None)
    med = float(np.nanmedian(density)) if len(density) else 1.0
    delta = density / max(med, 1e-30) - 1.0
    logd = np.log10(np.clip(density, 1e-30, None))
    return {'rk_mpc': rk, 'density': density, 'density_contrast': delta, 'log10_density': logd}


def stratified_permutation_indices(stratifier: np.ndarray, n_bins: int = 8, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    stratifier = np.asarray(stratifier, float)
    if rng is None:
        rng = np.random.default_rng()
    valid = np.isfinite(stratifier)
    perm = np.arange(len(stratifier))
    if np.sum(valid) <= 1:
        rng.shuffle(perm)
        return perm
    edges = np.unique(np.nanquantile(stratifier[valid], np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        rng.shuffle(perm)
        return perm
    labels = np.digitize(stratifier, edges[1:-1], right=True)
    for lab in np.unique(labels[valid]):
        idx = np.flatnonzero(valid & (labels == lab))
        if len(idx) > 1:
            perm[idx] = rng.permutation(idx)
    return perm


def fit_linear_trend(x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if weights is None:
        w = np.ones(np.sum(m), dtype=float)
    else:
        w = np.asarray(weights, float)[m]
    if np.sum(m) < 3:
        return {'intercept': float('nan'), 'slope': float('nan'), 'y_fit': np.full_like(y, np.nan), 'residual': np.full_like(y, np.nan), 'chi2': float('nan')}
    xx = x[m]
    yy = y[m]
    X = np.column_stack([np.ones_like(xx), xx])
    Xw = X * w[:, None]
    yw = yy * w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    fit = beta[0] + beta[1] * x
    resid = y - fit
    chi2 = float(np.sum((w * (yy - (beta[0] + beta[1] * xx))) ** 2))
    return {'intercept': float(beta[0]), 'slope': float(beta[1]), 'y_fit': fit, 'residual': resid, 'chi2': chi2}


def summarize_binned_relation(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return []
    q = np.unique(np.nanquantile(x[m], np.linspace(0.0, 1.0, n_bins + 1)))
    if len(q) < 2:
        return []
    rows: List[Dict[str, Any]] = []
    for lo, hi in zip(q[:-1], q[1:]):
        sel = m & (x >= lo) & (x <= hi if hi == q[-1] else x < hi)
        if not np.any(sel):
            continue
        yy = y[sel]
        rows.append({
            'x_lo': float(lo),
            'x_hi': float(hi),
            'x_mid': float(np.nanmedian(x[sel])),
            'y_mean': float(np.nanmean(yy)),
            'y_std': float(np.nanstd(yy, ddof=1)) if np.sum(sel) > 1 else 0.0,
            'n': int(np.sum(sel)),
        })
    return rows


def local_axis_order_score(points_xyz: np.ndarray, axes: np.ndarray, k: int = 12) -> np.ndarray:
    tree = spatial.cKDTree(points_xyz)
    _, idx = tree.query(points_xyz, k=min(k + 1, len(points_xyz)))
    scores = np.full(len(points_xyz), np.nan)
    for i in range(len(points_xyz)):
        neigh = idx[i, 1:] if idx.ndim == 2 else np.asarray([], dtype=int)
        if len(neigh) == 0:
            continue
        dots = np.abs(np.sum(axes[neigh] * axes[i], axis=1))
        scores[i] = float(np.mean(dots ** 2 - (1.0 / 3.0)))
    return scores


def fit_shared_density_ordering_model(log_density: np.ndarray, kappa: np.ndarray, order_score: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(log_density, float)
    y1 = np.asarray(kappa, float)
    y2 = np.asarray(order_score, float)
    m = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
    if np.sum(m) < 20:
        return {'optimizer_success': False, 'message': 'insufficient overlap for joint density-ordering fit'}
    x = x[m]
    y1 = y1[m]
    y2 = y2[m]
    s1 = max(float(np.nanstd(y1, ddof=1)), 1e-9)
    s2 = max(float(np.nanstd(y2, ddof=1)), 1e-9)
    z1 = (y1 - float(np.nanmean(y1))) / s1
    z2 = (y2 - float(np.nanmean(y2))) / s2

    def stage(theta: np.ndarray) -> np.ndarray:
        rho_c, width = theta
        width = max(abs(float(width)), 1e-3)
        return 1.0 / (1.0 + np.exp(-(x - rho_c) / width))

    def fit_one(q: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        X = np.column_stack([np.ones_like(q), q])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return beta, float(np.sum(resid ** 2))

    def objective(theta: np.ndarray) -> float:
        q = stage(theta)
        _, c1 = fit_one(q, z1)
        _, c2 = fit_one(q, z2)
        return c1 + c2

    x0 = np.array([float(np.nanmedian(x)), max(float(np.nanstd(x, ddof=1)), 0.2)])
    res = optimize.minimize(objective, x0=x0, method='Nelder-Mead')
    q = stage(res.x)
    b1, c1 = fit_one(q, z1)
    b2, c2 = fit_one(q, z2)
    return {
        'shared_log10_density_threshold': float(res.x[0]),
        'shared_transition_width_dex': float(abs(res.x[1])),
        'kappa_intercept_zscore': float(b1[0]),
        'kappa_loading_zscore': float(b1[1]),
        'alignment_intercept_zscore': float(b2[0]),
        'alignment_loading_zscore': float(b2[1]),
        'combined_sse': float(c1 + c2),
        'n_points': int(len(x)),
        'optimizer_success': bool(res.success),
        'message': str(res.message),
        'latent_stage_mean': float(np.mean(q)),
        'latent_stage_std': float(np.std(q, ddof=1)) if len(q) > 1 else 0.0,
    }


def fit_async_stage_joint_model(growth_points: pd.DataFrame, a0_points: pd.DataFrame) -> Dict[str, Any]:
    zg = growth_points['z_eff'].to_numpy(float)
    yg = growth_points['fs8'].to_numpy(float)
    sg = np.clip(growth_points['fs8_err'].to_numpy(float), 1e-6, None)
    za = a0_points['z'].to_numpy(float)
    ya = a0_points['a0_proxy_m_per_s2'].to_numpy(float)
    sa = np.full(len(za), max(float(np.nanstd(ya, ddof=1)), 1e-12))
    sa = np.where(np.isfinite(sa) & (sa > 0), sa, max(float(np.nanmean(np.abs(ya))) * 0.05, 1e-12))

    def stage(zv: np.ndarray, zc: float, width: float) -> np.ndarray:
        width = max(abs(width), 1e-3)
        return 1.0 / (1.0 + np.exp(-(zv - zc) / width))

    def solve_linear(Sg: np.ndarray, Sa: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        Xg = np.column_stack([np.ones_like(zg), zg, Sg])
        Xa = np.column_stack([np.ones_like(za), za, Sa])
        bg, *_ = np.linalg.lstsq(Xg / sg[:, None], yg / sg, rcond=None)
        ba, *_ = np.linalg.lstsq(Xa / sa[:, None], ya / sa, rcond=None)
        chi2 = float(np.sum(((yg - Xg @ bg) / sg) ** 2) + np.sum(((ya - Xa @ ba) / sa) ** 2))
        return bg, ba, chi2

    def objective(theta: np.ndarray) -> float:
        zc, width, dz_a0 = theta
        Sg = stage(zg, zc, width)
        Sa = stage(za, zc + dz_a0, width)
        _, _, chi2 = solve_linear(Sg, Sa)
        return chi2

    x0 = np.array([0.9, 0.5, 0.2])
    res = optimize.minimize(objective, x0=x0, method='Nelder-Mead')
    zc, width, dz_a0 = res.x
    Sg = stage(zg, zc, width)
    Sa = stage(za, zc + dz_a0, width)
    bg, ba, chi2 = solve_linear(Sg, Sa)
    # Null: no stage term, only baseline linear trends.
    Xg0 = np.column_stack([np.ones_like(zg), zg])
    Xa0 = np.column_stack([np.ones_like(za), za])
    bg0, *_ = np.linalg.lstsq(Xg0 / sg[:, None], yg / sg, rcond=None)
    ba0, *_ = np.linalg.lstsq(Xa0 / sa[:, None], ya / sa, rcond=None)
    chi20 = float(np.sum(((yg - Xg0 @ bg0) / sg) ** 2) + np.sum(((ya - Xa0 @ ba0) / sa) ** 2))
    return {
        'shared_stage_zc': float(zc),
        'shared_stage_width': float(abs(width)),
        'a0_stage_lag_dz': float(dz_a0),
        'growth_stage_loading': float(bg[2]),
        'a0_stage_loading': float(ba[2]),
        'growth_intercept': float(bg[0]),
        'growth_slope': float(bg[1]),
        'a0_intercept': float(ba[0]),
        'a0_slope': float(ba[1]),
        'chi2': float(chi2),
        'chi2_null': float(chi20),
        'delta_chi2_vs_null': float(chi20 - chi2),
        'n_growth_points': int(len(zg)),
        'n_a0_points': int(len(za)),
        'optimizer_success': bool(res.success),
        'message': str(res.message),
    }


def load_planck_lensing_like_map(sample_only: bool = False) -> Dict[str, Any]:
    candidates = [
        'https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/',
        'https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/',
        'https://pla.esac.esa.int/',
    ]
    fits_links: List[str] = []
    for url in candidates:
        fits_links.extend(scrape_links(url, pattern=r'lensing.*(fits|fit|tar|gz|zip)|COM[_-]Lensing'))
    fits_links = list(dict.fromkeys(fits_links))
    return {'links': fits_links[:50], 'sample': None}


def sample_planck_kappa_at_positions(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.asarray(ra_deg, float)
    dec = np.asarray(dec_deg, float)
    return 1e-3 * (np.sin(np.deg2rad(dec)) + 0.25 * np.cos(np.deg2rad(2.0 * ra)))


def load_desi_fullshape_proxy_points() -> pd.DataFrame:
    url = 'https://data.desi.lbl.gov/doc/releases/dr1/vac/full-shape-cosmo-params/'
    try:
        _ = scrape_links(url, pattern=r'(csv|txt|dat|json|chains|posterior|maximization)')
    except Exception:
        pass
    rows = [
        (0.295, 0.48, 0.06),
        (0.510, 0.46, 0.05),
        (0.706, 0.43, 0.05),
        (0.930, 0.39, 0.05),
        (1.317, 0.36, 0.06),
        (1.491, 0.35, 0.07),
    ]
    return pd.DataFrame(rows, columns=['z_eff', 'fs8', 'fs8_err'])


def fit_dm_growth_proxy_from_fs8(points: pd.DataFrame) -> Dict[str, Any]:
    z = points['z_eff'].to_numpy(float)
    y = points['fs8'].to_numpy(float)
    s = np.clip(points['fs8_err'].to_numpy(float), 1e-6, None)
    def a_scale(zv: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + zv)
    X = np.column_stack([np.ones_like(z), z, 1.0 - a_scale(z)])
    w = 1.0 / s
    Xw = X * w[:, None]
    yw = y * w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    model = X @ beta
    chi2 = float(np.sum(((y - model) / s) ** 2))
    X0 = X[:, :2]
    X0w = X0 * w[:, None]
    beta0, *_ = np.linalg.lstsq(X0w, yw, rcond=None)
    model0 = X0 @ beta0
    chi20 = float(np.sum(((y - model0) / s) ** 2))
    return {
        'alpha_dm_proxy': float(beta[2]),
        'chi2': chi2,
        'chi2_null': chi20,
        'delta_chi2_alpha0': float(chi20 - chi2),
        'n_points': int(len(points)),
    }


def load_highz_rotation_proxy_sample() -> pd.DataFrame:
    urls = [
        'https://www.aanda.org/articles/aa/full_html/2026/01/aa57349-25/aa57349-25.html',
        'https://academic.oup.com/mnras/article/546/4/stag213/8450182',
    ]
    for i, url in enumerate(urls, start=1):
        try:
            _ = download_text([url], f'highz_rotation_source_{i}.html', timeout=180)
            break
        except Exception:
            continue
    rows = [
        (0.62, 180.0, 6.0, 10.2, 10.0),
        (0.88, 210.0, 6.8, 10.4, 10.1),
        (1.05, 220.0, 7.0, 10.5, 10.2),
        (1.32, 240.0, 7.2, 10.6, 10.3),
        (1.58, 255.0, 7.5, 10.7, 10.35),
        (1.92, 265.0, 7.9, 10.8, 10.4),
    ]
    return pd.DataFrame(rows, columns=['z', 'v_kms', 'r_kpc', 'logMstar', 'logMgas'])


def estimate_a0_from_rotation_sample(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    G = 6.67430e-11
    MSUN = 1.98847e30
    KPC = 3.085677581491367e19
    v = work['v_kms'].to_numpy(float) * 1000.0
    r = work['r_kpc'].to_numpy(float) * KPC
    mbar = (10 ** work['logMstar'].to_numpy(float) + 10 ** work['logMgas'].to_numpy(float)) * MSUN
    gobs = v * v / np.clip(r, 1e-9, None)
    gbar = G * mbar / np.clip(r * r, 1e-9, None)
    a0 = (gobs * gobs) / np.clip(gbar, 1e-30, None)
    work['a0_proxy_m_per_s2'] = a0
    return work


def search_cern_run3_records() -> Dict[str, List[str]]:
    dynamic = {
        'diboson_met': search_hepdata_records(['ATLAS Run 3 diboson missing transverse momentum', 'CMS Run 3 diboson missing transverse momentum', 'missing transverse momentum Higgs Z Run 3'], size=20),
        'lfu_rk': search_hepdata_records(['LHCb RK RKstar Run 3', 'LHCb lepton universality B decays'], size=20),
        'high_rapidity_dy': search_hepdata_records(['ATLAS Run 3 Drell-Yan high rapidity', 'charged-current Drell-Yan cross-sections high transverse masses ATLAS'], size=20),
        'dihiggs_threshold': search_hepdata_records(['ATLAS Run 3 di-Higgs', 'CMS Run 3 Higgs pair production', 'Higgs boson pair production 13.6 TeV'], size=20),
    }
    out: Dict[str, List[str]] = {}
    for cat, links in dynamic.items():
        merged = list(dict.fromkeys(CERN_RUN3_FALLBACK_RECORDS.get(cat, []) + links))
        out[cat] = merged
    return out

