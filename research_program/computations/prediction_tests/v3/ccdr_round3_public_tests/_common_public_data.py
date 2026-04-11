#!/usr/bin/env python3
from __future__ import annotations

import io
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
from scipy import linalg, optimize, signal, sparse, stats
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
PDG_LEPTON_PDF_URLS = [
    "https://pdg.lbl.gov/2024/tables/rpp2024-sum-leptons.pdf",
]
PDG_QUARK_PDF_URLS = [
    "https://pdg.lbl.gov/2024/tables/rpp2024-sum-quarks.pdf",
]
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


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; CCDRRound3PublicTests/1.0; +https://openai.com)",
        "Accept": "*/*",
    }


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def download_bytes(urls: Sequence[str], cache_name: str, timeout: int = 180, force: bool = False) -> bytes:
    target = DATA_CACHE / cache_name
    if target.exists() and not force:
        return target.read_bytes()
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=_headers())
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

def fetch_sdss_galaxy_sample(z_min: float = 0.02, z_max: float = 0.12, max_rows: int = 15000, seed: int = 1234) -> pd.DataFrame:
    queries = [
        f"SELECT TOP {max_rows} p.ra, p.dec, s.z FROM PhotoObjAll p JOIN SpecObjAll s ON s.bestObjID = p.objID WHERE s.class = 'GALAXY' AND s.z BETWEEN {z_min:.6f} AND {z_max:.6f} AND p.type = 3 ORDER BY NEWID()",
        f"SELECT TOP {max_rows} p.ra, p.dec, s.z FROM PhotoObj p JOIN SpecObj s ON s.bestObjID = p.objID WHERE s.class = 'GALAXY' AND s.z BETWEEN {z_min:.6f} AND {z_max:.6f} ORDER BY p.objID",
    ]
    last_error = None
    for base in SDSS_SKYSERVER_SQL_BASES:
        for i, query in enumerate(queries):
            params = urllib.parse.urlencode({"cmd": query, "format": "csv"})
            url = f"{base}?{params}"
            cache = f"sdss_sample_{i}_{abs(hash((base,query))) % (10**8)}.csv"
            try:
                txt = download_text([url], cache, timeout=240)
                df = pd.read_csv(io.StringIO(txt))
                cols = {c.lower(): c for c in df.columns}
                ra = pick_column(df.columns, ["ra"])
                dec = pick_column(df.columns, ["dec"])
                z = pick_column(df.columns, ["z"])
                out = pd.DataFrame({"ra": df[ra].astype(float), "dec": df[dec].astype(float), "z": df[z].astype(float)})
                out = out[np.isfinite(out["ra"]) & np.isfinite(out["dec"]) & np.isfinite(out["z"])]
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


def estimate_filament_axes_knn(points_xyz: np.ndarray, k: int = 12) -> np.ndarray:
    from scipy.spatial import cKDTree
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

def _normalize_sparc_rar_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for col in out.columns:
        # pandas compatibility: some versions reject errors="ignore" here.
        # Convert only fully numeric columns and leave label/text columns intact.
        try:
            out[col] = pd.to_numeric(out[col], errors="raise")
        except Exception:
            pass

    # Public SPARC RAR tables sometimes expose gbar/gobs in log10(m/s^2).
    # Detect that case conservatively and convert to linear units.
    for base in ["gbar", "gobs"]:
        if base not in out.columns:
            continue
        vals = pd.to_numeric(out[base], errors="coerce").to_numpy(float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        # Typical linear accelerations are positive and tiny; log10 values are
        # usually negative, around -13 to -8 in SI units.
        if np.nanmax(finite) < 0.0 and np.nanmin(finite) > -30.0:
            out[base] = np.power(10.0, vals)
            err_col = f"e_{base}"
            if err_col in out.columns:
                err_vals = pd.to_numeric(out[err_col], errors="coerce").to_numpy(float)
                err_fin = err_vals[np.isfinite(err_vals)]
                # If the uncertainties also look logarithmic, convert dex-like
                # errors into approximate linear absolute errors.
                if err_fin.size and np.nanmax(np.abs(err_fin)) < 5.0:
                    lin = np.power(10.0, vals)
                    out[err_col] = np.log(10.0) * lin * np.abs(err_vals)
    return out


def load_sparc_rar() -> pd.DataFrame:
    from astropy.io import ascii
    raw = download_text(SPARC_RAR_URLS, "SPARC_RAR.mrt", timeout=180)
    table = ascii.read(raw, format="cds")
    return _normalize_sparc_rar_df(table.to_pandas())


def load_sparc_table1() -> pd.DataFrame:
    from astropy.io import ascii
    raw = download_text(SPARC_TABLE1_URLS, "SPARC_Table1.mrt", timeout=180)
    table = ascii.read(raw, format="cds")
    out = table.to_pandas()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def rar_relation(gbar: np.ndarray, a0: float) -> np.ndarray:
    x = np.sqrt(np.clip(gbar / a0, 1e-30, None))
    return gbar / (1.0 - np.exp(-x))


def fit_rar_hierarchical_like(df: pd.DataFrame, galaxy_col: str, gobs_col: str, gbar_col: str, offset_prior_dex: float = 0.08) -> Dict[str, Any]:
    work = df[[galaxy_col, gobs_col, gbar_col]].copy()
    work[gobs_col] = pd.to_numeric(work[gobs_col], errors="coerce")
    work[gbar_col] = pd.to_numeric(work[gbar_col], errors="coerce")
    work = work.dropna().copy()
    work = work[(work[gobs_col] > 0) & (work[gbar_col] > 0)].reset_index(drop=True)
    if len(work) == 0:
        raise RuntimeError(
            f"No usable positive RAR points after cleaning. Columns were {galaxy_col}, {gobs_col}, {gbar_col}; "
            "this usually means the public SPARC file is in log10 units and was not converted correctly."
        )
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
    bounds = [(-12.0, -8.0)] + [(None, None)] * len(galaxies)
    res = optimize.minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
    offsets = res.x[1:]
    return {
        "best_a0_m_per_s2": float(10.0 ** res.x[0]),
        "n_points": int(len(work)),
        "n_galaxies": int(len(galaxies)),
        "offset_rms_dex": float(np.sqrt(np.mean(offsets ** 2))) if len(offsets) else 0.0,
        "optimizer_success": bool(res.success),
        "optimizer_message": str(res.message),
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
        txt = download_text([url], cache_name=Path(urllib.parse.urlparse(url).path).name or "table.txt", timeout=180)
    except Exception:
        return None
    if "yaml" in url.lower() or txt.lstrip().startswith("independent_variables"):
        return None
    for sep in [",", "\t", r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(txt), sep=sep, engine="python")
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] >= 2:
                return num
        except Exception:
            continue
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
