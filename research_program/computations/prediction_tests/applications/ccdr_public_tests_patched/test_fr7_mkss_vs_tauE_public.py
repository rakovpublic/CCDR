from __future__ import annotations

"""FR7 public-data proxy test: M_KSS-like confinement margin vs tau_E.

Prediction in note: M_KSS should correlate with tau_E across tokamak databases.
Careful interpretation: M_KSS is a *margin above a minimum transport floor*;
better confinement should mean smaller margin and larger tau_E, i.e. an expected
negative correlation if the proxy is chi/chi_min.  But chi_proxy=a^2/tau_E
contains tau_E by construction, so this script treats that as a mechanical
sanity check only.

Patch 15 improvements:
- Handles the public HDB OSF transposed CSV layout via _common_public_data.
- Tests *all* recognized H-factor columns, not just the first match.
- Adds permutation p-values for H-factor vs residual confinement.
- Adds device-stratified residual diagnostics when a tokamak/device column exists.
- Keeps the final verdict conservative: mechanical chi_proxy alone cannot
  confirm FR7; support requires a positive H-factor residual check or a real
  independent transport/viscosity/diffusivity column.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common_public_data import (
    ensure_dir, json_dump, osf_find_download, read_public_table, save_plot,
    spearman_rank_corr, structured_report, best_matching_column_fuzzy,
    repair_header_if_needed,
)

ROOT = Path(__file__).resolve().parent
OUT = ensure_dir(ROOT / 'outputs' / 'fr7')
HBAR = 1.054571817e-34
M_D = 3.3435837724e-27
CHI_MIN = HBAR / (4 * np.pi * M_D)

TAU_CANDS = ['TAUTH','TAUE','TAU_E','TAU','energy confinement time','tauth s','taue s']
A_CANDS = ['AMIN','A_MINOR','MINOR_RADIUS','minor radius','amin m','a m']
R_CANDS = ['RGEO','RMAJOR','R_MAJOR','MAJOR_RADIUS','major radius','rgeo m']
IP_CANDS = ['IP','IPLA','IPL','PLASMA_CURRENT','current ma','plasma current']
BT_CANDS = ['BT','BTOR','BT0','TOROIDAL_FIELD','toroidal field']
NE_CANDS = ['NEL','NEBAR','DENSITY','LINE_AVG_DENSITY','line averaged density','nebar']
POWER_CANDS = ['PLTH','PLOSS','PIN','POH','PNBI','PECH','PICRH','AUX_POWER','HEATING_POWER','power loss','input power']
KAPPA_CANDS = ['KAREA','KAPPA','ELONG','ELONGATION','KAPPA95']
MEFF_CANDS = ['MEFF','AIMASS','MASS','AMASS','isotope mass']
Q95_CANDS = ['Q95','QCYL','Q_EDGE']
DEVICE_CANDS = ['TOK','TOKAMAK','DEVICE','MACHINE']
HFACTOR_CANDS = ['H98Y2','H98','HIPB98Y2','HIPB98','HITER96','H89','H89P','HFACTOR','H_FACTOR','CONF_ENH']
INDEPENDENT_TRANSPORT_CANDS = ['CHI','CHIE','CHII','XIE','XII','XION','XELEC','ETA_S','VISCOSITY','DIFFUSIVITY']


def _norm(x) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(x).lower())


def matching_columns_fuzzy(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    aliases = [_norm(c) for c in candidates if len(_norm(c)) >= 2]
    out: list[str] = []
    for col in df.columns:
        n = _norm(col)
        if len(n) < 2:
            continue
        hit = any(a == n or (len(a) >= 3 and a in n) or (len(n) >= 4 and n in a) for a in aliases)
        if hit and col not in out:
            out.append(str(col))
    return out


def log_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = 400, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    if n < 20:
        return {'median': float('nan'), 'ci95': [float('nan'), float('nan')]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(spearman_rank_corr(x[idx], y[idx]))
    vals = np.asarray(vals, float)
    return {'median': float(np.nanmedian(vals)), 'ci95': [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]}


def permutation_spearman_pvalue(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                                n_perm: int = 300, seed: int = 17, alternative: str = 'greater') -> dict:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    obs = float(spearman_rank_corr(x, y))
    if len(x) < 20 or not np.isfinite(obs):
        return {'observed': obs, 'p_value': float('nan'), 'n_perm': 0, 'alternative': alternative}
    hits = 0
    if groups is None:
        for _ in range(n_perm):
            xp = x.copy(); rng.shuffle(xp)
            rp = spearman_rank_corr(xp, y)
            if (alternative == 'greater' and rp >= obs) or (alternative == 'less' and rp <= obs):
                hits += 1
    else:
        groups = np.asarray(groups)
        unique = [g for g in pd.unique(groups) if str(g).lower() not in {'nan', 'none', '', 'unknown'}]
        valid_grouped = len(unique) >= 2 and any((groups == g).sum() >= 8 for g in unique)
        if not valid_grouped:
            return permutation_spearman_pvalue(x, y, None, n_perm, seed, alternative)
        for _ in range(n_perm):
            xp = x.copy()
            for g in unique:
                idx = np.where(groups == g)[0]
                if len(idx) >= 2:
                    xp[idx] = rng.permutation(xp[idx])
            rp = spearman_rank_corr(xp, y)
            if (alternative == 'greater' and rp >= obs) or (alternative == 'less' and rp <= obs):
                hits += 1
    return {'observed': obs, 'p_value': float((hits + 1) / (n_perm + 1)), 'n_perm': int(n_perm), 'alternative': alternative}


def numeric_series(df: pd.DataFrame, col: str | None, mask=None) -> np.ndarray | None:
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors='coerce')
    if mask is not None:
        s = s.loc[mask]
    return s.to_numpy(float)


def finite_positive(x: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    return np.isfinite(x) & (x > 0)


def build_baseline_features(df: pd.DataFrame, base_mask, columns: dict[str, str | None]) -> tuple[np.ndarray | None, list[str], np.ndarray]:
    """Build log-feature matrix for confinement baseline controls."""
    raw_features: list[np.ndarray] = []
    names: list[str] = []
    local_valid = np.ones(int(base_mask.sum()), dtype=bool)
    for name in ['R','a','Ip','Bt','ne','power','kappa','meff','q95']:
        arr = numeric_series(df, columns.get(name), mask=base_mask)
        good = finite_positive(arr)
        if arr is None or good is None or good.sum() < 250:
            continue
        vals = np.log(arr)
        if np.nanstd(vals[good]) < 1e-6:
            continue
        raw_features.append(vals)
        names.append(name)
        local_valid &= good
    if not raw_features or local_valid.sum() < 250:
        return None, names, local_valid
    X = np.column_stack([f[local_valid] for f in raw_features])
    return X, names, local_valid


def device_stratified_summary(x: np.ndarray, y: np.ndarray, devices: np.ndarray | None, min_n: int = 40) -> dict | None:
    if devices is None:
        return None
    rows = []
    for dev in pd.unique(devices):
        label = str(dev)
        if label.lower() in {'nan', 'none', '', 'unknown'}:
            continue
        idx = np.where(devices == dev)[0]
        if len(idx) < min_n:
            continue
        rho = spearman_rank_corr(x[idx], y[idx])
        if np.isfinite(rho):
            rows.append({'device': label, 'n': int(len(idx)), 'rho': float(rho)})
    if not rows:
        return None
    weights = np.array([r['n'] for r in rows], float)
    rhos = np.array([r['rho'] for r in rows], float)
    return {
        'min_n_per_device': min_n,
        'n_devices_used': int(len(rows)),
        'weighted_mean_rho': float(np.average(rhos, weights=weights)),
        'median_rho': float(np.median(rhos)),
        'fraction_positive': float(np.mean(rhos > 0)),
        'devices': sorted(rows, key=lambda r: r['n'], reverse=True)[:25],
    }


def residual_hfactor_check(df: pd.DataFrame, base_mask, c_tau: str, c_h: str,
                           aux_cols: dict[str, str | None], device_col: str | None = None) -> dict | None:
    """H-factor vs residual tau_E after baseline controls."""
    tau = numeric_series(df, c_tau, mask=base_mask)
    h = numeric_series(df, c_h, mask=base_mask)
    if tau is None or h is None:
        return None
    good0 = finite_positive(tau) & finite_positive(h)
    if good0.sum() < 250:
        return None

    raw_rho = spearman_rank_corr(h[good0], tau[good0])
    base2 = pd.Series(False, index=df.index)
    base_index = df.index[base_mask]
    base2.loc[base_index[good0]] = True

    tau2 = numeric_series(df, c_tau, mask=base2)
    h2 = numeric_series(df, c_h, mask=base2)
    X, feature_names, good_controls = build_baseline_features(df, base2, aux_cols)
    if X is None:
        return {
            'column': c_h,
            'n_used_raw': int(good0.sum()),
            'spearman_raw_hfactor_vs_tauE': float(raw_rho),
            'residual_check_available': False,
            'reason': 'not enough positive/variable baseline controls for residual H-factor test',
            'interpretation': 'Raw H vs raw tau_E is confounded by device and operating scale; do not use it alone.',
        }

    tau3 = tau2[good_controls]
    h3 = h2[good_controls]
    log_tau = np.log(tau3)
    log_h = np.log(h3)
    tau_resid = log_residual(log_tau, X)
    h_resid = log_residual(log_h, X)

    rho_h_vs_tau_resid = spearman_rank_corr(log_h, tau_resid)
    rho_h_resid_vs_tau_resid = spearman_rank_corr(h_resid, tau_resid)
    boot = bootstrap_spearman(log_h, tau_resid)

    devices3 = None
    if device_col is not None and device_col in df.columns:
        dev_all = df.loc[base2, device_col].astype(str).to_numpy()
        devices3 = dev_all[good_controls]
    perm_global = permutation_spearman_pvalue(log_h, tau_resid, None, alternative='greater')
    perm_device = permutation_spearman_pvalue(log_h, tau_resid, devices3, alternative='greater') if devices3 is not None else None
    strat = device_stratified_summary(log_h, tau_resid, devices3)

    device_ok = True
    if strat is not None and strat['n_devices_used'] >= 2:
        device_ok = strat['weighted_mean_rho'] > 0.10 and strat['fraction_positive'] >= 0.55

    p_ok = np.isfinite(perm_global['p_value']) and perm_global['p_value'] <= 0.05
    if rho_h_vs_tau_resid > 0.25 and boot['ci95'][0] > 0 and p_ok and device_ok:
        verdict = 'support_like_hfactor_tracks_residual_confinement'
    elif rho_h_vs_tau_resid > 0.10 and (p_ok or boot['ci95'][0] > -0.02):
        verdict = 'weak_support_hfactor_tracks_residual_confinement'
    elif rho_h_vs_tau_resid < -0.10:
        verdict = 'against_hfactor_residual_check'
    else:
        verdict = 'hfactor_residual_check_inconclusive'

    return {
        'column': c_h,
        'n_used_raw': int(good0.sum()),
        'spearman_raw_hfactor_vs_tauE': float(raw_rho),
        'residual_check_available': True,
        'n_used_residual': int(len(tau3)),
        'baseline_features': feature_names,
        'spearman_hfactor_vs_residual_tauE': float(rho_h_vs_tau_resid),
        'bootstrap_hfactor_vs_residual_tauE': boot,
        'permutation_global_positive': perm_global,
        'permutation_device_stratified_positive': perm_device,
        'device_stratified_summary': strat,
        'spearman_residual_hfactor_vs_residual_tauE': float(rho_h_resid_vs_tau_resid),
        'verdict': verdict,
        'interpretation': 'Positive means H-factor identifies confinement above baseline machine/operating scaling; still a proxy, not direct eta/s.',
    }


def rank_hfactor_result(result: dict | None) -> tuple[int, float, int]:
    if not result:
        return (-1, float('-inf'), 0)
    order = {
        'support_like_hfactor_tracks_residual_confinement': 4,
        'weak_support_hfactor_tracks_residual_confinement': 3,
        'hfactor_residual_check_inconclusive': 2,
        'against_hfactor_residual_check': 1,
    }
    rho = result.get('spearman_hfactor_vs_residual_tauE', result.get('spearman_raw_hfactor_vs_tauE', float('-inf')))
    n = result.get('n_used_residual', result.get('n_used_raw', 0))
    return (order.get(result.get('verdict'), 0), float(rho) if np.isfinite(rho) else float('-inf'), int(n))


def main():
    try:
        table = osf_find_download(['drwcq','hrqcf'], r'.*\.(xlsx|xls|csv|txt|dat)$', OUT/'HDB_public_table')
        df = read_public_table(table)
    except Exception as e:
        out = structured_report('FR7','source_unavailable_or_unparseable', reason=repr(e), attempted_nodes=['drwcq','hrqcf'])
        json_dump(out, OUT/'fr7_report.json'); print(json.dumps(out, indent=2)); return

    df = repair_header_if_needed(df, ['TAU','AMIN','RGEO','H98','H89','TOK'])
    c_tau = best_matching_column_fuzzy(df, TAU_CANDS)
    c_a = best_matching_column_fuzzy(df, A_CANDS)
    c_dev = best_matching_column_fuzzy(df, DEVICE_CANDS)
    hfactor_cols = matching_columns_fuzzy(df, HFACTOR_CANDS)
    c_h = hfactor_cols[0] if hfactor_cols else None
    independent_cols = [c for c in matching_columns_fuzzy(df, INDEPENDENT_TRANSPORT_CANDS) if c not in {c_tau, c_a}]
    c_ind = independent_cols[0] if independent_cols else None
    aux_cols = {
        'R': best_matching_column_fuzzy(df, R_CANDS),
        'a': c_a,
        'Ip': best_matching_column_fuzzy(df, IP_CANDS),
        'Bt': best_matching_column_fuzzy(df, BT_CANDS),
        'ne': best_matching_column_fuzzy(df, NE_CANDS),
        'power': best_matching_column_fuzzy(df, POWER_CANDS),
        'kappa': best_matching_column_fuzzy(df, KAPPA_CANDS),
        'meff': best_matching_column_fuzzy(df, MEFF_CANDS),
        'q95': best_matching_column_fuzzy(df, Q95_CANDS),
    }

    if c_tau is None or c_a is None:
        out = structured_report('FR7','not_executable_missing_columns', n_rows=int(len(df)),
                                columns_found={'tauE':c_tau,'a':c_a,'device':c_dev,'h_factor_first':c_h,
                                               'h_factor_all':hfactor_cols,'independent_transport_all':independent_cols, **aux_cols},
                                sample_columns=[str(c) for c in list(df.columns)[:100]], verdict='no_physics_verdict')
        json_dump(out, OUT/'fr7_report.json'); print(json.dumps(out, indent=2)); return

    tau_s = pd.to_numeric(df[c_tau], errors='coerce')
    a_s = pd.to_numeric(df[c_a], errors='coerce')
    mask = (tau_s > 0) & (a_s > 0)
    tau = tau_s[mask].to_numpy(float)
    a = a_s[mask].to_numpy(float)
    if len(tau) < 100:
        out = structured_report('FR7','not_executable_too_few_rows', n_valid=int(len(tau)), columns_found={'tauE':c_tau,'a':c_a})
        json_dump(out, OUT/'fr7_report.json'); print(json.dumps(out, indent=2)); return

    chi_proxy = a*a/tau
    mkss_proxy = chi_proxy/CHI_MIN
    rho_raw = spearman_rank_corr(mkss_proxy, tau)
    boot = bootstrap_spearman(mkss_proxy, tau)
    perm_mech = permutation_spearman_pvalue(mkss_proxy, tau, None, alternative='less')

    # Report the tautological baseline explicitly: log(M)=2log(a)-log(tau)-const.
    log_tau = np.log(tau); log_a = np.log(a); log_m = np.log(mkss_proxy)
    residual_tau_after_size = log_residual(log_tau, log_a.reshape(-1,1))
    residual_m_after_size = log_residual(log_m, log_a.reshape(-1,1))
    rho_size_controlled = spearman_rank_corr(residual_m_after_size, residual_tau_after_size)

    hfactor_results = []
    for col in hfactor_cols:
        res = residual_hfactor_check(df, mask, c_tau, col, aux_cols, c_dev)
        if res is not None:
            hfactor_results.append(res)
    hfactor_results = sorted(hfactor_results, key=rank_hfactor_result, reverse=True)
    hfactor_result = hfactor_results[0] if hfactor_results else None

    independent_transport_results = []
    for col in independent_cols:
        if col in {c_tau, c_a}:
            continue
        q = pd.to_numeric(df.loc[mask, col], errors='coerce').to_numpy(float)
        qm = np.isfinite(q) & (q > 0)
        if qm.sum() >= 100:
            independent_transport_results.append({
                'column': col,
                'n_used': int(qm.sum()),
                'spearman_independent_transport_vs_tauE': float(spearman_rank_corr(q[qm], tau[qm])),
                'bootstrap': bootstrap_spearman(q[qm], tau[qm]),
                'interpretation': 'stronger than chi_proxy only if column is a real transport/viscosity/diffusivity measure independent of tau_E',
            })
    independent_transport_result = independent_transport_results[0] if independent_transport_results else None

    fig, ax = plt.subplots(figsize=(7,4.6))
    ax.scatter(mkss_proxy, tau, s=5, alpha=.18)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('M_KSS proxy = (a^2/tau_E)/chi_min')
    ax.set_ylabel('tau_E')
    ax.grid(alpha=.25)
    save_plot(fig, OUT/'fr7_proxy_plot.png'); plt.close(fig)

    h_verdict = (hfactor_result or {}).get('verdict')
    if independent_transport_result is not None:
        verdict = 'candidate_independent_transport_test_available_review_column_semantics'
        confirmation_level = 'review needed; independent transport-like column found'
    elif h_verdict == 'support_like_hfactor_tracks_residual_confinement':
        verdict = 'support_like_residual_hfactor_plus_mechanical_mkss'
        confirmation_level = 'moderate proxy support; H-factor tracks residual tau_E after baseline controls and device-stratified/null checks where available, but no direct eta/s column'
    elif h_verdict == 'weak_support_hfactor_tracks_residual_confinement':
        verdict = 'weak_support_residual_hfactor_plus_mechanical_mkss'
        confirmation_level = 'weak proxy support; improved H-factor check is positive but not decisive'
    elif h_verdict == 'against_hfactor_residual_check':
        verdict = 'no_clean_support_hfactor_residual_negative_mechanical_mkss_only'
        confirmation_level = 'not confirmed; residual H-factor check is negative'
    elif rho_raw < -0.2:
        verdict = 'weak_mechanical_support_expected_anticorrelation_only'
        confirmation_level = 'not confirmed; only tautological chi_proxy anticorrelation is present'
    else:
        verdict = 'no_support_for_expected_anticorrelation'
        confirmation_level = 'not confirmed'

    out = structured_report(
        'FR7','ok', n_used=int(len(tau)), chi_min_m2_s=float(CHI_MIN),
        columns_found={'tauE':c_tau,'a':c_a,'device':c_dev,'h_factor_first':c_h,
                       'h_factor_all':hfactor_cols,'independent_transport_all':independent_cols, **aux_cols},
        expected_sign='negative for M_KSS margin vs tau_E: lower transport margin should accompany larger confinement time',
        spearman_mkss_proxy_vs_tauE=float(rho_raw), bootstrap_spearman=boot,
        permutation_mkss_proxy_vs_tauE_negative=perm_mech,
        size_controlled_residual_spearman=float(rho_size_controlled),
        size_controlled_note='Do not interpret as independent evidence: log(M_proxy)=2log(a)-log(tau_E)-const, so residualization can create near-deterministic anticorrelation.',
        hfactor_result=hfactor_result, hfactor_results=hfactor_results,
        independent_transport_result=independent_transport_result,
        independent_transport_results=independent_transport_results,
        verdict=verdict,
        confirmation_level=confirmation_level,
        caveat='A decisive FR7 public test needs an M_KSS/edge-transport proxy not algebraically built from tau_E, or published eta/s/chi_edge columns. H-factor residual tests reduce confounding but remain proxy evidence.'
    )
    json_dump(out, OUT/'fr7_report.json'); print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
