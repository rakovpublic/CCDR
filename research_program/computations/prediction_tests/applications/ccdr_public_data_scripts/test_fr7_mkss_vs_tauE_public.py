from __future__ import annotations

"""FR7 public-data proxy test: M_KSS-like confinement margin vs energy confinement time.

Prediction in note: M_KSS should correlate with tau_E across tokamak databases.
Careful interpretation: M_KSS is a *margin above a minimum transport floor*;
better confinement should mean smaller margin and larger tau_E, i.e. an expected
negative correlation if the proxy is chi/chi_min.  However chi_proxy=a^2/tau_E
contains tau_E by construction, so the script reports this as weak/mechanical
unless an independent transport or confinement-quality column is found.

Patch 13 improvement: the H-factor check is no longer raw H vs raw tau_E only.
Raw H-factor can be anti-correlated with tau_E because device size, field,
density, heating power, isotope mass, and shape dominate tau_E.  The improved
check compares H-factor against *baseline-scaling residual confinement*:
    log(tau_E) ~ log(R), log(a), log(Ip), log(Bt), log(ne), log(P), log(kappa), ...
Then it tests whether log(H) tracks the residual.  This is much closer to the
engineering question: does the independent confinement-quality factor identify
shots with better confinement than expected for their machine/operating point?
"""
import json
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
    """Build log-feature matrix for confinement baseline controls.

    Returns X, names, valid_mask_within_base.  Uses only finite-positive controls.
    Drops controls with too little finite coverage or nearly no variation.
    """
    raw_features: list[np.ndarray] = []
    names: list[str] = []
    local_valid = np.ones(int(base_mask.sum()), dtype=bool)
    candidate_names = ['R','a','Ip','Bt','ne','power','kappa','meff','q95']
    for name in candidate_names:
        arr = numeric_series(df, columns.get(name), mask=base_mask)
        good = finite_positive(arr)
        if arr is None or good is None or good.sum() < 250:
            continue
        vals = np.log(arr)
        # Reject constants / almost constants.  They make residuals unstable.
        if np.nanstd(vals[good]) < 1e-6:
            continue
        raw_features.append(vals)
        names.append(name)
        local_valid &= good
    if not raw_features or local_valid.sum() < 250:
        return None, names, local_valid
    X = np.column_stack([f[local_valid] for f in raw_features])
    return X, names, local_valid


def residual_hfactor_check(df: pd.DataFrame, base_mask, c_tau: str, c_h: str | None, aux_cols: dict[str, str | None]) -> dict | None:
    """Improved H-factor test.

    Raw H vs raw tau is reported, but the decisive quantity is H vs residual tau
    after baseline controls.  If H is a valid confinement enhancement factor, it
    should correlate positively with residual tau_E, even if raw H vs raw tau_E
    is weak or negative due cross-machine confounding.
    """
    if c_h is None:
        return None
    tau = numeric_series(df, c_tau, mask=base_mask)
    h = numeric_series(df, c_h, mask=base_mask)
    if tau is None or h is None:
        return None
    good0 = finite_positive(tau) & finite_positive(h)
    if good0.sum() < 250:
        return None

    # Raw diagnostic.
    raw_rho = spearman_rank_corr(h[good0], tau[good0])

    # Baseline residual diagnostic using available machine/operating controls.
    # Rebuild a mask that includes H and tau first; then add controls.
    base2 = base_mask.copy()
    base2.loc[base2.index[base2]] = good0  # align within base subset
    tau2 = numeric_series(df, c_tau, mask=base2)
    h2 = numeric_series(df, c_h, mask=base2)
    cols_for_baseline = aux_cols.copy()
    X, feature_names, good_controls = build_baseline_features(df, base2, cols_for_baseline)
    if X is None:
        return {
            'column': c_h,
            'n_used_raw': int(good0.sum()),
            'spearman_raw_hfactor_vs_tauE': float(raw_rho),
            'residual_check_available': False,
            'reason': 'not enough positive/variable baseline controls for residual H-factor test',
            'interpretation': 'Raw H vs raw tau_E is confounded by device and operating scale; do not use it alone.'
        }

    tau3 = tau2[good_controls]
    h3 = h2[good_controls]
    log_tau = np.log(tau3)
    log_h = np.log(h3)
    tau_resid = log_residual(log_tau, X)

    # Two views: H vs residual tau, and residual H after same controls vs residual tau.
    h_resid = log_residual(log_h, X)
    rho_h_vs_tau_resid = spearman_rank_corr(log_h, tau_resid)
    rho_h_resid_vs_tau_resid = spearman_rank_corr(h_resid, tau_resid)
    boot = bootstrap_spearman(log_h, tau_resid)

    if rho_h_vs_tau_resid > 0.25 and boot['ci95'][0] > 0:
        verdict = 'support_like_hfactor_tracks_residual_confinement'
    elif rho_h_vs_tau_resid > 0.10:
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
        'spearman_residual_hfactor_vs_residual_tauE': float(rho_h_resid_vs_tau_resid),
        'verdict': verdict,
        'interpretation': 'Use residual check over raw H-vs-tau: positive means H-factor identifies confinement above baseline machine/operating scaling.'
    }


def main():
    try:
        table = osf_find_download(['drwcq','hrqcf'], r'.*\.(xlsx|xls|csv|txt|dat)$', OUT/'HDB_public_table')
        df = read_public_table(table)
    except Exception as e:
        out = structured_report('FR7','source_unavailable_or_unparseable', reason=repr(e), attempted_nodes=['drwcq','hrqcf'])
        json_dump(out, OUT/'fr7_report.json'); print(json.dumps(out, indent=2)); return

    df = repair_header_if_needed(df, ['TAU','AMIN','H98','H89','TOK'])
    c_tau = best_matching_column_fuzzy(df, TAU_CANDS)
    c_a = best_matching_column_fuzzy(df, A_CANDS)
    c_dev = best_matching_column_fuzzy(df, DEVICE_CANDS)
    c_h = best_matching_column_fuzzy(df, HFACTOR_CANDS)
    c_ind = best_matching_column_fuzzy(df, INDEPENDENT_TRANSPORT_CANDS)
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
                                columns_found={'tauE':c_tau,'a':c_a,'device':c_dev,'h_factor':c_h,'independent_transport':c_ind, **aux_cols},
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

    # Report the tautological baseline explicitly: log(M)=2log(a)-log(tau)-const.
    log_tau = np.log(tau); log_a = np.log(a); log_m = np.log(mkss_proxy)
    residual_tau_after_size = log_residual(log_tau, log_a.reshape(-1,1))
    residual_m_after_size = log_residual(log_m, log_a.reshape(-1,1))
    rho_size_controlled = spearman_rank_corr(residual_m_after_size, residual_tau_after_size)

    hfactor_result = residual_hfactor_check(df, mask, c_tau, c_h, aux_cols)

    independent_transport_result = None
    if c_ind is not None and c_ind not in {c_tau, c_a}:
        q = pd.to_numeric(df.loc[mask, c_ind], errors='coerce').to_numpy(float)
        qm = np.isfinite(q) & (q > 0)
        if qm.sum() >= 100:
            independent_transport_result = {
                'column': c_ind,
                'n_used': int(qm.sum()),
                'spearman_independent_transport_vs_tauE': float(spearman_rank_corr(q[qm], tau[qm])),
                'interpretation': 'stronger than chi_proxy only if column is a real transport/viscosity/diffusivity measure independent of tau_E',
            }

    # Plot proxy for diagnostics.
    fig, ax = plt.subplots(figsize=(7,4.6))
    ax.scatter(mkss_proxy, tau, s=5, alpha=.18)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('M_KSS proxy = (a^2/tau_E)/chi_min')
    ax.set_ylabel('tau_E')
    ax.grid(alpha=.25)
    save_plot(fig, OUT/'fr7_proxy_plot.png'); plt.close(fig)

    # Improved verdict logic. Mechanical chi proxy alone cannot confirm FR7.
    h_verdict = (hfactor_result or {}).get('verdict')
    if independent_transport_result is not None:
        verdict = 'candidate_independent_transport_test_available_review_column_semantics'
        confirmation_level = 'review needed; independent transport-like column found'
    elif h_verdict == 'support_like_hfactor_tracks_residual_confinement':
        verdict = 'support_like_residual_hfactor_plus_mechanical_mkss'
        confirmation_level = 'moderate proxy support; H-factor tracks residual tau_E after baseline controls, but no direct eta/s column'
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
        columns_found={'tauE':c_tau,'a':c_a,'device':c_dev,'h_factor':c_h,'independent_transport':c_ind, **aux_cols},
        expected_sign='negative for M_KSS margin vs tau_E: lower transport margin should accompany larger confinement time',
        spearman_mkss_proxy_vs_tauE=float(rho_raw), bootstrap_spearman=boot,
        size_controlled_residual_spearman=float(rho_size_controlled),
        size_controlled_note='Do not interpret as independent evidence: log(M_proxy)=2log(a)-log(tau_E)-const, so residualization can create near-deterministic anticorrelation.',
        hfactor_result=hfactor_result, independent_transport_result=independent_transport_result,
        verdict=verdict,
        confirmation_level=confirmation_level,
        caveat='A decisive FR7 public test needs an M_KSS/edge-transport proxy not algebraically built from tau_E, or published eta/s/chi_edge columns. Improved H-factor residual check reduces, but does not remove, proxy limitations.'
    )
    json_dump(out, OUT/'fr7_report.json'); print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
