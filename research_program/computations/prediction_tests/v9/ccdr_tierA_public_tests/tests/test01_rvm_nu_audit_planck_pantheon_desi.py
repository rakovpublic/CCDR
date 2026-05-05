#!/usr/bin/env python3
from _common_public import *


def main():
    args = build_parser('T01 RVM ν audit with Pantheon+ covariance and DESI DR2 BAO public products').parse_args()
    cache = ensure_dir(args.cache); outdir = ensure_dir(args.outdir)
    res = result_template('T01', ['P1','P39','CL3'], 'Audit ν-like RVM signal stability using public Pantheon+ covariance and DESI DR2 BAO covariance products.')
    res['prediction_names'] = ['P1 — RVM BAO scale shift', 'P39 — secular Λ-drift from bulk cooling', 'CL3 — ν triangle/cross-link']
    res['falsification_logic'] = {
        'confirm_like': 'A covariance-aware BAO+SN public-data likelihood prefers a positive, stable ν-like parameter without bound hits.',
        'falsify_like': 'ν-like signal disappears under covariance-aware DR2/SN cuts or is carried by bounded diagnostics.'
    }

    df, att = load_pantheon(cache, timeout=args.timeout, force=args.force, max_rows=None)
    res['data_sources'].extend(att)
    cov, catt = load_pantheon_covariance(cache, timeout=args.timeout, force=args.force)
    res['data_sources'].extend(catt)
    sn_cov = fit_pantheon_nu_cov_like(df, cov)
    res['metrics']['pantheon_sn_covariance_audit'] = sn_cov
    # Keep the old diagonal proxy as a guardrail/debug comparator.
    res['metrics']['pantheon_sn_diagonal_proxy'] = fit_pantheon_nu_like(df)

    paths, att2 = github_download_matching('CobayaSampler','bao_data', cache, [r'desi_bao_dr2', r'\.dat$', r'\.txt$'], timeout=args.timeout, force=args.force, max_files=100)
    res['data_sources'].extend(att2)
    dr2_cov = summarize_dr2_bao_cov_likelihood(paths, max_rows=args.max_rows)
    dr2_trend = summarize_dr2_bao_trend(paths, max_rows=args.max_rows)
    res['metrics']['desi_dr2_covariance_gls_screen'] = dr2_cov
    res['metrics']['desi_dr2_mean_vector_screen'] = dr2_trend
    dr2_obs = desi_dr2_observablewise_gls_stability(paths, max_rows=args.max_rows)
    res['metrics']['desi_dr2_observablewise_stability'] = dr2_obs
    desi_model = fit_desi_dr2_simplified_rvm_likelihood(paths, max_rows=args.max_rows)
    res['metrics']['desi_dr2_model_likelihood'] = desi_model
    res['metrics']['joint_sn_desi_model_likelihood'] = fit_joint_sn_desi_rvm_model_likelihood(df, cov, paths, max_rows=args.max_rows)

    sn_bound = is_bound_hit_metric(sn_cov)
    sn_dchi = sn_cov.get('delta_chi2_vs_nu0') if isinstance(sn_cov, dict) else None
    sn_nu = sn_cov.get('nu_like') if isinstance(sn_cov, dict) else None
    bao_dchi = dr2_cov.get('delta_chi2_linear_vs_constant') if isinstance(dr2_cov, dict) else None
    bao_z = dr2_cov.get('slope_zscore') if isinstance(dr2_cov, dict) else None
    res['metrics']['interpretation_rules'] = {
        'sn_covariance_available': isinstance(sn_cov, dict) and sn_cov.get('mode') == 'full_public_covariance',
        'sn_proxy_bound_hit': bool(sn_bound),
        'sn_delta_chi2_vs_nu0': sn_dchi,
        'sn_nu_like': sn_nu,
        'desi_covariance_delta_chi2_linear_vs_constant': bao_dchi,
        'desi_covariance_slope_zscore': bao_z,
        'desi_observablewise_robust_negative_sign': bool(dr2_obs.get('robust_negative_sign')) if isinstance(dr2_obs, dict) else False,
        'desi_observablewise_n_groups_valid': dr2_obs.get('n_groups_valid') if isinstance(dr2_obs, dict) else None,
        'full_likelihood_run': bool(isinstance(sn_cov, dict) and 'covariance' in str(sn_cov.get('mode')) and isinstance(dr2_cov, dict) and dr2_cov.get('mode')),
        'note': 'v9.7 adds a simplified DESI model-vector likelihood; still not a full Boltzmann/MCMC likelihood.'
    }

    if isinstance(sn_cov, dict) and sn_cov.get('error') and isinstance(dr2_cov, dict) and dr2_cov.get('error'):
        res['status'] = 'data_limited'
    elif sn_bound or (isinstance(desi_model, dict) and (desi_model.get('nu_grid_bound_hit') or desi_model.get('schema') == 'fallback_unknown_schema')):
        res['status'] = 'diagnostic_only'
        res['warnings'].append('SN or DESI ν-like model hit the allowed grid boundary or lacked explicit schema; no RVM evidence claimed.')
    elif sn_dchi is not None and sn_dchi > 4 and sn_nu is not None and sn_nu > 0 and bao_dchi is not None and bao_dchi > 4 and isinstance(dr2_obs, dict) and dr2_obs.get('robust_negative_sign'):
        res['status'] = 'covariance_suggestive'
    elif bao_dchi is not None and bao_dchi > 4 and isinstance(dr2_obs, dict) and not dr2_obs.get('robust_negative_sign'):
        res['status'] = 'covariance_diagnostic_observablewise_unstable'
        res['warnings'].append('DESI all-row covariance trend is not robust across observable groups; no P39/CL3 support claimed.')
    elif sn_dchi is not None or bao_dchi is not None:
        res['status'] = 'covariance_diagnostic_only'
    else:
        res['status'] = 'data_limited'
    res['notes'].append('v9.8 adds schema-based DESI+SN model-likelihood diagnostics with leave-one-out checks; trend screens remain non-confirmatory unless model likelihood is stable and non-bound-hit.')
    write_result(res, outdir)

if __name__ == '__main__':
    main()
