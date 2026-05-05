#!/usr/bin/env python3
from _common_public import *


def main():
    args = build_parser('T02 DESI DR2-only secular Λ-drift covariance screen').parse_args()
    cache = ensure_dir(args.cache); outdir = ensure_dir(args.outdir)
    res = result_template('T02', ['P39'], 'DR2-only BAO covariance-aware trend screen for secular Λ-drift from bulk cooling.')
    res['prediction_names'] = ['P39 — secular Λ-drift from bulk cooling']
    res['falsification_logic'] = {
        'confirm_like': 'DR2-only covariance-aware residual trend has the predicted sign and significant Δχ² improvement over a constant residual.',
        'falsify_like': 'DR2-only covariance-aware residual proxy has zero/opposite trend or sign is not robust across observables.'
    }
    paths, att = github_download_matching('CobayaSampler','bao_data', cache, [r'desi_bao_dr2', r'\.dat$', r'\.txt$'], timeout=args.timeout, force=args.force, max_files=100)
    res['data_sources'].extend(att)
    covtrend = summarize_dr2_bao_cov_likelihood(paths, max_rows=args.max_rows)
    trend = summarize_dr2_bao_trend(paths, max_rows=args.max_rows)
    res['metrics']['desi_dr2_covariance_gls'] = covtrend
    res['metrics']['desi_dr2_mean_vector_crosscheck'] = trend
    obs = desi_dr2_observablewise_gls_stability(paths, max_rows=args.max_rows)
    res['metrics']['desi_dr2_observablewise_stability'] = obs
    model_like = fit_desi_dr2_simplified_rvm_likelihood(paths, max_rows=args.max_rows)
    res['metrics']['desi_dr2_model_likelihood'] = model_like
    dchi = covtrend.get('delta_chi2_linear_vs_constant') if isinstance(covtrend, dict) else None
    zscore = covtrend.get('slope_zscore') if isinstance(covtrend, dict) else None
    slope = covtrend.get('slope_norm_per_z') if isinstance(covtrend, dict) else None
    if isinstance(model_like, dict) and (model_like.get('nu_grid_bound_hit') or model_like.get('schema') == 'fallback_unknown_schema'):
        res['status']='model_likelihood_diagnostic_bound_or_schema_limited'
        res['warnings'].append('Schema-based DESI model likelihood hit ν grid boundary or schema was not explicit; trend screens are not counted as evidence.')
    elif isinstance(covtrend, dict) and covtrend.get('error'):
        # Fallback to old mean-vector screen, but label it correctly.
        rho = trend.get('spearman', {}).get('rho')
        pval = trend.get('spearman', {}).get('pvalue')
        if trend.get('n_points', 0) >= 4:
            res['status'] = 'mean_vector_fallback_null' if not (pval is not None and pval < 0.05) else 'mean_vector_fallback_suggestive'
        else:
            res['status'] = 'data_limited'
        res['warnings'].append('DESI DR2 covariance pair was not usable; fell back to v9.4 mean-vector screen.')
    elif dchi is not None and zscore is not None and abs(zscore) >= 2 and dchi > 4:
        # Sign convention remains provisional; negative slope was the v9.4 predicted direction.
        if slope is not None and slope < 0 and isinstance(obs, dict) and obs.get('robust_negative_sign'):
            res['status'] = 'covariance_suggestive'
        elif slope is not None and slope < 0:
            res['status'] = 'covariance_diagnostic_observablewise_unstable'
            res['warnings'].append('All-row covariance trend is negative, but observable-wise stability failed or labels were unavailable.')
        else:
            res['status'] = 'covariance_tension_or_null'
    elif dchi is not None:
        res['status'] = 'null'
    else:
        res['status'] = 'data_limited'
    res['notes'].append('v9.8 includes schema-based DESI model-vector likelihood plus leave-one-out checks; trend screens are diagnostic only unless model likelihood and observable stability agree.')
    write_result(res, outdir)

if __name__ == '__main__':
    main()
