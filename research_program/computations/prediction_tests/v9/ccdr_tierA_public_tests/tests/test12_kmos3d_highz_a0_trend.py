#!/usr/bin/env python3
from _common_public import *


def main():
    args=build_parser('T12 KMOS3D/KROSS high-z a0 trend proxy with quality cuts').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T12',['§6.1b','P36'],'Estimate high-z acceleration proxy from public KMOS3D/KROSS catalogue products and test trend with quality cuts.')
    res['prediction_names']=['§6.1b — high-z a0 → cH0','P36 — MOND-sequence ν extractor / CL1']
    res['falsification_logic']={'confirm_like':'quality-cut high-z acceleration proxy is above a matched local SPARC scale and increases with redshift','falsify_like':'quality-cut proxy trend vanishes/decreases or offset disappears under controls'}
    df,att=load_kmos_or_kross(cache,timeout=args.timeout,allow_large=args.allow_large,force=args.force,max_rows=args.max_rows); res['data_sources'].extend(att)
    if df is None:
        res['warnings'].append('No parseable compact KMOS3D/KROSS catalogue found; KMOS cubes require --allow-large and custom cube extraction.')
        write_result(res,outdir); return
    summ=highz_acceleration_summary(df)
    res['metrics']['highz_acceleration'] = summ
    res['metrics']['galaxy_or_object_bootstrap'] = highz_group_bootstrap_summary(df, seed=args.seed)
    survey_model = highz_split_by_survey_model(df, seed=args.seed)
    res['metrics']['survey_split_hierarchical_object_model'] = survey_model
    res['metrics']['prediction_split_interpretation']={'P36a_highz_offset':'evaluated independently from trend','P36b_monotonic_a0_z_trend':'requires row-level significance and object-bootstrap support','survey_split_groups': list((survey_model.get('groups') or {}).keys()) if isinstance(survey_model, dict) else []}
    if summ.get('error'):
        res['warnings'].append('Catalogue parsed but high-z acceleration columns were not identifiable.')
        res['status']='data_limited'; write_result(res,outdir); return
    q=summ.get('quality_cut', {})
    raw=summ.get('raw', {})
    slope=q.get('log_g_vs_z_coef', [None])[0] if q.get('log_g_vs_z_coef') else None
    pval=q.get('spearman_z_g', {}).get('pvalue') if isinstance(q.get('spearman_z_g'), dict) else None
    mean=q.get('mean_a_proxy_m_s2')
    boot=res['metrics'].get('galaxy_or_object_bootstrap', {})
    boot_p16=boot.get('slope_boot_p16') if isinstance(boot, dict) else None
    # Keep the status conservative: high mean alone is weak; trend needs significance.
    if q.get('n',0) < 30:
        res['status']='data_limited'
    elif slope is not None and slope > 0 and pval is not None and pval < 0.05 and (boot_p16 is None or boot_p16 > 0):
        res['status']='suggestive'
        res['metrics']['prediction_split_interpretation']['P36a_highz_offset_status']='support'
        res['metrics']['prediction_split_interpretation']['P36b_monotonic_a0_z_trend_status']='suggestive'
    elif slope is not None and slope > 0 and pval is not None and pval < 0.05:
        res['status']='row_level_suggestive_bootstrap_unconfirmed'
        res['metrics']['prediction_split_interpretation']['P36a_highz_offset_status']='partial'
        res['metrics']['prediction_split_interpretation']['P36b_monotonic_a0_z_trend_status']='bootstrap_unconfirmed'
        res['warnings'].append('Row-level trend is positive, but galaxy/object bootstrap does not keep the slope positive at p16.')
    elif mean is not None and mean > 1.2e-10:
        res['status']='offset_only_no_robust_trend'
        res['metrics']['prediction_split_interpretation']['P36a_highz_offset_status']='weak_partial_support'
        res['metrics']['prediction_split_interpretation']['P36b_monotonic_a0_z_trend_status']='not_confirmed'
        res['warnings'].append('High-z mean acceleration is above local Milgrom scale, but redshift trend is not significant under quality cuts.')
    else:
        res['status']='null'
        res['metrics']['prediction_split_interpretation']['P36a_highz_offset_status']='not_supported'
        res['metrics']['prediction_split_interpretation']['P36b_monotonic_a0_z_trend_status']='not_supported'
    res['metrics']['interpretation']={'raw_n':raw.get('n'),'quality_n':q.get('n'),'quality_slope_log10g_per_z':slope,'quality_spearman_pvalue':pval,'group_bootstrap':res['metrics'].get('galaxy_or_object_bootstrap'),'survey_split_summary':survey_model,'bootstrap_slope_p16_positive': bool(boot_p16 is not None and boot_p16 > 0),'offset_above_milgrom': bool(mean is not None and mean > 1.2e-10),'trend_confirmed': bool(slope is not None and slope>0 and pval is not None and pval<0.05 and (boot_p16 is None or boot_p16>0))}
    write_result(res,outdir)
if __name__=='__main__': main()
