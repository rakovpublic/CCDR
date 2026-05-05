#!/usr/bin/env python3
from _common_public import *


def main():
    args=build_parser('T13 three-point ν extractor: SPARC + high-z + SN').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T13',['P36','CL1'],'Combine local SPARC a0, quality-cut high-z acceleration proxy, and Pantheon+ ν-like audit into a ν consistency diagnostic.')
    res['prediction_names']=['P36 — MOND-sequence ν extractor / CL1']
    res['falsification_logic']={'confirm_like':'quality-cut high-z, local SPARC, and SN/BAO ν brackets agree within order-of-magnitude tolerance without bound hits','falsify_like':'ν brackets disagree by orders of magnitude or one bracket is bound-dominated'}
    spaths,att=load_sparc_rotmods(cache,timeout=args.timeout,force=args.force,allow_large=True); res['data_sources'].extend(att)
    sparc=fit_sparc_a0(spaths,max_galaxies=80)
    sparc_points=sparc_point_accelerations(spaths,max_galaxies=80)
    pdf,att2=load_pantheon(cache,timeout=args.timeout,force=args.force,max_rows=None); res['data_sources'].extend(att2)
    pcov,catt=load_pantheon_covariance(cache,timeout=args.timeout,force=args.force); res['data_sources'].extend(catt)
    sn=fit_pantheon_nu_cov_like(pdf,pcov)
    hdf,att3=load_kmos_or_kross(cache,timeout=args.timeout,allow_large=args.allow_large,force=args.force,max_rows=args.max_rows); res['data_sources'].extend(att3)
    high=highz_acceleration_summary(hdf) if hdf is not None else {'error':'no_highz_catalog'}
    high_boot=highz_group_bootstrap_summary(hdf, seed=args.seed) if hdf is not None else {'error':'no_highz_catalog'}
    q=high.get('quality_cut', {}) if isinstance(high,dict) else {}
    high_mean=q.get('mean_a_proxy_m_s2') or high.get('raw',{}).get('mean_a_proxy_m_s2') if isinstance(high,dict) else None
    zmean=q.get('z_mean') or high.get('raw',{}).get('z_mean') if isinstance(high,dict) else None
    nu_gap=None
    if high_mean and sparc.get('a0_best_m_s2') and zmean:
        nu_gap=float(np.log(high_mean/sparc['a0_best_m_s2'])/(3*np.log1p(zmean)))
    # Only use the SN bracket if it did not hit the bound.
    sn_nu=sn.get('nu_like') if isinstance(sn,dict) and not sn.get('hit_nu_bound') else None
    vals=[v for v in [nu_gap, sn_nu] if v is not None and np.isfinite(v) and v>0]
    res['metrics']={'sparc_local':sparc,'sparc_point_distribution':sparc_points,'highz_proxy':high,'highz_group_bootstrap':high_boot,'sn_nu_like_covariance':sn,'nu_gap_proxy_quality_cut':nu_gap,'sn_bracket_used': sn_nu is not None,'note':'ν_gap_proxy remains a screening diagnostic; v9.6 refuses to count bound-hit SN covariance proxy as a triangulation bracket.'}
    if len(vals)>=2:
        ratio=max(vals)/min(vals); res['metrics']['positive_nu_bracket_ratio']=float(ratio)
        res['status']='suggestive' if ratio<10 else 'inconclusive'
    elif nu_gap is not None:
        res['status']='single_bracket_only'
        res['warnings'].append('Only MOND-sequence/high-z ν proxy is usable; SN proxy is bound-hit or unavailable.')
    else:
        res['status']='data_limited'
    if isinstance(sn,dict) and sn.get('hit_nu_bound'):
        res['warnings'].append('SN-only ν-like proxy hit its conservative bound and is excluded from bracket-ratio scoring.')
    write_result(res,outdir)
if __name__=='__main__': main()
