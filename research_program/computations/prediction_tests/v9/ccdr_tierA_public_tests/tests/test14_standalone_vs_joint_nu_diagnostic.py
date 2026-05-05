#!/usr/bin/env python3
from _common_public import *


def main():
    args=build_parser('T14 standalone-vs-joint ν diagnostic with high-z quality controls').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T14',['OP11','P36'],'Diagnose standalone MOND-sequence instability by comparing SPARC stability with raw vs quality-cut high-z proxy.')
    res['prediction_names']=['OP11 — MOND-sequence standalone extractor instability','P36 — MOND-sequence ν extractor / CL1']
    res['falsification_logic']={'confirm_like':'SPARC local anchor is stable while high-z proxy instability is traceable to explicit quality/mapping cuts','falsify_like':'no stable local or high-z ν-like extraction is possible'}
    spaths,att=load_sparc_rotmods(cache,timeout=args.timeout,force=args.force,allow_large=True); res['data_sources'].extend(att)
    sparc_all=fit_sparc_a0(spaths)
    sparc_sub=fit_sparc_a0(spaths,max_galaxies=50)
    hdf,att2=load_kmos_or_kross(cache,timeout=args.timeout,allow_large=args.allow_large,force=args.force,max_rows=args.max_rows); res['data_sources'].extend(att2)
    high=highz_acceleration_summary(hdf) if hdf is not None else {'error':'no_highz_catalog'}
    high_boot=highz_group_bootstrap_summary(hdf, seed=args.seed) if hdf is not None else {'error':'no_highz_catalog'}
    ratio=None
    if sparc_all.get('a0_best_m_s2') and sparc_sub.get('a0_best_m_s2'):
        ratio=float(sparc_sub['a0_best_m_s2']/sparc_all['a0_best_m_s2'])
    raw_mean=high.get('raw',{}).get('mean_a_proxy_m_s2') if isinstance(high,dict) else None
    q_mean=high.get('quality_cut',{}).get('mean_a_proxy_m_s2') if isinstance(high,dict) else None
    high_cut_shift=float(q_mean/raw_mean) if raw_mean and q_mean else None
    res['metrics']={'sparc_all':sparc_all,'sparc_first50':sparc_sub,'sparc_subset_ratio':ratio,'highz_summary':high,'highz_group_bootstrap':high_boot,'quality_to_raw_highz_mean_ratio': high_cut_shift}
    if ratio and 0.7 < ratio < 1.3 and isinstance(high,dict) and not high.get('error'):
        res['status']='methodology_finding'
    elif ratio and not (0.7 < ratio < 1.3):
        res['status']='local_anchor_unstable'
    elif isinstance(high,dict) and high.get('error'):
        res['status']='data_limited'
    else:
        res['status']='null'
    write_result(res,outdir)
if __name__=='__main__': main()
