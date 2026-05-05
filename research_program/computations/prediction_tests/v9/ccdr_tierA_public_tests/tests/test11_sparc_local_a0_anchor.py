#!/usr/bin/env python3
from _common_public import *

def main():
    args=build_parser('T11 SPARC local a0 anchor').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T11',['§6.1a'],'Refit a local MOND/RAR acceleration scale from public SPARC rotmod files.')
    res['falsification_logic']={'confirm_like':'best a0 is close to Milgrom 1.2e-10 m/s² with small RMS','falsify_like':'no stable local a0 emerges from public SPARC mass models'}
    paths,att=load_sparc_rotmods(cache,timeout=args.timeout,force=args.force,allow_large=True); res['data_sources'].extend(att)
    fit=fit_sparc_a0(paths,max_galaxies=None)
    res['metrics']['sparc_a0_fit']=fit
    res['metrics']['sparc_robustness_matrix']=sparc_robustness_matrix(paths,seed=args.seed)
    res['metrics']['sparc_metadata_split_status']=res['metrics']['sparc_robustness_matrix'].get('physical_metadata_splits',{}).get('status') if isinstance(res['metrics'].get('sparc_robustness_matrix'),dict) else None
    a0=fit.get('a0_best_m_s2') if isinstance(fit,dict) else None
    if a0 and np.isfinite(a0):
        ratio=a0/1.2e-10; res['metrics']['a0_over_milgrom']=float(ratio)
        res['status']='confirm_like' if 0.5<ratio<2.0 else 'suggestive' if 0.2<ratio<5 else 'null'
    write_result(res,outdir)
if __name__=='__main__': main()
